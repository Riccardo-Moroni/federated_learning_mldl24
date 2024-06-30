# %%
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import SGD
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from torch.utils.data import TensorDataset

from shakespeare_utils import shakespeare_data_pruning, tokenize_encode, concat_dict_values
from collections import OrderedDict
import math
import copy

from CharRNN import CharRNN

import sys
sys.path.append('../') # necessary to import from parant directory
from client_selector import ClientSelector

from scipy.optimize import minimize, LinearConstraint

import wandb
wandb.login()


# %%
K = 100

params = {
    'K': K,
    'C': 0.3,
    'B': 100,
    'J': 8,
    'lr_server': 5e-1,
    'lr_client': 5e-1,
    'method': 'MGDA',
    'tau': 1e-3,
    'gamma': 0.1,
    'participation': 'uniform',
    'rounds': 2000,
    'weight_decay':4e-4,
    'crop_amount':4000
}

import sys
sys.path.append('../')
from client_selector import ClientSelector

client_selector = ClientSelector(params)

wandb.init(
    project='fl_shakespeare',
    name=f"federated {params['method']} C:{params['C']} J:{params['J']} lr:{params['lr_client']} partecipation:{params['participation']} weight_decay:{params['weight_decay']} epochs:{params['rounds']}",
    config= params
)

# %%
json_train_path = '../../datasets/shakespeare/train/all_data_niid_0_keep_0_train_9.json'
json_test_path = '../../datasets/shakespeare/test/all_data_niid_0_keep_0_test_9.json'

X_train_pruned, Y_train_pruned, X_test_pruned, Y_test_pruned = shakespeare_data_pruning(json_train_path, json_test_path, crop_amount=params['crop_amount'])
# data is already  split at this point {user1:[2000],...,user100:[2000]}
 
# %%
# define dictionary and chart to int mapping
train_sentence = ' '.join(' '.join(single_user_list) for single_user_list in X_train_pruned.values())
vocab = sorted(set(train_sentence))
vocab.append('<OOV>')
char_to_idx = {char: idx for idx, char in enumerate(vocab)}

# %%
# from dict to nested list
X_train_list = [X_train_pruned[key] for key in sorted(X_train_pruned.keys())]
Y_train_list = [Y_train_pruned[key] for key in sorted(Y_train_pruned.keys())]

X_train_enc, Y_train_enc, X_test_enc, Y_test_enc = [],[],[],[]
for user in tqdm(range(len(X_train_list))):
    X_train_enc.append(tokenize_encode(X_train_list[user], vocab, char_to_idx))
    Y_train_enc.append(tokenize_encode(Y_train_list[user], vocab, char_to_idx))

# We can discard the information about the user for the test data, since the testing is centralized 
X_test_concat = concat_dict_values(X_test_pruned)
Y_test_concat = concat_dict_values(Y_test_pruned)
X_test_enc = np.array(tokenize_encode(X_test_concat, vocab, char_to_idx)) # (100, 2000, 80) 
Y_test_enc = np.array(tokenize_encode(Y_test_concat, vocab, char_to_idx)).squeeze(-1) # (100, 2000, 1) --> (100, 2000,)

# to tensor
X_train_tensor = torch.tensor(X_train_enc, dtype=torch.long).cuda() # (100, 2000, 80)
Y_train_tensor = torch.tensor(Y_train_enc, dtype=torch.long).squeeze(-1).cuda() # (100, 2000, 1) --> (100, 2000,)

X_test_tensor = torch.tensor(X_test_enc, dtype=torch.long).cuda()
Y_test_tensor = torch.tensor(Y_test_enc, dtype=torch.long).cuda()

test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=100)

print(X_train_tensor.shape, Y_train_tensor.shape)
print(X_test_tensor.shape, Y_test_tensor.shape)

# train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)

train_datasets = []
for k in range(K):
    X_k = X_train_tensor[k].cuda()
    Y_k = Y_train_tensor[k].cuda()
    train_datasets.append(TensorDataset(X_k, Y_k))

# %%
model_params = {
    'vocab_size' : len(vocab),
    'embed_dim' : 8,
    'lstm_units' : 256,
}
model = CharRNN(vocab_size = model_params['vocab_size'], embed_dim = model_params['embed_dim'], lstm_units=model_params['lstm_units']).cuda()

criterion = nn.CrossEntropyLoss().cuda()
# server_optimizer = SGD(model.parameters(), lr=params['lr_client'], weight_decay=4e-4)

# %%
# !
def w_norm(coeffs, clients_gradients_normalized):

    w = np.sum([coeffs[i]*clients_gradients_normalized[i] for i in range(len(clients_gradients_normalized))], axis=0)
    w_l2_norm = np.linalg.norm(w)
    gradients = [2 * w.dot(clients_gradients_normalized[i]) for i in range(len(clients_gradients_normalized))]
    return w_l2_norm, gradients

def magical_gradient(clients_gradients): 
    n = len(clients_gradients)
    coeffs = np.full(n, 1/n)

    clients_gradients = list(clients_gradients)  # Convert generator to list

    res = minimize(
        w_norm, 
        coeffs, 
        args = (clients_gradients,),
        jac=True, 
        bounds=[(0.,1.) for _ in range(n)], 
        constraints=[LinearConstraint(A=[[1] * n], lb = 1., ub = 1.)] 
    )

    # this is the gradient produced by the minimization of the convex linear combination (so it's the shortest convex linear combination)
    grad_mo = torch.tensor(sum(res.x[i]*clients_gradients[i] for i in range(n))).cuda()
    return grad_mo

# %%

test_freq = 50

def test(model):
    model.eval()
    test_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss = test_loss / len(test_loader)
    test_accuracy = 100. * correct / total
    print(f'Test Loss: {test_loss:.6f} Acc: {test_accuracy:.2f}%')
    return test_accuracy, test_loss

def client_update(model, k, params):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=params['lr_client'], weight_decay=4e-4)
    loader = DataLoader(train_datasets[k], batch_size=params['B'], shuffle=True)

    client_loss, client_correct, client_total = 0, 0, 0
    i = 0
    client_gradients = []
    for i in range(params['J']):
        
        for batch_idx, (inputs, targets) in enumerate(loader):
            # inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            client_loss += loss.item()
            _, predicted = outputs.max(1)
            client_total += targets.size(0)
            client_correct += predicted.eq(targets).sum().item()

            loss.backward()

            # extract the gradient
            client_grad = []
            for parameter in model.parameters():
                client_grad += parameter.grad.reshape(-1).tolist()
            client_grad = np.array(client_grad)

            client_gradients.append(client_grad)

            optimizer.step()
            i += 1

            if i >= params['J']:
                client_loss = client_loss / params['J']
                client_accuracy = 100. * client_correct / client_total

                client_gradients = np.array(client_gradients)
                client_gradient_avg = np.mean(client_gradients, axis = 0) # average the gradients of the different J steps
                client_gradient_norm = client_gradient_avg / np.linalg.norm(client_gradient_avg)

                return client_accuracy, client_loss, client_gradient_norm
            
    client_gradients = np.array(client_gradients)
    client_gradient_avg = np.mean(client_gradients, axis = 0) # average the gradients of the different J steps
    client_gradient_norm = client_gradient_avg / np.linalg.norm(client_gradient_avg)
            
    client_loss = client_loss / params['J'] # average loss in J steps 
    client_accuracy = 100. * client_correct / client_total

    return client_accuracy, client_loss, client_gradient_norm

def train(model, params):
    server_optimizer = SGD(model.parameters(), lr=params['lr_server'], weight_decay=4e-4)
    server_optimizer.zero_grad(set_to_none=True)

    test_accuracies, test_losses = [], []
    train_accuracies, train_losses = [], []

    for t in range(params['rounds']):
        print(f"Epoch {t+1}")
        round_loss, round_accuracy = 0, 0
        s = client_selector.sample()

        clients_grad_normalized = []
        for k in s:
            client_accuracy, client_loss, client_gradient_norm = client_update(copy.deepcopy(model), k, params)
            round_loss += client_loss
            round_accuracy += client_accuracy
            clients_grad_normalized.append(client_gradient_norm)
        round_loss_avg = round_loss / len(s)
        round_accuracy_avg = round_accuracy / len(s)
        train_accuracies.append(round_loss_avg)
        train_losses.append(round_accuracy_avg)

        grad_mo = magical_gradient(clients_grad_normalized).cuda()

        # distribute the gradient on the parameters
        idx = 0
        for name, par in model.named_parameters():
            shape = tuple(par.data.shape)
            tot_len = np.prod(shape).astype(int) # shape[0]*shape[1]
            par.grad = grad_mo[idx:(idx + tot_len)].reshape(shape).to(torch.float) # setting the gradients!
            
            idx += tot_len

        server_optimizer.step()

        print(f'Train Loss: {round_loss_avg:.6f} Acc: {round_accuracy_avg:.2f}%')

        if t % test_freq == 0 or t == params['rounds']-1:
            acc, loss = test(model)
            test_accuracies.append(acc)
            test_losses.append(loss)
            wandb.log({'acc': acc, 'loss': loss, 'round': t})
    
    return test_accuracies, test_losses

# %%
accuracies, losses = train(model, params)

# %%
with open('./pickles/results/federated/accuracies.pkl', 'wb') as f:
    pickle.dump(accuracies, f)
with open('./pickles/results/federated/losses.pkl', 'wb') as f:
    pickle.dump(losses, f)

# %%
plt.xlabel('rounds')
plt.ylabel('accuracy')
plt.plot(accuracies, label=params['method'], marker='')
plt.legend()
# %%
