# %%
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import SGD
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import pickle
from torch.utils.data import TensorDataset
import random
from shakespeare_utils import shakespeare_data_pruning, tokenize_encode, concat_dict_values
from collections import OrderedDict
import math
import copy

from CharRNN import CharRNN

import sys
sys.path.append('../') # necessary to import from parant directory
from client_selector import ClientSelector

# import wandb
# wandb.login()


# %%
K = 100

params = {
    'K': K,
    'C': 0.3,
    'B': 100,
    'J': 8,
    # 'lr_server': 1e-1,
    'lr_client': 5e-1,
    'method': 'fedavg',
    'tau': 1e-3,
    'gamma': 0.1,
    'participation': 'uniform',
    'rounds': 2000
}

import sys
sys.path.append('../')
from client_selector import ClientSelector

client_selector = ClientSelector(params)

# wandb.init(
#     project='fl',
#     name=f'federated_shakespeare',
#     config= params
# )

# %%
json_train_path = '../../datasets/shakespeare/train/all_data_niid_0_keep_0_train_9.json'
json_test_path = '../../datasets/shakespeare/test/all_data_niid_0_keep_0_test_9.json'

X_train_pruned, Y_train_pruned, X_test_pruned, Y_test_pruned = shakespeare_data_pruning(json_train_path, json_test_path, crop_amount=3000)
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
X_train_tensor = torch.tensor(X_train_enc, dtype=torch.long) # (100, 2000, 80)
Y_train_tensor = torch.tensor(Y_train_enc, dtype=torch.long).squeeze(-1) # (100, 2000, 1) --> (100, 2000,)
X_test_tensor = torch.tensor(X_test_enc, dtype=torch.long)
Y_test_tensor = torch.tensor(Y_test_enc, dtype=torch.long)

test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=100)

print(X_train_tensor.shape, Y_train_tensor.shape)
print(X_test_tensor.shape, Y_test_tensor.shape)

# train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)


# %%
# utils
def reduce_w(w_list, f):
    return OrderedDict([
            (key, f([x[key] for x in w_list])) for key in w_list[0].keys()
        ])


def tensor_sum(tensors_list):
    return torch.sum(torch.stack(tensors_list), dim=0)


def w_norm2(w):
    res = 0
    for key in w.keys():
        res += torch.linalg.vector_norm(w[key]) ** 2
    return math.sqrt(res)


def fed_adagrad(v, delta, params):
    delta_norm2 = w_norm2(delta)
    return v + delta_norm2


def fed_yogi(v, delta, params):
    delta_norm2 = w_norm2(delta)
    return v - (1-params['beta2']) * delta_norm2 * torch.sign(v - delta_norm2)


def fed_adam(v, delta, params):
    delta_norm2 = w_norm2(delta)
    return params['beta2'] * v + (1-params['beta2']) * delta_norm2


methods = {
    'adagrad': fed_adagrad,
    'yogi': fed_yogi,
    'adam': fed_adam
}

# %%
model_params = {
    'vocab_size' : len(vocab),
    'embed_dim' : 8,
    'lstm_units' : 256,
}
model = CharRNN(vocab_size = model_params['vocab_size'], embed_dim = model_params['embed_dim'], lstm_units=model_params['lstm_units']).cuda()
model.to('cuda')

criterion = nn.CrossEntropyLoss().cuda()
optimizer = SGD(model.parameters(), lr=params['lr_client'], weight_decay=4e-4)

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
    X_k = X_train_tensor[k]
    Y_k = Y_train_tensor[k]
    train_dataset_k = TensorDataset(X_k, Y_k)
    loader = DataLoader(train_dataset_k, batch_size=params['B'], shuffle=True)

    client_loss, client_correct, client_total = 0, 0, 0
    i = 0
    for i in range(params['J']):
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            client_loss += loss.item()
            _, predicted = outputs.max(1)
            client_total += targets.size(0)
            client_correct += predicted.eq(targets).sum().item()

            loss.backward()
            optimizer.step()
            i += 1

            if i >= params['J']:
                client_loss = client_loss / params['J']
                client_accuracy = 100. * client_correct / client_total

                return model.state_dict(), client_accuracy, client_loss
            
    client_loss = client_loss / params['J'] # average loss in J steps 
    client_accuracy = 100. * client_correct / client_total

    return model.state_dict(), client_accuracy, client_loss

def train(model, params):
    test_accuracies, test_losses = [], []
    train_accuracies, train_losses = [], []
    v = params['tau'] ** 2
    w = model.state_dict()
    m = reduce_w([w], lambda x: torch.mul(x[0], 0.0))
    for t in range(params['rounds']):
        round_loss, round_accuracy = 0, 0
        s = client_selector.sample()
        # print(sorted(s))
        w_clients = []

        for k in s:
            update, client_accuracy, client_loss = client_update(copy.deepcopy(model), k, params)
            w_clients.append(update)
            round_loss += client_loss
            round_accuracy += client_accuracy
        round_loss_avg = round_loss / len(s)
        round_accuracy_avg = client_accuracy / len(s)
        train_accuracies.append(round_loss_avg)
        train_losses.append(round_accuracy_avg)

        if params['method'] == 'fedavg':
            w = reduce_w(
                w_clients,
                lambda x: tensor_sum(x) / len(w_clients)
            )
        else:
            deltas = [
                reduce_w(
                    [w, w_client],
                    lambda x: x[1] - x[0]
                ) for w_client in w_clients
            ]

            delta = reduce_w(
                deltas,
                lambda x: tensor_sum(x) / len(deltas)
            )

            m = reduce_w(
                [m, delta],
                lambda x: params['beta1'] * x[0] + (1-params['beta1']) * x[1]
            )

            v = methods[params['method']](v, delta, params)
            w = reduce_w(
                [w, m],
                lambda x: x[0] + params['lr_server'] * x[1] / (math.sqrt(v) + params['tau'])
            )
        
        model.load_state_dict(w)
        print(f'Train Loss: {round_loss_avg:.6f} Acc: {round_accuracy_avg:.2f}%')

        if t % test_freq == 0 or t == params['rounds']-1:
            acc, loss = test(model)
            test_accuracies.append(acc)
            test_losses.append(loss)
            # wandb.log({'acc': acc, 'loss': loss, 'round': t})

        # acc, loss = test(model)
        # test_accuracies.append(acc)
        # test_losses.append(loss)
        # wandb.log({'acc': acc, 'loss': loss, 'round': t})
    
    return test_accuracies, test_losses, train_accuracies, train_losses

# %%
accuracies, losses = train(model, params)

# %%
plt.xlabel('rounds')
plt.ylabel('accuracy')
plt.plot(accuracies, label=params['method'], marker='')
plt.legend()
# %%
