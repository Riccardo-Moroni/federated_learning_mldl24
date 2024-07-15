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

import wandb
wandb.login()

K = 100

# params = {
#     'K': K,
#     'C': 0.1,
#     'B': 10,
#     'J': 5,
#     # 'lr_server': 1e-1,
#     'lr_client': 5e-1,
#     'method': 'fedavg',
#     'tau': 1e-3,
#     'gamma': 0.1,
#     'participation': 'uniform',
#     'rounds': 2000,
#     'weight_decay':0,
#     'crop_amount':2500
# }

import sys
sys.path.append('../')
from client_selector import ClientSelector

# client_selector = ClientSelector(params)

# wandb.init(
#     project='fl_shakespeare',
#     name=f"federated_mod_{params['method']}_C:{params['C']}_J:{params['J']}_lr:{params['lr_client']}_partecipation:{params['participation']}_weight_decay:{params['weight_decay']}_epochs:{params['rounds']}",
#     config= params
# )


# json_train_path = '../../datasets/shakespeare/train/all_data_niid_0_keep_0_train_9.json'
# json_test_path = '../../datasets/shakespeare/test/all_data_niid_0_keep_0_test_9.json'

# X_train_pruned, Y_train_pruned, X_test_pruned, Y_test_pruned = shakespeare_data_pruning(json_train_path, json_test_path, crop_amount=2500)
# data is already  split at this point {user1:[crop_amount],...,user100:[crop_amount]}

IID=False

if IID:
    # iid
    json_train_path_iid = '../../datasets/shakespeare_iid/train/all_data_iid_0_0_keep_0_train_9.json'
    json_test_path_iid = '../../datasets/shakespeare_iid/test/all_data_iid_0_0_keep_0_test_9.json'

    X_train_pruned, Y_train_pruned, X_test_pruned, Y_test_pruned = shakespeare_data_pruning(json_train_path_iid, json_test_path_iid, crop_amount=2500) 
else:
    # non-IID
    json_train_path = '../../datasets/shakespeare/train/all_data_niid_0_keep_0_train_9.json'
    json_test_path = '../../datasets/shakespeare/test/all_data_niid_0_keep_0_test_9.json'

    X_train_pruned, Y_train_pruned, X_test_pruned, Y_test_pruned = shakespeare_data_pruning(json_train_path, json_test_path, crop_amount=2500) 
 

# define dictionary and char to int mapping
train_sentence = ' '.join(' '.join(single_user_list) for single_user_list in X_train_pruned.values())
vocab = sorted(set(train_sentence))
vocab.append('<OOV>')
char_to_idx = {char: idx for idx, char in enumerate(vocab)}


# from dict to nested list, each outer element is a client
X_train_list = [X_train_pruned[key] for key in sorted(X_train_pruned.keys())]
Y_train_list = [Y_train_pruned[key] for key in sorted(Y_train_pruned.keys())]

X_train_enc, Y_train_enc, X_test_enc, Y_test_enc = [],[],[],[]
for user in tqdm(range(len(X_train_list))):
    X_train_enc.append(tokenize_encode(X_train_list[user], vocab, char_to_idx))
    Y_train_enc.append(tokenize_encode(Y_train_list[user], vocab, char_to_idx))

# We can discard the information about the user for the test data, since the testing is centralized 
X_test_concat = concat_dict_values(X_test_pruned)
Y_test_concat = concat_dict_values(Y_test_pruned)
X_test_enc = np.array(tokenize_encode(X_test_concat, vocab, char_to_idx)) # (100(users), crop_amount(sentences per user), 80(characters per sentence)) 
Y_test_enc = np.array(tokenize_encode(Y_test_concat, vocab, char_to_idx)).squeeze(-1) # (100, crop_amount, 1) --> (100, crop_amount,)

# to tensor
X_train_tensor = torch.tensor(X_train_enc, dtype=torch.long) # (100, crop_amount, 80)
Y_train_tensor = torch.tensor(Y_train_enc, dtype=torch.long).squeeze(-1) # (100, crop_amount, 1) --> (100, crop_amount,)

X_test_tensor = torch.tensor(X_test_enc, dtype=torch.long).cuda()
Y_test_tensor = torch.tensor(Y_test_enc, dtype=torch.long).cuda()

# 
X_val_client_tensor = X_train_tensor[:, 2000:2250].cuda() # (100, 250, 80)
Y_val_client_tensor = Y_train_tensor[:, 2000:2250].cuda() # (100, 250, 1)

X_val_server_tensor = X_train_tensor[:, 2250:2500].reshape(100*250, 80).cuda()
Y_val_server_tensor = Y_train_tensor[:, 2250:2500].reshape(100*250, 1).squeeze(-1).cuda()

val_dataset_server = TensorDataset(X_val_server_tensor, Y_val_server_tensor)
val_loader_server = DataLoader(val_dataset_server, batch_size=100)

# delete the validation set from train set
X_train_tensor = X_train_tensor[:, 0:2000].cuda()
Y_train_tensor = Y_train_tensor[:, 0:2000].cuda()

test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=100)

print(X_train_tensor.shape, Y_train_tensor.shape)
print(X_val_server_tensor.shape, Y_val_server_tensor.shape)
print(X_test_tensor.shape, Y_test_tensor.shape)

# train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)

train_datasets, val_datasets_clients = [], []
for k in range(K):
    X_k = X_train_tensor[k]
    Y_k = Y_train_tensor[k]
    X_k_val = X_val_client_tensor[k]
    Y_k_val = Y_val_client_tensor[k]
    train_datasets.append(TensorDataset(X_k, Y_k))
    val_datasets_clients.append(TensorDataset(X_k_val, Y_k_val))

# utils
def reduce_w(w_list, f):
    return OrderedDict([
            (key, f([x[key] for x in w_list])) for key in w_list[0].keys()
        ])


def tensor_sum(tensors_list, weights=None):
    if weights:
      return torch.sum(torch.stack([t*w for t, w in zip(tensors_list, weights)]), dim=0)
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


model_params = {
    'vocab_size' : len(vocab),
    'embed_dim' : 8,
    'lstm_units' : 256,
}
model = CharRNN(vocab_size = model_params['vocab_size'], embed_dim = model_params['embed_dim'], lstm_units=model_params['lstm_units']).cuda()
model.to('cuda')

criterion = nn.CrossEntropyLoss().cuda()
# optimizer = SGD(model.parameters(), lr=params['lr_client'], weight_decay=4e-4)

test_freq = 50

def test(model, loader):
    model.eval()
    test_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            # inputs, targets = inputs.cuda(), targets.cuda()
            # print(len(inputs))
            # print(inputs.shape)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss = test_loss / len(loader)
    test_accuracy = 100. * correct / total
    # print(f'Test Loss: {test_loss:.6f} Acc: {test_accuracy:.2f}%')
    return test_accuracy, test_loss

def client_update(model, k, params):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=params['lr_client'], weight_decay=params['weight_decay'])
    train_loader = DataLoader(train_datasets[k], batch_size=params['B'], shuffle=True)
    client_val_loader = DataLoader(val_datasets_clients[k], batch_size=params['B'], shuffle=False)

    client_loss, client_correct, client_total = 0, 0, 0
    i = 0
    for i in range(params['J']):
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # inputs, targets = inputs.cuda(), targets.cuda()
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
                client_train_loss = client_loss / params['J']
                client_train_accuracy = 100. * client_correct / client_total

                return model.state_dict(), client_train_accuracy, client_train_loss, *test(model, client_val_loader)
            
    client_train_loss = client_loss / params['J'] # average loss in J steps 
    client_train_accuracy = 100. * client_correct / client_total

    return model.state_dict(), client_train_accuracy, client_train_loss, *test(model, client_val_loader)

def train(model, params, client_selector, test_freq):
    test_accuracies, test_losses = [], []
    train_accuracies, train_losses = [], []
    # v = params['tau'] ** 2
    w = model.state_dict()
    m = reduce_w([w], lambda x: torch.mul(x[0], 0.0))
    for t in range(params['rounds']):
        print(f"Epoch {t+1}")
        round_loss, round_accuracy = 0, 0
        s = client_selector.sample()
        # print(sorted(s))

        w_clients = []
        w_val_losses = []
        for k in s:
            update, client_train_accuracy, client_train_loss, client_val_acc, client_val_loss = client_update(copy.deepcopy(model), k, params)
            w_clients.append(update)
            round_loss += client_train_loss
            round_accuracy += client_train_accuracy
            w_val_losses.append(client_val_loss)

        w_sum = sum(w_val_losses)
        weights = [l/w_sum for l in w_val_losses]

        round_loss_avg = round_loss / len(s)
        round_accuracy_avg = round_accuracy / len(s)
        train_accuracies.append(round_loss_avg)
        train_losses.append(round_accuracy_avg)

        if params['method'] == 'fedavg':
            w = reduce_w(
                w_clients,
                lambda x: tensor_sum(x, weights)
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
            val_acc_server, val_loss_server = test(model, val_loader_server)
            test_acc, test_loss = test(model, test_loader)
            test_accuracies.append(test_acc)
            test_losses.append(test_loss)
            wandb.log({'val_acc':val_acc_server, 'val_loss':val_loss_server, 'test_acc': test_acc, 'test_loss': test_loss, 'round': t})

        # acc, loss = test(model, test_loader)
        # test_accuracies.append(acc)
        # test_losses.append(loss)
        # wandb.log({'acc': acc, 'loss': loss, 'round': t})
    
    return test_accuracies, test_losses

# %%
# accuracies, losses = train(model, params)

K = 100

# client_selector = ClientSelector(params)

sweep_config = {
    'method': 'grid',
    'name': 'Federated grid continuation C=0.3',
    'metric': {
        'name': 'test_acc',
        'goal': 'maximize'
    },
    'parameters': {
        'method': {
            'values': ['fedavg'] #['fedavg', 'adagrad', 'yogi', 'adam']
        },
        'C': {
            'values': [0.1] # [0.1, 0.2, 0.3]
        },
        'J': {
            'values': [5] # [1, 5, 10, 20], [4, 8, 16] 
        },
        'lr': {
            'values': [1.5] #[0.5, 1, 1.5]
        },
        'B': {
            'values': [10] #[10, 50]
        },
        'weight_decay': {
            'values':[0] # [0, 4e-4]
        }, 
        # 'momentum': {
        #     'values': [0.9] # [0, 0.9]
        # },
        'rounds': {
            'values': [2000] # [2000, 1000, 500]
        },
        # 'scheduler': {
        #     'values': ['CosineAnnealingLR']
        # },
        # 'beta1': {
        #     'values': [0.9]
        # },
        # 'tau': {
        #     'values': [1e-2]
        # },
        'participation': {
            'values': ['uniform'] # ['uniform', 'gamma-skewed'] # TODO pachinko? 
        },
        # 'gamma': {
        #     'values': [0.5, 1, 5, 10], #0.5, 1, 5, 10
        # },

    }
}

criterion = nn.CrossEntropyLoss().cuda() 

test_freq = 50

K = 100

def sweep_train():
    with wandb.init() as run:
        config = run.config

        # if config.J == 8:
        #     config.rounds = 1000

        # if config.J == 16:
        #     config.rounds = 500

        run_name = f"method:{config.method} | C:{config.C} | J:{config.J} | lr_client:{config.lr} | batch_size:{config.B} | weight_decay:{config.weight_decay} | rounds:{config.rounds} | participation:{config.participation}"
        if config.participation !='uniform':
            run_name += f" | gamma:{config.gamma}"
        
        run.name = run_name
        
        params = {
            'K': K,
            'method': config.method,
            'C': config.C,
            'J': config.J,
            'lr_client': config.lr,
            'B': config.B,
            'weight_decay': config.weight_decay,
            # 'momentum': config.momentum,
            'rounds': config.rounds,
            # 'scheduler': config.scheduler,
            # 'beta1': config.beta1,
            # 'tau': config.tau,
            'participation': config.participation,
            # 'gamma': config.gamma,
        }

        client_selector = ClientSelector(params)

        model = CharRNN(vocab_size=model_params['vocab_size'], embed_dim=model_params['embed_dim'], lstm_units=model_params['lstm_units']).cuda()

        accuracies, losses = train(model=model, params=params, client_selector=client_selector, test_freq=test_freq)
        
        wandb.finish()
        del model
        torch.cuda.empty_cache()

sweep_id = wandb.sweep(sweep_config, project='Federated_MOD_Shakespeare_2')
wandb.agent(sweep_id, function=sweep_train)

# %%
# with open('./pickles/results/federated/accuracies.pkl', 'wb') as f:
#     pickle.dump(accuracies, f)
# with open('./pickles/results/federated/losses.pkl', 'wb') as f:
#     pickle.dump(losses, f)

# # %%
# plt.xlabel('rounds')
# plt.ylabel('accuracy')
# plt.plot(accuracies, label=params['method'], marker='')
# plt.legend()
# %%
