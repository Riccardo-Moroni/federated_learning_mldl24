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

from CharRNN import CharRNN

import wandb
wandb.login()

# %% 
def data_pruning(json_train_path, json_test_path, crop_amount=2000, n_clients=100):
    """
    Reduces the dimension of LEAF dataset.
    Samples 'n_clients' clients among those having at least 'crop_amount' training samples.
    Each client is given 'crop_amount' number of contigous training samples

    Returns:
        - 4 dictionaries (X_train, Y_train, X_test, Y_test) 
    """
    rand_seed=0
    with open(json_train_path) as train_json_data:
        train_dict = json.load(train_json_data)
    with open(json_test_path) as test_json_data:
        test_dict = json.load(test_json_data)

    users_complete = train_dict['users']

    X_train_cropped, Y_train_cropped, X_test_cropped, Y_test_cropped = {}, {}, {}, {}

    i=0
    for k in train_dict['user_data'].keys():
        if train_dict['num_samples'][i] > crop_amount:
            np.random.seed(rand_seed)
            start = np.random.randint(len(train_dict['user_data'][k]['x'])-crop_amount)
            X_train_cropped[k] = train_dict['user_data'][k]['x'][start:start+crop_amount]
            Y_train_cropped[k] = train_dict['user_data'][k]['y'][start:start+crop_amount]
            X_test_cropped[k] = test_dict['user_data'][k]['x'][start:start+crop_amount]
            Y_test_cropped[k] = test_dict['user_data'][k]['y'][start:start+crop_amount]
            rand_seed+=1
            i+=1
        else:
            i+=1

    users_selected = random.sample(list(X_train_cropped.keys()), n_clients)

    X_train = {key: X_train_cropped[key] for key in users_selected}
    Y_train = {key: Y_train_cropped[key] for key in users_selected}
    X_test = {key: X_test_cropped[key] for key in users_selected}
    Y_test = {key: Y_test_cropped[key] for key in users_selected}

    return X_train, Y_train, X_test, Y_test


def concat_dict_values(my_dict):
    concat = []
    for v in my_dict.values():
        if isinstance(v, list):
            concat.extend(v)
        else:
            concat.append(v)
    return concat

json_train_path = '../../datasets/shakespeare/train/all_data_niid_0_keep_0_train_9.json'
json_test_path = '../../datasets/shakespeare/test/all_data_niid_0_keep_0_test_9.json'

X_train_pruned, Y_train_pruned, X_test_pruned, Y_test_pruned = data_pruning(json_train_path, json_test_path)
X_train = concat_dict_values(X_train_pruned)
Y_train = concat_dict_values(Y_train_pruned)
X_test = concat_dict_values(X_test_pruned)
Y_test = concat_dict_values(Y_test_pruned)

# %%

# save X and Y in './pickles/' folder
with open('./pickles/X_train.pkl', 'wb') as f:
    pickle.dump(X_train, f)
with open('./pickles/Y_train.pkl', 'wb') as f:
    pickle.dump(Y_train, f)
with open('./pickles/X_test.pkl', 'wb') as f:
    pickle.dump(X_test, f)
with open('./pickles/Y_test.pkl', 'wb') as f:
    pickle.dump(Y_test, f)

# %%
# Load pickles
with open('./pickles/X_train.pkl', 'rb') as f:
    X_train = pickle.load(f)
with open('./pickles/Y_train.pkl', 'rb') as f:
    Y_train = pickle.load(f)
with open('./pickles/X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)
with open('./pickles/Y_test.pkl', 'rb') as f:
    Y_test = pickle.load(f)

# %%
train_sentence = ' '.join(X_train)
vocab_train = sorted(set(train_sentence))
vocab_train.append('<OOV>')

char_to_idx = {char: idx for idx, char in enumerate(vocab_train)}

# %%
# create a ./tensors/ folder in which to save the (encoded) tensors 

def tokenize_encode(my_list, char_to_idx):
    oov_token = len(vocab_train)-1
    new_list = []
    for sentence in tqdm(my_list):
        characters = list(sentence)

        encoded = []
        for char in characters:
            if char in char_to_idx:
                encoded.append(char_to_idx[char])
            else:
                encoded.append(oov_token)
    
        new_list.append(encoded)
    return new_list

X_train_enc = np.array(tokenize_encode(X_train, char_to_idx))
Y_train_enc = np.array(tokenize_encode(Y_train, char_to_idx)).flatten()
X_test_enc = np.array(tokenize_encode(X_test, char_to_idx))
Y_test_enc = np.array(tokenize_encode(Y_test, char_to_idx)).flatten()

X_train_tensor = torch.tensor(X_train_enc, dtype=torch.long)
Y_train_tensor = torch.tensor(Y_train_enc, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_enc, dtype=torch.long)
Y_test_tensor = torch.tensor(Y_test_enc, dtype=torch.long)


# %%
# save tensors
torch.save(X_train_tensor,'./tensors/X_train_tensor.pt')
torch.save(Y_train_tensor,'./tensors/Y_train_tensor.pt')
torch.save(X_test_tensor,'./tensors/X_test_tensor.pt')
torch.save(Y_test_tensor,'./tensors/Y_test_tensor.pt')

# %%
# load tensors
X_train_tensor = torch.load('./tensors/X_train_tensor.pt')
Y_train_tensor = torch.load('./tensors/Y_train_tensor.pt')
X_test_tensor = torch.load('./tensors/X_test_tensor.pt')
Y_test_tensor = torch.load('./tensors/Y_test_tensor.pt')

# %%
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

# training parameters
train_params = {
    'batch_size' : 100,
    'lr' : 1e-1,
    'epochs' : 10,
    'momentum': 0.9,
}

train_loader = DataLoader(train_dataset, train_params['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, train_params['batch_size'])

# %%
# model parameters
model_params = {
    'vocab_size' : len(vocab_train),
    'embed_dim' : 8,
    'lstm_units' : 256,
}

all_params = train_params.copy()
all_params.update(model_params)
wandb.init(
    project='fl',
    name=f'centralized_shakespeare',
    config= all_params
)

model = CharRNN(vocab_size = model_params['vocab_size'], embed_dim = model_params['embed_dim'], lstm_units=model_params['lstm_units']).cuda()
model.to('cuda')
print(model)

criterion = nn.CrossEntropyLoss().cuda()
optimizer = SGD(model.parameters(), lr=train_params['lr'], momentum=train_params['momentum'], weight_decay=4e-4)

def train(model):
    accuracies, losses = [], []
    for t in range(train_params['epochs']):
        model.train()
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), f"./saved_models/{train_params['epochs']}epochs_weights.pt")

        # test (after each single epoch)
        acc, loss = test(model)
        wandb.log({'acc': acc, 'loss': loss, 'epoch': t})
        accuracies.append(acc)
        losses.append(loss)

    return accuracies, losses

def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Testing...")
        for batch_idx, (inputs, targets) in progress_bar: 
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


# %%
accuracies, losses = train(model)

# %%
# testing one batch on cpu (debug)

# model = CharRNN(vocab_size = model_params['vocab_size'], embed_dim = model_params['embed_dim'], lstm_units=model_params['lstm_units'])
# model.load_state_dict(torch.load('./saved_models/1epochs_weights.pth'))
# model.eval()
# test_loss = 0
# correct = 0
# total = 0
# for batch_idx, (inputs, targets) in enumerate(test_loader):
#     outputs = model(inputs)
#     loss = criterion(outputs, targets)
#     test_loss += loss.item()
#     _, predicted = outputs.max(1)
#     total += targets.size(0)
#     correct += predicted.eq(targets).sum().item()
#     break
