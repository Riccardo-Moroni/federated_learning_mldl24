# %%
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import SGD, lr_scheduler
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import itertools
import pickle
import torch.optim as optim
from torch.utils.data import TensorDataset

from CharRNN import CharRNN

# import wandb
# wandb.login()

# %% 
def x_y_from_path(json_path:str):
    with open(json_path) as json_data:
        data_dict = json.load(json_data)
    X , Y = [], []
    for k in data_dict['user_data'].keys():
        X.append(data_dict['user_data'][k]['x'])
        Y.append(data_dict['user_data'][k]['y'])
    X = list(itertools.chain(*X))
    Y = list(itertools.chain(*Y))
    return X,Y

json_train_path = '../../datasets/shakespeare/train/all_data_niid_0_keep_0_train_9.json'
json_test_path = '../../datasets/shakespeare/test/all_data_niid_0_keep_0_test_9.json'

X_train, Y_train = x_y_from_path(json_train_path)
X_test, Y_test = x_y_from_path(json_test_path)

# save X and Y in ./pickles/ folder
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
idx_to_char = {idx: char for idx, char in enumerate(vocab_train)}

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
    'batch_size' : 150,
    'lr' : 1e-1,
    'epochs' : 3,
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

model = CharRNN(vocab_size = model_params['vocab_size'], embed_dim = model_params['embed_dim'], lstm_units=model_params['lstm_units']).cuda()
model.to('cuda')
print(model)

criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=train_params['lr'], momentum=train_params['momentum'], weight_decay=4e-4)

test_freq = 50

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
