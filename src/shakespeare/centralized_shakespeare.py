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

def substitute_oov(sentences, char_to_idx):
    substituted_sentences = []
    for s in sentences:
        new_s = ''.join([char if char in char_to_idx.keys() else '' for char in s])
        substituted_sentences.append(new_s)
    return substituted_sentences

X_test = substitute_oov(X_test, char_to_idx)
Y_test = substitute_oov(Y_test, char_to_idx)

# %%
# create a ./tensors/ folder in which to save the (encoded) tensors 
def encode(data, dict):
    x = []
    for i in tqdm(range(len(data))):
            x.append(np.array([dict[char] for char in data[i]]))
    return x

X_train_enc = encode(X_train, char_to_idx)
Y_train_enc = encode(Y_train, char_to_idx)
X_test_enc = encode(X_test, char_to_idx)
Y_test_enc = encode(Y_test, char_to_idx)

X_train_tensor = torch.tensor(np.array(X_train_enc), dtype=torch.long)
Y_train_tensor = torch.tensor(np.array(Y_train_enc), dtype=torch.long)
X_test_tensor = torch.tensor(np.array(X_test_enc), dtype=torch.long)
Y_test_tensor = torch.tensor(np.array(Y_test_enc), dtype=torch.long)

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
    'epochs' : 1,
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
    accuracies = []
    losses = []
    for t in tqdm(range(train_params['epochs'])):
        model.train()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

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
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader

def load_json_files(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), 'r') as f:
                client_data = json.load(f)
                data.extend(client_data)
    return data

train_data = load_json_files('../../datasets/shakespeare/tensorflow_version/shakespeare_data/train')
test_data = load_json_files('../../datasets/shakespeare/tensorflow_version/shakespeare_data/test')

sequence_length = 80
def create_sequences(data, sequence_length):
    inputs, targets = [], []
    for snippet in data:
        text = snippet['text']
        for i in range(len(text) - sequence_length):
            inputs.append(text[i:i + sequence_length])
            targets.append(text[i + sequence_length])
    return inputs, targets

train_inputs, train_targets = create_sequences(train_data, sequence_length)
test_inputs, test_targets = create_sequences(test_data, sequence_length)

# %%
# Encode
all_text = ''.join([snippet['text'] for snippet in train_data])
vocab = sorted(set(all_text))
char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = {idx: char for idx, char in enumerate(vocab)}

def encode(text, char2idx):
    return [char2idx[char] for char in text]

train_inputs_enc = [encode(seq, char2idx) for seq in train_inputs]
train_targets_enc = [char2idx[char] for char in train_targets]
test_inputs_enc = [encode(seq, char2idx) for seq in test_inputs]
test_targets_enc = [char2idx[char] for char in test_targets]

class ShakespeareDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])

train_dataset = ShakespeareDataset(train_inputs_enc, train_targets_enc)
test_dataset = ShakespeareDataset(test_inputs_enc, test_targets_enc)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
