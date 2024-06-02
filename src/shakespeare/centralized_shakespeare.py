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

from CharRNN import CharRNN

# import wandb
# wandb.login()

# %% 
# Creating pickels (avoid if already have pickles)
all_data_json_path = '../../datasets/shakespeare/all_data/all_data.json'
with open(all_data_json_path) as json_data:
    all_data_dict = json.load(json_data)

X , Y = [], []
for k in all_data_dict['user_data'].keys():
    X.append(all_data_dict['user_data'][k]['x'])
    Y.append(all_data_dict['user_data'][k]['y'])
X = list(itertools.chain(*X))
Y = list(itertools.chain(*Y))

# saving pickles of X and Y
with open('./pickles/X.pkl', 'wb') as f:
    pickle.dump(X, f)
with open('./pickles/Y.pkl', 'wb') as f:
    pickle.dump(Y, f)

long_sentence = ' '.join(X)

# all_data_txt_path = '../../datasets/shakespeare/raw_data/raw_data.txt'
# all_data_txt = open(all_data_txt_path, mode='r').read()

vocab = sorted(set(long_sentence))

# %%
# Load pickles
with open('./pickles/X.pkl', 'rb') as f:
    X = pickle.load(f)
with open('./pickles/Y.pkl', 'rb') as f:
    Y = pickle.load(f)

# %%
