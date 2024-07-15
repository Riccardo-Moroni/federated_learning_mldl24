# %%
# setting up for grid search
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import SGD, lr_scheduler
from tqdm import tqdm

from torch.utils.data import TensorDataset

from shakespeare_utils import shakespeare_data_pruning, tokenize_encode
from CharRNN import CharRNN

import wandb
wandb.login()

torch.cuda.empty_cache()

def concat_dict_values(my_dict):
    concat = []
    for v in my_dict.values():
        if isinstance(v, list):
            concat.extend(v)
        else:
            concat.append(v)
    return concat

def split_concat_dict_values(my_dict):
    first_2000_sentences = []
    remaining_500_sentences = []
    
    for v in my_dict.values():
        # if isinstance(v, list):
            first_2000_sentences.extend(v[:1800])
            remaining_500_sentences.extend(v[1800:])
        # else:
            # raise ValueError("split_concat_dict_values() Error")
    
    return first_2000_sentences, remaining_500_sentences

def train(model, optimizer, epochs, train_loader, val_loader, test_loader, criterion, scheduler=None):
    accuracies, losses = [], []
    for t in range(epochs):
        print(f"Epoch {t+1}:")
        model.train()
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        if scheduler != None:
            scheduler.step()


        val_acc, val_loss = test(model, val_loader, criterion)

        test_acc, test_loss = test(model, test_loader, criterion)
        wandb.log({'val_acc': val_acc, 'val_loss': val_loss, 'test_acc': test_acc, 'test_loss': test_loss, 'epoch': t})

        accuracies.append(test_acc)
        losses.append(test_loss)

    return accuracies, losses

def test(model, test_loader, criterion):
    model.eval()
    test_loss, correct, total = 0,0,0

    with torch.no_grad():
        progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Testing...")
        for batch_idx, (inputs, targets) in progress_bar: 
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

IID=1

if IID:
    # iid
    json_train_path_iid = '../../datasets/shakespeare_iid/train/all_data_iid_0_0_keep_0_train_9.json'
    json_test_path_iid = '../../datasets/shakespeare_iid/test/all_data_iid_0_0_keep_0_test_9.json'
    # 2500 sentences per client retrieved, the last 500 will be our valuation set. 
    X_train_pruned, Y_train_pruned, X_test_pruned, Y_test_pruned = shakespeare_data_pruning(json_train_path_iid, json_test_path_iid, crop_amount=2000) 
else:
    # non-IID
    json_train_path = '../../datasets/shakespeare/train/all_data_niid_0_keep_0_train_9.json'
    json_test_path = '../../datasets/shakespeare/test/all_data_niid_0_keep_0_test_9.json'
    # 2500 sentences per client retrieved, the last 500 will be our valuation set. 
    X_train_pruned, Y_train_pruned, X_test_pruned, Y_test_pruned = shakespeare_data_pruning(json_train_path, json_test_path, crop_amount=2000) 

X_train, X_val = split_concat_dict_values(X_train_pruned) # (100, 1800, 80),  (100, 200, 80)
Y_train, Y_val = split_concat_dict_values(Y_train_pruned) # (100, 1800, 1),  (100, 200, 1)
X_test = concat_dict_values(X_test_pruned)
Y_test = concat_dict_values(Y_test_pruned)

# Define the vocabulary and the character to index mapping
train_sentence = ' '.join(X_train)
vocab_train = sorted(set(train_sentence))
vocab_train.append('<OOV>')

char_to_idx = {char: idx for idx, char in enumerate(vocab_train)}

X_train_enc = np.array(tokenize_encode(X_train, vocab_train, char_to_idx))
Y_train_enc = np.array(tokenize_encode(Y_train, vocab_train, char_to_idx)).flatten() # (100, 1800, 1) --> (100, 1800,)

X_val_enc = np.array(tokenize_encode(X_val, vocab_train, char_to_idx))
Y_val_enc = np.array(tokenize_encode(Y_val, vocab_train, char_to_idx)).flatten() #(100, 200, 1) --> (100, 200,)

X_test_enc = np.array(tokenize_encode(X_test, vocab_train, char_to_idx))
Y_test_enc = np.array(tokenize_encode(Y_test, vocab_train, char_to_idx)).flatten() # (some_number_depending_on_chose_clients,)

# create train, val, test tensors
X_train_tensor = torch.tensor(X_train_enc, dtype=torch.long).cuda()
Y_train_tensor = torch.tensor(Y_train_enc, dtype=torch.long).cuda()

X_val_tensor = torch.tensor(X_val_enc, dtype=torch.long).cuda()
Y_val_tensor = torch.tensor(Y_val_enc, dtype=torch.long).cuda()

X_test_tensor = torch.tensor(X_test_enc, dtype=torch.long).cuda()
Y_test_tensor = torch.tensor(Y_test_enc, dtype=torch.long).cuda()

train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

param_grid = {
    "lr": [2e-2, 5e-2, 1e-1], #'lr': [2e-2, 5e-2, 1e-1, 2e-1 ,5e-1, 1], 
    'batch_size': [100, 200], #'batch_size': [10, 20 , 50, 100, 200],
    'weight_decay': [0], #'weight_decay': [0, 4e-4],
    'momentum': [0.9], #'momentum': [0, 0.9],
    'epochs' : [25], # depending on the batch size, each epoch can take from 30s to some minutes
    'lr_scheduler': ["CosineAnnealingLR"] # 'lr_scheduler': ["CosineAnnealingLR", "PolynomialLR","ExponentialLR", None]
}

model_params = { # no gird search over these
    'vocab_size' : len(vocab_train),
    'embed_dim' : 8,
    'lstm_units' : 256,
}

# %%
# gridsearch by for loops

# for batch_size in param_grid["batch_size"]: 

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size)

#     criterion = nn.CrossEntropyLoss().cuda() 

#     for lr in param_grid["lr"]: 
#         for weight_decay in param_grid["weight_decay"]: 
#             for momentum in param_grid["momentum"]: 
#                 for epochs in param_grid["epochs"]: 
#                     for scheduler in param_grid["lr_scheduler"]: 
#                         model = CharRNN(vocab_size = model_params['vocab_size'], embed_dim = model_params['embed_dim'], lstm_units=model_params['lstm_units']).cuda()

#                         wandb_run_name = f"lr:{lr} | batch_size:{batch_size} | weight_decay:{weight_decay} | momentum:{momentum} | lr_scheduler:{scheduler}"
#                         print(wandb_run_name)

#                         all_params = {
#                             'lr': lr,
#                             'batch_size': batch_size,
#                             'weight_decay': weight_decay,
#                             'momentum': momentum,
#                             'epochs': epochs,
#                             'lr_scheduler': scheduler,
#                             **model_params
#                         }

#                         wandb.init(
#                             project='Centralized_Shakespeare_gridsearch',
#                             name= wandb_run_name,
#                             config= all_params
#                         )

#                         optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

#                         if scheduler == "CosineAnnealingLR": 
#                             scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
#                         elif scheduler == "PolynomialLR":
#                             scheduler = lr_scheduler.PolynomialLR(optimizer, total_iters=20, power=2)
#                         elif scheduler == "ExponentialLR":
#                             scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
#                         else:
#                             scheduler = None

#                         accuracies, losses = train(model, optimizer, epochs, train_loader, test_loader, criterion, scheduler)
#                         wandb.finish()

#                         del model 

# torch.cuda.empty_cache()

# %%
# grid search via wandb sweeps

sweep_config = {
    'method': 'grid',
    'name': "Centralized grid | lr:[2e-2, 5e-2, 1e-1, 5e-1, 1] | batch_size:[10, 50, 100, 200] | momentum:[0, 0.9] | scheduler:['CosineAnnealingLR', 'ExponentialLR', None]",
    'metric': {
        'name': 'test_acc',
        'goal': 'maximize'
    },
    'parameters': {
        'lr': {
            'values': [5e-2, 1e-1] # 2e-2, 5e-2, 1e-1, 5e-1, 1
        },
        'batch_size': {
            'values': [50, 100, 200] # 10, 50, 100, 200
        },
        'weight_decay': {
            'values':[0] # 0, 4e-4
        }, 
        'momentum': {
            'values': [0, 0.9]
        },
        'epochs': {
            'values': [20]
        },
        'scheduler': {
            'values': ['CosineAnnealingLR', "ExponentialLR", None]
        },
    }
}

criterion = nn.CrossEntropyLoss().cuda() 

def sweep_train():
    with wandb.init() as run:
        config = run.config

        run.name = f"lr:{config.lr} | batch_size:{config.batch_size} | momentum:{config.momentum} | scheduler:{config.scheduler}"

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

        model = CharRNN(vocab_size=model_params['vocab_size'], embed_dim=model_params['embed_dim'], lstm_units=model_params['lstm_units']).cuda()

        optimizer = SGD(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, momentum=config.momentum)

        if config.scheduler == "CosineAnnealingLR":
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
        elif config.scheduler == "PolynomialLR":
            scheduler = lr_scheduler.PolynomialLR(optimizer, total_iters=20, power=2)
        elif config.scheduler == "ExponentialLR":
            scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        else:
            scheduler = None

        accuracies, losses = train(model=model, 
                                   optimizer=optimizer, 
                                   epochs=config.epochs, 
                                   train_loader=train_loader, 
                                   val_loader=val_loader, 
                                   test_loader=test_loader,
                                   criterion=criterion, 
                                   scheduler=scheduler)

        wandb.finish()
        del model

# %%
# 
sweep_id = wandb.sweep(sweep_config, project='Centralized_Shakespeare_gridsearch_2')
wandb.agent(sweep_id, function=sweep_train)

 # %%