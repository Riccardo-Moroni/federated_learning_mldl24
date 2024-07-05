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
# wandb.login()

torch.cuda.empty_cache()

def concat_dict_values(my_dict):
    concat = []
    for v in my_dict.values():
        if isinstance(v, list):
            concat.extend(v)
        else:
            concat.append(v)
    return concat

def train(model, optimizer, epochs, train_loader, test_loader, criterion, scheduler=None):
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

        # test (after each single epoch)
        acc, loss = test(model, test_loader, criterion)
        wandb.log({'test_acc': acc, 'test_loss': loss, 'epoch': t})

        accuracies.append(acc)
        losses.append(loss)

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

json_train_path = '../../datasets/shakespeare/train/all_data_niid_0_keep_0_train_9.json'
json_test_path = '../../datasets/shakespeare/test/all_data_niid_0_keep_0_test_9.json'

X_train_pruned, Y_train_pruned, X_test_pruned, Y_test_pruned = shakespeare_data_pruning(json_train_path, json_test_path)
X_train = concat_dict_values(X_train_pruned)
Y_train = concat_dict_values(Y_train_pruned)
X_test = concat_dict_values(X_test_pruned)
Y_test = concat_dict_values(Y_test_pruned)

# Define the vocabulary and the character to index mapping
train_sentence = ' '.join(X_train)
vocab_train = sorted(set(train_sentence))
vocab_train.append('<OOV>')

char_to_idx = {char: idx for idx, char in enumerate(vocab_train)}

X_train_enc = np.array(tokenize_encode(X_train, vocab_train, char_to_idx))
Y_train_enc = np.array(tokenize_encode(Y_train, vocab_train, char_to_idx)).flatten()
X_test_enc = np.array(tokenize_encode(X_test, vocab_train, char_to_idx))
Y_test_enc = np.array(tokenize_encode(Y_test, vocab_train, char_to_idx)).flatten()

X_train_tensor = torch.tensor(X_train_enc, dtype=torch.long).cuda()
Y_train_tensor = torch.tensor(Y_train_enc, dtype=torch.long).cuda()
X_test_tensor = torch.tensor(X_test_enc, dtype=torch.long).cuda()
Y_test_tensor = torch.tensor(Y_test_enc, dtype=torch.long).cuda()

train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

param_grid = {
    "lr": [2e-2, 5e-2, 1e-1], #'lr': [2e-2, 5e-2, 1e-1, 2e-1 ,5e-1, 1], 
    'batch_size': [100, 200], #'batch_size': [10, 20 , 50, 100, 200],
    'weight_decay': [0], #'weight_decay': [0, 4e-4],
    'momentum': [0.9], #'momentum': [0, 0.9],
    'epochs' : [25],
    'lr_scheduler': ["CosineAnnealingLR"] # 'lr_scheduler': ["CosineAnnealingLR", "PolynomialLR","ExponentialLR", None]
}

model_params = { # no gird search over these
    'vocab_size' : len(vocab_train),
    'embed_dim' : 8,
    'lstm_units' : 256,
}

# %%
# gridsearch
# No multiprocessing

for batch_size in param_grid["batch_size"]: 

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss().cuda() 

    for lr in param_grid["lr"]: 
        for weight_decay in param_grid["weight_decay"]: 
            for momentum in param_grid["momentum"]: 
                for epochs in param_grid["epochs"]: 
                    for scheduler in param_grid["lr_scheduler"]: 
                        model = CharRNN(vocab_size = model_params['vocab_size'], embed_dim = model_params['embed_dim'], lstm_units=model_params['lstm_units']).cuda()

                        wandb_run_name = f"lr:{lr} | batch_size:{batch_size} | weight_decay:{weight_decay} | momentum:{momentum} | lr_scheduler:{scheduler}"
                        print(wandb_run_name)

                        all_params = {
                            'lr': lr,
                            'batch_size': batch_size,
                            'weight_decay': weight_decay,
                            'momentum': momentum,
                            'epochs': epochs,
                            'lr_scheduler': scheduler,
                            **model_params
                        }

                        wandb.init(
                            project='Centralized_Shakespeare_gridsearch',
                            name= wandb_run_name,
                            config= all_params
                        )

                        optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

                        if scheduler == "CosineAnnealingLR": 
                            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
                        elif scheduler == "PolynomialLR":
                            scheduler = lr_scheduler.PolynomialLR(optimizer, total_iters=20, power=2)
                        elif scheduler == "ExponentialLR":
                            scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
                        else:
                            scheduler = None

                        accuracies, losses = train(model, optimizer, epochs, train_loader, test_loader, criterion, scheduler)
                        wandb.finish()

                        del model 

torch.cuda.empty_cache()

# %%
# multiprocessing gridsearch
# DANGER ZONE. Warning: you could allocate more than what your gpu can handle!  

import torch.multiprocessing as mp
import logging
import time

def run_experiment(config):
    batch_size, lr, weight_decay, momentum, epochs, scheduler = config
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = CharRNN(vocab_size=model_params['vocab_size'], embed_dim=model_params['embed_dim'], lstm_units=model_params['lstm_units']).cuda()

    wandb_run_name = f"lr:{lr} | batch_size:{batch_size} | weight_decay:{weight_decay} | momentum:{momentum} | lr_scheduler:{scheduler}"
    print(wandb_run_name)

    all_params = {
        'lr': lr,
        'batch_size': batch_size,
        'weight_decay': weight_decay,
        'momentum': momentum,
        'epochs': epochs,
        'lr_scheduler': scheduler,
        **model_params
    }

    wandb.init(
        project='Centralized_Shakespeare_gridsearch',
        name=wandb_run_name,
        config=all_params
    )

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

    if scheduler == "CosineAnnealingLR": 
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    elif scheduler == "PolynomialLR":
        scheduler = lr_scheduler.PolynomialLR(optimizer, total_iters=20, power=2)
    elif scheduler == "ExponentialLR":
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    else:
        scheduler = None

    accuracies, losses = train(model, optimizer, epochs, train_loader, test_loader, criterion, scheduler)
    wandb.finish()

def worker_process(config):
    logging.info(f"Running config: {config}")
    try:
        run_experiment(config)
    except Exception as e:
        logging.info(f"Error in configuration {config}: {e}")

def main():
    wandb.login()
    wandb.setup()

    configs = [(batch_size, lr, weight_decay, momentum, epochs, scheduler) 
               for batch_size in param_grid["batch_size"]
               for lr in param_grid["lr"]
               for weight_decay in param_grid["weight_decay"]
               for momentum in param_grid["momentum"]
               for epochs in param_grid["epochs"]
               for scheduler in param_grid["lr_scheduler"]]
    
    # # attempt 1
    # print(f"Total configurations: {len(configs)}")
    # for config in configs:
    #     print(f"Config: {config}")

    # with mp.Pool(processes=5) as pool:
    #     pool.map(worker_process, configs)
    ###########################################

    # # attempt 2
    # # Apply asynchronously with callback
    # pool = mp.Pool(processes=2)
    # results = []
    # for config in configs:
    #     results.append(pool.apply_async(worker_process, args=(config,)))

    # # Wait for all processes to finish
    # for result in results:
    #     result.get()
    ############################################

    # attempt 3
    with mp.Pool(processes=5) as pool:
        results = []
        for config in configs:
            print(config)
            results.append(pool.apply_async(worker_process, args=(config,)))
            time.sleep(1)

        # Wait for all processes to finish
        for result in results:
            result.get()


if __name__ == "__main__":
    mp.set_start_method('spawn')  
    main() 
# none of these attempts worked for me, multiple processes are instanciated but all running the same config, although they really shouldn't
# even more strange is the fact that vocab_size, which shouldn't really change among the different config runs, changes. 

# %%
# grid search via wandb sweeps

import time
import multiprocessing

sweep_config = {
    'method': 'grid',
    'name': 'Centralized grid lr:[0.02, 0.05, 0.1] batch_size:[100, 200]',
    'metric': {
        'name': 'test_acc',
        'goal': 'maximize'
    },
    'parameters': {
        'lr': {
            'values': [2e-2, 5e-2, 1e-1]
        },
        'batch_size': {
            'values': [100, 200]
        },
        'weight_decay': {
            'values':[0]
        }, 
        'momentum': {
            'values': [0.9]
        },
        'epochs': {
            'values': [25]
        },
        'scheduler': {
            'values': ['CosineAnnealingLR']
        },
    }
}

criterion = nn.CrossEntropyLoss().cuda() 

def sweep_train():
    with wandb.init() as run:
        config = run.config

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
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
                                   test_loader=test_loader, 
                                   criterion=criterion, 
                                   scheduler=scheduler)

        wandb.finish()
        del model

sweep_id = wandb.sweep(sweep_config, project='Centralized_Shakespeare_gridsearch')
wandb.agent(sweep_id, function=sweep_train)

# %%