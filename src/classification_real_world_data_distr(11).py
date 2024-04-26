# %%
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.ops import MLP
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR100
from torch.optim import SGD
import copy
from tqdm import tqdm
from collections import OrderedDict
import random
import matplotlib.pyplot as plt

# %%
# import data

K = 100
S = np.array(range(K))

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
])

train_dataset = CIFAR100('datasets/cifar100', train=True, download=True, transform=preprocess)
test_dataset = CIFAR100('datasets/cifar100', train=False, download=True, transform=preprocess)

test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# %%
# split data, build clients

iid = True
samples_per_client = int(len(train_dataset) / K)
shards_per_client = 2

def split_data(dataset, iid=True):
    if iid:
        return torch.utils.data.random_split(train_dataset, [samples_per_client] * K)
    else:
        sorted_dataset = sorted(train_dataset, key=lambda x: x[1])
        shard_size = int(samples_per_client / shards_per_client)
        shards = [
            torch.utils.data.Subset(
                sorted_dataset,
                range(i*shard_size, (i+1)*shard_size)
            )
            for i in range(K*shards_per_client)
        ]

        random.shuffle(shards)

        return [
            torch.utils.data.ConcatDataset([shards[2*i], shards[2*i+1]])
            for i in range(K)
        ]


client_datasets = split_data(train_dataset, iid)
assert len(client_datasets) == K
assert len(client_datasets[0]) == samples_per_client
assert iid or all([0 < len(set(map(lambda x: x[1], client_datasets[i]))) <= 4 for i in range(K)])

# %%
# init model, criterion
from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

class LeNet5_circa(nn.Module):
    def __init__(self):
        super( LeNet5_circa, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(5 * 5 * 64, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 100)

    def forward(self, x):
        x = self.pool(self.conv1(x).relu())
        x = self.pool(self.conv2(x).relu())
        x = torch.flatten(x, 1)
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        x = self.fc3(x)
        x = nn.functional.softmax(x)
        return x
    

model_lenet = LeNet5_circa().cuda()
print(model_lenet)
count_parameters(model_lenet)

optimizer = SGD(model_lenet.parameters(), lr=0.01, momentum=0.9, weight_decay=4e-4)
criterion = torch.nn.CrossEntropyLoss().cuda()
# %%
# Training

T = 100
test_freq = 20

def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

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
    return test_accuracy


def client_update(model, k, w, params):
    model.train()
    optimizer = torch.optim.SGD(model_lenet.parameters(), lr=0.01, momentum=0.9, weight_decay=4e-4)
    loader = DataLoader(client_datasets[k], batch_size=params['B'])

    for i in range(params['E']):
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    return model.state_dict()


def train(model, params):
    accuracies = []
    w = model.state_dict()
    for t in tqdm(range(T)):
        m = int(max(params['C']*K, 1))
        s = np.random.choice(S, m, replace=False)

        w_clients = []
        for k in s:
            w_clients.append(client_update(copy.deepcopy(model), k, w, params))

        w = OrderedDict([
            (
                key,
                sum(map(lambda x: x[key], w_clients)) / len(w_clients)
            ) for key in w_clients[0].keys()
        ])

        model.load_state_dict(w)

        if t % test_freq == 0 or t == T-1:
            accuracies.append(test(model))

    return accuracies

params = {'C': 0.2, 'B': 10, 'E': 1}
accuracies_lenet = train(model_lenet, params)

# %%
# plot
plt.xlabel('rounds')
plt.ylabel('accuracy')
xx = np.arange(0, T + test_freq, test_freq)
plt.plot(xx, accuracies_lenet, label='LeNet-5 (11)', marker='.')
plt.legend()