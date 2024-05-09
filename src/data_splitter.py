# %%
import pandas as pd
import torch
import networkx as nx
import numpy as np
from tqdm import tqdm
import random

import numpy as np
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader, Subset

from torchvision import transforms

# %%

class DataSplitter: 
    def __init__(self, params):
        self.dataset = params['dataset']
        self.K = params['K']
        
        if params.get('transform') is not None: # TODO may be useful to change the this class' functions in order to return an array of data loaders
            self.transform =  params['transform']
        
        if params.get("n_labels") is None:
            self.n_labels = None
        else:
            self.n_labels = params["n_labels"]
        
        if params.get("samples_per_client") is None:
            self.samples_per_client = int(len(self.dataset) / self.K)
        else:
            self.samples_per_client = params["samples_per_client"]
            if self.samples_per_client > len(self.dataset): 
                raise ValueError(f"samples_per_client cannot be greater than len(dataset) ({self.samples_per_client}>{len(self.dataset)})" )
        
        if self.K > len(self.dataset):
            raise ValueError(f"Number of clients (K={self.K}) cannot be greater than the number of samples in the dataset (len(dataset)={len(self.dataset)}).")
        if self.n_labels != None:
            if self.K* self.n_labels > len(self.dataset):
                raise ValueError(f"Number of clients multiplied by the number of labels requred per client (K*n_labels = {self.K}*{self.n_labels} = {self.K*self.n_labels}) cannot be greater than the number of samples in the (len(dataset)={len(self.dataset)}).")
        
    def idd_split(self): 
        return torch.utils.data.random_split(self.dataset, [self.samples_per_client] * self.K)
    
    def pachinko_allocation_split_cifar100(self, alpha=0.1, beta=10):  # TODO implement n_labels = ?
        """Generates a pachinko allocation of cifar100 dataset (loaded with load_dataset() function coming from datasets library (huggingface) because torchvision.datasets.CIFAR100 does not natively provide coarse_labels)
        Each sample in the dataset should have "coarse_label" and a "fine_label" attribute. 

        Args:
            alpha (float, optional): Dirichlet parameter for the coarse labels. Defaults to 0.1.
            beta (float, optional): Dirichlet parameter for the fine labels. Defaults to 10.

        Returns:
            list: a list of lists. Each element of the outer list is a list of samples (eg. images for cifar100).
        """
        dataset_list = []
        for sample in self.dataset:
            dataset_list.append({
                "img": sample["img"],
                "coarse_label": sample["coarse_label"], 
                "fine_label": sample["fine_label"],
            })
        # Convert the list of dictionaries to a Pandas DataFrame
        dataset_pandas = pd.DataFrame(dataset_list)
        H = nx.DiGraph()
        H.add_node("Root")

        # add all the nodes
        # for cifar100 it will be:
        
        # #nodes  |     names                        |  data inside
        ###############################################################################   
        # 1       |      "Root":                     |  {"dir_alpha_prior":[...]} --> 
        # 20      |      range(20)  (coarse):        |  {"dir_beta_prior": [...]} --> 
        # 100     |      f"{coarse}_{fine}":         |  {} --> 
        # 50000   |      f"{coarse}_{fine}_{index}:  |  {"img_data": ..., "label": ...}

        # = 50121 nodes 

        for i in range(len(dataset_pandas)): 
            r = dataset_pandas.iloc[i]

            img_data = r['img']#.tobytes()
            coarse = r['coarse_label']
            fine = r['fine_label']

            label = f"{coarse}_{fine}"
            id = f"{label}_{i}"
            node_attributes = {"img_data":img_data, "label":label}

            H.add_node(id, **node_attributes)

            if H.has_node(coarse): # has coarse => if has fine then just add the img, otherwise add fine, add img and link properly
                if H.has_node(label):
                    H.add_edge(label, id)
                else: 
                    H.add_edge(coarse, label)
                    H.add_edge(label, id)
            else: 
                H.add_edge("Root", coarse)
                H.add_edge(coarse, label)
                H.add_edge(label, id)
        
        def renormalize(arr, i):
            result = np.delete(arr, i)
            remaining_sum = np.sum(result)
            if remaining_sum != 0:
                result = result / remaining_sum
            return result

        all_clients_datasets = []

        for m in tqdm(range(self.K)):
            c_mult = np.random.dirichlet(alpha=np.full(len(list(H.successors("Root"))), alpha))

            f_mult = {} # distributions over the fine labels (one distr per each coarse)
            for c in list(H.successors("Root")):
                f_mult[c] = np.random.dirichlet(alpha=np.full(len(list(H.successors(c))), beta))

            Dms = []
            while len(Dms) < self.samples_per_client:
                c_sampled = np.random.choice(list(H.successors("Root")), p=c_mult) # [0-19]
                f_sampled = np.random.choice(list(H.successors(c_sampled)), p=f_mult[c_sampled]) # f"{coarse}_{fine}"

                leaf_sampled = random.choice(list(H.successors(f_sampled))) # f"{coarse}_{fine}_{dfindex}"
                Dms.append([H.nodes[leaf_sampled]["img_data"], H.nodes[leaf_sampled]["label"]])
                H.remove_node(leaf_sampled)

                if len(list(H.successors(f_sampled))) == 0:
                    # renormalize the mult on the fine
                    f_mult[c_sampled] = renormalize(f_mult[c_sampled], list(H.successors(c_sampled)).index(f_sampled))
                    H.remove_node(f_sampled)

                    if len(list(H.successors(c_sampled))) == 0 : 
                        # renormalize the mult
                        c_mult = renormalize(c_mult, list(H.successors("Root")).index(c_sampled))
                        H.remove_node(c_sampled)

            all_clients_datasets.append(Dms)
        
        return all_clients_datasets
    
    def non_iid_split(self):
        """
        works with torchvision.datasets.CIFAR100
        """
        sorted_dataset = sorted(self.dataset, key=lambda x: x[1])
        shards_per_client = 2 #
        shard_size = int(self.samples_per_client / shards_per_client)
        shards = [
            torch.utils.data.Subset(
                sorted_dataset,
                range(i*shard_size, (i+1)*shard_size)
            )
            for i in range(self.K*shards_per_client)
        ]

        random.shuffle(shards)

        return [
            torch.utils.data.ConcatDataset([shards[2*i], shards[2*i+1]])
            for i in range(self.K)
        ]
    
    
    def sort_it_and_it_will_make_sense(self):
        """
        works with torchvision.datasets.CIFAR100
        """
        
        def find_clients_with_label(dictionary, label):
            clients_with_label = []
            for client_id, labels in dictionary.items():
                if label in labels:
                    clients_with_label.append(client_id)
            return clients_with_label

        samples_per_client = int(len(self.dataset)/self.K)
        print("samples_per_client:", samples_per_client)

        sorted_dataset = sorted(self.dataset, key=lambda x: x[1])

        cifar100_df = pd.DataFrame(sorted_dataset, columns=["img", "targets"])

        samples_per_label = int(samples_per_client/self.n_labels)
        print("samples_per_label:", samples_per_label)
        clients = []

        client_labels_dict = {} # labels owned by each client

        for c in tqdm(range(self.K)):
            client_data = []
            clients_labels = list(cifar100_df["targets"].unique()[:self.n_labels])
            print("clients labels:", clients_labels)

            client_labels_dict[c] = clients_labels

            for label in clients_labels:

                label_data = cifar100_df[cifar100_df["targets"] == label]

                if len(label_data) >= samples_per_label: # add
                    sampled_indices = label_data.sample(samples_per_label).index
                    client_data.extend(label_data.loc[sampled_indices].values.tolist())
                    cifar100_df = cifar100_df.drop(sampled_indices)
                    remaining_label_data = cifar100_df[cifar100_df["targets"] == label]

                    if len(remaining_label_data) < samples_per_label: # add also the remaining to the client
                        client_data.extend(remaining_label_data.values.tolist())
                        cifar100_df = cifar100_df[cifar100_df["targets"] != label]

                else:
                    print("Warning: a client is getting all the data belonging to a label, which is less than len(dataset)/(K*n_labels)")
                    sampled_indices = label_data.index
                    client_data.extend(label_data.values.tolist())
                    cifar100_df = cifar100_df.drop(sampled_indices)

            clients.append(client_data)

        return clients, cifar100_df

    
    

# %%
# testing paching allocation

from datasets import load_dataset
import matplotlib.pyplot as plt

cifar_train, cifar_test = load_dataset("cifar100", split = ["train", "test"])
print(cifar_train.shape, cifar_test.shape)
        
params = {'K':500, 'dataset':cifar_train}
        
data_split_pachinko = DataSplitter(params).pachinko_allocation_split_cifar100()

uniques = [ len(set([data_split_pachinko[i][j][1] for j in range(len(data_split_pachinko[i]))])) for i in range(len(data_split_pachinko))]

plt.hist(uniques, bins=20)
plt.title("CIFAR-100 Client Label Distribution")
plt.xlabel("Number of unique labels")
plt.ylabel("Frequency")
plt.show() 

        
# %%
# testing sort_it_and_it_will_make_sense()

params = {"dataset":CIFAR100(root='../torchvision_datasets', train=True, download=True), "K":100, "n_labels":5}
non_iid_nlabels_clients, unused_data = DataSplitter(params).sort_it_and_it_will_make_sense()
# %%
