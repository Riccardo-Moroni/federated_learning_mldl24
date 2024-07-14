# MLDL24 Federated Learning Project: Data and System Heterogeneity Effects on Federated Learning and a Multi-Objective Approach

## FedAvg
Location: `src/fedavg.ipynb` for CIFAR-100 and `src/shakespeare/federated_shakespeare.py` for Shakespeare dataset.

Parameters are stored in `params`:
- `K` number of clients
- `C` portion of clients selected in each round
- `B` local batch size
- `J` number of local steps
- `lr_client` client training rate
- `participation` mode of client participation: `uniform` or `skewed`
- `gamma` scale of skewness in the case of skewed participation, otherwise ignored
- `rounds` number of communication rounds

Additional parameters concerning data splitting for CIFAR-100 are stored in `data_split_params`:
- `split_method` method of splitting data: `iid` or `non-iid`
- `n_labels` number of labels per client in the case of non-IID data distribution


## Loss-biased FedAvg
Location: `src/fed_loss.ipynb` for CIFAR-100 and `src/shakespeare/federated_shakespeare_mod.py` for Shakespeare dataset.

Same parameter structure as for FedAvg.

## MGDA 
Location: `src/fed_mgda.ipynb` for CIFAR-100 and `src/shakespeare/federated_shakespeare_mod.py` for Shakespeare dataset.

Same parameter structure as for FedAvg with additional parameter:
- `lr_server` server learning rate
