# %%
import json
import numpy as np
import random
from tqdm import tqdm

def shakespeare_data_pruning(json_train_path, json_test_path, crop_amount=2000, n_clients=100):
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
            X_test_cropped[k] = test_dict['user_data'][k]['x']
            Y_test_cropped[k] = test_dict['user_data'][k]['y']
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


def tokenize_encode(my_list, vocab, char_to_idx):
    oov_index = len(vocab)-1
    new_list = []
    for sentence in my_list:
        characters = list(sentence)

        encoded = []
        for char in characters:
            if char in char_to_idx:
                encoded.append(char_to_idx[char])
            else:
                encoded.append(oov_index)
    
        new_list.append(encoded)
    return new_list

def concat_dict_values(my_dict):
    concat = []
    for v in my_dict.values():
        if isinstance(v, list):
            concat.extend(v)
        else:
            concat.append(v)
    return concat
# %%
