import json
from pathlib import Path
from collections import OrderedDict
from itertools import repeat
import pandas as pd
import os
import numpy as np
from glob import glob
import math

import torch
import torch.nn as nn

SEED = 123
np.random.seed(SEED)


def load_shhs_folds(np_data_path, n_folds):
    np.random.seed(SEED)

    files = sorted(glob(os.path.join(np_data_path, "*.npz")))
    npzfiles = np.asarray(files , dtype='<U200')
    np.random.shuffle(npzfiles)
    splited_files = np.array_split(npzfiles, n_folds)
    folds_data = {}
    for fold_id in range(n_folds):
        test_file = splited_files[fold_id]
        if fold_id+1<n_folds:
            valid_file = splited_files[fold_id+1]
        else:
            valid_file = splited_files[0]        

        train_file = list(set(npzfiles) - set(valid_file)- set(test_file))
        folds_data[fold_id] = {'train': train_file, 
                               'valid': valid_file, 
                               'test': test_file}
    return folds_data


def load_edf_folds(np_data_path, n_folds):
    np.random.seed(SEED)
    
    files = sorted(glob(os.path.join(np_data_path, "*.npz")))

    files_dict = dict()
    for i in files:
        file_name = os.path.split(i)[-1] 
        file_num = file_name[3:5]
        if file_num not in files_dict:
            files_dict[file_num] = [i]
        else:
            files_dict[file_num].append(i)
    
    files_pairs = []
    for i, key in enumerate(files_dict):
        files_pairs.append(files_dict[key])
    np.random.shuffle(files_pairs)
    splited_files = np.array_split(files_pairs, n_folds)

    folds_data = {}
    for fold_id in range(n_folds):
        test_file = splited_files[fold_id].tolist()
        if '20' in np_data_path:
            n_valid = 4        
        else:
            n_valid = 1
    
        if fold_id+n_valid < n_folds:
            valid_list = splited_files[fold_id+1:fold_id+1+n_valid]
            train_1 = splited_files[:fold_id]
            train_2 = splited_files[fold_id+1+n_valid:]
            train_list = train_1+train_2
    
        else: 
            valid_1 = splited_files[fold_id+1:]
            valid_2 = splited_files[:n_valid-len(valid_1)]
            valid_list = valid_1+valid_2
            train_list = splited_files[n_valid-len(valid_1):fold_id]
        
        valid_file = [sublist.tolist() for sublist in valid_list]
        train_file = [sublist.tolist() for sublist in train_list]  
        valid_file = sum(valid_file,[])
        train_file = sum(train_file,[])
            
        folds_data[fold_id] = {'train': train_file, 
                               'valid': valid_file, 
                               'test': test_file}
    return folds_data


def load_folds_semi_spervised_reduced(data_dir_sup, data_dir_unsup, n_folds):
    np.random.seed(SEED)
    
    edf_20_files = sorted(glob(os.path.join(data_dir_sup, "*.npz")))
    edf_78_files = sorted(glob(os.path.join(data_dir_unsup, "*.npz")))
    
    n_valid = 4
    files_dict = dict()
    for i in edf_20_files:
        file_name = os.path.split(i)[-1] 
        file_num = file_name[3:5]
        if file_num not in files_dict:
            files_dict[file_num] = [i]
        else:
            files_dict[file_num].append(i)
    
    files_pairs = []
    for i, key in enumerate(files_dict):
        files_pairs.append(files_dict[key])
    np.random.shuffle(files_pairs)
    splited_files = np.array_split(files_pairs, n_folds)
    
    train_edf_78 = list()        
    for i in edf_78_files:
        file_name = os.path.split(i)[-1] 
        file_num = file_name[3:5]
        if file_num not in files_dict:
            train_edf_78.append(i)

    folds_data = {}
    for fold_id in range(n_folds):
        test_file = sum(splited_files[fold_id].tolist(),[])
        
        if fold_id+n_valid < n_folds:   
            valid_list = splited_files[fold_id+1:fold_id+1+n_valid]
        else: 
            valid_1 = splited_files[fold_id+1:]
            valid_2 = splited_files[:n_valid-len(valid_1)]
            valid_list = valid_1+valid_2
        
        valid_file = [sum(sublist.tolist(), []) for sublist in valid_list]
        valid_file = sum(valid_file,[])
        train_edf_20 = list(set(edf_20_files) - set(valid_file) -set(test_file))
                
        folds_data[fold_id] = {'train_sup': train_edf_20, 
                               'train_unsup': train_edf_78,
                               'valid': valid_file, 
                               'test': test_file}
    return folds_data


torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
def weights_init_normal(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.Conv1d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.BatchNorm1d:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
        
        
def calc_class_weight(labels_count):
    total = np.sum(labels_count)
    class_weight = dict()
    num_classes = len(labels_count)

    factor = 1 / num_classes
    mu = [factor * 1.5, factor * 2, factor * 1.5, factor, factor * 1.5] # THESE CONFIGS ARE FOR SLEEP-EDF-20 ONLY

    for key in range(num_classes):
        score = math.log(mu[key] * total / float(labels_count[key]))
        class_weight[key] = score if score > 1.0 else 1.0
        class_weight[key] = round(class_weight[key] * mu[key], 2)

    class_weight = [class_weight[i] for i in range(num_classes)]

    return class_weight


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
        
        
