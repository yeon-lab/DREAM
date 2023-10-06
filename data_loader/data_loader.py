import os
import torch
import numpy as np
from torch.utils.data import Dataset


class SleepDataLoader(Dataset):
    def __init__(self, config, files, phase):
        self.seq_len = config['hyper_params']['seq_len']
        self.files = files
        self.counts = None
        self.phase = phase
        self.n_domains = None
        self.n_data = None
        self.domain_idx = 0
        
        self.inputs, self.labels, self.epochs = self.split_dataset()
            
        print(phase+' info:')
        if self.n_data == len(self.inputs) and self.n_data == len(self.labels):
            print('n data:', self.n_data)
            print('n domains:', self.n_domains)
        else:
            raise Exception("data length does not match")
        
    def __len__(self):
        return len(self.epochs)

    def __getitem__(self, idx):
        file_idx, domain_idx, idx, seq_len = self.epochs[idx]
        inputs = self.inputs[file_idx][idx*seq_len:(idx+1)*seq_len]
        inputs = torch.from_numpy(inputs).float()
        labels = self.labels[file_idx][idx*seq_len:(idx+1)*seq_len]
        labels = torch.from_numpy(labels).long()
        return inputs, labels, domain_idx
        
    def split_dataset(self):
        inputs, labels, epochs = [], [], []
        all_ys = np.array([])
        
        file_idx = 0
        for file_list in self.files:
            for file in file_list:
                npz_file = np.load(file)
                inputs.append(npz_file['x'])
                labels.append(npz_file['y'])
                all_ys = np.append(all_ys, npz_file['y'])
        
                epoch_size = len(npz_file['x']) // self.seq_len
                for i in range(epoch_size):
                    epochs.append([file_idx, domain_idx, i, self.seq_len])
                file_idx += 1
            self.domain_idx += 1

        if self.phase == 'train':
            all_ys = all_ys.tolist()
            num_classes = len(np.unique(all_ys))
            counts = [all_ys.count(i) for i in range(num_classes)]
            self.counts = counts
            
        self.n_domains = self.domain_idx
        self.n_data = file_idx
        
        return inputs, labels, epochs
