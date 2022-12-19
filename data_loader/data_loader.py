import os
import torch
import numpy as np
from torch.utils.data import Dataset


class SleepDataLoader(Dataset):
    def __init__(self, config, files, d_type, phase):
        self.seq_len = config['hyper_params']['seq_len']
        self.files = files
        self.counts = None
        self.phase = phase
        self.n_domains = 0
        self.n_data = 0
        self.check_shape = True
        
        if d_type == 'edf':
            self.inputs, self.labels, self.epochs = self.split_dataset_edf()
        elif d_type == 'shhs':
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
        
        if self.check_shape and self.phase == 'train':
            print('\nx shape: {}'.format(inputs.shape))
            print('y shape: {}\n'.format(labels.shape))
            print('-'*100)
            self.check_shape = False

        return inputs, labels, domain_idx
            

    def split_dataset(self):

        inputs, labels, epochs = [], [], []
        all_ys = np.array([])

        for file_idx, file in enumerate(self.files):
            npz_file = np.load(file)
            inputs.append(npz_file['x'])
            labels.append(npz_file['y'])
            all_ys = np.append(all_ys, npz_file['y'])
            
            epoch_size = len(npz_file['x']) // self.seq_len
            for i in range(epoch_size):
                epochs.append([file_idx, file_idx, i, self.seq_len])
                
        if self.phase == 'train':
            all_ys = all_ys.tolist()
            num_classes = len(np.unique(all_ys))
            counts = [all_ys.count(i) for i in range(num_classes)]
            self.counts = counts
            
        self.n_domains = file_idx+1
        self.n_data = file_idx+1

        return inputs, labels, epochs
        
    def split_dataset_edf(self):

        inputs, labels, epochs = [], [], []
        all_ys = np.array([])
        
        file_idx = 0
        for domain_idx, file_list in enumerate(self.files):
            for file in file_list:
                npz_file = np.load(file)
                inputs.append(npz_file['x'])
                labels.append(npz_file['y'])
                all_ys = np.append(all_ys, npz_file['y'])
        
                epoch_size = len(npz_file['x']) // self.seq_len
                for i in range(epoch_size):
                    epochs.append([file_idx, domain_idx, i, self.seq_len])
                    
                file_idx += 1

                
        if self.phase == 'train':
            all_ys = all_ys.tolist()
            num_classes = len(np.unique(all_ys))
            counts = [all_ys.count(i) for i in range(num_classes)]
            self.counts = counts
            
        self.n_domains = domain_idx+1
        self.n_data = file_idx
        
        return inputs, labels, epochs
