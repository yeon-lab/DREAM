import os
import torch
import numpy as np
from torch.utils.data import Dataset


class SleepDataLoader(Dataset):
    def __init__(self, config, files, m_type, train=True):
        self.seq_len = config['hyper_params']['seq_len']
        self.m_type = m_type
        self.files = files
        self.train=train
        self.counts=None
        self.check_shape=True

        self.inputs, self.labels, self.epochs = self.split_dataset()
        print('x length:', len(self.inputs))
        print('y length:', len(self.labels))
        
    def __len__(self):
        return len(self.epochs)

    def __getitem__(self, idx):
        file_idx, idx, seq_len = self.epochs[idx]
        
        if self.m_type == 'Utime':
            inputs = self.inputs[file_idx][idx*seq_len:(idx+1)*seq_len]
            inputs = torch.from_numpy(inputs).float()
            inputs = inputs.reshape(1,-1)
            
            labels = self.labels[file_idx][idx*seq_len:(idx+1)*seq_len]
            labels = torch.from_numpy(labels).long()
        elif self.m_type == 'AttnSleep':
            inputs = self.inputs[file_idx][idx]
            inputs = torch.from_numpy(inputs).float()
            inputs = inputs.reshape(1,-1)
    
            labels = self.labels[file_idx][idx]
            labels = torch.tensor(labels).long()
            
        elif self.m_type == 'IITNet':
            inputs = self.inputs[file_idx][idx:idx+seq_len]
            inputs = torch.from_numpy(inputs).float()
            
            labels = self.labels[file_idx][idx:idx+seq_len]
            labels = torch.from_numpy(labels).long()
            labels = labels[-1]
            
        elif self.m_type == 'Proposed_method':
            inputs = self.inputs[file_idx][idx:idx+seq_len]
            inputs = torch.from_numpy(inputs).float()
            
            labels = self.labels[file_idx][idx:idx+seq_len]
            labels = torch.from_numpy(labels).long()
            
            domains = file_idx
            if self.check_shape:
                print('\nmodel type: {}'.format(self.m_type))
                print('x shape: {}'.format(inputs.shape))
                print('y shape: {}\n'.format(labels.shape))
                self.check_shape = False
            return inputs, labels, domains
            
            
        if self.train and self.check_shape:
            print('\nmodel type: {}'.format(self.m_type))
            print('x shape: {}'.format(inputs.shape))
            print('y shape: {}\n'.format(labels.shape))
            self.check_shape = False

        return inputs, labels

    def split_dataset(self):

        inputs, labels, epochs = [], [], []
        all_ys = np.array([])

        for file_idx, file in enumerate(self.files):
            npz_file = np.load(file)
            inputs.append(npz_file['x'])
            labels.append(npz_file['y'])
            all_ys = np.append(all_ys, npz_file['y'])
            
            if self.m_type == 'Utime':
                epoch_size = len(npz_file['x']) // self.seq_len
                for i in range(epoch_size):
                    epochs.append([file_idx, i, self.seq_len])
                    
            elif self.m_type == 'AttnSleep':
                epoch_size = len(npz_file['x'])
                for i in range(epoch_size):
                    epochs.append([file_idx, i, self.seq_len])
                    
            elif self.m_type == 'IITNet':
                epoch_size = len(npz_file['x']) - self.seq_len + 1
                for i in range(epoch_size):
                    epochs.append([file_idx, i, self.seq_len])
                    
            elif self.m_type == 'Proposed_method':
                epoch_size = len(npz_file['x']) - self.seq_len + 1
                for i in range(epoch_size):
                    epochs.append([file_idx, i, self.seq_len])
                
        if self.train:
            all_ys = all_ys.tolist()
            num_classes = len(np.unique(all_ys))
            counts = [all_ys.count(i) for i in range(num_classes)]
            self.counts = counts

        return inputs, labels, epochs