import torch
import torch.nn as nn
from torch.nn import functional as F
import cv2
import numpy as np
import random

def conv3(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

##### Decoders
class Decoder_ResNet(nn.Module):
    def __init__(self, zd_dim, zy_dim, sampling_rate):
        super(Decoder_ResNet, self).__init__()
        self.upsample1=nn.Upsample(scale_factor=2)
        if sampling_rate == 100:
            self.dfc2 = nn.Linear(zd_dim + zy_dim, 6016)
            self.bn2 = nn.BatchNorm1d(6016)
            self.dfc1 = nn.Linear(6016, 32*1*94)
            self.bn1 = nn.BatchNorm1d(32*1*94)
            self.dconv3 = nn.ConvTranspose1d(32, 16, 3, padding = 1)
            self.dconv2 = nn.ConvTranspose1d(16, 16, 5, padding = 3)
            self.dconv1 = nn.ConvTranspose1d(16, 1, 12, stride = 4, padding =0)
        else: ### sampling_rate == 125
            self.dfc2 = nn.Linear(zd_dim + zy_dim, 7552)
            self.bn2 = nn.BatchNorm1d(7552)
            self.dfc1 = nn.Linear(7552, 32*1*117)
            self.bn1 = nn.BatchNorm1d(32*1*117)
            self.upsample1=nn.Upsample(scale_factor=2)
            self.dconv3 = nn.ConvTranspose1d(32, 16, 3, padding = 1)
            self.dconv2 = nn.ConvTranspose1d(16, 16, 5, padding = 2)
            self.dconv1 = nn.ConvTranspose1d(16, 1, 12, stride = 4, padding = 1)
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, zy, zd): 
        x = torch.cat((zd, zy), dim=-1)
        batch_size = x.shape[0]
        x = self.dfc2(x)
        x = F.relu(self.bn2(x))
        x = self.dfc1(x)
        x = F.relu(self.bn1(x))
        x = x.view(batch_size,32,-1)
        x = self.upsample1(x)
        x = F.relu(self.dconv3(x))
        x = self.upsample1(x)
        x = F.relu(self.dconv2(x))
        x = self.upsample1(x)
        x = torch.sigmoid(self.dconv1(x))
        return x
    
#### prior network
class p_decoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(p_decoder, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(in_dim, out_dim, bias=False), nn.BatchNorm1d(out_dim), nn.ReLU())
        self.fc21 = nn.Sequential(nn.Linear(out_dim, out_dim))
        self.fc22 = nn.Sequential(nn.Linear(out_dim, out_dim), nn.Softplus())
        torch.nn.init.xavier_uniform_(self.fc1[0].weight)
        torch.nn.init.xavier_uniform_(self.fc21[0].weight)
        self.fc21[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc22[0].weight)
        self.fc22[0].bias.data.zero_()
    def forward(self, x):
        x = self.fc1(x)
        loc = self.fc21(x)
        scale = self.fc22(x) + 1e-7
        return loc, scale


# Encoders
class Encoder_ResNet(nn.Module):
    def __init__(self, out_dim, sampling_rate):
        super(Encoder_ResNet, self).__init__()
        self.layer_config_dict = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3]
        }
        self.inplanes = 16
        self.layers = self.layer_config_dict[50]

        self.initial_layer = nn.Sequential(
            nn.Conv1d(1, 16, 7, 2, 3, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(3, 2, 1))
        self.layer1 = self._make_layer(Bottleneck, 16, self.layers[0], stride=1, first=True)
        self.layer2 = self._make_layer(Bottleneck, 16, self.layers[1], stride=2)
        self.layer3 = self._make_layer(Bottleneck, 32, self.layers[2], stride=2)
        self.layer4 = self._make_layer(Bottleneck, 32, self.layers[3], stride=2)
        self.maxpool = nn.MaxPool1d(3, 2, 1)
        self.dropout = nn.Dropout(p=0.01)
        
        x = torch.rand((2,1,sampling_rate*30))      
        x = self.initial_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.dropout(x)
        x = x.view(2,-1)
        
        self.fc11 = nn.Sequential(nn.Linear(x.shape[1], out_dim))
        self.fc12 = nn.Sequential(nn.Linear(x.shape[1], out_dim), nn.Softplus())
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        torch.nn.init.xavier_uniform_(self.fc11[0].weight)
        self.fc11[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc12[0].weight)
        self.fc12[0].bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, first=False):
        downsample = None
        if (stride != 1 and first is False) or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.initial_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.dropout(x)
        x = x.view(batch_size, -1)
        loc = self.fc11(x)
        scale = self.fc12(x) + 1e-7
        return loc, scale   # (batch_size, out_dim)

# Auxiliary networks
class aux_layer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(aux_layer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.zero_()
    def forward(self, x):
        h = F.relu(x)
        loc = self.fc(h)
        return loc


class Augmentation:
    def __init__(self, sampling_rate):
        self.size = sampling_rate*30
        
    def permute(self,signal, pieces):
        """
        signal: numpy array (batch x window)
        pieces: number of segments along time
        """
        signal = signal.cpu().numpy()
        signal = signal.T
        pieces = int(np.ceil(np.shape(signal)[0] / (np.shape(signal)[0] // pieces)).tolist()) 
        piece_length = int(np.shape(signal)[0] // pieces)
        sequence = list(range(0, pieces))
        np.random.shuffle(sequence)

        permuted_signal = np.reshape(signal[:(np.shape(signal)[0] // pieces * pieces)],
                                     (pieces, piece_length,-1)).tolist()

        tail = signal[(np.shape(signal)[0] // pieces * pieces):]
        permuted_signal = np.asarray(permuted_signal)[sequence]
        permuted_signal = np.concatenate(permuted_signal, axis=0)
        permuted_signal = np.concatenate((permuted_signal,tail[:,0]), axis=0)
        permuted_signal = permuted_signal[:,None]
        permuted_signal = permuted_signal.T
        return permuted_signal
    
    def crop_resize(self, signal, size):
        signal = signal.cpu().numpy()
        signal = signal.T
        signal_shape = signal.shape
        size = signal.shape[0] * size
        size = int(size)
        start = random.randint(0, signal.shape[0]-size)
        crop_signal = signal[start:start + size,:]

        crop_signal = cv2.resize(crop_signal, (1, self.size), interpolation=cv2.INTER_LINEAR)

        crop_signal = crop_signal.T
        return crop_signal
