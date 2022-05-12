import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as dist
import random
import numpy as np
from TorchCRF import CRF
from torch.autograd import Variable

def contrastive_loss(features, y, n_classes=5):
    batch_size = features.shape[0]
    features = features.view(batch_size, -1)
    features = F.normalize(features, dim=1)
    mat = torch.matmul(features, features.T).detach().cpu().numpy()
    loss_all = (np.exp(mat/1).sum().item()-len(features))/2
    loss = 0
    for class_ in range(n_classes):
        index, = torch.where(y == class_)
        index = index.cpu().numpy()
        mat_ = torch.matmul(features[index], features[index].T).detach().cpu().numpy()
        loss += (np.exp(mat_/1).sum().item()-len(index))/2

    cont_loss= -np.log(loss/loss_all)

    return cont_loss
    

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
        
        self.fc = nn.Sequential(nn.Linear(x.shape[1], out_dim))

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)   


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
        
        x = self.fc(x)

        return x   # (batch_size, out_dim)



# Auxiliary tasks
class aux_layer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(aux_layer, self).__init__()

        self.fc = nn.Linear(in_dim, out_dim)

        torch.nn.init.xavier_normal_(self.fc.weight)
        self.fc.bias.data.zero_()

    def forward(self, x):
        x = F.relu(x)
        x = self.fc(x)
        return x


class DIVA(nn.Module):
    def __init__(self, zy_dim, config, d_type):
        super(DIVA, self).__init__()
        self.zy_dim = zy_dim
        self.y_dim = config['hyper_params']['num_classes']
        self.seq_len = config['hyper_params']['seq_len']
        
        if d_type == 'edf':
            self.sampling_rate = 100
        elif d_type == 'shhs':
            self.sampling_rate = 125
            
        self.qzy = Encoder_ResNet(self.zy_dim, self.sampling_rate)
        self.qy = aux_layer(self.zy_dim, self.y_dim)


    def forward(self, x):
        features = self.qzy.forward(x)
        preds = F.softmax(self.qy(features), dim=1)

        return features, preds
        

    def get_losses(self, x, y, d):        
        class_loss, conts_loss = 0, 0
        
        for i in range(self.seq_len):
            x_input = x[:, i].view(x.size(0),1, -1)  
            y_target = y[:, i]

            features, preds = self.forward(x_input)

            class_loss += F.cross_entropy(preds, y_target)        
            conts_loss += contrastive_loss(features, y_target)
   
        all_loss = (class_loss+conts_loss)/self.seq_len

        return all_loss, class_loss/self.seq_len, conts_loss/self.seq_len
        
        
    def get_features(self, x):
        batch_size = x.size(0)
        f_seq = []
        for i in range(self.seq_len):            
            x_input = x[:, i].view(batch_size, 1, -1)
            features = self.qzy.forward(x_input)
            f_seq.append(features.view(batch_size,1, -1))

        out = torch.cat(f_seq, dim=1)
        out = out.permute(0,2,1)   # (batch_size, n_feat, len)   
        return out

    def predict(self, x):
        batch_size = x.size(0)
        f_seq = []
        with torch.no_grad():
            for i in range(self.seq_len):   
                x_input = x[:, i].view(batch_size, 1, -1)
                features = self.qzy.forward(x_input)
                preds = F.softmax(self.qy(features), dim=1)
                f_seq.append(preds.view(batch_size,1, -1))
                
            out = torch.cat(f_seq, dim=1)
            out = out.permute(0,2,1)           # (batch_size, n_class, len)      
        return out
    
    
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
    
class TCN(nn.Module):
    def __init__(self, input_size, num_channels, config, n_classes=5):
        super(TCN, self).__init__()
        
        self.batch_size = config["data_loader"]["args"]["batch_size"]
        
        params = config['hyper_params']
        self.is_CFR =  params['is_CFR']
        
        if self.is_CFR  is True:
            self.mask = torch.ones((self.batch_size, config['hyper_params']['seq_len'])).byte().cuda()
            self.crf = CRF(n_classes)
        else: 
            self.criterion = nn.CrossEntropyLoss()
            self.softmax = nn.Softmax(dim=2) 

        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=params['kernel_size'], dropout=params['dropout_rate'])
        self.linear = nn.Linear(num_channels[-1], n_classes)
        
        
    def forward(self, x):
        """Inputs have to have dimension (batch_size, n_features, sequence_length)"""
        x = self.tcn(x)  # input should have dimension (N, C= n_feature of qzy, L)
        x = self.linear(x.transpose(1, 2))
        return x  # (N_batch, Length, Class)
        
    def get_loss(self, x, y): 
        x = self.forward(x)  # out: (N_batch, Length, Class)

        if self.is_CFR is True:
            mask = self.mask[:len(y)]
            loss = self.crf.forward(x, y, mask)  # y: (batch_size, sequence_size), mask: (batch_size, sequence_size), out: (batch_size, sequence_size, num_labels)
            loss = -loss.mean()
        else:
            x = self.softmax(x)  # (N_batch, Length, Class)
            x = x.permute(0,2,1) # (N_batch, Class, L)
            loss =  self.criterion(x, y) # input:(N, C, L), out:(N,L)
            
        return loss
    
    def predict(self, x):
        x = self.forward(x) # out: (N_batch, Length, Class)
        if self.is_CFR is True:
            mask = self.mask[:len(x)]
            x = self.crf.viterbi_decode(x, mask)
        else:
            x = self.softmax(x)  # (N_batch, Length, Class)
            x = x.permute(0,2,1) # (N_batch, Class, L)
            x = x.data.max(1)[1].cpu()
        return x
       
        
        
        