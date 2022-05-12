import torch
import torch.nn as nn
from torch.autograd import Variable

def conv3(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv1(out)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


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


class ResNetFeature(nn.Module):

    def __init__(self, config):

        super(ResNetFeature, self).__init__()

        self.layer_config_dict = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3]
        }
        self.inplanes = 16
        self.config = config
        self.layers = self.layer_config_dict[config['hyper_params']['num_layers']]
        if config['hyper_params']['num_layers'] == 18 or config['hyper_params']['num_layers'] == 34:
            block = BasicBlock
        elif config['hyper_params']['num_layers'] == 50 or config['hyper_params']['num_layers'] == 101 or config['hyper_params']['nun_layers'] == 152:
            block = Bottleneck
        else:
            raise NotImplementedError("num layers '{}' is not in layer config".format(config['hyper_params']['num_layers']))

        self.initial_layer = nn.Sequential(
            nn.Conv1d(1, 16, 7, 2, 3, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(3, 2, 1))

        self.layer1 = self._make_layer(block, 16, self.layers[0], stride=1, first=True)
        self.layer2 = self._make_layer(block, 16, self.layers[1], stride=2)
        self.layer3 = self._make_layer(block, 32, self.layers[2], stride=2)
        self.layer4 = self._make_layer(block, 32, self.layers[3], stride=2)
        self.maxpool = nn.MaxPool1d(3, 2, 1)

        self.dropout = nn.Dropout(p=config['hyper_params']['dropout_rate'])
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
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
        f_seq = []

        for i in range(self.config['hyper_params']['seq_len']):
            f = self.initial_layer(x[:, i].view(x.size(0), 1, -1))
            f = self.layer1(f)
            f = self.layer2(f)
            f = self.maxpool(f)
            f = self.layer3(f)
            f = self.layer4(f)
            f = self.dropout(f)
            f_seq.append(f.permute(0, 2, 1))

        out = torch.cat(f_seq, dim=1)

        return out
        
        
class ResNetFeature_RNN(nn.Module):

    def __init__(self, config):

        super(ResNetFeature_RNN, self).__init__()

        self.layer_config_dict = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3]
        }
        self.inplanes = 16
        self.config = config
        self.layers = self.layer_config_dict[config['hyper_params']['num_layers']]
        if config['hyper_params']['num_layers'] == 18 or config['hyper_params']['num_layers'] == 34:
            block = BasicBlock
        elif config['hyper_params']['num_layers'] == 50 or config['hyper_params']['num_layers'] == 101 or config['hyper_params']['nun_layers'] == 152:
            block = Bottleneck
        else:
            raise NotImplementedError("num layers '{}' is not in layer config".format(config['hyper_params']['num_layers']))

        self.initial_layer = nn.Sequential(
            nn.Conv1d(1, 16, 7, 2, 3, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(3, 2, 1))

        self.layer1 = self._make_layer(block, 16, self.layers[0], stride=1, first=True)
        self.layer2 = self._make_layer(block, 16, self.layers[1], stride=2)
        self.layer3 = self._make_layer(block, 32, self.layers[2], stride=2)
        self.layer4 = self._make_layer(block, 32, self.layers[3], stride=2)
        self.maxpool = nn.MaxPool1d(3, 2, 1)

        self.dropout = nn.Dropout(p=config['hyper_params']['dropout_rate'])
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
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
        f = self.initial_layer(x)
        f = self.layer1(f)
        f = self.layer2(f)
        f = self.maxpool(f)
        f = self.layer3(f)
        f = self.layer4(f)
        f = self.dropout(f)
        f = f.permute(0, 2, 1)

        return f
        
        

class LSTM(nn.Module):
    def __init__(self, config, n_layer=2, bidirectional=True):
        super(LSTM, self).__init__()
        
        self.n_direct = int(bidirectional)+1
        self.input_dim = 128
        self.hidden_size = config['hyper_params']['hidden_dim']
        
        self.feature = ResNetFeature_RNN(config)
        

        self.lstm = nn.LSTM(input_size = self.input_dim, hidden_size = self.hidden_size, batch_first=True, 
        num_layers = n_layer, bidirectional=bidirectional)

        self.fc = nn.Linear(self.hidden_size*self.n_direct, config['hyper_params']['num_classes'])

        self.softmax = nn.Softmax(dim=1)
        

    def forward(self, x): #in: (batch_size, seq_len, n_feat) if batch_first=True
        #out:(batch_size, seq_len, hidden*n_direct). / h_n&c_n :(n_direct*n_layer, batch_size, n_hidden)
        x = self.feature(x)
        out, (h_n, c_n) = self.lstm(x)
        out = self.fc(out[:,-1]) 
        out = self.softmax(out)
        return out

class GRU(nn.Module):
    def __init__(self, config, n_layer=2, bidirectional=True):
        super(GRU, self).__init__()
        
        self.n_direct = int(bidirectional)+1
        self.input_dim = 128
        self.hidden_size = config['hyper_params']['hidden_dim']
        
        self.feature = ResNetFeature_RNN(config)

        self.gru = nn.GRU(input_size = self.input_dim, hidden_size = self.hidden_size, num_layers = n_layer, 
                            batch_first=True, bidirectional=bidirectional)

        self.fc = nn.Linear(self.hidden_size*self.n_direct, config['hyper_params']['num_classes'])

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        #out:(batch_size, seq_len, hidden*n_direct). / h_n&c_n :(n_direct*n_layer, batch_size, n_hidden)
        x = self.feature(x)
        out, _ = self.gru(x)
        out = self.fc(out[:,-1]) 
        out = self.softmax(out)
        return out