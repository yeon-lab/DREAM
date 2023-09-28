import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch_metric_learning import losses

##### SupervisedContrastiveLoss
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self):
        super(SupervisedContrastiveLoss, self).__init__()
        self.tau = 0.07
    def forward(self, feature_vectors, labels):
        # Normalize feature vectors
        logits = F.normalize(feature_vectors, p=2, dim=1)             
        return losses.NTXentLoss(temperature=self.tau)(logits, torch.squeeze(labels))

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
    def __init__(self, zd_dim, zx_dim, zy_dim, sampling_rate):
        super(Decoder_ResNet, self).__init__()
        self.upsample1=nn.Upsample(scale_factor=2)
        if sampling_rate == 100:
            self.dfc2 = nn.Linear(zd_dim + zx_dim + zy_dim, 6016)
            self.bn2 = nn.BatchNorm1d(6016)
            self.dfc1 = nn.Linear(6016, 32*1*94)
            self.bn1 = nn.BatchNorm1d(32*1*94)
            self.dconv3 = nn.ConvTranspose1d(32, 16, 3, padding = 1)
            self.dconv2 = nn.ConvTranspose1d(16, 16, 5, padding = 3)
            self.dconv1 = nn.ConvTranspose1d(16, 1, 12, stride = 4, padding =0)
        else: ### sampling_rate == 125
            self.dfc2 = nn.Linear(zd_dim + zx_dim + zy_dim, 7552)
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

    def forward(self, zx, zy, zd): 
        if zx is None:
            x = torch.cat((zd, zy), dim=-1)
        else:
            x = torch.cat((zd, zx, zy), dim=-1)
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
