import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as dist
import random
import numpy as np
from TorchCRF import CRF


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



       
    
### Follows model as seen in LEARNING ROBUST REPRESENTATIONS BY PROJECTING SUPERFICIAL STATISTICS OUT

# Decoders
class Decoder_ResNet(nn.Module):
    def __init__(self, zd_dim, zx_dim, zy_dim, sampling_rate):
        super(Decoder_ResNet, self).__init__()
        
        self.upsample1=nn.Upsample(scale_factor=2)
        
        if sampling_rate == 100:
            self.dfc3 = nn.Linear(zd_dim + zx_dim + zy_dim, 6016)
            self.bn3 = nn.BatchNorm1d(6016)
            self.dfc2 = nn.Linear(6016, 6016)
            self.bn2 = nn.BatchNorm1d(6016)
            self.dfc1 = nn.Linear(6016, 32*1*94)
            self.bn1 = nn.BatchNorm1d(32*1*94)
            self.dconv5 = nn.ConvTranspose1d(32, 32, 3, padding = 0)
            self.dconv4 = nn.ConvTranspose1d(32, 96, 3, padding = 1)
            self.dconv3 = nn.ConvTranspose1d(96, 48, 3, padding = 2)
            self.dconv2 = nn.ConvTranspose1d(48, 16, 5, padding = 3)
            self.dconv1 = nn.ConvTranspose1d(16, 1, 12, stride = 4, padding =0)
        else: 
            self.dfc3 = nn.Linear(zd_dim + zx_dim + zy_dim, 7552)
            self.bn3 = nn.BatchNorm1d(7552)
            self.dfc2 = nn.Linear(7552, 7552)
            self.bn2 = nn.BatchNorm1d(7552)
            self.dfc1 = nn.Linear(7552, 32*1*117)
            self.bn1 = nn.BatchNorm1d(32*1*117)
            self.upsample1=nn.Upsample(scale_factor=2)
            self.dconv5 = nn.ConvTranspose1d(32, 32, 3, padding = 0)
            self.dconv4 = nn.ConvTranspose1d(32, 96, 3, padding = 1)
            self.dconv3 = nn.ConvTranspose1d(96, 48, 3, padding = 2)
            self.dconv2 = nn.ConvTranspose1d(48, 16, 5, padding = 2)
            self.dconv1 = nn.ConvTranspose1d(16, 1, 12, stride = 4, padding = 1)

    def forward(self,zx, zy, zd): 
        if zx is None:
            x = torch.cat((zd, zy), dim=-1)
        else:
            x = torch.cat((zd, zx, zy), dim=-1)
            
        batch_size = x.shape[0]
        x = self.dfc3(x)
        x = F.relu(self.bn3(x))
        x = self.dfc2(x)
        x = F.relu(self.bn2(x))
        x = self.dfc1(x)
        x = F.relu(self.bn1(x))
        x = x.view(batch_size,32,-1)
        x=self.upsample1(x)
        x = self.dconv5(x)
        x = F.relu(x)
        x = F.relu(self.dconv4(x))
        x = F.relu(self.dconv3(x))
        x=self.upsample1(x)
        x = self.dconv2(x)
        x = F.relu(x)
        x=self.upsample1(x)
        x = self.dconv1(x)
        x = torch.sigmoid(x)
        return x
    

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




# Auxiliary tasks
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

class DIVA(nn.Module):
    def __init__(self, zd_dim, zy_dim, n_domains, config, d_type):
        super(DIVA, self).__init__()
        self.zd_dim = zd_dim
        self.zx_dim = 0
        self.zy_dim = zy_dim
        self.d_dim = n_domains
        self.y_dim = config['hyper_params']['num_classes']
        self.seq_len = config['hyper_params']['seq_len']
        
        if d_type == 'edf':
            self.sampling_rate = 100
        elif d_type == 'shhs':
            self.sampling_rate = 125
            

        self.px = Decoder_ResNet(self.zd_dim, self.zx_dim, self.zy_dim, self.sampling_rate)
        self.pzd = p_decoder(self.d_dim, self.zd_dim)
        self.pzy = p_decoder(self.y_dim, self.zy_dim)

        self.qzd = Encoder_ResNet(self.zd_dim, self.sampling_rate)
        if self.zx_dim != 0:
            self.qzx = Encoder_ResNet(self.zx_dim, self.sampling_rate)
        self.qzy = Encoder_ResNet(self.zy_dim, self.sampling_rate)

        # auxiliary
        self.qd = aux_layer(self.zd_dim, self.d_dim)
        self.qy = aux_layer(self.zy_dim, self.y_dim)

        self.aux_loss_multiplier_y = config['hyper_params']['aux_loss_y']
        self.aux_loss_multiplier_d = config['hyper_params']['aux_loss_d']

        self.beta_d = config['hyper_params']['beta_d']
        self.beta_x = config['hyper_params']['beta_x']
        self.beta_y = config['hyper_params']['beta_y']
        self.const_weight = config['hyper_params']['const_weight']

    def forward(self, x, y, d):
        # Encode
        zd_q_loc, zd_q_scale = self.qzd(x)
        if self.zx_dim != 0:
            zx_q_loc, zx_q_scale = self.qzx(x)
        zy_q_loc, zy_q_scale = self.qzy(x)

        # Reparameterization trick
        qzd = dist.Normal(zd_q_loc, zd_q_scale)
        zd_q = qzd.rsample()
        if self.zx_dim != 0:
            qzx = dist.Normal(zx_q_loc, zx_q_scale)
            zx_q = qzx.rsample()
        else:
            qzx = None
            zx_q = None

        qzy = dist.Normal(zy_q_loc, zy_q_scale)
        zy_q = qzy.rsample()

        # Decode
        x_recon = self.px(zx=zx_q, zy=zy_q, zd=zd_q)

        zd_p_loc, zd_p_scale = self.pzd(d)

        if self.zx_dim != 0:
            zx_p_loc, zx_p_scale = torch.zeros(zd_p_loc.size()[0], self.zx_dim).cuda(),\
                                   torch.ones(zd_p_loc.size()[0], self.zx_dim).cuda()
        zy_p_loc, zy_p_scale = self.pzy(y)

        # Reparameterization trick
        pzd = dist.Normal(zd_p_loc, zd_p_scale)
        if self.zx_dim != 0:
            pzx = dist.Normal(zx_p_loc, zx_p_scale)
        else:
            pzx = None
        pzy = dist.Normal(zy_p_loc, zy_p_scale)

        # Auxiliary losses
        d_hat = self.qd(zd_q)
        y_hat = self.qy(zy_q)

        return x_recon, d_hat, y_hat, qzd, pzd, zd_q, qzx, pzx, zx_q, qzy, pzy, zy_q, zy_q_loc

    def get_losses(self, x, y, d):        
        DIVA_losses, class_y_losses, conts_losses = 0, 0, 0
        
        d_target = d
        d_input = F.one_hot(d, num_classes= self.d_dim).float()

        for i in range(self.seq_len):
            x_input = x[:, i].view(x.size(0),1, -1)  
            y_target = y[:, i]
            y_input = F.one_hot(y_target, num_classes= self.y_dim).float() 
                  
            x_recon, d_hat, y_hat, qzd, pzd, zd_q, qzx, pzx, zx_q, qzy, pzy, zy_q, features = self.forward(x_input, y_input, d_input)

            CE_x = F.mse_loss(x_recon, x_input, reduction='sum')


            zd_p_minus_zd_q = torch.sum(pzd.log_prob(zd_q) - qzd.log_prob(zd_q))
            if self.zx_dim != 0:
                KL_zx = torch.sum(pzx.log_prob(zx_q) - qzx.log_prob(zx_q))
            else:
                KL_zx = 0

            zy_p_minus_zy_q = torch.sum(pzy.log_prob(zy_q) - qzy.log_prob(zy_q))

            CE_d = F.cross_entropy(d_hat, d_target, reduction='sum')
            CE_y = F.cross_entropy(y_hat, y_target, reduction='sum')

            DIVA_losses += CE_x \
               - self.beta_d * zd_p_minus_zd_q \
               - self.beta_x * KL_zx \
               - self.beta_y * zy_p_minus_zy_q \
               + self.aux_loss_multiplier_d * CE_d \
               + self.aux_loss_multiplier_y * CE_y
            
            class_y_losses += CE_y        
            conts_losses += contrastive_loss(features, y_target)*self.const_weight
   
        
        all_losses = (DIVA_losses+conts_losses)/self.seq_len
            
        return all_losses, class_y_losses/self.seq_len, conts_losses/self.seq_len

    def get_features(self, x):
        batch_size = x.size(0)
        f_seq = []
        with torch.no_grad():
            for i in range(self.seq_len):            
                x_input = x[:, i].view(batch_size, 1, -1)
                features, _ = self.qzy.forward(x_input)
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
                features, _ = self.qzy.forward(x_input)
                y_hat = F.softmax(self.qy(features), dim=1)
                f_seq.append(y_hat.view(batch_size,1, -1))
                
            out = torch.cat(f_seq, dim=1)
            out = out.permute(0,2,1)           # (batch_size, n_feat, len)      
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
    def __init__(self, input_size, num_channels, config, kernel_size=4, dropout=0.1, n_classes=5):
        super(TCN, self).__init__()
        
        self.batch_size = config["data_loader"]["args"]["batch_size"]
        self.mask = torch.ones((self.batch_size, config['hyper_params']['seq_len'])).byte()
        
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], n_classes)
        self.crf = CRF(n_classes)
        
    def forward(self, x):
        """Inputs have to have dimension (N, C_in, L_in) -> (batch_size, n_features, sequence_length)"""
        x = self.tcn(x)  # input should have dimension (N, C= n_feature of qzy, L)
        x = self.linear(x.transpose(1, 2))
        return x  # (N_batch, Length, Class)
        
    def get_loss(self, x, y): 
        mask = self.mask[:len(y)]
        x = self.forward(x)  # out: (N_batch, Length, Class)
        losses = self.crf.forward(x, y, mask)  # y: (batch_size, sequence_size), mask: (batch_size, sequence_size), out: (batch_size, sequence_size, num_labels)
        
        return losses.mean()
    
    def predict(self, x):
        mask = self.mask[:len(x)]
        x = self.forward(x) # out: (N_batch, Length, Class)
        x = self.crf.viterbi_decode(x, mask)
        
        return x
       
        
        
        