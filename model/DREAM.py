import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as dist
import random
import numpy as np
from TorchCRF import CRF
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

class VAE(nn.Module):
    def __init__(self, zd_dim, zy_dim, n_domains, config, d_type):
        super(VAE, self).__init__()
        SEED = 1111
        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
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
            
        self.contrastive_loss = SupervisedContrastiveLoss()
            
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
        VAE_losses, conts_losses = 0, 0

        
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

            VAE_losses += CE_x \
               - self.beta_d * zd_p_minus_zd_q \
               - self.beta_x * KL_zx \
               - self.beta_y * zy_p_minus_zy_q \
               + self.aux_loss_multiplier_d * CE_d \
               + self.aux_loss_multiplier_y * CE_y
            
            conts_losses += self.contrastive_loss(features, y_target)*self.const_weight
   
        return (VAE_losses+conts_losses)/self.seq_len


    def get_features(self, x):
        batch_size = x.size(0)
        f_seq = []
        for i in range(self.seq_len):            
            x_input = x[:, i].view(batch_size, 1, -1)
            features, _ = self.qzy.forward(x_input)
            f_seq.append(features.view(batch_size,1, -1))

        out = torch.cat(f_seq, dim=1)
        return out # (batch_size,len, n_feat)   
    
    def predict(self, x):
        batch_size = x.size(0)
        f_seq = []
        with torch.no_grad():
            for i in range(self.seq_len):   
                x_input = x[:, i].view(batch_size, 1, -1)
                zy, _ = self.qzy.forward(x_input)
                alpha = F.softmax(self.qy(zy), dim=1)
                res, ind = torch.topk(alpha, 1)
                y = x_input.new_zeros(alpha.size())
                y = y.scatter_(1, ind, 1.0)
                f_seq.append(y.view(batch_size,1, -1))
                
            out = torch.cat(f_seq, dim=1)
            out = out.permute(0,2,1)           # (batch_size, n_class, len)      
        return out
    

    
class Transformer(nn.Module):
    def __init__(self, input_size, config, n_layer=4, n_classes=5):
        super(Transformer, self).__init__()
        SEED = 1111
        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        self.hidden_dim = input_size
        self.batch_size = config["data_loader"]["args"]["batch_size"]
        self.dim_feedforward = config['hyper_params']['dim_feedforward']
        self.is_CFR =  config['hyper_params']['is_CFR']
        try:
            self.n_layer = config['hyper_params']['n_layers']
        except:
            self.n_layer = n_layer

        if self.is_CFR  is True:
            self.mask = torch.ones((self.batch_size, config['hyper_params']['seq_len'])).byte().cuda()
            self.crf = CRF(n_classes)
        else: 
            self.criterion = nn.CrossEntropyLoss()
            self.softmax = nn.Softmax(dim=2) 
            
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=8, batch_first=True, dim_feedforward=self.dim_feedforward) 
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.n_layer)
        self.fc = nn.Linear(self.hidden_dim, n_classes)
        
    def forward(self, x): #in: (batch, seq, feature) if batch_first=True
        #out:(batch_size, seq_len, feature).
        x = self.transformer_encoder(x)
        x = self.fc(x)
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
       
