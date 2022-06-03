import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as dist
import random
import numpy as np
from TorchCRF import CRF
from pytorch_metric_learning import losses

import numpy as np
import torch
import math
import cv2

class Transform:
    def __init__(self, sampling_rate=100):
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

def Self_SupervisedContrastiveLoss(x, criterion):
    LARGE_NUM = 1e9
    temperature = 0.1
    x = F.normalize(x, dim=-1)

    num = int(x.shape[0] / 2)
    hidden1, hidden2 = torch.split(x, num)

    hidden1_large = hidden1
    hidden2_large = hidden2
    labels = torch.arange(0,num).to('cuda')
    masks = F.one_hot(torch.arange(0,num), num).to('cuda')


    logits_aa = torch.matmul(hidden1, hidden1_large.T) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = torch.matmul(hidden2, hidden2_large.T) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = torch.matmul(hidden1, hidden2_large.T) / temperature
    logits_ba = torch.matmul(hidden2, hidden1_large.T) / temperature


    loss_a = criterion(torch.cat([logits_ab, logits_aa], 1),
        labels)
    loss_b = criterion(torch.cat([logits_ba, logits_bb], 1),
        labels)
    loss = torch.mean(loss_a + loss_b)
    return loss
    

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



       
    
### Follows model as seen in LEARNING ROBUST REPRESENTATIONS BY PROJECTING SUPERFICIAL STATISTICS OUT

# Decoders
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
        else: 
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

class DIVA(nn.Module):
    def __init__(self, zd_dim, zy_dim, n_domains, config, d_type):
        super(DIVA, self).__init__()
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
        self.transformer= Transform()
        self.criterion = nn.CrossEntropyLoss()
            

        self.px = Decoder_ResNet(self.zd_dim, self.zx_dim, self.zy_dim, self.sampling_rate)
        self.pzy = p_decoder(self.y_dim, self.zy_dim)
        self.pzd = p_decoder(self.d_dim, self.zd_dim)
        
        self.qzy = Encoder_ResNet(self.zy_dim, self.sampling_rate)
        self.qzd = Encoder_ResNet(self.zd_dim, self.sampling_rate)
        if self.zx_dim != 0:
            self.qzx = Encoder_ResNet(self.zx_dim, self.sampling_rate)

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
        zd_q_loc, zd_q_scale = self.qzd(x)        # Encode
        qzd = dist.Normal(zd_q_loc, zd_q_scale)   # Reparameterization trick
        zd_q = qzd.rsample() 

        if self.zx_dim != 0:
            zx_q_loc, zx_q_scale = self.qzx(x)
            qzx = dist.Normal(zx_q_loc, zx_q_scale)
            zx_q = qzx.rsample()
        else:
            qzx = None
            zx_q = None

        zy_q_loc, zy_q_scale = self.qzy(x)          # Encode
        qzy = dist.Normal(zy_q_loc, zy_q_scale)     # Reparameterization trick
        zy_q = qzy.rsample()

        # Decode
        x_recon = self.px(zx=zx_q, zy=zy_q, zd=zd_q)
        
        zd_p_loc, zd_p_scale = self.pzd(d)
        pzd = dist.Normal(zd_p_loc, zd_p_scale)
        d_hat = self.qd(zd_q)

        if self.zx_dim != 0:
            zx_p_loc, zx_p_scale = torch.zeros(zd_p_loc.size()[0], self.zx_dim).cuda(),\
                                   torch.ones(zd_p_loc.size()[0], self.zx_dim).cuda()
            pzx = dist.Normal(zx_p_loc, zx_p_scale)
        else:
            pzx = None
            

        if y is not None:
            zy_p_loc, zy_p_scale = self.pzy(y)
        else:
             # Create labels and repeats of zy_q and qzy
            y_onehot = torch.eye(self.y_dim)
            y_onehot = y_onehot.repeat(1, 100)
            y_onehot = y_onehot.view(-1, self.y_dim).cuda()

            zy_q = zy_q.repeat(10, 1)
            zy_q_loc, zy_q_scale = zy_q_loc.repeat(10, 1), zy_q_scale.repeat(10, 1)
            qzy = dist.Normal(zy_q_loc, zy_q_scale)
            
            zy_p_loc, zy_p_scale = self.pzy(y_onehot)
        
        pzy = dist.Normal(zy_p_loc, zy_p_scale)
        y_hat = self.qy(zy_q)

        return x_recon, d_hat, y_hat, qzd, pzd, zd_q, qzx, pzx, zx_q, qzy, pzy, zy_q, zy_q_loc

    def get_losses(self, x, y, d):        
        DIVA_losses, conts_losses, CE_class, CE_domain = 0, 0, 0, 0
        KL_domain, KL_class, reconst_losses = 0, 0, 0
        
        d_target = d
        d_input = F.one_hot(d, num_classes= self.d_dim).float()

        for i in range(self.seq_len):
            x_input = x[:, i].view(x.size(0),1, -1)
            if y is not None:
                y_target = y[:, i]
                y_input = F.one_hot(y_target, num_classes= self.y_dim).float() 
                  
                x_recon, d_hat, y_hat, qzd, pzd, zd_q, qzx, pzx, zx_q, qzy, pzy, zy_q, features = self.forward(x=x_input, y=y_input, d=d_input)
    
                CE_x = F.mse_loss(x_recon, x_input, reduction='sum')
                CE_d = F.cross_entropy(d_hat, d_target, reduction='sum')
    
                zd_p_minus_zd_q = torch.sum(pzd.log_prob(zd_q) - qzd.log_prob(zd_q))
                
                if self.zx_dim != 0:
                    KL_zx = torch.sum(pzx.log_prob(zx_q) - qzx.log_prob(zx_q))
                else:
                    KL_zx = 0
            
                CE_y = F.cross_entropy(y_hat, y_target, reduction='sum')
                zy_p_minus_zy_q = torch.sum(pzy.log_prob(zy_q) - qzy.log_prob(zy_q))
                
                DIVA_losses += CE_x \
                   - self.beta_d * zd_p_minus_zd_q \
                   - self.beta_x * KL_zx \
                   - self.beta_y * zy_p_minus_zy_q \
                   + self.aux_loss_multiplier_d * CE_d \
                   + self.aux_loss_multiplier_y * CE_y
                
                CE_class += CE_y    
                CE_domain += CE_d
                reconst_losses += CE_x
                
                conts_losses += self.contrastive_loss(features, y_target)*self.const_weight
                
                KL_domain += zd_p_minus_zd_q
                KL_class += zy_p_minus_zy_q
                
            else:
                y_input = None
                
                
                x_croped = self.transformer.crop_resize(x_input,random.uniform(0.25,0.75))
                x_croped  = torch.FloatTensor(x_croped).cuda()
                
                x_permuted = self.transformer.permute(x_input,random.randint(5,20))
                x_permuted  = torch.FloatTensor(x_permuted).cuda()
                 
                feature_set = torch.tensor([]).cuda()
                for x_transformed in [x_croped, x_permuted]:
                    x_recon, d_hat, y_hat, qzd, pzd, zd_q, qzx, pzx, zx_q, qzy, pzy, zy_q, features = self.forward(x=x_transformed, y=y_input, d=d_input)
                    
                    feature_set = torch.cat((feature_set,features), dim=0)
                    
                    CE_x = F.mse_loss(x_recon, x_input, reduction='sum')
                    CE_d = F.cross_entropy(d_hat, d_target, reduction='sum')
        
                    zd_p_minus_zd_q = torch.sum(pzd.log_prob(zd_q) - qzd.log_prob(zd_q))
                    
                    if self.zx_dim != 0:
                        KL_zx = torch.sum(pzx.log_prob(zx_q) - qzx.log_prob(zx_q))
                    else:
                        KL_zx = 0
                        
                    y_onehot = torch.eye(self.y_dim)
                    y_onehot = y_onehot.repeat(1, 100)
                    y_onehot = y_onehot.view(-1, self.y_dim).cuda()
                    
                    alpha_y = F.softmax(y_hat, dim=-1)
                    qy = dist.OneHotCategorical(alpha_y)
                    prob_qy = torch.exp(qy.log_prob(y_onehot))
                    
                    zy_p_minus_zy_q = torch.sum(pzy.log_prob(zy_q) - qzy.log_prob(zy_q), dim=-1)
        
                    marginal_zy_p_minus_zy_q = torch.sum(prob_qy * zy_p_minus_zy_q)
        
                    prior_y = torch.tensor(1/10).cuda()
                    prior_y_minus_qy = torch.log(prior_y) - qy.log_prob(y_onehot)
                    marginal_prior_y_minus_qy = torch.sum(prob_qy * prior_y_minus_qy)
        
                    DIVA_losses += (CE_x \
                           - self.beta_d * zd_p_minus_zd_q \
                           - self.beta_x * KL_zx \
                           - self.beta_y * marginal_zy_p_minus_zy_q \
                           - marginal_prior_y_minus_qy \
                           + self.aux_loss_multiplier_d * CE_d)*0.5                      
           
                    
                conts_losses += Self_SupervisedContrastiveLoss(feature_set, self.criterion)*self.const_weight
                

        all_losses = (DIVA_losses+conts_losses)/self.seq_len
        

        return all_losses, DIVA_losses/self.seq_len, CE_class/self.seq_len, CE_domain/self.seq_len, conts_losses/self.seq_len, KL_domain/self.seq_len, KL_class/self.seq_len, reconst_losses/self.seq_len

  
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
                features, _ = self.qzy.forward(x_input)
                alpha = F.softmax(self.qy(features), dim=1)
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
       
