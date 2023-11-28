import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as dist
import random
from TorchCRF import CRF

import torch
from model.base import Decoder_ResNet, Encoder_ResNet, p_decoder, aux_layer, Augmentation
from model.loss import SupervisedContrastiveLoss, Self_SupervisedContrastiveLoss
SEED = 1111
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

class VAE(nn.Module):
    def __init__(self, config, n_domains, sampling_rate):
        super(VAE, self).__init__()
        self.d_dim = n_domains
        params = config['hyper_params']
        self.zd_dim = params['zd_dim']
        self.zy_dim = params['zy_dim']
        self.y_dim = params['num_classes']
        self.seq_len = params['seq_len']
        self.sampling_rate = sampling_rate
            
        self.contrastive_loss = SupervisedContrastiveLoss()
        self.augmentor = Augmentation(sampling_rate=sampling_rate)
        self.criterion = nn.CrossEntropyLoss()
            
        self.px = Decoder_ResNet(self.zd_dim, self.zy_dim, self.sampling_rate)
        self.pzy = p_decoder(self.y_dim, self.zy_dim)
        self.pzd = p_decoder(self.d_dim, self.zd_dim)
        
        self.qzy = Encoder_ResNet(self.zy_dim, self.sampling_rate)
        self.qzd = Encoder_ResNet(self.zd_dim, self.sampling_rate)

        # auxiliary
        self.qd = aux_layer(self.zd_dim, self.d_dim)
        self.qy = aux_layer(self.zy_dim, self.y_dim)

        self.aux_loss_multiplier_y = params['aux_loss_y']
        self.aux_loss_multiplier_d = params['aux_loss_d']

        self.beta_y, self.beta_d = params['beta_y'], params['beta_d']
        self.const_weight = params['const_weight']

        
    def forward(self, x, y, d):
        zd_q_loc, zd_q_scale = self.qzd(x)
        zy_q_loc, zy_q_scale = self.qzy(x)

        qzd = dist.Normal(zd_q_loc, zd_q_scale)
        zd_q = qzd.rsample()
        qzy = dist.Normal(zy_q_loc, zy_q_scale)
        zy_q = qzy.rsample()

        # Decode
        x_recon = self.px(zy=zy_q, zd=zd_q)
        
        zd_p_loc, zd_p_scale = self.pzd(d)
        pzd = dist.Normal(zd_p_loc, zd_p_scale)
        d_hat = self.qd(zd_q)            

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

        return x_recon, d_hat, y_hat, qzd, pzd, zd_q, qzy, pzy, zy_q, zy_q_loc

    def get_losses(self, x, y, d):        
        VAE_losses, conts_losses = 0, 0
        
        d_target = d
        d_input = F.one_hot(d, num_classes= self.d_dim).float()

        for i in range(self.seq_len):
            x_sample = x[:, i].view(x.size(0),1, -1)                  
            if y is not None:
                y_target = y[:, i]
                y_input = F.one_hot(y_target, num_classes= self.y_dim).float() 
                  
                x_recon, d_hat, y_hat, qzd, pzd, zd_q, qzy, pzy, zy_q, features = self.forward(x=x_input, y=y_input, d=d_input)
    
                CE_x = F.mse_loss(x_recon, x_input, reduction='sum')
                CE_d = F.cross_entropy(d_hat, d_target, reduction='sum')
                zd_p_minus_zd_q = torch.sum(pzd.log_prob(zd_q) - qzd.log_prob(zd_q))
                zy_p_minus_zy_q = torch.sum(pzy.log_prob(zy_q) - qzy.log_prob(zy_q))
            
                CE_y = F.cross_entropy(y_hat, y_target, reduction='sum')
                
                VAE_losses += CE_x \
                   - self.beta_d * zd_p_minus_zd_q \
                   - self.beta_y * zy_p_minus_zy_q \
                   + self.aux_loss_multiplier_d * CE_d \
                   + self.aux_loss_multiplier_y * CE_y
                
                conts_losses += self.contrastive_loss(features, y_target)*self.const_weight
                
            else:
                y_input = None
                x_croped = self.augmentor.crop_resize(x_sample,random.uniform(0.25,0.75))
                x_croped  = torch.FloatTensor(x_croped).to(y.device)
                x_permuted = self.augmentor.permute(x_sample,random.randint(5,20))
                x_permuted  = torch.FloatTensor(x_permuted).to(y.device)
                x_set = [x_croped, x_permuted]
                 
                feature_set = torch.tensor([]).cuda()
                for x_transformed in x_set:
                    x_recon, d_hat, y_hat, qzd, pzd, zd_q, qzy, pzy, zy_q, features = self.forward(x=x_transformed, y=y_input, d=d_input)
                    
                    feature_set = torch.cat((feature_set,features), dim=0)
                    
                    CE_x = F.mse_loss(x_recon, x_input, reduction='sum')
                    CE_d = F.cross_entropy(d_hat, d_target, reduction='sum')
        
                    zd_p_minus_zd_q = torch.sum(pzd.log_prob(zd_q) - qzd.log_prob(zd_q))

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
        
                    VAE_losses += (CE_x \
                           - self.beta_d * zd_p_minus_zd_q \
                           - self.beta_y * marginal_zy_p_minus_zy_q \
                           - marginal_prior_y_minus_qy \
                           + self.aux_loss_multiplier_d * CE_d)*0.5                      
                conts_losses += Self_SupervisedContrastiveLoss(feature_set, self.criterion)*self.const_weight

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
    

################################################################################
################################################################################

class Transformer(nn.Module):
    def __init__(self, config, n_layer=4, n_classes=5):
        super(Transformer, self).__init__()
        params = config['hyper_params']
        self.hidden_dim = params['zy_dim']
        self.batch_size = config["data_loader"]["args"]["batch_size"]
        self.dim_feedforward = params['dim_feedforward']
        self.n_layer = params['n_layers']
        
        self.mask = torch.ones((self.batch_size, params['seq_len'])).byte().cuda()
        self.crf = CRF(n_classes)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=8, batch_first=True, dim_feedforward=self.dim_feedforward) 
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.n_layer)
        self.fc = nn.Linear(self.hidden_dim, n_classes)
        
    def forward(self, x): #in: (batch, seq, feature) if batch_first=True
        x = self.transformer_encoder(x)
        x = self.fc(x)
        return x  # (N_batch, Length, Class)
        
    def get_loss(self, x, y): 
        x = self.forward(x)  # out: (N_batch, Length, Class)
        mask = self.mask[:len(y)]
        loss = self.crf.forward(x, y, mask)  # y: (batch_size, sequence_size), mask: (batch_size, sequence_size), out: (batch_size, sequence_size, num_labels)
        loss = -loss.mean()
        return loss
    
    def predict(self, x):
        x = self.forward(x) # out: (N_batch, Length, Class)
        mask = self.mask[:len(x)]
        y = self.crf.viterbi_decode(x, mask)
        return y
       
