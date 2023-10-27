import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch_metric_learning import losses

def weighted_CrossEntropyLoss(output, target, classes_weights, device):
    cr = nn.CrossEntropyLoss(weight=torch.tensor(classes_weights).to(device))
    return cr(output, target)
    
    
def CrossEntropyLoss(output, target, classes_weights, device):
    cr = nn.CrossEntropyLoss()
    return cr(output, target)


##### SupervisedContrastiveLoss        
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, tau=0.1):
        super(SupervisedContrastiveLoss, self).__init__()
        self.tau = tau

    def forward(self, feature_vectors, labels):
        feature_vectors_normalized = F.normalize(feature_vectors, p=2, dim=1)
        logits = torch.div(
            torch.matmul(
                feature_vectors_normalized, torch.transpose(feature_vectors_normalized, 0, 1)
            ),
            self.tau,
        )
        return losses.NTXentLoss(temperature=0.07)(logits, torch.squeeze(labels))

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
