import torch
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score


#def accuracy(output, target):
#    with torch.no_grad():
#        pred = torch.argmax(output, dim=1)
#        assert pred.shape[0] == len(target)
#        correct = 0
#        correct += torch.sum(pred == target).item()
#    return correct / len(target)
def accuracy(output, target):
    return accuracy_score(target, output)

def confusion(output, target):
    return confusion_matrix(target,output )

def f1(output, target):
    #with torch.no_grad():
    #    pred = torch.argmax(output, dim=1)
    #assert output.shape[0] == len(target)
    return f1_score(target,output, average='macro')
#    return f1_score(target.data.cpu().numpy(), pred.cpu().numpy(),  average='macro')
