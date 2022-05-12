import torch
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score


def accuracy(output, target):
    return accuracy_score(target, output)

def confusion(output, target):
    return confusion_matrix(target,output )

def f1(output, target):
    return f1_score(target,output, average='macro')
