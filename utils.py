import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.datasets as dsets

def l2_distance(model, images, adv_images, labels, device="cuda"):
    outputs = model(adv_images)
    _, pre = torch.max(outputs.data, 1)
    corrects = (labels.to(device) == pre)
    delta = (adv_images - images.to(device)).view(len(images), -1)
    l2 = torch.norm(delta[~corrects], p=2, dim=1).mean()
    return l2

def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    #print("X size is ", X_exp.size())
    #print("partition size is ", partition, partition.size())
    return X_exp / partition  # 这里应用了广播机制

def tempsigmoid(x):
    nd=3.0
    temp=nd/torch.log(torch.tensor(9.0)) 
    return torch.sigmoid(x/(temp)) 