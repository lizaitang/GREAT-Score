from ast import parse
from unicodedata import name
from robustbench.data import load_cifar10
import torch
import os 
import os
from autoattack import AutoAttack
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import numpy as np
device = torch.device("cuda")
from torch.autograd import Variable
import dill
import argparse
import torch.nn.functional as F
import random
import foolbox as fb
import gc
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

device="cuda"
from robustbench.data import load_cifar10
from robustbench.utils import load_model, clean_accuracy
def l2_distance(model, images, adv_images, labels, device="cuda"):
    outputs = model(adv_images)
    _, pre = torch.max(outputs.data, 1)
    corrects = (labels.to(device) == pre)
    num_correct = torch.eq(labels.to(device), pre).sum().float().item()
    delta = (adv_images - images.to(device)).view(len(images), -1)
    l2 = torch.norm(delta[~corrects], p=2, dim=1).mean()
    return l2,num_correct
data=open("auto_attack.txt",'a')

path="samples.npz"
f=np.load(path)
train_X, train_y = f['x'], f['y']

f.close()
train_X = train_X.astype('float32')
train_X /= 255
#train_X=train_X-0.5

train_y = train_y.astype('int64')
train_X1=train_X[0:500]
train_y=train_y[0:500]

target_classes=np.zeros(500)

images=torch.from_numpy(train_X1)
labels=torch.from_numpy(train_y)
images=images[0:500]
labels=labels[0:500]
images1=images
labels1=labels
unique,count=np.unique(labels,return_counts=True)
data_count=dict(zip(unique,count))
print(data_count)
print(images.shape)


import datetime
import numpy as np
import warnings

warnings.filterwarnings(action='ignore')

import torch
import torch.nn as nn
import torch.optim as optim

#model_list=['Rebuffi2021Fixing_70_16_cutmix_ddpm','Engstrom2019Robustness','Rice2020Overfitting','Rony2019Decoupling','Ding2020MMA','Standard']
#model_list=['Rebuffi2021Fixing_70_16_cutmix_extra','Gowal2020Uncovering_extra','Rebuffi2021Fixing_28_10_cutmix_ddpm','Augustin2020Adversarial_34_10_extra','Augustin2020Adversarial_34_10']
#model_list=['Augustin2020Adversarial']

model_list=['Rebuffi2021Fixing_70_16_cutmix_extra','Gowal2020Uncovering_extra','Rebuffi2021Fixing_70_16_cutmix_ddpm','Rebuffi2021Fixing_28_10_cutmix_ddpm','Augustin2020Adversarial_34_10_extra','Sehwag2021Proxy','Augustin2020Adversarial_34_10']
# https://github.com/Harry24k/adversarial-attacks-pytorch
# https://github.com/Harry24k/adversarial-attacks-pytorch
import torchattacks
print("torchattacks %s"%(torchattacks.__version__))
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    #print("X size is ", X_exp.size())
    #print("partition size is ", partition, partition.size())
    return X_exp / partition  # 这里应用了广播机制

'''
for model_name in model_list:
    #model = load_model(model_name, norm='L2').to(device)
    model = load_model(model_name=model_name, dataset='cifar10', threat_model='L2').to(device)
    acc = clean_accuracy(model, images.to(device), labels.to(device))
    print('Model: {}'.format(model_name))
    print('- Standard Acc: {}'.format(acc))
'''

X_adv_data=train_X1
Y_data=train_y
for model_name in model_list:
    print('Model: {}'.format(model_name),file=data)
    model = load_model(model_name=model_name, dataset='cifar10', threat_model='L2').to(device)
    acc = clean_accuracy(model, images.to(device), labels.to(device))
    print('- Standard Acc: {}'.format(acc))
    difference=np.zeros(2000,dtype=float)
    seq = []
    dif=[]
    j=0
    num_correct= 0
    right=target_classes
    for idx in range(len(Y_data)):
                # load original image
                
                # load adversarial image
                image_adv = np.array(np.expand_dims(X_adv_data[idx], axis=0), dtype=np.float32)
                #image_adv = np.transpose(image_adv, (0, 3, 1, 2))
                # load label
                label = np.array(Y_data[idx], dtype=np.int64)

                # check bound
                
                
                # transform to torch.tensor
                data_adv = torch.from_numpy(image_adv).to(device)
                target = torch.from_numpy(label).to(device)
                
                
                # evluation
                X, y = Variable(data_adv, requires_grad=True), Variable(target)
                out = model(X)
                out1=out.detach().cpu().numpy()
                num_classes = len(out1)
                #print(out1[0])
                predicted_label =np.argmax(out1)
                least_likely_label = np.argmin(out1)
                start_class = 0 
                random_class = predicted_label
                top2_label = np.argsort(out1[0])[-2]
                new_seq = [least_likely_label, top2_label, predicted_label]
                #print(new_seq)
                
                random_class = random.randint(start_class, start_class + num_classes - 1)
                new_seq[2] = random_class
                #true_label = np.argmax(Y_data[idx])
                true_label =target

                information = []
                target_type = 0b0100
                predicted_label2=np.array(predicted_label)
                predicted_label2=torch.from_numpy(predicted_label2).to(device)
                out = softmax(out)
                
                #if true_label != predicted_label2:
                #print(1)
                # punk=1
                #else:
                if target_type & 0b10000:
                                for c in range(num_classes):
                                    if c != predicted_label:
                                        seq.append(c)
                                        information.append('class'+str(c))
                else:
                                if target_type & 0b0100:
                                    # least
                                    seq.append(new_seq[0])
                                    information.append('least')
                                    difference[idx]=out[0][predicted_label]-out[0][least_likely_label]
                                    
                                if target_type & 0b0001:
                                    # top-2
                                    seq.append(new_seq[1])
                                    #difference[idx]=out[0][predicted_label]-out[0][top2_label]
                                    difference[j]=out[0][predicted_label]-out[0][top2_label]
                                    j=j+1
                                    #dif.append(out[0][predicted_label]-out[0][top2_label])
                                    information.append('top2')
                                if target_type & 0b0010:
                                    # random
                                    seq.append(new_seq[2])
                                    difference[idx]=out[0][predicted_label]-out[0][random_class]
                                    information.append('random')
            
                #target_classes[idx]=new_seq[1]
                right[idx]=predicted_label
                #out = softmax(out)
                predicted_label1=np.array(predicted_label)
                predicted_label1=torch.from_numpy(predicted_label1).to(device)
                num_correct += torch.eq(predicted_label1, target).sum().float().item()
    target_classes=np.array(seq)
    target_classes=torch.from_numpy(target_classes).to(device)
    l2_seq=np.zeros(100,dtype=float)
    acc_seq=np.zeros(100,dtype=float)
    num=np.zeros(100,dtype=float)
    print(target_classes.shape)



    print(num_correct/2000,file=data)

    #print(target_classes)
    #difference=np.array(dif)
    #print(np.mean(difference[0:int(num_correct)-1]),file=data)
    print(np.mean(difference),file=data)


        
    from robustbench.data import load_cifar10

    x_test, y_test = load_cifar10(n_examples=50)

        
    print("- Foolbox")
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    atk = fb.attacks.L2CarliniWagnerAttack(binary_search_steps=1, initial_const=1,
                                        confidence=0, steps=100, stepsize=0.01)
    epsilons = [0.5]
    
    for epsilons1 in epsilons:
        adversary=AutoAttack(model,norm='L2',eps=epsilons1,version='standard')
        x_adv=adversary.run_standard_evaluation(images.to(device),labels.to(device),bs=25)
data.close()