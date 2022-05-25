from ast import parse
from unicodedata import name
from robustbench.data import load_cifar10
import torch
import os 
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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
import pandas as pd
import seaborn as sns
mpl.use('Agg')
import matplotlib.pyplot as plt
import math
from robustbench.data import load_cifar10



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
data=open("group.txt",'a')

#path="samples_dcgan.npz"
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
#images=images[100:400]
#labels=labels[100:400]
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



model_list=['Rebuffi2021Fixing_70_16_cutmix_extra','Gowal2020Uncovering_extra','Rebuffi2021Fixing_70_16_cutmix_ddpm','Rebuffi2021Fixing_28_10_cutmix_ddpm','Augustin2020Adversarial_34_10_extra','Sehwag2021Proxy','Augustin2020Adversarial_34_10','Rade2021Helper_R18_ddpm','Rebuffi2021Fixing_R18_cutmix_ddpm','Gowal2020Uncovering', 'Sehwag2021Proxy_R18', 'Wu2020Adversarial','Augustin2020Adversarial','Engstrom2019Robustness','Rice2020Overfitting','Rony2019Decoupling','Ding2020MMA']
#model_list=['Rebuffi2021Fixing_70_16_cutmix_extra','Gowal2020Uncovering_extra','Rebuffi2021Fixing_70_16_cutmix_ddpm','Rebuffi2021Fixing_28_10_cutmix_ddpm','Augustin2020Adversarial_34_10_extra','Augustin2020Adversarial_34_10','Rade2021Helper_R18_ddpm','Rebuffi2021Fixing_R18_cutmix_ddpm','Gowal2020Uncovering', 'Wu2020Adversarial','Augustin2020Adversarial','Engstrom2019Robustness','Rice2020Overfitting','Rony2019Decoupling','Ding2020MMA']
#model_list=['Rebuffi2021Fixing_70_16_cutmix_ddpm','Rebuffi2021Fixing_28_10_cutmix_ddpm','Sehwag2021Proxy','Rade2021Helper_R18_ddpm','Rebuffi2021Fixing_R18_cutmix_ddpm','Gowal2020Uncovering', 'Sehwag2021Proxy_R18', 'Wu2020Adversarial','Engstrom2019Robustness','Rice2020Overfitting','Rony2019Decoupling','Ding2020MMA']
#model_list=['Rebuffi2021Fixing_70_16_cutmix_extra','Gowal2020Uncovering_extra','Augustin2020Adversarial_34_10_extra','Augustin2020Adversarial_34_10','Gowal2020Uncovering', 'Wu2020Adversarial','Augustin2020Adversarial','Engstrom2019Robustness','Rice2020Overfitting','Rony2019Decoupling','Ding2020MMA']
#model_list=['Rebuffi2021Fixing_70_16_cutmix_extra','Gowal2020Uncovering_extra','Rebuffi2021Fixing_70_16_cutmix_ddpm','Rebuffi2021Fixing_28_10_cutmix_ddpm','Augustin2020Adversarial_34_10_extra','Sehwag2021Proxy','Augustin2020Adversarial_34_10','Rade2021Helper_R18_ddpm','Rebuffi2021Fixing_R18_cutmix_ddpm','Gowal2020Uncovering']
#model_list=['Rebuffi2021Fixing_70_16_cutmix_extra','Sehwag2021Proxy']
# https://github.com/Harry24k/adversarial-attacks-pytorch
import torchattacks
print("torchattacks %s"%(torchattacks.__version__))
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
'''
for model_name in model_list:
    #model = load_model(model_name, norm='L2').to(device)
    model = load_model(model_name=model_name, dataset='cifar10', threat_model='L2').to(device)
    acc = clean_accuracy(model, images.to(device), labels.to(device))
    print('Model: {}'.format(model_name))
    print('- Standard Acc: {}'.format(acc))
'''
#train_X1=train_X[100:400]
#train_y=train_y[100:400]
X_adv_data=train_X1
Y_data=train_y
y = np.zeros(12,dtype=float)
h=0
epsilons = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7]
result1=[]
for model_name in model_list:
    start1 = datetime.datetime.now()
    print('Model: {}'.format(model_name),file=data)
    model = load_model(model_name=model_name, dataset='cifar10', threat_model='L2').to(device)
    
    difference=np.zeros(500,dtype=float)

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
                #print(out,file=data)
              
                #out = tempsigmoid(out)
                #out=torch.sigmoid(out)
                #print(out,file=data)
                num_classes = len(out1)
                #print(out1[0])
                predicted_label =np.argmax(out1)
                least_likely_label = np.argmin(out1)
                start_class = 0 
                random_class = predicted_label
                top2_label = np.argsort(out1[0])[-2]
                #print(top2_label,file=data)
                #print(out,file=data)
                new_seq = [least_likely_label, top2_label, predicted_label]
                #print(new_seq)
                
                random_class = random.randint(start_class, start_class + num_classes - 1)
                new_seq[2] = random_class
                #true_label = np.argmax(Y_data[idx])
                true_label =target

                information = []
                target_type = 0b0001
                predicted_label2=np.array(predicted_label)
                predicted_label2=torch.from_numpy(predicted_label2).to(device)
               
                out = softmax(out)
                #out=(1+torch.tanh(out))/2
                if true_label != predicted_label2:
                 #print(1)
                 #punk=1
                 #seq.append(new_seq[1])
                 difference[j]=0
                 j=j+1
                else:
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
                                        #print(out,file=data)
                                    if target_type & 0b0010:
                                        # random
                                        seq.append(new_seq[2])
                                        difference[idx]=out[0][predicted_label]-out[0][random_class]
                                        information.append('random')
                
                    #target_classes[idx]=new_seq[1]
                   
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



    print(num_correct,file=data)
 
    #print(target_classes)
    #difference=np.array(dif)
    #print(math.sqrt(math.pi/2)*np.mean(difference[0:int(num_correct)]),file=data)
    #print(difference,file=data)
    print(math.sqrt(math.pi/2)*np.mean(difference),file=data)
    #a=math.sqrt(math.pi/2)*np.mean(difference[0:int(num_correct)])
    a=math.sqrt(math.pi/2)*np.mean(difference)
    b=a.item()
    result1.append(b)
    h=h+1
    #print(difference,file=data)
    end1 = datetime.datetime.now()
    print('time:({} ms)'.format(int((end1-start1).total_seconds())),file=data)

        
    '''
    
    print("- Foolbox")
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    atk = fb.attacks.L2CarliniWagnerAttack(binary_search_steps=1, initial_const=1,
                                        confidence=0, steps=100, stepsize=0.01)
            
    start1 = datetime.datetime.now()
    for idx in range(100):
        
        start = datetime.datetime.now()
        criterion = fb.criteria.TargetedMisclassification(target_classes[idx:idx+1])
        images1=images[idx:idx+1]
        labels1=labels[idx:idx+1]
        _, adv_images, _ = atk(fmodel, images1.to(device), criterion, epsilons=1)
        
        acc = clean_accuracy(model, adv_images, labels1)
        l2,num_correct = l2_distance(model, images1, adv_images, labels1, device=device)
        l2_seq[idx]=l2
        acc_seq[idx]=acc
        num[idx]=num_correct
        end = datetime.datetime.now()
        print('- Robust Acc: {} / L2: {:1.2} ({} ms)'.format(acc, l2,
                                                        int((end-start).total_seconds()*2000)),file=data)
    end1 = datetime.datetime.now()
    sum1=num*l2_seq
    sum2=np.sum(num)
    sum3=np.sum(sum1)
    l2_total=sum3/sum2
    print('- Robust Acc: {} / L2: {:1.2} ({} ms)'.format(np.mean(acc_seq), l2_total,
                                                        int((end1-start1).total_seconds()*2000)),file=data)'''



from scipy import stats
import numpy as np
'''x = np.array([69.33,
70,
72.33,
75.67,
70.67,
72,
68,
73,
73.33,
68.67,
])'''
'''
x = np.array([79,
78,
84,
85,
78,
83,
80,
82,
82,
82,
78,
78,
77,
67,
70,
68,
64])
'''

'''

x =np.array([87.20,
85.60,
86.20,
86.40,
86.40,
84.60,
85.20,
82.20,
81.80,
79.20,
77.60])
'''

'''
x =np.array([
90.60,
90.00,
89.20,
86.60,
87.60,
86.40,
88.60,
84.60,
82.20,
81.80,
79.20,
77.60])
'''



x= np.array([87.20,
85.60,
90.60,
90.00,
86.20,
89.20,
86.40,
86.60,
87.60,
86.40,
88.60,
84.60,
85.20,
82.20,
81.80,
79.20,
77.60])


z=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])

y=np.array(result1)




print('group1',file=data)
x1=[x[7],x[11],x[13],x[14],x[15],x[16]]
y1=[y[7],y[11],y[13],y[14],y[15],y[16]]
z1=[z[7],z[11],z[13],z[14],z[15],z[16]]


print(y1,file=data)
print(stats.spearmanr(x1, y1),file=data)
print(stats.spearmanr(x1, z1),file=data)
print(stats.spearmanr(y1, z1),file=data)
print(stats.kendalltau(x1, y1),file=data)
print(stats.kendalltau(x1, z1),file=data)
print(stats.kendalltau(y1, z1),file=data)
print(stats.weightedtau(x1, y1),file=data)
print(stats.weightedtau(x1, z1),file=data)
print(stats.weightedtau(y1, z1),file=data)

print('group2',file=data)


x2=[x[0],x[2],x[3],x[8]]
y2=[y[0],y[2],y[3],y[8]]
z2=[z[0],z[2],z[3],z[8]]

print(y,file=data)
print(stats.spearmanr(x2, y2),file=data)
print(stats.spearmanr(x2, z2),file=data)
print(stats.spearmanr(y2, z2),file=data)
print(stats.kendalltau(x2, y2),file=data)
print(stats.kendalltau(x2, z2),file=data)
print(stats.kendalltau(y2, z2),file=data)
print(stats.weightedtau(x2, y2),file=data)
print(stats.weightedtau(x2, z2),file=data)
print(stats.weightedtau(y2, z2),file=data)

print('group3',file=data)

x3=[x[1],x[9]]
y3=[y[1],y[9]]
z3=[z[1],z[9]]

print(y,file=data)
print(stats.spearmanr(x3, y3),file=data)
print(stats.spearmanr(x3, z3),file=data)
print(stats.spearmanr(y3, z3),file=data)
print(stats.kendalltau(x3, y3),file=data)
print(stats.kendalltau(x3, z3),file=data)
print(stats.kendalltau(y3, z3),file=data)
print(stats.weightedtau(x3, y3),file=data)
print(stats.weightedtau(x3, z3),file=data)
print(stats.weightedtau(y3, z3),file=data)


print('group4',file=data)
x4=[x[4],x[6],x[12]]
y4=[y[4],y[6],y[12]]
z4=[z[4],z[6],z[12]]

print(y,file=data)
print(stats.spearmanr(x4, y4),file=data)
print(stats.spearmanr(x4, z4),file=data)
print(stats.spearmanr(y4, z4),file=data)
print(stats.kendalltau(x4, y4),file=data)
print(stats.kendalltau(x4, z4),file=data)
print(stats.kendalltau(y4, z4),file=data)
print(stats.weightedtau(x4, y4),file=data)
print(stats.weightedtau(x4, z4),file=data)
print(stats.weightedtau(y4, z4),file=data)

print('group5',file=data)

x5=[x[5],x[10]]
y5=[y[5],y[10]]
z5=[z[5],z[10]]

print(y,file=data)
print(stats.spearmanr(x5, y5),file=data)
print(stats.spearmanr(x5, z5),file=data)
print(stats.spearmanr(y5, z5),file=data)
print(stats.kendalltau(x5, y5),file=data)
print(stats.kendalltau(x5, z5),file=data)
print(stats.kendalltau(y5, z5),file=data)
print(stats.weightedtau(x5, y5),file=data)
print(stats.weightedtau(x5, z5),file=data)
print(stats.weightedtau(y5, z5),file=data)


data.close()
