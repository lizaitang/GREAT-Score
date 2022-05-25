from ast import parse
from unicodedata import name
from robustbench.data import load_cifar10
import torch
import os 
import os
from autoattack import AutoAttack
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
import numpy as np
import matplotlib.pyplot as plt
import math
from robustbench.data import load_cifar10
from robustbench.utils import load_model, clean_accuracy
import datetime
import numpy as np
import warnings

warnings.filterwarnings(action='ignore')

import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
mpl.use('Agg')
from scipy import stats
import numpy as np



from utils import l2_distance,softmax,tempsigmoid







def main():
    parser = argparse.ArgumentParser(description='GREAT')
    # dataset: cifar-10 default and imagenet ,set -data_dir names according to the dataset
    # different group setting : 1,2,3,4,5 group and comment, follows and output.
    # set a sample directory contains the sample npz or records the command for creating the samples?
    # argument: sample size, activation function,dataset, epsilons, load all the models together, returns final great score and corresponding rank coefficient.
    # set the gpu maybe:
    # set the lower bound mode or autoattack accurcy mode, set a corresponding sample size for attack
    # set proper output format: time for each 
    # rewrite a good enough sample random image cw attack individual mode.
    # set different size to process the image. eg:255 -0.5
    # untargeted cw attack and targeted version
    # 
    parser.add_argument('--dataset',type=str,default='cifar10',description='the dataset want to evaluate on: ›cifar10 or imagenet')
    parser.add_argument('--activation_function',type=str,default='sigmoid',description='the activation function we used')
    parser.add_argument('--sample_size',type=int,default='1000',description= 'the generated model sample size')
    parser.add_argument('--data_path',type=str,default='samples.npz',description='the samples data path')
    
    parser.add_argument('--lr', type=float, default=0.01)
    
    

    # attack specific settings
    parser.add_argument('--lower_bound_eval_global', action="store_false",description='whether to run cw attack on overall images')
    parser.add_argument('--lb_n',type=int,default=1000,description='the sample size used to test the lower bound')
    parser.add_argument('--robust_accuracy', action="store_false", description='whether to run robust accuracy evaluation on over images')
    parser.add_argument('--ra_n',type=int,default=500,description='the sample size used to run robust accuracy evaluation')

    args = parser.parse_args()
    
    # load the generated images and process
    path="samples.npz"
    f=np.load(path)
    train_X, train_y = f['x'], f['y']
    f.close()
    train_X = train_X.astype('float32')
    train_X /= 255
    #train_X=train_X-0.5
    train_y = train_y.astype('int64')
    train_X1=train_X[0:2000]
    train_y=train_y[0:2000]
    target_classes=np.zeros(2000)
    images=torch.from_numpy(train_X1)
    labels=torch.from_numpy(train_y)
    X_adv_data=train_X1
    Y_data=train_y

    # set a array to store the great score for each model
    great_result1=[]

    # load different models according to the dataset
    if args.dataset.lower() == 'cifar10':
      model_list=['Rebuffi2021Fixing_70_16_cutmix_extra','Gowal2020Uncovering_extra','Rebuffi2021Fixing_70_16_cutmix_ddpm','Rebuffi2021Fixing_28_10_cutmix_ddpm','Augustin2020Adversarial_34_10_extra','Sehwag2021Proxy','Augustin2020Adversarial_34_10','Rade2021Helper_R18_ddpm','Rebuffi2021Fixing_R18_cutmix_ddpm','Gowal2020Uncovering', 'Sehwag2021Proxy_R18', 'Wu2020Adversarial','Augustin2020Adversarial','Engstrom2019Robustness','Rice2020Overfitting','Rony2019Decoupling','Ding2020MMA']
    else:
      model_list=['Salman2020Do_50_2','Salman2020Do_R50','Engstrom2019Robustness','Wong2020Fast','Salman2020Do_R18']
    
    for model_name in model_list:
            start1 = datetime.datetime.now()
            print('Model: {}'.format(model_name),file=data)
            if args.dataset.lower() == 'cifar10':
               model = load_model(model_name=model_name, dataset='cifar10', threat_model='L2').to(device)
            else:
               model = load_model(model_name=model_name, dataset='imagenet',threat_model='Linf').to(device)
            
            difference=np.zeros(2000,dtype=float)

            seq = []
            dif=[]
            j=0
            num_correct= 0
            right=target_classes
            for idx in range(len(Y_data)):
                        # load adversarial image
                        image_adv = np.array(np.expand_dims(X_adv_data[idx], axis=0), dtype=np.float32)
                        #image_adv = np.transpose(image_adv, (0, 3, 1, 2))
                        # load label
                        label = np.array(Y_data[idx], dtype=np.int64)
                        # transform to torch.tensor
                        data_adv = torch.from_numpy(image_adv).to(device)
                        target = torch.from_numpy(label).to(device)
                        
                        
                        # evluation
                        X, y = Variable(data_adv, requires_grad=True), Variable(target)
                        out = model(X)
                        out1=out.detach().cpu().numpy()
                        #print(out,file=data)
                        
                        #out = tempsigmoid(out)
                        # use different activation function according to 
                        if args.dataset.lower() == 'cifar10':
                           out=torch.sigmoid(out)
                        else:
                           out = softmax(out)
                        #print(out,file=data)
                        num_classes = len(out1)
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
                    
                        #out = softmax(out)
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
                                                difference[j]=math.sqrt(math.pi/2)*(out[0][predicted_label]-out[0][top2_label])
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




if __name__ == "__main__":
    cfgs, gpus_per_node, run_name, hdf5_path, rank = load_configs_initialize_training()

    if cfgs.RUN.distributed_data_parallel and cfgs.OPTIMIZATION.world_size > 1:
        mp.set_start_method("spawn", force=True)
        print("Train the models through DistributedDataParallel (DDP) mode.")
        try:
            torch.multiprocessing.spawn(fn=loader.load_worker,
                                        args=(cfgs,
                                              gpus_per_node,
                                              run_name,
                                              hdf5_path),
                                        nprocs=gpus_per_node)
        except KeyboardInterrupt:
            misc.cleanup()
    else:
        loader.load_worker(local_rank=rank,
                           cfgs=cfgs,
                           gpus_per_node=gpus_per_node,
                           run_name=run_name,
                           hdf5_path=hdf5_path)
