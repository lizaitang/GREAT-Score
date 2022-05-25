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

device = torch.device("cuda")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
mpl.use('Agg')
from scipy import stats
import numpy as np



from utils import l2_distance,softmax,tempsigmoid







def main():
    parser = argparse.ArgumentParser(description='cig-nn')
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
    parser = ArgumentParser(add_help=True)
    parser.add_argument('--dataset',type=str,default='cifar10',description='the dataset want to evaluate on: ›cifar10 or imagenet')
    parser.add_argument('--sample_size',type=int,default='1000',description= 'the generated model sample size')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=128)
    # put a layer norm right after input
    parser.add_argument('--layer_norm_first', action="store_true")
    # put layer norm between layers or not
    parser.add_argument('--use_ln', type=int,default=0)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--l2decay', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)

    #print device information if set true
    parser.add_argument('--reprod', action="store_true")
    parser.add_argument('--inductive', action="store_true")
    

    # attack specific settings
    parser.add_argument('--lower_bound_eval_global', action="store_false",description='whether to run cw attack on overall images')
    parser.add_argument('--lb_n',type=int,default=1000,description='the sample size used to test the lower bound')
    parser.add_argument('--robust_accuracy', action="store_false", description='whether to run robust accuracy evaluation on over images')
    parser.add_argument('--ra_n',type=int,default=500,description='the sample size used to ')

    # save and load best weights for final evaluation
    parser.add_argument('--best_weights', action="store_true")

    ######################## Adv. Training Setting ####################
    parser.add_argument('--step_size', type=float, default=1e-3)
    parser.add_argument('--m', type=int, default=3)
    parser.add_argument('--attack', type=str, default='vanilla')
    parser.add_argument('--pre_epochs', type=int, default=-1)

    ######################## Robustness Eval Setting ####################
    parser.add_argument('--eval_robo', action="store_true")
    # targeted attack else non-targeted
    parser.add_argument('--eval_target', action="store_true")
    # number of targets in each deg category
    parser.add_argument('--target_num', type=int, default=200) 
    
    # if evaluated in blackbox, the attacked graph will be loaded for evaluation
    parser.add_argument('--eval_robo_blk', action="store_true")







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
