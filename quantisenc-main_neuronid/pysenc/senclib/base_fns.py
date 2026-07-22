import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import fileinput

def accuracy(act,pred):
    return np.mean((act == pred).detach().cpu().numpy())

def bin2fp(bin_integer,bin_fraction):
    v = 0 
    for i,b in enumerate(bin_integer):
        v += b * (2 ** i)

    for i,b in enumerate(bin_fraction):
        v += b * 1 / (2 ** (i+1))
    return v


def fpmax(n=1,q=4):
    bin_integer  = [1 for i in range(n)]
    bin_fraction = [1 for i in range(q)]
    return bin2fp(bin_integer,bin_fraction)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def SearchReplaceStr(searchExp,replaceExp,file='example.txt'):
    for line in fileinput.input(file, inplace=1):
        if searchExp in line:
            print(replaceExp, end ='\n')
        else:
            print(line, end ='')
