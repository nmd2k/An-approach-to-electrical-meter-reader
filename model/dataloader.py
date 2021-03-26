#==========================================================================
#
#   Author      : NMD
#   Github      : https://github.com/manhdung20112000/
#   Email       : manhdung20112000@gmail.com
#   File        : dataloader.py
#   Created on  : 2021-3-22
#   Description : 
#
#==========================================================================

import os
import pandas as pd
import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import DataLoader, dataloader, random_split
from collections import OrderedDict
from model.configs import *
from model.utils import getTransform

def load_dataset(valid):
    train_transform = getTransform(INPUT_SIZE, True)
    test_transform  = getTransform(INPUT_SIZE, False)

    torch.manual_seed(RANDOM_SEED)
    # loader data from folder
    train_set = datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=train_transform)
    test_set = datasets.ImageFolder(root=TEST_DATA_PATH, transform=test_transform)

    if (valid):
        # split train_set into validation set and training set
        train_size = int(0.8*len(train_set))
        valid_size = len(train_set) - train_size
        train_set, valid_set = random_split(train_set, [train_size, valid_size])

        return dict(train=train_set, valid=valid_set, test=test_set)

    return dict(train=train_set, test=test_set)


def get_dataloader(dataset):
    torch.manual_seed(RANDOM_SEED)

    dataloader = dict(
        train=DataLoader(dataset['train'], batch_size=CLASSIFIER_BATCH_SIZE, shuffle=True),
        test=DataLoader(dataset['test'], batch_size=CLASSIFIER_BATCH_SIZE, shuffle=False)
    )

    if(SPLIT_DATASET):
        dataloader['valid']=DataLoader(dataset['valid'], batch_size=CLASSIFIER_BATCH_SIZE, shuffle=False),
        
    return dataloader

def get_sample_dataloader(dataset, train_sampler, valid_sampler ,batch_size=CLASSIFIER_BATCH_SIZE):
    sample_dataloader = dict(
        train=DataLoader(dataset, batch_size=CLASSIFIER_BATCH_SIZE, sampler=train_sampler),
        valid=DataLoader(dataset, batch_size=CLASSIFIER_BATCH_SIZE, sampler=valid_sampler),
    )
    return sample_dataloader
