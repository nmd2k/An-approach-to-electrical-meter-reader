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
from torch.utils.data import DataLoader, random_split
from collections import OrderedDict
from model.configs import CLASSIFIER_BATCH_SIZE, DATA_PATH, INPUT_SIZE, TEST_DATA_PATH, TRAIN_DATA_PATH
from model.utils import getTransform

def load_dataset():
    train_transform = getTransform(INPUT_SIZE, True)
    test_transform  = getTransform(INPUT_SIZE, False)

    torch.manual_seed(42)
    # loader data from folder
    train_set = datasets.ImageFolder(root=DATA_PATH, transform=train_transform)
    test_set = datasets.ImageFolder(root=DATA_PATH, transform=test_transform)

    # split train_set into validation set and training set
    train_size = int(0.8*len(train_set))
    valid_size = len(train_set) - train_size
    train_set, valid_set = random_split(train_set, [train_size, valid_size])

    return dict(train=train_set, valid=valid_set, test=test_set)


def get_dataloader(dataset):
    torch.manual_seed(42)
    dataLoaders = dict(
        train=torch.utils.data.DataLoader(dataset['train'], batch_size=CLASSIFIER_BATCH_SIZE, shuffle=True),
        valid=torch.utils.data.DataLoader(dataset['valid'], batch_size=CLASSIFIER_BATCH_SIZE, shuffle=False),
        test=torch.utils.data.DataLoader(dataset['test'], batch_size=CLASSIFIER_BATCH_SIZE, shuffle=False)
    )
    
    return dataLoaders
