#==========================================================================
#
#   Author      : NMD
#   Github      : https://github.com/manhdung20112000/
#   Email       : manhdung20112000@gmail.com
#   File        : classifier.py
#   Created on  : 2021-3-22
#   Description : build iamge classifier
#
#==========================================================================

import enum
from typing import NewType
import torch
import wandb
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import SubsetRandomSampler
from torch.nn.modules.activation import ReLU
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
import torch.optim as optim
from torch.optim import optimizer
from model.configs import *
import time
from model.utils import getTransform, save_func, train, test
from model.dataloader import get_sample_dataloader ,get_dataloader, load_dataset
from sklearn.model_selection import KFold
# Ignore excessive warnings
import logging
logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)

# build classify model
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        # self.pool = nn.MaxPool2d(2,2)
        # self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(16*5*5, 120)
        # self.fc2 = nn.Linear(120 ,84)
        # self.fc2 = nn.Linear(84 ,2)

        self.net = nn.Sequential(
            # conv block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),

            # conv block 2
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),

            # conv block 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.25),

            # convert 3D feature maps to 1D feature vectors
            nn.Flatten(),
            nn.Linear(in_features=17*17*64, out_features=128),
            nn.ReLU(),

            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            
            # 3 is 3 classes: digital; analog; other
            nn.Linear(in_features=128, out_features=3),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 16 * 5 * 5)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        x = self.net(x)
        return x

def cv_train_classifier(dataset, criterion=nn.CrossEntropyLoss() ,epochs = CLASSIFIER_EPOCHS, lr = CLASSIFIER_LEARNING_RATE):    
    # cuda 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    
    # save model input and hypermeters
    config = wandb.config
    config.learning_rate    = CLASSIFIER_LEARNING_RATE
    config.batch_size       = CLASSIFIER_BATCH_SIZE
    config.epochs           = CLASSIFIER_EPOCHS
    config.seed             = RANDOM_SEED

    # wandb.watch(model, log="all")

    # K-fold
    splits = KFold(n_splits=NUMBER_K_FOLD, random_state=RANDOM_SEED, shuffle=True)

    for fold, (train_idx, valid_idx) in enumerate(splits.split(dataset)):
        # plot result on wandb
        wandb.init(project=f'electric_meter_classifier_ver{VERSION}', reinit=True)
        wandb.run.name = f'CD_ID: {fold}'

        # init neural network and optimizer
        model = Classifier().to(device)
        optimizer = optim.Adam(params=model.parameters(), lr=lr)

        # split the data
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # get dataloader
        # dataloader = get_dataloader(dataset=dataset)
        dataloader = get_sample_dataloader(dataset=dataset, train_sampler=train_sampler, valid_sampler=valid_sampler)

        
        # training
        train(model=model, fold=fold, dataloader=dataloader, criterion=criterion, epochs=epochs, optim=optimizer, device=device)

def train_classifier():
    pass

def test_classifier(model, dataloader, device, criterion):
    test(model=model, device=device, dataloaders=dataloader, criterion=criterion)
        



