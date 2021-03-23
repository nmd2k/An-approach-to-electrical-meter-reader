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
import torch
import wandb
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
import torch.optim as optim
from torch.optim import optimizer
from model.configs import CLASSIFIER_EPOCHS, CLASSIFIER_LEARNING_RATE, INPUT_SIZE, SAVE_PATH, VERSION
import time
from model.utils import getTransform, save_func, train
from model.dataloader import get_dataloader
# Ignore excessive warnings
import logging
logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)

# build classify model
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120 ,84)
        self.fc2 = nn.Linear(84 ,2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def train_classifier(model, dataloader, device):
    # plot result on wandb
    wandb.init(project='Electric_meter_classifier')
    wandb.watch(model, log="all")

    # save model input and hypermeters
    config = wandb.config
    config.learning_rate = CLASSIFIER_LEARNING_RATE
    
    epochs = CLASSIFIER_EPOCHS
    lr = CLASSIFIER_LEARNING_RATE
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    
    # training
    train_losses, valid_losses, valid_errors = train(model=model, dataloader=dataloader, criterion=criterion, 
            save_func=save_func, epochs=epochs, optim=optimizer, device=device)

    wandb.save(f'classifier_ver{VERSION}.h5')
    return train_losses, valid_losses, valid_errors

def test():
    pass
        



