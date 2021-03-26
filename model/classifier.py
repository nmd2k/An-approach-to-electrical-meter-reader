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
from model.utils import getTransform, save_func, setup_wandb, train_eval, train_eval_epoch, test
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

def conv_layer(channel_in, channel_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv2d(channel_in, channel_out, kernel_size=k_size, padding=p_size),
        nn.BatchNorm2d(channel_out),
        nn.ReLU()
    )
    return layer

def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):

    layers = [ conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list)) ]
    layers += [ nn.MaxPool2d(kernel_size = pooling_k, stride = pooling_s)]
    return nn.Sequential(*layers)

def vgg_fc_layer(size_in, size_out):
    layer = nn.Sequential(
        nn.Linear(size_in, size_out),
        nn.BatchNorm1d(size_out),
        nn.ReLU()
    )
    return layer

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block(VGG_LAYER_1[0], VGG_LAYER_1[1], VGG_LAYER_1[2], VGG_LAYER_1[3], VGG_LAYER_1[4])
        self.layer2 = vgg_conv_block(VGG_LAYER_2[0], VGG_LAYER_2[1], VGG_LAYER_2[2], VGG_LAYER_2[3], VGG_LAYER_2[4])
        self.layer3 = vgg_conv_block(VGG_LAYER_3[0], VGG_LAYER_3[1], VGG_LAYER_3[2], VGG_LAYER_3[3], VGG_LAYER_3[4])
        self.layer4 = vgg_conv_block(VGG_LAYER_4[0], VGG_LAYER_4[1], VGG_LAYER_4[2], VGG_LAYER_4[3], VGG_LAYER_4[4])
        self.layer5 = vgg_conv_block(VGG_LAYER_5[0], VGG_LAYER_5[1], VGG_LAYER_5[2], VGG_LAYER_5[3], VGG_LAYER_5[4])

        # FC layers
        self.layer6 = vgg_fc_layer(VGG_FC_1[0], VGG_FC_1[1])
        self.layer7 = vgg_fc_layer(VGG_FC_2[0], VGG_FC_2[1])

        # Final layer
        self.layer8 = nn.Linear(VGG_FC_3[0], VGG_FC_3[1])

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        vgg16_features = self.layer5(out)
        out = vgg16_features.view(out.size(0), -1)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        return vgg16_features, out

def cv_train_classifier(dataset, criterion=nn.CrossEntropyLoss() ,epochs = CLASSIFIER_EPOCHS, lr = CLASSIFIER_LEARNING_RATE):
    # cuda 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # save model input and hypermeters
    config = setup_wandb()

    # wandb.watch(model, log="all")

    # K-fold
    splits = KFold(n_splits=NUMBER_K_FOLD, random_state=RANDOM_SEED, shuffle=True)

    # valid loss
    valid_losses = []

    for fold, (train_idx, valid_idx) in enumerate(splits.split(dataset)):
        # plot result on wandb
        wandb.init(project=f'electric_meter_classifier_ver{VERSION}', reinit=True)
        wandb.run.name = f'VGG16: {fold}'

        # init neural network and optimizer
        model = VGG16().to(device)
        optimizer = optim.Adam(params=model.parameters(), lr=lr)

        # split the data
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # get dataloader
        # dataloader = get_dataloader(dataset=dataset)
        dataloader = get_sample_dataloader(dataset=dataset, train_sampler=train_sampler, valid_sampler=valid_sampler)

        # training
        valid_loss = train_eval_epoch(model=model, fold=fold, dataloader=dataloader, criterion=criterion, epochs=epochs, optim=optimizer, device=device)
        valid_losses.append(valid_loss)

    # evaluate
    cv_score = np.sum(valid_losses, dtype = np.float32)
    print(f'\n\nCross Validation score (error): {cv_score/NUMBER_K_FOLD}\n\n')
    
    return cv_score

def train_classifier(dataset, criterion=nn.CrossEntropyLoss() ,epochs = CLASSIFIER_EPOCHS, lr = CLASSIFIER_LEARNING_RATE):
    # cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # save model input and hypermeters
    config = setup_wandb()

    # plot result
    wandb.init(project=f'Electric_meter_classifier')

    # init neural network and optimizer
    model = Classifier().to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=lr)

    # get dataloader
    dataloader = get_dataloader(dataset=dataset)

    # training and evalate in test set
    train_eval(model=model, dataloader=dataloader, criterion=criterion, epochs=epochs, optim=optimizer, device=device)

def test_classifier(model, dataloader, device, criterion):
    test(model=model, device=device, dataloaders=dataloader, criterion=criterion)
        



