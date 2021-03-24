#==========================================================================
#
#   Author      : NMD
#   Github      : https://github.com/manhdung20112000/
#   Email       : manhdung20112000@gmail.com
#   File        : train.py
#   Created on  : 2021-3-22
#   Description : train the full model
#
#==========================================================================
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import wandb
from model.dataloader import load_dataset, get_dataloader
from model.classifier import test_classifier, cv_train_classifier, train_classifier, Classifier
from model.configs import *

def main():
    # load the dataset
    dataset = load_dataset(valid=SPLIT_DATASET)
    if (SPLIT_DATASET):
        print(f"#Training: {len(dataset['train'])} \n#Valid: {len(dataset['valid'])} \n#Testing: {len(dataset['test'])}")
    else:
        print(f"#Training: {len(dataset['train'])} \n#Testing: {len(dataset['test'])}")

    # cross validation training
    cv_train_classifier(dataset=dataset['train'], epochs=10)
    
    # evaluate
    # test_classifier(model=classifier, dataset=dataset['test'], device=device, criterion=criterion)

if __name__ == '__main__':
    main()