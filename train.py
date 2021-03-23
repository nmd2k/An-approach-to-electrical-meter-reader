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
import torchvision
import matplotlib.pyplot as plt
import wandb
from model.dataloader import load_dataset, get_dataloader
from model.classifier import train_classifier, Classifier
from model.configs import CLASSIFIER_LEARNING_RATE

def main():
    # load the dataset
    dataset = load_dataset()
    print(f"#Training: {len(dataset['train'])} \n#Valid: {len(dataset['valid'])} \n#Testing: {len(dataset['test'])}")

    # get dataloader
    dataloader = get_dataloader(dataset=dataset)
    print(dataloader)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classifier = Classifier().to(device)

    #training
    train_losses, valid_losses, valid_errors = train_classifier(model=classifier, device=device, dataloader=dataloader)


if __name__ == '__main__':
    main()