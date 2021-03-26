#==========================================================================
#
#   Author      : NMD
#   Github      : https://github.com/manhdung20112000/
#   Email       : manhdung20112000@gmail.com
#   File        : utils.py
#   Created on  : 2021-3-22
#   Description : helper function
#
#==========================================================================

import torch
import wandb
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.transforms import ColorJitter, RandomAffine, RandomApply
from tqdm import tqdm
from model.configs import *

# create a transform function
def getTransform(input_size=INPUT_SIZE, isTrain=True):
    transform = [
        transforms.Resize((input_size, input_size)),
        transforms.Pad( (INPUT_SIZE-input_size)//2 )
        ]

    if isTrain:
        transform += [
            transforms.RandomApply([transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2
            )], p=0.6),
            transforms.RandomApply([transforms.RandomAffine(
                degrees=15,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=0.1
            )], p=0.6),
        ]

    transform += [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]

    transformFunction = transforms.Compose(transform)
    return transformFunction

def save_func(name, model):
    model_path = f'{SAVE_PATH}/{name}_{VERSION}.h5'
    torch.save(model.state_dict(), model_path)
    
    return f'{model_path}'

def setup_wandb():
    config = wandb.config
    config.learning_rate    = CLASSIFIER_LEARNING_RATE
    config.batch_size       = CLASSIFIER_BATCH_SIZE
    config.epochs           = CLASSIFIER_EPOCHS
    config.seed             = RANDOM_SEED

    return config

def train(model, device, optim, criterion, dataloader):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(dataloader['train']):
            # using cuda
            data, target = data.to(device), target.to(device)

            # zero the parameter gradients
            optim.zero_grad()

            # call a loss func
            pred = model(data)
            loss = criterion(pred, target)
            
            # forward + backward + optimize
            loss.backward()
            optim.step()

            # print statistics
            total_loss += loss.item()
        
        epoch_loss = total_loss / len(dataloader['train'])

        wandb.log({"Train Loss": epoch_loss})
        return epoch_loss
        
def valid(model, device, criterion, dataloader):
    model.eval()
    total_loss, total_error, total_count = 0, 0, 0

    example_images = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader['valid']):
            # using cuda
            data, target = data.to(device), target.to(device)

            # predict
            output = model(data)
            loss = criterion(output, target)

            # print statistics
            total_loss += loss.item()
            pred = torch.argmax(output, dim=1)
            total_error += torch.sum(pred != target).item()
            total_count += target.shape[0]

            # log wandb
            example_images.append(wandb.Image(data[0], caption=f"Pred: {pred[0].item()}"))

        epoch_loss = total_loss/(len(dataloader['valid']))
        epoch_error = total_error/total_count

        wandb.log({
            "Examples": example_images,
            "Valid Error": epoch_error,
            "Valid Loss": epoch_loss})
        return epoch_loss, epoch_error

def train_eval(model, dataloader, criterion, epochs, optim, device):
    train_losses, test_losses, test_errors = [], [], []
    progess_bar = tqdm(range(epochs), total=epochs)

    for epoch in progess_bar:
        train_losses.append(train(model=model, device=device, criterion=criterion, dataloader=dataloader, optim=optim))

        test_loss, test_error = test(model=model, device=device, criterion=criterion, dataloader=dataloader)
        test_losses.append(test_loss)
        test_errors.append(test_error)

        progess_bar.set_description(f'Loss score: {test_loss:.3f} | Error score: {test_error*100:.3f}%')

        # save model's weight
        if SAVE_WEIGHT:
            model_path = save_func(f'classifier', model)
            wandb.save(model_path)

def train_eval_epoch(model, dataloader, criterion, epochs, optim, device, fold=''):
    train_losses, valid_losses, valid_errors = [], [], []
    progess_bar = tqdm(range(epochs), total=epochs)
    best_score = 100000
    for epoch in progess_bar:
        train_losses.append(train(model=model, device=device, criterion=criterion, dataloader=dataloader, optim=optim))

        valid_loss, valid_error = valid(model=model, device=device, criterion=criterion, dataloader=dataloader)
        valid_losses.append(valid_loss)
        valid_errors.append(valid_error)

        progess_bar.set_description(f'Fold: {fold+1} | Loss score: {valid_loss:.3f} | Current Escore: {valid_error*100:.3f}%')

        #Error score: the lower the better
        if valid_error < best_score:
            best_score = valid_error
            progess_bar.set_description(f'Fold: {fold+1} | Loss score: {valid_loss:.3f} | New best Escore: {best_score*100:.3f}%')
            #save the model when the current epoch have lower loss than the previous
            if SAVE_WEIGHT:
                model_path = save_func(f'classifier_CV_{fold}', model)
                wandb.save(model_path)

    return valid_errors[-1]


def test(model, dataloader, criterion, device):
    model.eval()
    test_loss, test_error, test_count = 0, 0, 0

    example_images = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader['test']):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)
            
            pred = torch.argmax(output, dim=1)
            test_loss += loss.item()
            test_error += torch.sum(pred != target).item()
            test_count += target.shape[0]

            # log wandb
            for i in range(5):
                example_images.append(wandb.Image(data[i], caption=f"Pred: {pred[i].item()}"))
        
        epoch_loss = test_loss/(len(dataloader['test']))
        epoch_error = test_error/test_count

        wandb.log({
                    "Examples": example_images,
                    "Test Error": epoch_error,
                    "Test Loss": epoch_loss})

    return epoch_loss, epoch_error