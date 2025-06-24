import wandb
import argparse
import os
import time
import random
import numpy as np
from glob import glob
import cv2
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import albumentations as A



from data import SegRExtDataset
from utils import (
    seeding,shuffling, make_channel_first, make_channel_last, create_dir, epoch_time, print_and_save
)


from loss import DiceLoss, DiceBCELoss, IoUBCELoss, BCELoss, IoULoss, weighted_BCELoss, weighted_IoULoss, weighted_e_Loss, e_bce_loss, e_dice_loss
from Hybrid_Eloss import hybrid_e_loss

from model import SegRExtNet





def load_data(dataset_path, split=0.2):
    
    train_x = sorted(glob(os.path.join(dataset_path,"train/image","*.png")))
    train_y = sorted(glob(os.path.join(dataset_path,"train/mask", "*.png")))
    
    test_x = sorted(glob(os.path.join(dataset_path,"test/image", "*.png")))
    test_y = sorted(glob(os.path.join(dataset_path,"test/mask", "*.png")))
    
    valid_x = sorted(glob(os.path.join(dataset_path,"val/image", "*.png")))
    valid_y = sorted(glob(os.path.join(dataset_path,"val/mask", "*.png")))
    
    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)




def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0

    # Log gradients and model parameters
    wandb.watch(model)
    model.train()
    for i, (x, y) in enumerate(loader):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        yp = model(x)
        loss = loss_fn(yp, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss/len(loader)
    return epoch_loss

def evaluate(model, loader, loss_fn, device):
    epoch_loss = 0

    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            yp = model(x)
            loss = loss_fn(yp, y)
            epoch_loss += loss.item()

    epoch_loss = epoch_loss/len(loader)
    return epoch_loss

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    parser = argparse.ArgumentParser(description="Train the segmentation model with configurable paths and parameters.")

    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to save model checkpoints')
    parser.add_argument('--train_log_path', type=str, required=True, help='Path to save training logs')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training and validation')
    parser.add_argument('--num_epochs', type=int, default=40, help='Number of training epochs')

    args = parser.parse_args()



    """ Directories """
    # create_dir("files")

    """ Training logfile """
    train_log_path = args.train_log_path

    if os.path.exists(train_log_path):
        print("Log file exists")
    else:
        train_log = open(train_log_path, "w")
        train_log.write("\n")
        train_log.close()

    """ Load dataset """    
    dataset_path = args.dataset_path

    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset_path)
    
    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")


    train_x, train_y = shuffling(train_x, train_y)
    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print_and_save(train_log_path, data_str)

    """ Hyperparameters """
    size = (256, 256)
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    lr = 1e-4


    checkpoint_path = args.checkpoint_path

    
    """ Data augmentation: Transforms """
    transform =  A.Compose([
        A.Rotate(limit=25, p=0.3),
        A.HorizontalFlip(p=0.3),
        # A.VerticalFlip(p=0.3),
        A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32)
    ])

    """ Dataset and loader """
    train_dataset = SegRExtDataset(train_x, train_y, size, transform=transform)
    valid_dataset = SegRExtDataset(valid_x, valid_y, size, transform=None)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    """ Model """
    device = torch.device('cuda')
    model = SegRExtNet()

    model = model.to(device)

    optimizer = torch.optim.NAdam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=6, verbose=True)


    loss_fn = weighted_e_Loss
    loss_name ='weighted e_Loss'
    

    data_str = f"Hyperparameters:\nImage Size: {size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
    data_str += f"Optimizer: NAdam\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)

    """ Training the model. """
    best_valid_loss = float('inf')

    for epoch in range(num_epochs):
        start_time = time.time()
        wandb.init()

        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss = evaluate(model, valid_loader, loss_fn, device)
        scheduler.step(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
        print_and_save(train_log_path, data_str)
