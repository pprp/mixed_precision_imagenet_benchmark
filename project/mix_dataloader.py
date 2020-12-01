import os

import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def get_train_dataloader(root_path='../data', batch_size=32):
    train_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_datasets = ImageFolder(os.path.join(
        root_path, "train"), transform=train_trans)

    return DataLoader(train_datasets, batch_size=batch_size, shuffle=True)


def get_val_dataloader(root_path, batch_size):
    val_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_datasets = ImageFolder(os.path.join(
        root_path, "valid"), transform=val_trans)

    return DataLoader(valid_datasets, batch_size=batch_size, shuffle=True)
