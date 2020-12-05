import os

import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

optional_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def get_train_dataloader(root_path:str, batch_size: int, workers: int):
    train_trans = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),  # 随机裁剪224
        transforms.RandomHorizontalFlip(),  # 水平翻转
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_datasets = ImageFolder(os.path.join(
        root_path, "train"), transform=train_trans)

    return DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=workers)


def get_val_dataloader(root_path:str, batch_size: int, workers: int):
    val_trans = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_datasets = ImageFolder(os.path.join(
        root_path, "val"), transform=val_trans)

    return DataLoader(valid_datasets, batch_size=batch_size, shuffle=False, num_workers=workers)
