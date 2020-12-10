# coding:utf-8
# large borrowed from https://github.com/FlyEgle/cub_baseline/blob/355cc311b6/dataset/imagenet_dataset.py
import os

import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.datasets.folder import accimage_loader
from typing import Any
from PIL import Image

BASE_RESIZE_SIZE = 512
RESIZE_SIZE = 224
INPUT_SIZE = 224
BRIGHTNESS = 0.4
HUE = 0.1
CONTRAST = 0.4
SATURATION = 0.4


def get_train_dataloader(root_path: str, batch_size: int, workers: int):
    train_trans = transforms.Compose([
        transforms.RandomResizedCrop(224),  # 随机裁剪224
        transforms.RandomHorizontalFlip(),  # 水平翻转
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_datasets = ImageFolder(os.path.join(
        root_path, "train"), transform=train_trans)

    return DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=workers)


def get_mix_train_dataloader(root_path, batch_size, workers):
    train_trans = transforms.Compose([
        # transforms.RandomResizedCrop(self.INPUT_SIZE, scale=(0.2, 1.)),
        # transforms.Resize((self.BASE_RESIZE_SIZE, self.BASE_RESIZE_SIZE), Image.BILINEAR),
        transforms.RandomCrop(INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=BRIGHTNESS, contrast=CONTRAST, hue=HUE, saturation=SATURATION),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    train_datasets = ImageFolder(os.path.join(
        root_path, "train"), transform=train_trans, loader=mix_pil_loader)
    return DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=workers)


def get_val_dataloader(root_path: str, batch_size: int, workers: int):
    val_trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_datasets = ImageFolder(os.path.join(
        root_path, "val"), transform=val_trans)

    return DataLoader(valid_datasets, batch_size=batch_size, shuffle=False, num_workers=workers)


def get_mix_val_dataloader(root_path, batch_size, workers):
    val_trans = transforms.Compose([
        transforms.Resize(
            (448, 448)),
        transforms.CenterCrop(
            (224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_datasets = ImageFolder(os.path.join(
        root_path, "images"), transform=val_trans)
    return val_datasets


def mix_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return mix_pil_loader(path)


def resize_img(image, resize_size):
    # from 蒋神
    w, h = image.size
    scale = resize_size / float(min(h, w))
    resize_h, resize_w = int(h * scale), int(w * scale)
    image = image.resize((resize_w, resize_h), Image.BILINEAR)
    return image


def mix_pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = resize_img(img, resize_size=BASE_RESIZE_SIZE)

        return img.convert('RGB')


if __name__ == "__main__":
    # transss = transforms.Compose([
    #     transforms.RandomResizedCrop(224),  # 随机裁剪224
    #     transforms.RandomHorizontalFlip(),  # 水平翻转
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406],
    #                          [0.229, 0.224, 0.225])
    # ])

    # anthor_trans = transforms.Compose([
    #     # transforms.RandomResizedCrop(self.INPUT_SIZE, scale=(0.2, 1.)),
    #     # transforms.Resize((self.BASE_RESIZE_SIZE, self.BASE_RESIZE_SIZE), Image.BILINEAR),
    #     transforms.RandomCrop(INPUT_SIZE),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(degrees=15),
    #     transforms.ColorJitter(
    #         brightness=BRIGHTNESS, contrast=CONTRAST, hue=HUE, saturation=SATURATION),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406],
    #                          [0.229, 0.224, 0.225])
    # ])
    # dataset = MixImageNetTrainingDataset(
    #     root="D:\GitHub\SimpleCVReproduction\fine_grained_baseline\data\images\images",
    #     transforms=anthor_trans)
    # dataset = ImageFolder(root=r"D:\GitHub\SimpleCVReproduction\fine_grained_baseline\data\images\images",
    #                       transform=anthor_trans,
    #                       loader=mix_loader)

    dataloader = get_mix_val_dataloader(
        root_path=r"D:\GitHub\SimpleCVReproduction\fine_grained_baseline\data\images", batch_size=4, workers=1)
    for item in dataloader:
        print(item)
