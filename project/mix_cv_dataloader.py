# coding:utf-8
# https://github.com/FlyEgle/cub_baseline/blob/355cc311b6/dataset/imagenet_dataset.py
# https://www.kaggle.com/solomonk/pytorch-simplenet-augmentation-cnn-lb-0-945
# https://github.com/bearpelican/Experiments/blob/bcf10ef0dbaf56cc5f6202f504a80f8b1004c990/rectangular_images/validation_utils.py
import math
import os
import pickle
import random
from typing import Any

import cv2
import jpeg4py as jpeg
import numpy as np
import torch
import torchvision
import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import accimage_loader

# from cvtransforms import (CenterCrop, ColorJitter, Compose, Normalize,
#                           RandomHorizontalFlip, RandomResizedCrop,
#                           RandomRotation, Resize, ToCVImage, ToTensor)

# another library named albumentation
import albumentations as A


BASE_RESIZE_SIZE = 512
RESIZE_SIZE = 224
INPUT_SIZE = 224
BRIGHTNESS = 0.4
HUE = 0.4
CONTRAST = 0.4
SATURATION = 0.4
__imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}

'''
Add PCA noise with a coefficient sampled from a normal distribution N (0, 0.1).
'''


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


def get_mix_train_dataloader(root_path, batch_size, workers):
    train_trans = A.Compose([
        A.RandomResizedCrop(224),
        A.RandomHorizontalFlip(0.5),
        A.RandomRotation(degrees=15),
        A.ColorJitter(
            brightness=BRIGHTNESS, contrast=CONTRAST, hue=HUE, saturation=SATURATION),
        A.ToTensor(),
        Lighting(0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec']),
        A.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    ])
    train_datasets = ImageFolder(os.path.join(
        root_path, "train"), transform=train_trans, loader=mix_loader)
    # 内存充足的情况下，可以pin_memory,可以加速
    return DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)


def get_mix_val_dataloader(root_path, batch_size, workers):
    val_trans = A.Compose([
        A.Resize((448, 448)),
        A.CenterCrop(
            (224, 224)),
        A.ToTensor(),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_datasets = ImageFolder(os.path.join(
        root_path, "val"), transform=val_trans, loader=mix_loader)
    # 内存充足的情况下，可以pin_memory,可以加速
    return DataLoader(val_datasets, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)


def mix_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return mix_cv_loader(path)


def resize_img(image, resize_size):
    # from 蒋神
    # w, h = image.size # PIL
    h, w, _ = image.shape
    scale = resize_size / float(min(h, w))
    resize_h, resize_w = int(h * scale), int(w * scale)

    # image = image.resize((resize_w, resize_h), Image.BILINEAR) # PIL
    image = cv2.resize(image, (resize_w, resize_h),
                       interpolation=cv2.INTER_LINEAR)
    return image


def mix_cv_loader(path: str):
    # with open(path, 'rb') as f:
    # img = Image.open(f)

    # using jpeg4py to accelerate
    # img = cv2.imread(path)
    img = jpeg.JPEG(path).decode()  # 默认出来的就是RGB所以不用再转化了

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = resize_img(img, resize_size=BASE_RESIZE_SIZE)
    return img
