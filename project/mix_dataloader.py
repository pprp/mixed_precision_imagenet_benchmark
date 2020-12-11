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
import PIL
import PIL.Image as im
import PIL.ImageEnhance as ie
import torch
import torchvision
import tqdm
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import accimage_loader
from cvtransforms import *

'''
_all__ = ["Compose", "ToTensor", "ToCVImage",
           "Normalize", "Resize", "CenterCrop", "Pad",
           "Lambda", "RandomApply", "RandomOrder", "RandomChoice", "RandomCrop",
           "RandomHorizontalFlip", "RandomVerticalFlip", "RandomResizedCrop",
           "FiveCrop", "TenCrop", "LinearTransformation", "ColorJitter",
           "RandomRotation", "RandomAffine", "RandomAffine6", "RandomPerspective",
           "Grayscale", "RandomGrayscale",
           "RandomGaussianNoise", "RandomPoissonNoise", "RandomSPNoise"]
'''

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


class RandomResizedCrop:
    """Randomly crop a rectangle region whose aspect ratio is randomly sampled 
    in [3/4, 4/3] and area randomly sampled in [8%, 100%], then resize the cropped
    region into a 224-by-224 square image.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped (w / h)
        interpolation: Default: cv2.INTER_LINEAR: 
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation='linear'):

        self.methods = {
            "area": cv2.INTER_AREA,
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "lanczos4": cv2.INTER_LANCZOS4
        }

        self.size = (size, size)
        self.interpolation = self.methods[interpolation]
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img):
        # img = cv2.cvtColor(np.asarray(img),
        #                    cv2.COLOR_RGB2BGR)
        h, w, _ = img.shape

        area = w * h

        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            target_ratio = random.uniform(*self.ratio)

            output_h = int(round(math.sqrt(target_area * target_ratio)))
            output_w = int(round(math.sqrt(target_area / target_ratio)))

            if random.random() < 0.5:
                output_w, output_h = output_h, output_w

            if output_w <= w and output_h <= h:
                topleft_x = random.randint(0, w - output_w)
                topleft_y = random.randint(0, h - output_h)
                break

        if output_w > w or output_h > h:
            output_w = min(w, h)
            output_h = output_w
            topleft_x = random.randint(0, w - output_w)
            topleft_y = random.randint(0, h - output_w)

        cropped = img[topleft_y: topleft_y +
                      output_h, topleft_x: topleft_x + output_w]

        resized = cv2.resize(cropped, self.size,
                             interpolation=self.interpolation)

        # from cv2 to PIL
        # resized = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

        return resized

    def __repr__(self):
        for name, inter in self.methods.items():
            if inter == self.interpolation:
                inter_name = name

        interpolate_str = inter_name
        format_str = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_str += ', scale={0}'.format(tuple(round(s, 4)
                                                 for s in self.scale))
        format_str += ', ratio={0}'.format(tuple(round(r, 4)
                                                 for r in self.ratio))
        format_str += ', interpolation={0})'.format(interpolate_str)

        return format_str


def get_train_dataloader(root_path: str, batch_size: int, workers: int):
    train_trans = Compose([
        RandomResizedCrop(224),  # 随机裁剪224
        RandomHorizontalFlip(),  # 水平翻转
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_datasets = ImageFolder(os.path.join(
        root_path, "train"), transform=train_trans)

    return DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=workers)


def get_mix_train_dataloader(root_path, batch_size, workers):
    train_trans = Compose([
        # transforms.RandomResizedCrop(self.INPUT_SIZE, scale=(0.2, 1.)),
        # transforms.Resize((self.BASE_RESIZE_SIZE, self.BASE_RESIZE_SIZE), Image.BILINEAR),
        # transforms.RandomCrop(INPUT_SIZE),
        RandomResizedCrop(224),
        RandomHorizontalFlip(0.5),
        RandomRotation(degrees=15),
        ColorJitter(
            brightness=BRIGHTNESS, contrast=CONTRAST, hue=HUE, saturation=SATURATION),
        ToTensor(),
        Lighting(0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec']),
        Normalize([0.485, 0.456, 0.406],
                  [0.229, 0.224, 0.225])
    ])
    train_datasets = ImageFolder(os.path.join(
        root_path, "train"), transform=train_trans, loader=mix_pil_loader)
    # 内存充足的情况下，可以pin_memory,可以加速
    return DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)


def get_val_dataloader(root_path: str, batch_size: int, workers: int):
    val_trans = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_datasets = ImageFolder(os.path.join(
        root_path, "val"), transform=val_trans)

    return DataLoader(valid_datasets, batch_size=batch_size, shuffle=False, num_workers=workers)


def get_mix_val_dataloader(root_path, batch_size, workers):
    val_trans = Compose([
        Resize(
            (448, 448)),
        CenterCrop(
            (224, 224)),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_datasets = ImageFolder(os.path.join(
        root_path, "val"), transform=val_trans)
    # 内存充足的情况下，可以pin_memory,可以加速
    return DataLoader(val_datasets, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)


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


class RandomErasing(object):
    def __init__(self, EPSILON=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.EPSILON = EPSILON
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.EPSILON:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size()[2] and h <= img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    #img[0, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                    #img[1, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                    #img[2, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                    #img[:, x1:x1+h, y1:y1+w] = torch.from_numpy(np.random.rand(3, h, w))
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[1]
                    # img[0, x1:x1+h, y1:y1+w] = torch.from_numpy(np.random.rand(1, h, w))
                return img

        return img


def random_crop(img, boxes):
    '''Crop the given PIL image to a random size and aspect ratio.
    A crop of random size of (0.08 to 1.0) of the original size and a random
    aspect ratio of 3/4 to 4/3 of the original aspect ratio is made.
    Args:
      img: (PIL.Image) image to be cropped.
      boxes: (tensor) object boxes, sized [#ojb,4].
    Returns:
      img: (PIL.Image) randomly cropped image.
      boxes: (tensor) randomly cropped boxes.
    '''
    success = False
    for attempt in range(10):
        area = img.size[0] * img.size[1]
        target_area = random.uniform(0.56, 1.0) * area
        aspect_ratio = random.uniform(3. / 4, 4. / 3)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if random.random() < 0.5:
            w, h = h, w

        if w <= img.size[0] and h <= img.size[1]:
            x = random.randint(0, img.size[0] - w)
            y = random.randint(0, img.size[1] - h)
            success = True
            break

    # Fallback
    if not success:
        w = h = min(img.size[0], img.size[1])
        x = (img.size[0] - w) // 2
        y = (img.size[1] - h) // 2

    img = img.crop((x, y, x+w, y+h))
    boxes -= torch.Tensor([x, y, x, y])
    boxes[:, 0::2].clamp_(min=0, max=w-1)
    boxes[:, 1::2].clamp_(min=0, max=h-1)
    return img, boxes


class Grayscale(object):
    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Brightness(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Contrast(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class RandomOrder(object):
    """ Composes several transforms together in random order.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        if self.transforms is None:
            return img
        order = torch.randperm(len(self.transforms))
        for i in order:
            img = self.transforms[i](img)
        return img


class ColorJitter(RandomOrder):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.transforms = []
        if brightness != 0:
            self.transforms.append(Brightness(brightness))
        if contrast != 0:
            self.transforms.append(Contrast(contrast))
        if saturation != 0:
            self.transforms.append(Saturation(saturation))

# from fastai solution
# not avaliable now


class RectangularCropTfm(object):
    def __init__(self, idx2ar, target_size):
        self.idx2ar, self.target_size = idx2ar, target_size

    def __call__(self, img, idx):
        target_ar = self.idx2ar[idx]
        if target_ar < 1:
            w = int(self.target_size/target_ar)
            size = (w//8*8, self.target_size)
        else:
            h = int(self.target_size*target_ar)
            size = (self.target_size, h//8*8)
        return transforms.functional.center_crop(img, size)

# Step 1: sort images by aspect ratio


def sort_ar(data, valdir):
    idx2ar_file = data/'sorted_idxar.p'
    if os.path.isfile(idx2ar_file):
        return pickle.load(open(idx2ar_file, 'rb'))
    print('Creating AR indexes. Please be patient this may take a couple minutes...')
    val_dataset = torchvision.datasets.ImageFolder(valdir)
    sizes = [img[0].size for img in tqdm(val_dataset, total=len(val_dataset))]
    idx_ar = [(i, round(s[0]/s[1], 5)) for i, s in enumerate(sizes)]
    sorted_idxar = sorted(idx_ar, key=lambda x: x[1])
    pickle.dump(sorted_idxar, open(idx2ar_file, 'wb'))
    return sorted_idxar

# Step 2: chunk images by batch size. This way we can crop each image to the batch aspect ratio mean


def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))

# Step 3: map image index to batch aspect ratio mean so our transform function knows where to crop


def map_idx2ar(idx_ar_sorted, batch_size):
    ar_chunks = list(chunks(idx_ar_sorted, batch_size))
    idx2ar = {}
    ar_means = []
    for chunk in ar_chunks:
        idxs, ars = list(zip(*chunk))
        mean = round(np.mean(ars), 5)
        ar_means.append(mean)
        for idx in idxs:
            idx2ar[idx] = mean
    return idx2ar, ar_means


class RandomFlip(object):
    """Randomly flips the given PIL.Image with a probability of 0.25 horizontal,
                                                                0.25 vertical,
                                                                0.5 as is
    """

    def __call__(self, img):
        dispatcher = {
            0: img,
            1: img,
            2: img.transpose(im.FLIP_LEFT_RIGHT),
            3: img.transpose(im.FLIP_TOP_BOTTOM)
        }

        return dispatcher[random.randint(0, 3)]  # randint is inclusive


class RandomRotate(object):
    """Randomly rotate the given PIL.Image with a probability of 1/6 90°,
                                                                 1/6 180°,
                                                                 1/6 270°,
                                                                 1/2 as is
    """

    def __call__(self, img):
        dispatcher = {
            0: img,
            1: img,
            2: img,
            3: img.transpose(im.ROTATE_90),
            4: img.transpose(im.ROTATE_180),
            5: img.transpose(im.ROTATE_270)
        }

        return dispatcher[random.randint(0, 5)]  # randint is inclusive


class PILColorBalance(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        alpha = random.uniform(1 - self.var, 1 + self.var)
        return ie.Color(img).enhance(alpha)


class PILContrast(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        alpha = random.uniform(1 - self.var, 1 + self.var)
        return ie.Contrast(img).enhance(alpha)


class PILBrightness(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        alpha = random.uniform(1 - self.var, 1 + self.var)
        return ie.Brightness(img).enhance(alpha)


class PILSharpness(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        alpha = random.uniform(1 - self.var, 1 + self.var)
        return ie.Sharpness(img).enhance(alpha)


# Check ImageEnhancer effect: https://www.youtube.com/watch?v=_7iDTpTop04
# Not documented but all enhancements can go beyond 1.0 to 2
# Image must be RGB
# Use Pillow-SIMD because Pillow is too slow
class PowerPIL(RandomOrder):
    def __init__(self, rotate=True,
                 flip=True,
                 colorbalance=0.4,
                 contrast=0.4,
                 brightness=0.4,
                 sharpness=0.4):
        self.transforms = []
        if rotate:
            self.transforms.append(RandomRotate())
        if flip:
            self.transforms.append(RandomFlip())
        if brightness != 0:
            self.transforms.append(PILBrightness(brightness))
        if contrast != 0:
            self.transforms.append(PILContrast(contrast))
        if colorbalance != 0:
            self.transforms.append(PILColorBalance(colorbalance))
        if sharpness != 0:
            self.transforms.append(PILSharpness(sharpness))


if __name__ == "__main__":
    # dataloader = get_mix_val_dataloader(
    #     root_path=r"D:\GitHub\SimpleCVReproduction\fine_grained_baseline\data\images", batch_size=4, workers=1)
    # for item in dataloader:
    #     print(item)
    # RectangularCropTfm()
    pass
