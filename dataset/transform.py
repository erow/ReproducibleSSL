"""
Misc functions.

Mostly copy-paste from torchvision references or other public repos like DETR:
https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""

import os
import sys
import time
import math
import random
import datetime
import subprocess
from collections import defaultdict, deque

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from PIL import ImageFilter, ImageOps
from torchvision import transforms
from torchvision.transforms import Normalize, RandomResizedCrop, RandomHorizontalFlip, ColorJitter, ToTensor, RandomApply, RandomGrayscale, Compose
import gin
from PIL import Image

from typing import List

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
DEFAULT_CROP_RATIO = 224/256

class ToDevice(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, x:torch.Tensor):
        return x.to(self.device,non_blocking=True)
    
class GaussianBlur(nn.Module):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

class Solarization(nn.Module):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.1, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
        return img

class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class GrayScale(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2):
        self.p = p
        self.transf = transforms.Grayscale(3)
 
    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img

@gin.configurable()
class ThreeAugmentation(nn.Module):
    def __init__(self, img_size=224,color_jitter=0.4, src=False):
        super().__init__()
        img_size = img_size
        remove_random_resized_crop = src
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        primary_tfl = []
        scale=(0.08, 1.0)
        interpolation='bicubic'
        if remove_random_resized_crop:
            primary_tfl = [
                transforms.Resize(img_size, interpolation=3),
                transforms.RandomCrop(img_size, padding=4,padding_mode='reflect'),
                transforms.RandomHorizontalFlip()
            ]
        else:
            primary_tfl = [
                transforms.RandomResizedCrop(
                    img_size, scale=scale, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip()
            ]

            transforms.RandomSolarize
        secondary_tfl = [transforms.RandomChoice([GrayScale(p=1.0),
                                                Solarization(p=1.0),
                                                GaussianBlur(p=1.0)])]
    
        if color_jitter is not None and not color_jitter==0:
            secondary_tfl.append(transforms.ColorJitter(color_jitter, color_jitter, color_jitter))
        final_tfl = [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=torch.tensor(mean),
                    std=torch.tensor(std))
            ]
        self.transform =  transforms.Compose(primary_tfl+secondary_tfl+final_tfl)

    def forward(self,x):
        return self.transform(x)

@gin.configurable()
class SimpleAugmentation(nn.Module):
    def __init__(self,img_size=224,scale=(0.2, 1.0),):
        super().__init__()
         # simple augmentation
        self.transforms = Compose([
                RandomResizedCrop(img_size, scale=scale, interpolation=Image.BICUBIC),  # 3 is bicubic
                RandomHorizontalFlip(),
                ToTensor(),
                # ToDevice('cuda'),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    def forward(self,x):
        return self.transforms(x)
    
    def change_resolution(self,img_size):
        decoder = self.transforms[0]
        decoder.size=(img_size,img_size)


@gin.configurable()
class DataAugmentationDINO(nn.Module):
    def __init__(self,img_size=224, global_crops_scale=(0.4, 1.), local_crops_scale=(0.05, 0.4), local_crops_number=8):
        """Multi-view data augmentation
        Reference: https://github.com/facebookresearch/dino/blob/main/main_dino.py#L419

        Args:
            global_crops_scale (tuple, optional): _description_. Defaults to (0.4, 1.).
            local_crops_scale (tuple, optional): _description_. Defaults to (0.05, 0.4).
            local_crops_number (int, optional): _description_. Defaults to 8.
        
        Return:
            [2 x global views, local_crops_number x local views]
        """
        super().__init__()
        flip_and_color_jitter = Compose([
            RandomHorizontalFlip(p=0.5),
            RandomApply(
                [ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            RandomGrayscale(p=0.2),
        ])

        normalize = Compose([
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = Compose([
            RandomResizedCrop(img_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(5),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = Compose([
            RandomResizedCrop(img_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(5),
            Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = Compose([
            RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            RandomApply([GaussianBlur(5)],p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops