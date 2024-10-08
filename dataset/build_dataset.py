from torchvision import datasets
import numpy as np
import torch
import gin

from torchvision.datasets import VisionDataset 
from torchvision.datasets.folder import default_loader
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import os

from dataset.transform import SimpleAugmentation
def find_classes(directory: str):
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """

    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir() and entry.name[0]=='n')
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

def find_samples(path,split=None):
    classes,class_to_idx = find_classes(path)
    if split:
        split_list = open(split,'r').readlines()
        split_list = [i.strip('\n')for i in split_list]   
        for c in classes:
            if not c in split_list:
                del class_to_idx[c]
    samples = []
    for c,idx in class_to_idx.items():
        for file in os.listdir(os.path.join(path,c)):
            samples.append((os.path.join(path,c,file),idx))
    
    return samples,class_to_idx


class Folders(VisionDataset):
    def __init__(
        self,
        root: str,
        samples,
        class_to_idx,
        loader: Callable[[str], Any] = default_loader,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        classes = list(class_to_idx.keys())
        print(f"find classes: {len(classes)}")
        self.root = root
        self.samples = samples

        self.loader = loader
        self.classes = classes
        self.class_to_idx = class_to_idx
    
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
                    
        path, target = self.samples[index]
        path = os.path.join(self.root, path)
    
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

@gin.configurable(denylist=["args"])
def build_dataset(args,transform_fn=SimpleAugmentation):
    transform_train = transform_fn()
    if args.data_set == 'IMNET':
        # simple augmentation
        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    elif args.data_set == 'IF':
        dataset_train = datasets.ImageFolder(args.data_path, transform=transform_train)
    elif args.data_set == 'STL':
        dataset_train = datasets.STL10(args.data_path, split='train+unlabeled', transform=transform_train,download=True)
    return dataset_train

