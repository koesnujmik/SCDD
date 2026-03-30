from __future__ import print_function

import os
import socket
import numpy as np
import torchvision.datasets
from torch.utils.data import DataLoader
import torch.utils.data as data
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# refer to: https://github.com/pytorch/examples/blob/master/imagenet/main.py
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform_train = transforms.Compose([
    transforms.RandomCrop(64, padding=8),  # refer to the cifar case
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
    
class TinyImageNetValDataset(Dataset):
    def __init__(self, val_dir, transform=None):
        self.val_dir = os.path.join(val_dir, 'images')
        self.transform = transform
        annotation_file = os.path.join(val_dir, 'val_annotations.txt')
        
        self.image_labels = []
        with open(annotation_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split('\t')
                filename, label = parts[0], parts[1]
                self.image_labels.append((filename, label))

        # 构建类别名 -> 索引的映射
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(set([x[1] for x in self.image_labels])))}
        self.samples = [(os.path.join(self.val_dir, fname), self.class_to_idx[label]) for fname, label in self.image_labels]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def get_tinyimagenet_dataloaders(batch_size=64, num_workers=8, data_folder='./data/tinyimagenet', is_instance=False):
    # train set and loder
    train_set = torchvision.datasets.ImageFolder(root=data_folder + '/train', transform=transform_train)
    train_loader = DataLoaderX(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = TinyImageNetValDataset(val_dir=os.path.join(data_folder, 'val'), transform=transform_test)
    test_loader = DataLoaderX(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

    return train_loader, test_loader
