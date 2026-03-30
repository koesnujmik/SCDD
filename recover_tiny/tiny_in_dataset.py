from __future__ import print_function

import os
import socket
import numpy as np
import torchvision.datasets
from torch.utils.data import DataLoader
import torch.utils.data as data
import torch
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

from torch.utils.data import Dataset
class LongTailImageFolder(Dataset):
    def __init__(self, root, transform=None, num_cls=200, imbal_rate=0.01):
        self.transform = transform
        self.num_cls = num_cls
        self.imbal_rate = imbal_rate

        # Load entire dataset first
        full_dataset = torchvision.datasets.ImageFolder(root=root)
        x_all = np.array([np.array(img) for img, _ in full_dataset])
        y_all = np.array([label for _, label in full_dataset])
        self.classes = full_dataset.classes

        # Long-tail distribution computation
        per_class_num = []
        avg_per_class = x_all.shape[0] // num_cls
        for cls_idx in range(num_cls):
            factor = imbal_rate ** (cls_idx / (num_cls - 1))
            num_samples = int(avg_per_class * factor)
            per_class_num.append(num_samples)

        # Store sampled data
        new_x = []
        new_y = []
        for cls_idx in range(num_cls):
            indices = np.where(y_all == cls_idx)[0]
            sampled_idx = np.random.choice(indices, per_class_num[cls_idx], replace=False)
            for idx in sampled_idx:
                new_x.append(x_all[idx])
                new_y.append(cls_idx)

        self.data = new_x
        self.targets = new_y
        print(f"[LongTail] Imbalance ratio = {per_class_num[0] / per_class_num[-1]:.2f}")
        print(f"[LongTail] Per class sample numbers: {per_class_num}")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx]

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def get_tinyimagenet_dataloaders(batch_size=64, num_workers=8, data_folder='./data/tinyimagenet', is_instance=False):
    train_set = LongTailImageFolder(
        root=os.path.join(data_folder, 'train'),
        transform=transform_train,
        num_cls=200,                
        imbal_rate=0.01            
    )

    train_loader = DataLoaderX(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    # test set and loder
    test_set = torchvision.datasets.ImageFolder(root=data_folder + '/val', transform=transform_test)
    test_loader = DataLoaderX(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

    return train_loader, test_loader
