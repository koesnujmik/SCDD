import os
import random
import argparse
import collections
import numpy as np
import cifar100Imbanlance,cifar10Imbanlance
from PIL import Image
import shutil
from tqdm import tqdm
import torch
import torch.utils
from baseline import get_network as ti_get_network
import torch.nn as nn
import tinyImbalance
import torch.optim as optim
import torch.utils.data.distributed
import torch.nn.functional as F
from torchvision import transforms
from collections import OrderedDict
import torchvision.models as models
from synthesize.utils import *
from convnet import ConvNet

def convnet4(nclass, logger=None):
    width = int(128)
    model = ConvNet(nclass,
                        net_norm='instance',
                        net_depth=4,
                        net_width=width,
                        channel=3,
                        im_size=(64, 64))
    if logger is not None:
        logger(f"=> creating model convnet-4, norm: instance")
    return model

def convnet3(nclass, logger=None):
    width = int(128)
    model = ConvNet(nclass,
                        net_norm='instance',
                        net_depth=3,
                        net_width=width,
                        channel=3,
                        im_size=(32, 32))
    if logger is not None:
        logger(f"=> creating model convnet-3, norm: instance")
    return model


def init_images(args, model=None):
    args.imbanlance_rate = 0.01
    args.num_crop = 1
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            MultiRandomCrop(
                num_crop=args.num_crop, size=args.input_size, factor=args.factor
            ),
            normalize,
        ]
    )
    
    if args.subset == 'tinyimagenet':
        trainset = tinyImbalance.TinyImbanlance(transform=transform,imbanlance_rate=args.imbanlance_rate, train=True,file_path=os.path.join(''))
    elif args.subset == 'cifar10':
        trainset = cifar10Imbanlance.Cifar10Imbanlance(transform=transform,imbanlance_rate=args.imbanlance_rate, train=True,file_path='../expert/root')
    elif args.subset == 'cifar100':
        trainset = cifar100Imbanlance.Cifar100Imbanlance(transform=transform,imbanlance_rate=args.imbanlance_rate, train=True,file_path=os.path.join(''))
    else:
        pass

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.mipc,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )

    num = 0
    for c, (images, labels) in enumerate(tqdm(train_loader)):
        images = selector(
            args.ipc * args.factor**2,
            model,
            images,
            labels,
            args.input_size,
            m=args.num_crop,
            cls_id=num,
            method=args.selection_method,
            imbanlance_rate=args.imbanlance_rate,
        )
        num = num + 1
        images = mix_images(images, args.input_size, args.factor, args.ipc)
        save_images(args, denormalize(images), c)


def save_images(args, images, class_id):
    for id in range(images.shape[0]):
        dir_path = "{}/{:05d}".format(args.syn_data_path, class_id)
        place_to_store = dir_path + "/class{:05d}_id{:05d}.jpg".format(class_id, id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        pil_image.save(place_to_store)


def main(args):
    print(args)
    with torch.no_grad():
        if not os.path.exists(args.syn_data_path):
            os.makedirs(args.syn_data_path)
        else:
            shutil.rmtree(args.syn_data_path)
            os.makedirs(args.syn_data_path)

        if args.subset == 'tinyimagenet':
            model = convnet4(nclass=200)
        elif args.subset == 'cifar10':
            model = convnet3(nclass=10)
        else:
            model = convnet3(nclass=100)
        checkpoint = torch.load(args.pre_train_path, map_location='cuda')
        # checkpoint = torch.load("",map_location="cpu")
        
        checkpoint = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        if any(k.startswith('module.') for k in checkpoint):
            checkpoint = {k.replace('module.', '', 1): v for k, v in checkpoint.items()}

        model_teacher=(model)
        model_teacher.load_state_dict(checkpoint)

        model_teacher = nn.DataParallel(model_teacher).cuda()
        model_teacher.eval()
        for p in model_teacher.parameters():
            p.requires_grad = False

        init_images(args, model_teacher)


if __name__ == "__main__":
    pass
