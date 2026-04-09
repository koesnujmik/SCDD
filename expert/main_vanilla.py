import sys
import os
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
import random
from torch.backends import cudnn
import torch.nn.functional as F
from utils import util
from utils.util import *
from model import ResNet_cifar
from model import Resnet_LT
from model import ResNet_tiny
from imbalance_data import imageImbanlance, cifar10Imbanlance, cifar100Imbanlance, dataset_lt_data, tinyImbalance
import logging
from convnet import ConvNet
import datetime
import math
from sklearn.metrics import confusion_matrix
from Vanilla_Traner import Vanilla_Trainer

best_acc1 = 0

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


def get_model(args):
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'convnet':
        net = convnet3(nclass=args.num_classes)
    elif args.arch == 'convnet4':
        net = convnet4(nclass=args.num_classes)
    if args.arch == 'resnet50':
        net = Resnet_LT.resnext50_32x4d(num_classes=args.num_classes)
    elif args.arch == 'resnet18':
        net = ResNet_cifar.resnet18(num_class=args.num_classes)
    elif args.arch == 'resnet32':
        net = ResNet_cifar.resnet32(num_class=args.num_classes)
    elif args.arch == 'resnet34':
        net = ResNet_cifar.resnet34(num_class=args.num_classes)
    return net

def get_dataset(args):
    transform_train, transform_val = util.get_transform(args.dataset)
    if args.dataset == 'cifar10':
        trainset = cifar10Imbanlance.Cifar10Imbanlance(transform=transform_train, imbanlance_rate=args.imbanlance_rate, train=True, file_path=args.root)
        testset = cifar10Imbanlance.Cifar10Imbanlance(imbanlance_rate=args.imbanlance_rate, train=False, transform=transform_val, file_path=args.root)
        print("load cifar10")
        return trainset, testset

    if args.dataset == 'cifar100':
        trainset = cifar100Imbanlance.Cifar100Imbanlance(transform=transform_train, imbanlance_rate=args.imbanlance_rate, train=True, file_path=os.path.join(args.root, 'cifar-100-python/'))
        testset = cifar100Imbanlance.Cifar100Imbanlance(imbanlance_rate=args.imbanlance_rate, train=False, transform=transform_val, file_path=os.path.join(args.root, 'cifar-100-python/'))
        print("load cifar100")
        return trainset, testset

    if args.dataset == 'tiny':
        trainset = tinyImbalance.TinyImbanlance(transform=transform_train, imbanlance_rate=args.imbanlance_rate, train=True, file_path=os.path.join(args.root, 'tiny-imagenet-200/'))
        testset = tinyImbalance.TinyImbanlance(imbanlance_rate=args.imbanlance_rate, train=False, transform=transform_val, file_path=os.path.join(args.root, 'tiny-imagenet-200/'))
        print("load tiny")
        return trainset, testset

    if args.dataset == 'ImageNet-LT':
        trainset = dataset_lt_data.LT_Dataset(os.path.join(args.root, 'imagenet/'), args.dir_train_txt, transform_train)
        testset = dataset_lt_data.LT_Dataset(os.path.join(args.root, 'imagenet/'), args.dir_test_txt, transform_val)
        return trainset, testset

def main():
    args = parser.parse_args()
    print(args)
    curr_time = datetime.datetime.now()
    args.store_name = '#'.join(["dataset: " + args.dataset, "arch: " + args.arch, "imbanlance_rate: " + str(args.imbanlance_rate),
            datetime.datetime.strftime(curr_time, '%Y-%m-%d %H:%M:%S')])
    prepare_folders(args)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = True
    main_worker(args.gpu, args)

def main_worker(gpu, args):

    global best_acc1
    global train_cls_num_list

    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    model = get_model(args)
    _ = print_model_param_nums(model=model)
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.root_log + args.store_name, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger()
    logger.addHandler(fh)

    train_dataset, val_dataset = get_dataset(args)
    num_classes = len(np.unique(train_dataset.targets))
    assert num_classes == args.num_classes

    cls_num_list = [0] * num_classes
    for label in train_dataset.targets:
        cls_num_list[label] += 1
    train_cls_num_list = np.array(cls_num_list)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, persistent_workers=True,
                                               pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, persistent_workers=True, pin_memory=True)

    start_time = time.time()
    print("Training started!")
    trainer = Vanilla_Trainer(args, model=model, train_loader=train_loader, val_loader=val_loader,
                              per_class_num=train_cls_num_list, log=logging)
    trainer.train()
    end_time = time.time()
    print("It took {} to execute the program".format(hms_string(end_time - start_time)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Vanilla Training (no debiasing)")
    parser.add_argument('--dataset', type=str, default='cifar100', help="cifar10,cifar100,ImageNet-LT,iNaturelist2018")
    parser.add_argument('--root', type=str, default='root/', help="dataset setting")
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet34',
                        choices=('convnet', 'convnet4', 'convnet5', 'resnet18', 'resnet32', 'resnet34', 'resnet50', 'resnext50_32x4d'))
    parser.add_argument('--num_classes', default=100, type=int, help='number of classes')
    parser.add_argument('--imbanlance_rate', default=0.01, type=float, help='imbalance factor')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', dest='lr')
    parser.add_argument('--epochs', default=200, type=int, metavar='N')
    parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
    parser.add_argument('--wd', '--weight_decay', default=5e-3, type=float, metavar='W', dest='weight_decay')
    parser.add_argument('--dir_train_txt', default='', type=str)
    parser.add_argument('--dir_test_txt', default='', type=str)
    parser.add_argument('--seed', default=3407, type=int)
    parser.add_argument('-p', '--print_freq', default=1000, type=int, metavar='N')
    parser.add_argument('--gpu', default=None, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N')
    parser.add_argument('--root_log', type=str, default='root/log/')
    parser.add_argument('--root_model', type=str, default='root/model/')
    parser.add_argument('--store_name', type=str, default='')
    main()
