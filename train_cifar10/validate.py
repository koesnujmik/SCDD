import os
import sys
import argparse
import ResNet_cifar
from convnet import ConvNet
import numpy as np
from baseline import get_network as ti_get_network
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import AverageMeter, accuracy

import torch
import torch.nn as nn


normalize = transforms.Normalize([0.5071, 0.4867, 0.4408],
                                  [0.2675, 0.2565, 0.2761])


def convnet3(nclass):
    return ConvNet(nclass, net_norm='instance', net_depth=3,
                   net_width=128, channel=3, im_size=(32, 32))


def get_args():
    parser = argparse.ArgumentParser("Validate a trained student model on CIFAR-10")
    parser.add_argument('--weight-path', type=str, required=True,
                        help='path to student model checkpoint')
    parser.add_argument('--model', type=str, default='convnet',
                        help='student model name')
    parser.add_argument('--val-dir', type=str, default='../expert/root',
                        help='path to CIFAR-10 dataset root')
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--gpu-id', type=str, default='0')
    return parser.parse_args()


def validate(model, val_loader):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_function = nn.CrossEntropyLoss()
    num_classes = 10
    correct_per_class = torch.zeros(num_classes).cuda()
    total_per_class = torch.zeros(num_classes).cuda()

    model.eval()
    with torch.no_grad():
        for data, target in val_loader:
            target = target.type(torch.LongTensor)
            data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = loss_function(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            n = data.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            pred = output.argmax(dim=1)
            for cls in range(num_classes):
                cls_mask = (target == cls)
                correct_per_class[cls] += (pred[cls_mask] == cls).sum()
                total_per_class[cls] += cls_mask.sum()

    per_class_acc = 100.0 * correct_per_class / (total_per_class + 1e-6)

    print(f"loss = {objs.avg:.6f}\t"
          f"Top-1 acc = {top1.avg:.2f}%\t"
          f"Top-5 acc = {top5.avg:.2f}%")

    print("\nPer-class Accuracy:")
    for cls in range(num_classes):
        print(f"  Class {cls}: {per_class_acc[cls]:.2f}%")

    return top1.avg


def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # Load model
    if args.model == 'convnet':
        model = convnet3(nclass=10)
    elif args.model == 'resnet32':
        model = ResNet_cifar.resnet32(num_class=10)
    elif args.model == 'resnet18':
        import torchvision.models as models
        model = models.resnet18(pretrained=False, num_classes=10)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, 10)
    else:
        model = ti_get_network(args.model, channel=3, num_classes=10, im_size=(32, 32), dist=False)
    model = model.cuda()

    # Load checkpoint
    if not os.path.isfile(args.weight_path):
        print(f"Checkpoint not found: {args.weight_path}")
        sys.exit(1)

    checkpoint = torch.load(args.weight_path, map_location='cuda')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    if any(k.startswith('module.') for k in state_dict):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    epoch = checkpoint.get('epoch', '?')
    best_acc = checkpoint.get('best_acc1', '?')
    print(f"Loaded '{args.weight_path}'  (epoch={epoch}, saved best_acc1={best_acc})\n")

    # Validation dataset
    val_dataset = torchvision.datasets.CIFAR10(
        root=args.val_dir, train=False, download=False,
        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    validate(model, val_loader)


if __name__ == "__main__":
    main()
