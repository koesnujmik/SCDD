import os
import sys
import json
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
from synthesize.cluster_stats import extract_and_save_cluster_stats
from convnet import ConvNet

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(_REPO_ROOT, "train_cifar10"))
import ResNet_cifar


def seed_everything(seed):
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed, offset=0):
    if seed is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(offset))
    return generator


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


def build_cifar_model(arch_name, num_classes, input_size=32):
    if arch_name in ["convnet", "conv3"]:
        return convnet3(nclass=num_classes)
    if arch_name == "conv4":
        return convnet4(nclass=num_classes)
    if arch_name == "resnet18":
        return ResNet_cifar.resnet18(num_class=num_classes)
    if arch_name == "resnet32":
        return ResNet_cifar.resnet32(num_class=num_classes)
    if arch_name == "resnet34":
        return ResNet_cifar.resnet34(num_class=num_classes)
    raise ValueError(
        f"Unsupported teacher arch '{arch_name}'. "
        "Supported values: convnet, resnet18, resnet32, resnet34"
    )


def load_checkpoint_state_dict(path):
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def init_images(args, model=None):
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
        trainset = cifar100Imbanlance.Cifar100Imbanlance(transform=transform,imbanlance_rate=args.imbanlance_rate, train=True,file_path='../expert/root/cifar-100-python/')
    else:
        pass

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.mipc,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        worker_init_fn=seed_worker,
        generator=make_generator(args.seed, 0),
    )

    save_cluster_stats = bool(getattr(args, 'cluster_stat_path', None))
    if save_cluster_stats:
        if args.selection_method != 'kmeans':
            raise ValueError("--cluster-stat-path requires --selection-method kmeans")
        if args.factor != 1:
            raise ValueError("--cluster-stat-path currently supports --factor 1 only")
        if args.subset != 'cifar10':
            raise ValueError("--cluster-stat-path is implemented for --subset cifar10 only (v1)")
        os.makedirs(args.cluster_stat_path, exist_ok=True)
        manifest = {
            'schema_version': 1,
            'num_classes': int(args.nclass),
            'ipc': int(args.ipc),
            'factor': int(args.factor),
            'selection_method': args.selection_method,
            'mipc': int(args.mipc),
            'classes': {},
        }

    num = 0
    for c, (images, labels) in enumerate(tqdm(train_loader)):
        result = selector(
            args.ipc * args.factor**2,
            model,
            images,
            labels,
            args.input_size,
            m=args.num_crop,
            cls_id=num,
            method=args.selection_method,
            imbanlance_rate=args.imbanlance_rate,
            n_class=args.nclass,
            return_meta=save_cluster_stats,
        )
        if save_cluster_stats:
            images_sel, meta = result
            class_id = c
            keep_limit = meta['keep_limit']
            n_clusters = meta['n_clusters']
            cluster_labels = meta['cluster_labels']
            if keep_limit > 0 and n_clusters > 0 and cluster_labels is not None:
                # Cifar10Imbanlance.x is class-major with each class padded to mipc
                # entries (real + zero blanks). Real samples are the first
                # `keep_limit` rows of the class block.
                class_start = class_id * args.mipc
                class_raw = trainset.x[class_start:class_start + keep_limit]
                extract_and_save_cluster_stats(
                    teacher=model,
                    raw_x=class_raw,
                    cluster_labels=cluster_labels,
                    n_clusters=n_clusters,
                    class_id=class_id,
                    save_root=args.cluster_stat_path,
                    arch_alias=args.arch_name,
                )
            cluster_sizes = [int((cluster_labels == k).sum()) for k in range(n_clusters)] \
                if cluster_labels is not None else []
            entries = []
            for entry in meta['selected']:
                src_img_id = entry['source_img_id']
                cluster_id = int(cluster_labels[src_img_id]) if cluster_labels is not None else -1
                entries.append({
                    'ipc_id': entry['ipc_id'],
                    'source_img_id': src_img_id,
                    'source_aug_id': entry['source_aug_id'],
                    'cluster_id': cluster_id,
                })
            manifest['classes'][str(class_id)] = {
                'keep_limit': int(keep_limit),
                'n_clusters': int(n_clusters),
                'cluster_sizes': cluster_sizes,
                'selected': entries,
            }
            images = images_sel
        else:
            images = result
        num = num + 1
        images = mix_images(images, args.input_size, args.factor, args.ipc)
        save_images(args, denormalize(images), c)

    if save_cluster_stats:
        with open(os.path.join(args.cluster_stat_path, 'manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=2)


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
    seed_everything(args.seed)
    print(args)
    with torch.no_grad():
        if not os.path.exists(args.syn_data_path):
            os.makedirs(args.syn_data_path)
        else:
            shutil.rmtree(args.syn_data_path)
            os.makedirs(args.syn_data_path)

        model = build_cifar_model(args.arch_name, args.nclass, args.input_size)
        checkpoint = load_checkpoint_state_dict(args.pre_train_path)

        model_teacher = model
        model_teacher.load_state_dict(checkpoint)

        model_teacher = model_teacher.cuda()
        model_teacher.eval()
        for p in model_teacher.parameters():
            p.requires_grad = False

        init_images(args, model_teacher)


if __name__ == "__main__":
    pass
