"""Per-cluster activation statistics extraction for RLDD initial stage.

For each KMeans cluster (computed in `selector(method='kmeans', return_meta=True)`),
this module forwards the cluster's real CIFAR-10 images through the teacher and
records the same per-sample input statistics that recover_cifar10/utils.py
accumulates for its per-class running buffers
(`ConvFeatureHook.pre_hook_fn` / `BNFeatureHook.pre_hook_fn`, category-aware
section). The cluster value for each statistic is the mean of the per-sample
values across the cluster's samples — same reduction recover applies when it
divides its `category_running_*` accumulators by `counter[cls]` in `save()`.

To make the cluster targets directly comparable to the syn-image stats that the
recover hook computes, the same augmentation + normalization pipeline that
recover uses for real-stat collection is applied here:
RandomCrop(32, padding=4) + RandomHorizontalFlip + ToTensor + CIFAR Normalize.
"""

import os

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image


# Recover-side CIFAR-10 normalization (matches recover_cifar10/utils.py:74-78
# and the Normalize() applied to the train_loader in recover.py).
_RECOVER_MEAN = (0.5071, 0.4867, 0.4408)
_RECOVER_STD = (0.2675, 0.2565, 0.2761)

# Mirrors the transform used by recover_cifar10/recover.py when it builds
# `train_dataset` to collect real CIFAR statistics (line 212-218 area).
_REAL_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(_RECOVER_MEAN, _RECOVER_STD),
])


def _div_four_mul(v):
    v = int(v)
    m = v % 4
    return int(v // 4 * 4) + int(m > 0) * 4


class _ConvCaptureHook:
    """Per-sample dd/patch mean and var for one Conv2d input.

    Math mirrors the category-aware section of
    recover_cifar10/utils.py ConvFeatureHook.pre_hook_fn (line ~612-624):
        dd_mean   = input.mean([2, 3])                          # [B, C]
        dd_var    = input.view(B, C, -1).var(2, unbiased=False) # [B, C]
        patch_mean= class_new.mean([2])                         # [B, num_patches]
        patch_var = class_new.var([2], unbiased=False)
    where class_new comes from `b c (u h) (v w) -> b (u v) (c h w)` rearrange.
    """

    def __init__(self, module):
        self.handle = module.register_forward_hook(self._fn)
        self.last = None

    def _fn(self, module, input, output):
        x = input[0]
        bs, nch = x.shape[0], x.shape[1]
        dd_mean = x.mean([2, 3])                                          # [B, C]
        dd_var = x.view(bs, nch, -1).var(2, unbiased=False)               # [B, C]
        new_h, new_w = _div_four_mul(x.shape[2]), _div_four_mul(x.shape[3])
        new_x = F.interpolate(x, [new_h, new_w], mode="bilinear")
        class_new_x = einops.rearrange(
            new_x, "b c (u h) (v w) -> b (u v) (c h w)", h=4, w=4
        ).contiguous()                                                     # [B, num_patches, C*16]
        patch_mean = class_new_x.mean([2])                                 # [B, num_patches]
        patch_var = class_new_x.var([2], unbiased=False)
        self.last = {
            'dd_mean': dd_mean.detach().cpu(),
            'dd_var': dd_var.detach().cpu(),
            'patch_mean': patch_mean.detach().cpu(),
            'patch_var': patch_var.detach().cpu(),
        }

    def close(self):
        self.handle.remove()


class _BNCaptureHook:
    """Per-sample dd_mean/dd_var. Mirrors recover_cifar10/utils.py
    BNFeatureHook.pre_hook_fn category-aware section (line ~234-239)."""

    def __init__(self, module):
        self.handle = module.register_forward_hook(self._fn)
        self.last = None

    def _fn(self, module, input, output):
        x = input[0]
        bs, nch = x.shape[0], x.shape[1]
        dd_mean = x.mean([2, 3])                                          # [B, C]
        dd_var = x.view(bs, nch, -1).var(2, unbiased=False)               # [B, C]
        self.last = {
            'dd_mean': dd_mean.detach().cpu(),
            'dd_var': dd_var.detach().cpu(),
        }

    def close(self):
        self.handle.remove()


def _layer_full_name(model_class_name, arch_alias, module_name):
    """Same naming used by recover_*/recover.py when registering hooks."""
    return f"{model_class_name}_{arch_alias}={module_name}"


def _augmented_batch(raw_x_subset):
    """raw_x_subset: numpy uint8 [N, H, W, 3] → tensor [N, 3, 32, 32]
    after RandomCrop+Flip+ToTensor+CIFAR Normalize (same as recover stat
    collection). Each sample independently goes through the random crop/flip
    pipeline, matching how recover's DataLoader yields augmented samples."""
    tensors = []
    for i in range(raw_x_subset.shape[0]):
        pil = Image.fromarray(raw_x_subset[i])
        tensors.append(_REAL_TRANSFORM(pil))
    return torch.stack(tensors, 0)


def extract_and_save_cluster_stats(*, teacher, raw_x, cluster_labels, n_clusters,
                                   class_id, save_root, arch_alias='convnet'):
    """
    Forward each cluster's real CIFAR-10 images through the teacher with the
    recover-side stat-collection pipeline and save per-cluster activation
    statistics.

    For each cluster k:
      stats_k = mean over k's samples of per-sample input statistics
                (dd_mean, dd_var, patch_mean, patch_var for Conv2d;
                 dd_mean, dd_var for BatchNorm2d).
    This mirrors `category_running_*[cls] += per_sample_stat` followed by
    division by `counter[cls]` in recover_cifar10/utils.py save().

    Args:
        teacher:        nn.Module (DataParallel-wrapped or not).
        raw_x:          numpy uint8 [keep_limit, H, W, 3] — class's real images.
        cluster_labels: numpy int [keep_limit] in [0, n_clusters).
        n_clusters:     int.
        class_id:       int.
        save_root:      output root directory.
        arch_alias:     'convnet' (matches aux_teacher entry on recover side).
    """
    teacher_unwrapped = teacher.module if isinstance(teacher, nn.DataParallel) else teacher
    arch_class = teacher_unwrapped.__class__.__name__

    conv_hooks = []
    bn_hooks = []
    for name, module in teacher_unwrapped.named_modules():
        full_name = _layer_full_name(arch_class, arch_alias, name)
        if isinstance(module, nn.Conv2d):
            conv_hooks.append((full_name, _ConvCaptureHook(module)))
        elif isinstance(module, nn.BatchNorm2d):
            bn_hooks.append((full_name, _BNCaptureHook(module)))

    device = next(teacher_unwrapped.parameters()).device

    conv_buffers = {full_name: {'dd_mean': [], 'dd_var': [],
                                 'patch_mean': [], 'patch_var': []}
                    for full_name, _ in conv_hooks}
    bn_buffers = {full_name: {'dd_mean': [], 'dd_var': []}
                  for full_name, _ in bn_hooks}
    valid_cluster_mask = []

    teacher_unwrapped.eval()
    with torch.no_grad():
        for k in range(n_clusters):
            mask = cluster_labels == k
            count = int(mask.sum())
            if count == 0:
                # Empty cluster — record placeholder; backfilled with zeros
                # of the right shape after we know layer dims.
                valid_cluster_mask.append(False)
                for full_name, _ in conv_hooks:
                    for kk in ('dd_mean', 'dd_var', 'patch_mean', 'patch_var'):
                        conv_buffers[full_name][kk].append(None)
                for full_name, _ in bn_hooks:
                    for kk in ('dd_mean', 'dd_var'):
                        bn_buffers[full_name][kk].append(None)
                continue

            valid_cluster_mask.append(True)
            xk = _augmented_batch(raw_x[mask]).to(device)
            _ = teacher_unwrapped(xk)
            # Reduce per-sample → mean over cluster samples (matches recover's
            # category_running divided by counter[cls] in save()).
            for full_name, hook in conv_hooks:
                last = hook.last
                conv_buffers[full_name]['dd_mean'].append(last['dd_mean'].mean(0))
                conv_buffers[full_name]['dd_var'].append(last['dd_var'].mean(0))
                conv_buffers[full_name]['patch_mean'].append(last['patch_mean'].mean(0))
                conv_buffers[full_name]['patch_var'].append(last['patch_var'].mean(0))
            for full_name, hook in bn_hooks:
                last = hook.last
                bn_buffers[full_name]['dd_mean'].append(last['dd_mean'].mean(0))
                bn_buffers[full_name]['dd_var'].append(last['dd_var'].mean(0))

    for _, hook in conv_hooks:
        hook.close()
    for _, hook in bn_hooks:
        hook.close()

    def _backfill(buffers, keys):
        for full_name, buf in buffers.items():
            shapes = {kk: None for kk in keys}
            for kk in keys:
                for v in buf[kk]:
                    if v is not None:
                        shapes[kk] = v.shape
                        break
            for kk in keys:
                if shapes[kk] is None:
                    raise RuntimeError(
                        f"Layer {full_name} got no valid forward; cannot determine shape."
                    )
                buf[kk] = [torch.zeros(shapes[kk], dtype=torch.float32) if v is None else v
                           for v in buf[kk]]

    _backfill(conv_buffers, ['dd_mean', 'dd_var', 'patch_mean', 'patch_var'])
    _backfill(bn_buffers, ['dd_mean', 'dd_var'])

    for full_name, buf in conv_buffers.items():
        out_dir = os.path.join(save_root, "ConvFeatureHook", f"class_{class_id}", full_name)
        os.makedirs(out_dir, exist_ok=True)
        np.savez(
            os.path.join(out_dir, "cluster_running.npz"),
            cluster_dd_mean=torch.stack(buf['dd_mean'], 0).numpy(),
            cluster_dd_var=torch.stack(buf['dd_var'], 0).numpy(),
            cluster_patch_mean=torch.stack(buf['patch_mean'], 0).numpy(),
            cluster_patch_var=torch.stack(buf['patch_var'], 0).numpy(),
        )

    for full_name, buf in bn_buffers.items():
        out_dir = os.path.join(save_root, "BNFeatureHook", f"class_{class_id}", full_name)
        os.makedirs(out_dir, exist_ok=True)
        np.savez(
            os.path.join(out_dir, "cluster_running.npz"),
            cluster_dd_mean=torch.stack(buf['dd_mean'], 0).numpy(),
            cluster_dd_var=torch.stack(buf['dd_var'], 0).numpy(),
        )

    return valid_cluster_mask
