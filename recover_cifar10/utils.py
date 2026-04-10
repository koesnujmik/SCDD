'''This code is modified from https://github.com/liuzechun/Data-Free-NAS'''

import torch
from torch import distributed
import numpy as np
import torch.nn.functional as F
import os, sys, random
import einops
import copy
import multiprocessing
import torch.distributed as dist


def distributed_is_initialized():
    if distributed.is_available():
        if distributed.is_initialized():
            return True
    return False


def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr


def div_four_mul(v):
    v = int(v)
    m = v % 4
    return int(v // 4 * 4) + int(m > 0) * 4


def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)


def beta_policy(mom_fn):
    def _alr(optimizer, iteration, epoch, param, indx):
        mom = mom_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group[param][indx] = mom

    return _alr


def mom_cosine_policy(base_beta, warmup_length, epochs):
    def _beta_fn(iteration, epoch):
        if epoch < warmup_length:
            beta = base_beta * (epoch + 1) / warmup_length
        else:
            beta = base_beta
        return beta

    return beta_policy(_beta_fn)


def clip(image_tensor, use_fp16=False):
    '''
    adjust the input based on mean and variance
    '''
    if use_fp16:
        mean = np.array([0.5071, 0.4867, 0.4408], dtype=np.float16)
        std = np.array([0.2675, 0.2565, 0.2761], dtype=np.float16)
    else:
        mean = np.array([0.5071, 0.4867, 0.4408])
        std = np.array([0.2675, 0.2565, 0.2761])
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    return image_tensor


def denormalize(image_tensor, use_fp16=False):
    '''
    convert floats back to input
    '''
    if use_fp16:
        mean = np.array([0.5071, 0.4867, 0.4408], dtype=np.float16)
        std = np.array([0.2675, 0.2565, 0.2761], dtype=np.float16)
    else:
        mean = np.array([0.5071, 0.4867, 0.4408])
        std = np.array([0.2675, 0.2565, 0.2761])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

    return image_tensor


def safe_projection_loss(residual, direction, eps):
    residual = torch.nan_to_num(residual.reshape(-1).float(), nan=0.0, posinf=0.0, neginf=0.0)
    direction = torch.nan_to_num(direction.reshape(-1).float(), nan=0.0, posinf=0.0, neginf=0.0)
    if residual.numel() != direction.numel():
        raise RuntimeError(
            f"Projection shape mismatch: residual has {residual.numel()} elements, "
            f"direction has {direction.numel()} elements."
        )

    numer = torch.sum(residual * direction)
    denom = torch.sum(direction * direction)
    safe = torch.isfinite(numer) & torch.isfinite(denom) & (denom > eps)
    zeros = numer.new_zeros(())
    return torch.where(safe, numer.square() / denom.clamp_min(eps), zeros)


class EMA(object):
    def __init__(self, alpha, initial_value=None):
        self.alpha = alpha
        self.value = initial_value

    @torch.no_grad()
    def ema_update(self, x):
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * x


class BNFeatureHook():
    def __init__(self, module, save_path="./", training_momentum=0.4, name=None, gpu=0, flatness_weight=0,
                      category_aware = 'global', class_num_list=None):
        self.module = module
        if module is not None and name is not None:
            self.hook = module.register_forward_hook(self.post_hook_fn)
        else:
            raise ModuleNotFoundError("module and name can not be None!")
        self.dd_var = 0.
        self.dd_mean = 0.
        self.momentum = training_momentum
        self.bn_statics_list = []
        self.tea_tag = False
        self.return_tag = False
        self.flatness_weight = flatness_weight
        # BDPC fields
        self.bdpc_beta = 0.
        self.bdpc_eps = 1e-6
        self.bdpc_cur_iter = 0
        self.bdpc_max_iter = 1
        self.bdpc_schedule = True
        # Sub-component losses for logging
        self.loss_global_mean = torch.tensor(0.)
        self.loss_global_var = torch.tensor(0.)
        self.loss_class_mean = torch.tensor(0.)
        self.loss_class_var = torch.tensor(0.)
        self.loss_bdpc = torch.tensor(0.)
        self.bias_dir_mean_f1 = None    # (C,) global BN running_mean direction
        self.bias_dir_var_f1 = None     # (C,) global BN running_var direction
        self.bias_dir_cls_mean = None   # (num_classes, C) per-class mean direction
        self.bias_dir_cls_var = None    # (num_classes, C) per-class var direction
        for i in range(10):
            cls_dir = os.path.join(save_path, f"BNFeatureHook", f"class_{i}", name)
            if not os.path.exists(cls_dir):
                os.makedirs(cls_dir, exist_ok=True)
        self.category_save_path_list = [
            os.path.join(save_path, f"BNFeatureHook", f"class_{i}", name, "running.npz") for i in range(10)
        ]
        self.category_running_dd_var_list = [0. for i in range(10)]
        self.category_running_dd_mean_list = [0. for i in range(10)]
        self.load_tag = True
        self.category_aware = category_aware
        if class_num_list is not None:
            n = torch.tensor(class_num_list, dtype=torch.float)
            n_min, n_max = n.min(), n.max()
            alpha = (n - n_min) / (n_max - n_min)  # tail→0, head→1
            self.alpha_per_class = alpha.cuda(gpu)
        else:
            self.alpha_per_class = None
        if category_aware == "global":
            for i, category_save_path in enumerate(self.category_save_path_list):
                if os.path.exists(category_save_path):
                    npz_file = np.load(category_save_path)
                    self.load_tag = True & self.load_tag
                    self.category_running_dd_var_list[i] = torch.from_numpy(npz_file["running_dd_var"]).cuda(gpu)
                    self.category_running_dd_mean_list[i] = torch.from_numpy(npz_file["running_dd_mean"]).cuda(gpu)
                else:
                    self.load_tag = False
                    self.counter = [0 for i in range(10)]
            if self.load_tag:
                self.category_running_dd_var_list = torch.stack(self.category_running_dd_var_list,0)
                self.category_running_dd_mean_list = torch.stack(self.category_running_dd_mean_list,0)
            
    def set_ori(self):
        self.tea_tag = False
    
    def set_return(self):
        self.return_tag = True
    
    def remove_return(self):
        self.return_tag = False
    
    def set_tea(self):
        self.tea_tag = True

    def set_label(self,targets):
        """
        targets: (B,)
        This function used to acquire the category information within this batch.
        """
        self.targets = targets

    def set_bias_direction(self, mean_f1, var_f1, cls_mean, cls_var):
        """Set precomputed bias direction vectors for BDPC loss."""
        self.bias_dir_mean_f1 = mean_f1    # (C,)
        self.bias_dir_var_f1 = var_f1       # (C,)
        self.bias_dir_cls_mean = cls_mean   # (num_classes, C)
        self.bias_dir_cls_var = cls_var     # (num_classes, C)

    def set_hook(self, pre=True):
        if hasattr(self, "hook"):
            self.close()
        if pre:
            self.hook = self.module.register_forward_hook(self.pre_hook_fn)
        else:
            self.hook = self.module.register_forward_hook(self.post_hook_fn)

    def save(self):
        for i, category_save_path in enumerate(self.category_save_path_list):
            self.category_running_dd_mean_list[i] = self.category_running_dd_mean_list[i]/self.counter[i]
            self.category_running_dd_var_list[i] = self.category_running_dd_var_list[i]/self.counter[i]
            npz_file = {"running_dd_var": self.category_running_dd_var_list[i].cpu().numpy() if isinstance(self.category_running_dd_var_list[i],
                                                                                      torch.Tensor) else self.category_running_dd_var_list[i],
                        "running_dd_mean": self.category_running_dd_mean_list[i].cpu().numpy() if isinstance(self.category_running_dd_mean_list[i],
                                                                                        torch.Tensor) else self.category_running_dd_mean_list[i]}
            np.savez(category_save_path, **npz_file)
        self.category_running_dd_var_list = torch.stack(self.category_running_dd_var_list, 0)
        self.category_running_dd_mean_list = torch.stack(self.category_running_dd_mean_list, 0)

    @torch.no_grad()
    def pre_hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        bs = input[0].shape[0]
        input_0 = input[0]
        # category-aware
        dd_mean = input_0.mean([2, 3])
        dd_var = input_0.view(bs, nch, -1).var(2, unbiased=False)
        for i in range(bs):
            c_m,c_v,cls = dd_mean[i],dd_var[i],self.targets[i].int().item()
            self.category_running_dd_var_list[cls] += c_v
            self.category_running_dd_mean_list[cls] += c_m
            self.counter[cls] += 1

    def post_hook_fn(self, module, input, output):
        if self.return_tag:
            return
        if self.category_aware == "global":
            nch = input[0].shape[1]
            bs = input[0].shape[0]
            input_0 = input[0]
            mean = input_0.mean([0, 2, 3])
            var = (input_0.permute(1, 0, 2, 3).contiguous().reshape([nch, -1])).var(1, unbiased=False)
            with torch.no_grad():
                if isinstance(self.dd_var, int):
                    self.dd_var = var
                    self.dd_mean = mean
                else:
                    self.dd_var = self.momentum * self.dd_var + (1 - self.momentum) * var
                    self.dd_mean = self.momentum * self.dd_mean + (1 - self.momentum) * mean
            syn_mean = self.dd_mean + mean - mean.detach()
            syn_var = self.dd_var + var - var.detach()
            loss_global_mean = torch.norm(module.running_mean.data - syn_mean, 2)
            loss_global_var = torch.norm(module.running_var.data - syn_var, 2)
            if self.alpha_per_class is not None:
                batch_cls = self.targets.long()
                mean_alpha = self.alpha_per_class[batch_cls].mean()
                unique_cls = batch_cls.unique()
                loss_class_mean = sum((1 - self.alpha_per_class[c]) * torch.norm(self.category_running_dd_mean_list[c] - syn_mean, 2) for c in unique_cls) / len(unique_cls)
                loss_class_var = sum((1 - self.alpha_per_class[c]) * torch.norm(self.category_running_dd_var_list[c] - syn_var, 2) for c in unique_cls) / len(unique_cls)
                r_feature = (loss_global_mean + loss_global_var) * mean_alpha + loss_class_mean + loss_class_var
            else:
                category_dd_var = self.category_running_dd_var_list[self.targets.long()].mean(0)
                category_dd_mean = self.category_running_dd_mean_list[self.targets.long()].mean(0)
                loss_class_mean = torch.norm(category_dd_mean - syn_mean, 2)
                loss_class_var = torch.norm(category_dd_var - syn_var, 2)
                r_feature = (loss_global_mean + loss_global_var + loss_class_mean + loss_class_var) * 0.5
            self.loss_global_mean = loss_global_mean.detach()
            self.loss_global_var = loss_global_var.detach()
            self.loss_class_mean = loss_class_mean.detach() if isinstance(loss_class_mean, torch.Tensor) else torch.tensor(float(loss_class_mean))
            self.loss_class_var = loss_class_var.detach() if isinstance(loss_class_var, torch.Tensor) else torch.tensor(float(loss_class_var))
            # BDPC projection loss
            if self.bdpc_beta > 0 and self.bias_dir_mean_f1 is not None:
                # form1: global bias direction
                bdpc_loss = safe_projection_loss(syn_mean - module.running_mean.data, self.bias_dir_mean_f1, self.bdpc_eps) \
                          + safe_projection_loss(syn_var  - module.running_var.data,  self.bias_dir_var_f1, self.bdpc_eps)
                # form2: per-class bias direction
                unique_cls = self.targets.long().unique()
                bdpc_f2 = sum(
                    safe_projection_loss(syn_mean - self.category_running_dd_mean_list[c], self.bias_dir_cls_mean[c], self.bdpc_eps) +
                    safe_projection_loss(syn_var  - self.category_running_dd_var_list[c],  self.bias_dir_cls_var[c], self.bdpc_eps)
                    for c in unique_cls
                ) / len(unique_cls)
                iter_scale = (self.bdpc_cur_iter / self.bdpc_max_iter) ** 2 if self.bdpc_schedule else 1.0
                bdpc_val = self.bdpc_beta * iter_scale * (bdpc_loss + bdpc_f2)
                r_feature = r_feature + bdpc_val
                self.loss_bdpc = bdpc_val.detach()
            else:
                self.loss_bdpc = torch.tensor(0.)
            self.r_feature = r_feature
        else:
            if self.tea_tag:
                nch = input[0].shape[1]
                bs = input[0].shape[0]
                input_0 = input[0]
                mean = input_0.mean([0, 2, 3])
                var = (input_0.permute(1, 0, 2, 3).contiguous().reshape([nch, -1])).var(1, unbiased=False)
                self.category_running_dd_var_list = var
                self.category_running_dd_mean_list = mean
            else:
                nch = input[0].shape[1]
                bs = input[0].shape[0]
                input_0 = input[0]
                mean = input_0.mean([0, 2, 3])
                var = (input_0.permute(1, 0, 2, 3).contiguous().reshape([nch, -1])).var(1, unbiased=False)
                with torch.no_grad():
                    if isinstance(self.dd_var, int):
                        self.dd_var = var
                        self.dd_mean = mean
                    else:
                        self.dd_var = self.momentum * self.dd_var + (1 - self.momentum) * var
                        self.dd_mean = self.momentum * self.dd_mean + (1 - self.momentum) * mean
                syn_mean = self.dd_mean + mean - mean.detach()
                syn_var = self.dd_var + var - var.detach()
                category_dd_var = self.category_running_dd_var_list
                category_dd_mean = self.category_running_dd_mean_list
                loss_global_mean = torch.norm(module.running_mean.data - syn_mean, 2)
                loss_global_var = torch.norm(module.running_var.data - syn_var, 2)
                loss_class_mean = torch.norm(category_dd_mean - syn_mean, 2)
                loss_class_var = torch.norm(category_dd_var - syn_var, 2)
                r_feature = (loss_global_mean + loss_global_var + loss_class_mean + loss_class_var) * 0.5
                self.loss_global_mean = loss_global_mean.detach()
                self.loss_global_var = loss_global_var.detach()
                self.loss_class_mean = loss_class_mean.detach()
                self.loss_class_var = loss_class_var.detach()
                # BDPC projection loss (local mode)
                if self.bdpc_beta > 0 and self.bias_dir_mean_f1 is not None:
                    # form1: global bias direction
                    bdpc_loss = safe_projection_loss(syn_mean - module.running_mean.data, self.bias_dir_mean_f1, self.bdpc_eps) \
                              + safe_projection_loss(syn_var  - module.running_var.data,  self.bias_dir_var_f1, self.bdpc_eps)
                    # form2: use local teacher stats as target, single "class" direction
                    bdpc_loss += safe_projection_loss(syn_mean - category_dd_mean, self.bias_dir_cls_mean[self.targets.long()[0]], self.bdpc_eps) \
                               + safe_projection_loss(syn_var  - category_dd_var,  self.bias_dir_cls_var[self.targets.long()[0]], self.bdpc_eps)
                    iter_scale = (self.bdpc_cur_iter / self.bdpc_max_iter) ** 2 if self.bdpc_schedule else 1.0
                    bdpc_val = self.bdpc_beta * iter_scale * bdpc_loss
                    r_feature = r_feature + bdpc_val
                    self.loss_bdpc = bdpc_val.detach()
                else:
                    self.loss_bdpc = torch.tensor(0.)
                self.r_feature = r_feature

    def close(self):
        self.hook.remove()

import collections


class ConvFeatureHook():
    def __init__(self, module=None, save_path="./", data_number=50000, name=None, gpu=0, training_momentum=0.4,
                 drop_rate=0.4, flatness_weight=0, category_aware = 'global', class_num_list=None):
        self.counter = collections.defaultdict(int)
        self.module = module
        if module is not None and name is not None:
            self.hook = module.register_forward_hook(self.post_hook_fn)
        else:
            raise ModuleNotFoundError("module and name can not be None!")
        self.data_number = data_number
        self.dd_var = 0.
        self.dd_mean = 0.
        self.patch_var = 0.
        self.patch_mean = 0.
        self.flatness_weight = flatness_weight
        self.momentum = training_momentum  # origin = 0.2
        self.drop_rate = drop_rate  # 0.0 0.4 0.8
        # BDPC fields
        self.bdpc_beta = 0.
        self.bdpc_eps = 1e-6
        self.bdpc_cur_iter = 0
        self.bdpc_max_iter = 1
        self.bdpc_schedule = True
        # Sub-component losses for logging
        self.loss_global_mean = torch.tensor(0.)
        self.loss_global_var = torch.tensor(0.)
        self.loss_class_mean = torch.tensor(0.)
        self.loss_class_var = torch.tensor(0.)
        self.loss_bdpc = torch.tensor(0.)
        self.bias_dir_dd_mean_f1 = None       # (C,) global
        self.bias_dir_dd_var_f1 = None        # (C,)
        self.bias_dir_patch_mean_f1 = None    # (num_patches,)
        self.bias_dir_patch_var_f1 = None     # (num_patches,)
        self.bias_dir_cls_dd_mean = None      # (num_classes, C)
        self.bias_dir_cls_dd_var = None       # (num_classes, C)
        self.bias_dir_cls_patch_mean = None   # (num_classes, num_patches)
        self.bias_dir_cls_patch_var = None    # (num_classes, num_patches)
        dir = os.path.join(save_path, "ConvFeatureHook", name)
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)
        self.conv_statics_list = []
        self.tea_tag = False
        self.return_tag = False
        self.save_path = os.path.join(save_path, "ConvFeatureHook", name, "running.npz")
        if os.path.exists(self.save_path):
            npz_file = np.load(self.save_path)
            self.load_tag = True
            self.running_dd_var = torch.from_numpy(npz_file["running_dd_var"]).cuda(gpu)
            self.running_dd_mean = torch.from_numpy(npz_file["running_dd_mean"]).cuda(gpu)
            self.running_patch_var = torch.from_numpy(npz_file["running_patch_var"]).cuda(gpu)
            self.running_patch_mean = torch.from_numpy(npz_file["running_patch_mean"]).cuda(gpu)
        else:
            self.load_tag = False
            self.running_dd_var = 0.
            self.running_dd_mean = 0.
            self.running_patch_var = 0.
            self.running_patch_mean = 0.

        self.category_aware = category_aware
        if class_num_list is not None:
            n = torch.tensor(class_num_list, dtype=torch.float)
            n_min, n_max = n.min(), n.max()
            alpha = 1 - ((n - n_min) / (n_max - n_min))  # tail→1, head→0 (reverse)
            self.alpha_per_class = alpha.cuda(gpu)
        else:
            self.alpha_per_class = None
        if category_aware == "global":
            self.category_save_path_list = [
                os.path.join(save_path, f"ConvFeatureHook", f"class_{i}", name, "running.npz") for i in range(10)
            ]
            for i in range(10):
                cls_dir = os.path.join(save_path, f"ConvFeatureHook", f"class_{i}", name)
                if not os.path.exists(cls_dir):
                    os.makedirs(cls_dir, exist_ok=True)
            self.category_running_dd_var_list = [0. for i in range(10)]
            self.category_running_dd_mean_list = [0. for i in range(10)]
            self.category_running_patch_var_list = [0. for i in range(10)]
            self.category_running_patch_mean_list = [0. for i in range(10)]

            for i, category_save_path in enumerate(self.category_save_path_list):
                if os.path.exists(category_save_path):
                    npz_file = np.load(category_save_path)
                    self.load_tag = True & self.load_tag
                    self.category_running_dd_var_list[i] = torch.from_numpy(npz_file["running_dd_var"]).cuda(gpu)
                    self.category_running_dd_mean_list[i] = torch.from_numpy(npz_file["running_dd_mean"]).cuda(gpu)
                    patch_stats_version = int(npz_file["patch_stats_version"]) if "patch_stats_version" in npz_file.files else 1
                    if patch_stats_version >= 2:
                        self.category_running_patch_var_list[i] = torch.from_numpy(npz_file["running_patch_var"]).cuda(gpu)
                        self.category_running_patch_mean_list[i] = torch.from_numpy(npz_file["running_patch_mean"]).cuda(gpu)
                    else:
                        self.category_running_patch_var_list[i] = torch.from_numpy(npz_file["running_patch_mean"]).cuda(gpu)
                        self.category_running_patch_mean_list[i] = torch.from_numpy(npz_file["running_patch_var"]).cuda(gpu)
                else:
                    self.load_tag = False
                    self.counter = [0 for i in range(10)]
            
            if self.load_tag:
                self.category_running_dd_var_list = torch.stack(self.category_running_dd_var_list,0)
                self.category_running_dd_mean_list = torch.stack(self.category_running_dd_mean_list,0)
                self.category_running_patch_var_list = torch.stack(self.category_running_patch_var_list,0)
                self.category_running_patch_mean_list = torch.stack(self.category_running_patch_mean_list,0)

    def set_ori(self):
        self.tea_tag = False
    
    def set_tea(self):
        self.tea_tag = True

    def set_label(self,targets):
        self.targets = targets

    def set_bias_direction(self, dd_mean_f1, dd_var_f1, patch_mean_f1, patch_var_f1,
                           cls_dd_mean, cls_dd_var, cls_patch_mean, cls_patch_var):
        """Set precomputed bias direction vectors for BDPC loss."""
        self.bias_dir_dd_mean_f1 = dd_mean_f1          # (C,)
        self.bias_dir_dd_var_f1 = dd_var_f1             # (C,)
        self.bias_dir_patch_mean_f1 = patch_mean_f1     # (num_patches,)
        self.bias_dir_patch_var_f1 = patch_var_f1       # (num_patches,)
        self.bias_dir_cls_dd_mean = cls_dd_mean         # (num_classes, C)
        self.bias_dir_cls_dd_var = cls_dd_var           # (num_classes, C)
        self.bias_dir_cls_patch_mean = cls_patch_mean   # (num_classes, num_patches)
        self.bias_dir_cls_patch_var = cls_patch_var     # (num_classes, num_patches)

    def set_return(self):
        self.return_tag = True

    def remove_return(self):
        self.return_tag = False

    def save(self):
        npz_file = {"running_dd_var": self.running_dd_var.cpu().numpy() if isinstance(self.running_dd_var,
                                                                                      torch.Tensor) else self.running_dd_var,
                    "running_dd_mean": self.running_dd_mean.cpu().numpy() if isinstance(self.running_dd_mean,
                                                                                        torch.Tensor) else self.running_dd_mean,
                    "running_patch_var": self.running_patch_var.cpu().numpy() if isinstance(self.running_patch_var,
                                                                                            torch.Tensor) else self.running_patch_var,
                    "running_patch_mean": self.running_patch_mean.cpu().numpy() if isinstance(self.running_patch_mean,
                                                                                              torch.Tensor) else self.running_patch_mean,
                    "patch_stats_version": np.array(2, dtype=np.int64)}
        np.savez(self.save_path, **npz_file)

        for i, category_save_path in enumerate(self.category_save_path_list):
            self.category_running_dd_mean_list[i] = self.category_running_dd_mean_list[i]/self.counter[i]
            self.category_running_dd_var_list[i] = self.category_running_dd_var_list[i]/self.counter[i]
            self.category_running_patch_mean_list[i] = self.category_running_patch_mean_list[i]/self.counter[i]
            self.category_running_patch_var_list[i] = self.category_running_patch_var_list[i]/self.counter[i]
            npz_file = {"running_dd_var": self.category_running_dd_var_list[i].cpu().numpy() if isinstance(self.category_running_dd_var_list[i],
                                                                                      torch.Tensor) else self.category_running_dd_var_list[i],
                        "running_dd_mean": self.category_running_dd_mean_list[i].cpu().numpy() if isinstance(self.category_running_dd_mean_list[i],
                                                                                        torch.Tensor) else self.category_running_dd_mean_list[i],
                        "running_patch_var": self.category_running_patch_var_list[i].cpu().numpy() if isinstance(self.category_running_patch_var_list[i],
                                                                                            torch.Tensor) else self.category_running_patch_var_list[i],
                        "running_patch_mean": self.category_running_patch_mean_list[i].cpu().numpy() if isinstance(self.category_running_patch_mean_list[i],
                                                                                              torch.Tensor) else self.category_running_patch_mean_list[i],
                        "patch_stats_version": np.array(2, dtype=np.int64)}
            np.savez(category_save_path, **npz_file)
        self.category_running_dd_var_list = torch.stack(self.category_running_dd_var_list, 0)
        self.category_running_dd_mean_list = torch.stack(self.category_running_dd_mean_list, 0)
        self.category_running_patch_var_list = torch.stack(self.category_running_patch_var_list, 0)
        self.category_running_patch_mean_list = torch.stack(self.category_running_patch_mean_list, 0)

    def set_hook(self, pre=True):
        if hasattr(self, "hook"):
            self.close()
        if pre:
            self.hook = self.module.register_forward_hook(self.pre_hook_fn)
        else:
            self.hook = self.module.register_forward_hook(self.post_hook_fn)

    @torch.no_grad()
    def pre_hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        bs = input[0].shape[0]
        input_0 = input[0]
        dd_mean = input_0.mean([0, 2, 3])
        dd_var = (input_0.permute(1, 0, 2, 3).contiguous().reshape([nch, -1])).var(1, unbiased=False)
        new_h, new_w = div_four_mul(input_0.shape[2]), div_four_mul(input_0.shape[3])
        new_input_0 = F.interpolate(input_0, [new_h, new_w], mode="bilinear")
        class_new_input_0 = einops.rearrange(new_input_0, "b c (u h) (v w) -> b (u v) (c h w)", h=4, w=4).contiguous()
        new_input_0 = einops.rearrange(new_input_0, "b c (u h) (v w) -> (u v) (b c h w)", h=4, w=4).contiguous()
        patch_mean = new_input_0.mean([1])
        patch_var = new_input_0.var([1], unbiased=False)
        self.running_dd_var += (dd_var * bs / self.data_number)
        self.running_dd_mean += (dd_mean * bs / self.data_number)
        self.running_patch_var += (patch_var * bs / self.data_number)
        self.running_patch_mean += (patch_mean * bs / self.data_number)

        # category-aware
        dd_mean = input_0.mean([2, 3])
        dd_var = input_0.view(bs, nch, -1).var(2, unbiased=False)
        patch_mean = class_new_input_0.mean([2])
        patch_var = class_new_input_0.var([2], unbiased=False)

        for i in range(bs):
            c_m,c_v,p_m,p_v,cls = dd_mean[i],dd_var[i],patch_mean[i],patch_var[i],self.targets[i].int().item()
            self.category_running_dd_var_list[cls] += c_v
            self.category_running_dd_mean_list[cls] += c_m
            self.category_running_patch_mean_list[cls] += p_m
            self.category_running_patch_var_list[cls] += p_v
            self.counter[cls] += 1


    def post_hook_fn(self, module, input, output):
        if self.return_tag:
            return
        if self.category_aware == "local":
            if self.tea_tag:
                nch = input[0].shape[1]
                bs = input[0].shape[0]
                input_0 = input[0]
                dd_mean = input_0.mean([0, 2, 3])
                dd_var = (input_0.permute(1, 0, 2, 3).contiguous().reshape([nch, -1])).var(1, unbiased=False)
                new_h, new_w = div_four_mul(input_0.shape[2]), div_four_mul(input_0.shape[3])
                new_input_0 = F.interpolate(input_0, [new_h, new_w], mode="bilinear")
                new_input_0 = einops.rearrange(new_input_0, "b c (u h) (v w) -> (u v) (b c h w)", h=4, w=4).contiguous()
                patch_mean = new_input_0.mean([1])
                patch_var = new_input_0.var([1], unbiased=False)        
                self.category_running_dd_var_list = dd_var
                self.category_running_dd_mean_list = dd_mean
                self.category_running_patch_mean_list = patch_mean
                self.category_running_patch_var_list = patch_var
            else:
                if random.random() > (1. - self.drop_rate):
                    self.r_feature = torch.Tensor([0.]).to(input[0].device)
                    return
                nch = input[0].shape[1]
                input_0 = input[0]
                dd_mean = input_0.mean([0, 2, 3])
                dd_var = (input_0.permute(1, 0, 2, 3).contiguous().reshape([nch, -1])).var(1, unbiased=False)
                new_h, new_w = div_four_mul(input_0.shape[2]), div_four_mul(input_0.shape[3])
                new_input_0 = F.interpolate(input_0, [new_h, new_w], mode="bilinear")
                new_input_0 = einops.rearrange(new_input_0, "b c (u h) (v w) -> (u v) (b c h w)", h=4, w=4).contiguous()
                patch_mean = new_input_0.mean([1])
                patch_var = new_input_0.var([1], unbiased=False)        
                with torch.no_grad():
                    if isinstance(self.dd_var, int):
                        self.dd_var = dd_var
                        self.dd_mean = dd_mean
                        self.patch_var = patch_var
                        self.patch_mean = patch_mean
                    else:
                        self.dd_var = self.momentum * self.dd_var + (1 - self.momentum) * dd_var
                        self.dd_mean = self.momentum * self.dd_mean + (1 - self.momentum) * dd_mean
                        self.patch_var = self.momentum * self.patch_var + (1 - self.momentum) * patch_var
                        self.patch_mean = self.momentum * self.patch_mean + (1 - self.momentum) * patch_mean

                syn_dd_mean = self.dd_mean + dd_mean - dd_mean.detach()
                syn_dd_var = self.dd_var + dd_var - dd_var.detach()
                syn_patch_mean = self.patch_mean + patch_mean - patch_mean.detach()
                syn_patch_var = self.patch_var + patch_var - patch_var.detach()
                category_dd_var = self.category_running_dd_var_list
                category_dd_mean = self.category_running_dd_mean_list
                category_patch_var = self.category_running_patch_var_list
                category_patch_mean = self.category_running_patch_mean_list
                loss_global_mean = torch.norm(self.running_dd_mean - syn_dd_mean, 2) + torch.norm(self.running_patch_mean - syn_patch_mean, 2)
                loss_global_var = torch.norm(self.running_dd_var - syn_dd_var, 2) + torch.norm(self.running_patch_var - syn_patch_var, 2)
                loss_class_mean = torch.norm(category_dd_mean - syn_dd_mean, 2) + torch.norm(category_patch_mean - syn_patch_mean, 2)
                loss_class_var = torch.norm(category_dd_var - syn_dd_var, 2) + torch.norm(category_patch_var - syn_patch_var, 2)
                r_feature = (loss_global_mean + loss_global_var + loss_class_mean + loss_class_var) * 0.5
                self.loss_global_mean = loss_global_mean.detach()
                self.loss_global_var = loss_global_var.detach()
                self.loss_class_mean = loss_class_mean.detach()
                self.loss_class_var = loss_class_var.detach()
                # BDPC projection loss (local mode)
                if self.bdpc_beta > 0 and self.bias_dir_dd_mean_f1 is not None:
                    # form1: global bias direction
                    bdpc_loss = safe_projection_loss(syn_dd_mean    - self.running_dd_mean,    self.bias_dir_dd_mean_f1, self.bdpc_eps) \
                              + safe_projection_loss(syn_dd_var     - self.running_dd_var,     self.bias_dir_dd_var_f1, self.bdpc_eps) \
                              + safe_projection_loss(syn_patch_mean - self.running_patch_mean, self.bias_dir_patch_mean_f1, self.bdpc_eps) \
                              + safe_projection_loss(syn_patch_var  - self.running_patch_var,  self.bias_dir_patch_var_f1, self.bdpc_eps)
                    # form2: local teacher stats as target, single class direction
                    cls_idx = self.targets.long()[0]
                    bdpc_loss += safe_projection_loss(syn_dd_mean    - category_dd_mean,    self.bias_dir_cls_dd_mean[cls_idx], self.bdpc_eps) \
                               + safe_projection_loss(syn_dd_var     - category_dd_var,     self.bias_dir_cls_dd_var[cls_idx], self.bdpc_eps) \
                               + safe_projection_loss(syn_patch_mean - category_patch_mean, self.bias_dir_cls_patch_mean[cls_idx], self.bdpc_eps) \
                               + safe_projection_loss(syn_patch_var  - category_patch_var,  self.bias_dir_cls_patch_var[cls_idx], self.bdpc_eps)
                    iter_scale = (self.bdpc_cur_iter / self.bdpc_max_iter) ** 2 if self.bdpc_schedule else 1.0
                    bdpc_val = self.bdpc_beta * iter_scale * bdpc_loss
                    r_feature = r_feature + bdpc_val
                    self.loss_bdpc = bdpc_val.detach()
                else:
                    self.loss_bdpc = torch.tensor(0.)
                self.r_feature = r_feature
        else:
            if random.random() > (1. - self.drop_rate):
                self.r_feature = torch.Tensor([0.]).to(input[0].device)
                return
            nch = input[0].shape[1]
            bs = input[0].shape[0]
            input_0 = input[0]
            dd_mean = input_0.mean([0, 2, 3])
            dd_var = (input_0.permute(1, 0, 2, 3).contiguous().reshape([nch, -1])).var(1, unbiased=False)
            new_h, new_w = div_four_mul(input_0.shape[2]), div_four_mul(input_0.shape[3])
            new_input_0 = F.interpolate(input_0, [new_h, new_w], mode="bilinear")
            new_input_0 = einops.rearrange(new_input_0, "b c (u h) (v w) -> (u v) (b c h w)", h=4, w=4).contiguous()
            patch_mean = new_input_0.mean([1])
            patch_var = new_input_0.var([1], unbiased=False)        
            with torch.no_grad():
                if isinstance(self.dd_var, int):
                    self.dd_var = dd_var
                    self.dd_mean = dd_mean
                    self.patch_var = patch_var
                    self.patch_mean = patch_mean
                else:
                    self.dd_var = self.momentum * self.dd_var + (1 - self.momentum) * dd_var
                    self.dd_mean = self.momentum * self.dd_mean + (1 - self.momentum) * dd_mean
                    self.patch_var = self.momentum * self.patch_var + (1 - self.momentum) * patch_var
                    self.patch_mean = self.momentum * self.patch_mean + (1 - self.momentum) * patch_mean

            syn_dd_mean = self.dd_mean + dd_mean - dd_mean.detach()
            syn_dd_var = self.dd_var + dd_var - dd_var.detach()
            syn_patch_mean = self.patch_mean + patch_mean - patch_mean.detach()
            syn_patch_var = self.patch_var + patch_var - patch_var.detach()
            loss_global_mean = torch.norm(self.running_dd_mean - syn_dd_mean, 2) + torch.norm(self.running_patch_mean - syn_patch_mean, 2)
            loss_global_var = torch.norm(self.running_dd_var - syn_dd_var, 2) + torch.norm(self.running_patch_var - syn_patch_var, 2)
            if self.alpha_per_class is not None:
                batch_cls = self.targets.long()
                mean_alpha = self.alpha_per_class[batch_cls].mean()
                unique_cls = batch_cls.unique()
                loss_class_mean = sum((1 - self.alpha_per_class[c]) * (torch.norm(self.category_running_dd_mean_list[c] - syn_dd_mean, 2) + torch.norm(self.category_running_patch_mean_list[c] - syn_patch_mean, 2)) for c in unique_cls) / len(unique_cls)
                loss_class_var = sum((1 - self.alpha_per_class[c]) * (torch.norm(self.category_running_dd_var_list[c] - syn_dd_var, 2) + torch.norm(self.category_running_patch_var_list[c] - syn_patch_var, 2)) for c in unique_cls) / len(unique_cls)
                r_feature = (loss_global_mean + loss_global_var) * mean_alpha + loss_class_mean + loss_class_var
            else:
                category_dd_var = self.category_running_dd_var_list[self.targets.long()].mean(0)
                category_dd_mean = self.category_running_dd_mean_list[self.targets.long()].mean(0)
                category_patch_var = self.category_running_patch_var_list[self.targets.long()].mean(0)
                category_patch_mean = self.category_running_patch_mean_list[self.targets.long()].mean(0)
                loss_class_mean = torch.norm(category_dd_mean - syn_dd_mean, 2) + torch.norm(category_patch_mean - syn_patch_mean, 2)
                loss_class_var = torch.norm(category_dd_var - syn_dd_var, 2) + torch.norm(category_patch_var - syn_patch_var, 2)
                r_feature = (loss_global_mean + loss_global_var + loss_class_mean + loss_class_var) * 0.5
            self.loss_global_mean = loss_global_mean.detach()
            self.loss_global_var = loss_global_var.detach()
            self.loss_class_mean = loss_class_mean.detach() if isinstance(loss_class_mean, torch.Tensor) else torch.tensor(float(loss_class_mean))
            self.loss_class_var = loss_class_var.detach() if isinstance(loss_class_var, torch.Tensor) else torch.tensor(float(loss_class_var))
            # BDPC projection loss (global mode)
            if self.bdpc_beta > 0 and self.bias_dir_dd_mean_f1 is not None:
                # form1: global bias direction
                bdpc_loss = safe_projection_loss(syn_dd_mean    - self.running_dd_mean,    self.bias_dir_dd_mean_f1, self.bdpc_eps) \
                          + safe_projection_loss(syn_dd_var     - self.running_dd_var,     self.bias_dir_dd_var_f1, self.bdpc_eps) \
                          + safe_projection_loss(syn_patch_mean - self.running_patch_mean, self.bias_dir_patch_mean_f1, self.bdpc_eps) \
                          + safe_projection_loss(syn_patch_var  - self.running_patch_var,  self.bias_dir_patch_var_f1, self.bdpc_eps)
                # form2: per-class bias direction
                unique_cls = self.targets.long().unique()
                bdpc_f2 = sum(
                    safe_projection_loss(syn_dd_mean    - self.category_running_dd_mean_list[c],    self.bias_dir_cls_dd_mean[c], self.bdpc_eps) +
                    safe_projection_loss(syn_dd_var     - self.category_running_dd_var_list[c],     self.bias_dir_cls_dd_var[c], self.bdpc_eps) +
                    safe_projection_loss(syn_patch_mean - self.category_running_patch_mean_list[c], self.bias_dir_cls_patch_mean[c], self.bdpc_eps) +
                    safe_projection_loss(syn_patch_var  - self.category_running_patch_var_list[c],  self.bias_dir_cls_patch_var[c], self.bdpc_eps)
                    for c in unique_cls
                ) / len(unique_cls)
                iter_scale = (self.bdpc_cur_iter / self.bdpc_max_iter) ** 2 if self.bdpc_schedule else 1.0
                bdpc_val = self.bdpc_beta * iter_scale * (bdpc_loss + bdpc_f2)
                r_feature = r_feature + bdpc_val
                self.loss_bdpc = bdpc_val.detach()
            else:
                self.loss_bdpc = torch.tensor(0.)
            self.r_feature = r_feature

    def close(self):
        self.hook.remove()


class SimpleHook():
    def __init__(self, module=None):
        self.module = module
        if module is not None:
            self.hook = module.register_forward_hook(self.post_hook_fn)
        else:
            raise ModuleNotFoundError("module and name can not be None!")
        
    def post_hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        input_0 = input[0]
        dd_mean = input_0.mean([0, 2, 3])
        dd_var = (input_0.permute(1, 0, 2, 3).contiguous().reshape([nch, -1])).var(1, unbiased=False)
        self.dd_mean = dd_mean
        self.dd_var = dd_var

    def close(self):
        self.hook.remove()
            
class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


def get_image_prior_losses(inputs_jit):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0

    return loss_var_l1, loss_var_l2


from torchvision.datasets import ImageFolder


class PreImgPathCache(ImageFolder):
    def __init__(self, root, transforms):
        super(PreImgPathCache, self).__init__(root, transform=transforms)
        self.label2img = [[] for _ in range(len(self.classes))]
        self.label_ptr = [0 for _ in range(len(self.classes))]  

        for img_path, label in self.imgs:
            self.label2img[label].append(img_path)

    def random_img_sample(self, idx):
        imgpaths = self.label2img[idx]
        new_idx = np.random.choice(len(imgpaths), (1,), replace=False)[0]
        imgpath = imgpaths[new_idx]
        sample = self.loader(imgpath)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def sequential_img_sample(self, idx):
        imgpaths = self.label2img[idx]
        ptr = self.label_ptr[idx]
        imgpath = imgpaths[ptr % len(imgpaths)]  # 超过就从头开始
        self.label_ptr[idx] += 1

        sample = self.loader(imgpath)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

class GenerateGaussianDisturb():
    def __init__(self,model,mean_std=[0,0.05]):
        self.model = copy.deepcopy(model)
        self.ws = copy.deepcopy(self.model.state_dict())
        self.mean = mean_std[0]
        self.std = mean_std[1]
    
    @torch.no_grad()
    def generate_disturb_parameters(self, new_model,mean_std=None):
        if mean_std is not None:
            self.mean, self.std = mean_std
        new_model_named_parameters = dict(new_model.named_parameters())
        self._normalized_gaussian_noise()
        m = self.std
        for name, param in self.model.named_parameters():
            new_model_named_parameters[name].data = param.data + m * self.x[name]

    @torch.no_grad()
    def _normalized_gaussian_noise(self):
        self.x = self._find_direction(self.model)
        self.x = self._normalize_filter(self.x, self.ws)
        self.x = self._ignore_bn(self.x)

    @torch.no_grad()
    def _find_direction(self, model):
        x = {}
        for name, param in model.named_parameters():
            x[name] = torch.randn_like(param.data)
        return x

    @torch.no_grad()
    def _normalize_filter(self, bs, ws):
        # TODO: normalize
        bs = {k: v.float() for k, v in bs.items()}
        ws = {k: v.float() for k, v in ws.items()}
        norm_bs = {}
        for k in bs:
            ws_norm = torch.norm(ws[k], dim=0, keepdim=True)
            bs_norm = torch.norm(bs[k], dim=0, keepdim=True)
            norm_bs[k] = ws_norm / (bs_norm + 1e-7) * bs[k]  # random * true_norm / rand_norm
        return norm_bs

    @torch.no_grad()
    def _ignore_bn(self, ws):
        ignored_ws = {}
        for k in ws:
            if len(ws[k].size()) < 2:
                ignored_ws[k] = torch.zeros(size=ws[k].size(), device=ws[k].device)
            else:
                ignored_ws[k] = ws[k]
        return ignored_ws
    

def noise_schedule(niter,titer,sigma_max,sigma_min=0,rho=7):
    nsigma = sigma_max ** (1/rho) + (niter/titer) * (sigma_min ** (1/rho) - sigma_max ** (1/rho))
    return nsigma ** (rho)

class ShufflePatches(torch.nn.Module):
    def shuffle_weight(self, img, factor):
        h, w = img.shape[1:]
        th, tw = h // factor, w // factor
        patches = []
        for i in range(factor):
            i = i * tw
            if i != factor - 1:
                patches.append(img[..., i : i + tw])
            else:
                patches.append(img[..., i:])
        indices = np.random.choice(list(range(len(patches))),len(patches),replace=False)
        new_patches = []
        for indice in indices:
            new_patches.append(patches[indice])
        img = torch.cat(patches, -1)
        return img

    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, img):
        img = self.shuffle_weight(img, self.factor)
        img = img.permute(0, 2, 1)
        img = self.shuffle_weight(img, self.factor)
        img = img.permute(0, 2, 1)
        return img
