'''This code is modified from https://github.com/liuzechun/Data-Free-NAS'''

import os
import random
import argparse
import collections
import json
import time
import faulthandler

import wandb
from tqdm import tqdm
import numpy as np
import torchvision.datasets
import ResNet_cifar
from PIL import Image

import torch.multiprocessing as mp
import torch
import torch.utils
import torch.nn as nn
import torch.optim as optim
import cifar10Imbanlance
from torchvision import transforms
import torch.utils.data.distributed
import multiprocessing
import torch.distributed as dist
import torchvision.models as models
from utils import *
import models as ti_models
from baseline import get_network as ti_get_network

faulthandler.enable(all_threads=True)


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


def SupConLoss(features, labels, temperature=0.1):
    """Supervised Contrastive Loss (Khosla et al. 2020).
    features: [N, dim] unnormalized embeddings
    labels:   [N] integer class labels
    """
    import torch.nn.functional as F
    features = F.normalize(features, dim=1)
    N = features.shape[0]
    device = features.device
    sim = torch.matmul(features, features.T) / temperature
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()
    mask_self = torch.eye(N, dtype=torch.bool, device=device)
    labels_col = labels.view(-1, 1)
    mask_pos = (labels_col == labels_col.T).float().masked_fill(mask_self, 0)
    exp_sim = torch.exp(sim) * (~mask_self).float()
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    num_pos = mask_pos.sum(dim=1)
    valid = num_pos > 0
    if not valid.any():
        return torch.tensor(0., device=device, requires_grad=True)
    mean_log_prob_pos = (mask_pos * log_prob).sum(dim=1) / (num_pos + 1e-8)
    return -mean_log_prob_pos[valid].mean()


class ApplyTransformToPair:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img1, img2):
        seed = random.randint(0,int(1e13))
        torch.manual_seed(seed)
        img1_transformed = self.transform(img1)

        torch.manual_seed(seed)
        img2_transformed = self.transform(img2)

        return img1_transformed, img2_transformed


def build_cifar_model(arch_name, num_classes):
    if arch_name == "convnet":
        return ti_models.model_dict["convnet"](num_classes=num_classes)
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


def random_backbone(args,gpu):
    random_init_backbone = []
    for name in args.aux_teacher:
        model = build_cifar_model(name, num_classes=10)
        random_init_backbone.append(model.cuda(gpu).eval())
    for _random_init_backbone in random_init_backbone:
        for p in _random_init_backbone.parameters():
            p.requires_grad = False
    return random_init_backbone



def main_worker(gpu, ngpus_per_node, args, model_teacher, model_verifier, ipc_id_range):
    args.gpu = gpu
    print("Use GPU: {} for training".format(args.gpu))
    if args.distributed:
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    else:
        args.rank = 0

    torch.cuda.set_device(args.gpu)
    worker_seed = int(args.seed) + int(args.rank)
    seed_everything(worker_seed)
    model_teacher = [_model_teacher.cuda(gpu).eval() for _model_teacher in model_teacher]

    for _model_teacher in model_teacher:
        for p in _model_teacher.parameters():
            p.requires_grad = False
    model_verifier = model_verifier.cuda(gpu)
    model_verifier.eval()
    for p in model_verifier.parameters():
        p.requires_grad = False
    hook_for_display = lambda x, y: validate(x, y, model_verifier)

    if gpu == 0 and getattr(args, 'wandb_project', None):
        wandb.login(key=getattr(args, 'wandb_api_key', None))
        run_name = getattr(args, 'wandb_run_name', None) or args.exp_name
        wandb.init(project=args.wandb_project, name=run_name, resume="never")
        wandb.config.update({f"recover/{k}": v for k, v in vars(args).items()
                             if k not in ("wandb_api_key",)})
        run_sh_path = os.path.join(os.path.dirname(__file__), "..", "run.sh")
        if os.path.exists(run_sh_path):
            wandb.save(os.path.abspath(run_sh_path), base_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    save_every = 100
    batch_size = args.batch_size
    best_cost = 1e4
    load_tag_dict = [True for i in range(len(model_teacher))]
    loss_r_feature_layers = [[] for _ in range(len(model_teacher))]
    load_tag = True
    
    if args.adaptive_alpha:
        _data_num = 50000 // 10
        class_num_list = [int(_data_num * (args.imbanlance_rate ** (i / 9))) for i in range(10)]
    else:
        class_num_list = None

    # Resolve cluster matching state once: read manifest and build the
    # (num_classes, ipc) -> cluster_id map. We require manifest existence
    # whenever cluster_loss_weight > 0 and a path is given so that silent
    # mis-pairings between initial and recover artifacts surface immediately.
    cluster_stat_path = getattr(args, 'cluster_stat_path', None)
    cluster_loss_weight = float(getattr(args, 'cluster_loss_weight', 0.0))
    ipc_to_cluster = None
    if cluster_stat_path is not None and cluster_loss_weight > 0:
        manifest_path = os.path.join(cluster_stat_path, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"--cluster-loss-weight>0 but manifest not found at {manifest_path}"
            )
        with open(manifest_path) as f:
            manifest = json.load(f)
        if int(manifest.get('num_classes', 10)) != 10:
            raise ValueError(
                f"recover_cifar10 expects 10 classes, manifest reports "
                f"{manifest.get('num_classes')}"
            )
        if int(manifest.get('ipc', args.ipc_number)) != int(args.ipc_number):
            raise ValueError(
                f"manifest ipc={manifest.get('ipc')} != recover ipc={args.ipc_number}"
            )
        ipc_to_cluster = torch.full((10, args.ipc_number), -1, dtype=torch.long)
        for cls_id_str, cls_data in manifest['classes'].items():
            c = int(cls_id_str)
            for entry in cls_data['selected']:
                ipc_to_cluster[c, int(entry['ipc_id'])] = int(entry['cluster_id'])
        print(f"[Cluster] Loaded manifest with weight={cluster_loss_weight} from {manifest_path}")

    for i, (_model_teacher) in enumerate(model_teacher):
        for name, module in _model_teacher.named_modules():
            full_name = str(_model_teacher.__class__.__name__) + "_" + str(args.aux_teacher[i]) + "=" + name
            if isinstance(module, nn.BatchNorm2d):
                _hook_module = BNFeatureHook(module,save_path=args.statistic_path,
                                            name=full_name,
                                            gpu=gpu,training_momentum=args.training_momentum,
                                            flatness_weight=args.flatness_weight,
                                            category_aware=args.category_aware,
                                            class_num_list=class_num_list,
                                            num_classes=10,
                                            enable_struct=(args.struct_weight > 0),
                                            cluster_stat_path=cluster_stat_path,
                                            ipc_to_cluster=ipc_to_cluster,
                                            cluster_loss_weight=cluster_loss_weight)
                _hook_module.set_hook(pre=True)
                load_tag = load_tag & _hook_module.load_tag
                load_tag_dict[i] = load_tag_dict[i] & _hook_module.load_tag
                loss_r_feature_layers[i].append(_hook_module)

            elif isinstance(module, nn.Conv2d):
                _hook_module = ConvFeatureHook(module, save_path=args.statistic_path,
                                               name=full_name,
                                               gpu=gpu, training_momentum=args.training_momentum,
                                               drop_rate=args.drop_rate,
                                               flatness_weight=args.flatness_weight,
                                               category_aware=args.category_aware,
                                               class_num_list=class_num_list,
                                               num_classes=10,
                                               enable_struct=(args.struct_weight > 0),
                                               cluster_stat_path=cluster_stat_path,
                                               ipc_to_cluster=ipc_to_cluster,
                                               cluster_loss_weight=cluster_loss_weight)
                _hook_module.set_hook(pre=True)
                load_tag = load_tag & _hook_module.load_tag
                load_tag_dict[i] = load_tag_dict[i] & _hook_module.load_tag
                loss_r_feature_layers[i].append(_hook_module)


    sub_batch_size = int(batch_size // ngpus_per_node)

    initial_img_cache = PreImgPathCache(args.initial_img_dir,transforms=transforms.Compose([
                                                             transforms.Resize((32,32)),
                                                             transforms.RandomHorizontalFlip(),
                                                             transforms.ToTensor(),
                                                              transforms.Normalize([0.5071, 0.4867, 0.4408],
                                                                                     [0.2675, 0.2565, 0.2761])],
                                                             ))
    if args.category_aware == "local":
        original_img_cache = PreImgPathCache(os.path.join(args.train_data_path,"train"), transforms=transforms.Compose([
                                                                transforms.Resize((32,32)),
                                                                transforms.RandomHorizontalFlip(),
                                                                transforms.ToTensor(),
                                                                transforms.Normalize([0.5071, 0.4867, 0.4408],
                                                                                        [0.2675, 0.2565, 0.2761])]
                                                                ))

    if not load_tag:
        train_dataset = cifar10Imbanlance.Cifar10Imbanlance(transform=transforms.Compose([
                                                          transforms.RandomCrop(32, padding=4),
                                                          transforms.RandomHorizontalFlip(),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize([0.5071, 0.4867, 0.4408],
                                                                               [0.2675, 0.2565, 0.2761])
                                                      ]),imbanlance_rate=args.imbanlance_rate, train=True,file_path=args.train_data_path)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   num_workers=4,
                                                   batch_size=64,
                                                   drop_last=False,
                                                   shuffle=True,
                                                   worker_init_fn=seed_worker,
                                                   generator=make_generator(args.seed, args.rank * 1000))

        with torch.no_grad():
            for j, _model_teacher in enumerate(model_teacher):
                if not load_tag_dict[j]:
                    print(f"Conduct backbone {args.aux_teacher[j]} statistics")

                    total_seen = 0  # 初始化累计样本数
                    time1 = time.time()
                    for i, (data, targets) in tqdm(enumerate(train_loader)):
                        B = data.size(0)

                        data = data.cuda(gpu)
                        targets = targets.cuda(gpu)

                        for _loss_t_feature_layer in loss_r_feature_layers[j]:
                            _loss_t_feature_layer.set_label(targets)
                            if hasattr(_loss_t_feature_layer, "momentum"):
                                _loss_t_feature_layer.momentum = B / float(total_seen + B + 1e-6)

                        _ = _model_teacher(data)  

                        total_seen += B  
                    for _loss_t_feature_layer in loss_r_feature_layers[j]:
                        _loss_t_feature_layer.save()

        print("Training Statistic Information Is Successfully Saved")
    else:
        print("Training Statistic Information Is Successfully Load")

    for j in range(len(loss_r_feature_layers)):
        for _loss_t_feature_layer in loss_r_feature_layers[j]:
            _loss_t_feature_layer.set_hook(pre=False)
    
    if args.bdpc_beta > 0:
        for i, (_model_teacher) in enumerate(model_teacher):
            hook_idx = 0
            for name, module in _model_teacher.named_modules():
                full_name = str(_model_teacher.__class__.__name__) + "_" + str(args.aux_teacher[i]) + "=" + name
                if isinstance(module, nn.BatchNorm2d):
                    _hook = loss_r_feature_layers[i][hook_idx]
                    # form1: BN running stats bias direction
                    bn_path = os.path.join(args.biased_statistic_path, "BNFeatureHook", full_name, "bn_running.npz")
                    bn_npz = np.load(bn_path)
                    bias_dir_mean_f1 = torch.from_numpy(bn_npz["running_mean"]).cuda(gpu) - module.running_mean.data
                    bias_dir_var_f1 = torch.from_numpy(bn_npz["running_var"]).cuda(gpu) - module.running_var.data
                    # form2: per-class bias direction
                    cls_mean_dirs = []
                    cls_var_dirs = []
                    for c in range(10):
                        biased_cls_path = os.path.join(args.biased_statistic_path, "BNFeatureHook",
                                                       f"class_{c}", full_name, "running.npz")
                        biased_cls_npz = np.load(biased_cls_path)
                        biased_cls_mean = torch.from_numpy(biased_cls_npz["running_dd_mean"]).cuda(gpu)
                        biased_cls_var = torch.from_numpy(biased_cls_npz["running_dd_var"]).cuda(gpu)
                        cls_mean_dirs.append(biased_cls_mean - _hook.category_running_dd_mean_list[c])
                        cls_var_dirs.append(biased_cls_var - _hook.category_running_dd_var_list[c])
                    _hook.set_bias_direction(
                        bias_dir_mean_f1, bias_dir_var_f1,
                        torch.stack(cls_mean_dirs, 0), torch.stack(cls_var_dirs, 0))
                    _hook.bdpc_beta = args.bdpc_beta
                    _hook.bdpc_schedule = args.bdpc_schedule
                    hook_idx += 1
                elif isinstance(module, nn.Conv2d):
                    _hook = loss_r_feature_layers[i][hook_idx]
                    # form1: global Conv stats bias direction
                    biased_conv_path = os.path.join(args.biased_statistic_path, "ConvFeatureHook",
                                                    full_name, "running.npz")
                    biased_conv_npz = np.load(biased_conv_path)
                    dd_mean_f1 = torch.from_numpy(biased_conv_npz["running_dd_mean"]).cuda(gpu) - _hook.running_dd_mean
                    dd_var_f1 = torch.from_numpy(biased_conv_npz["running_dd_var"]).cuda(gpu) - _hook.running_dd_var
                    patch_mean_f1 = torch.from_numpy(biased_conv_npz["running_patch_mean"]).cuda(gpu) - _hook.running_patch_mean
                    patch_var_f1 = torch.from_numpy(biased_conv_npz["running_patch_var"]).cuda(gpu) - _hook.running_patch_var
                    # form2: per-class Conv stats bias direction
                    cls_dd_mean_dirs = []
                    cls_dd_var_dirs = []
                    cls_patch_mean_dirs = []
                    cls_patch_var_dirs = []
                    for c in range(10):
                        biased_cls_path = os.path.join(args.biased_statistic_path, "ConvFeatureHook",
                                                       f"class_{c}", full_name, "running.npz")
                        biased_cls_npz = np.load(biased_cls_path)
                        patch_stats_version = int(biased_cls_npz["patch_stats_version"]) if "patch_stats_version" in biased_cls_npz.files else 1
                        if patch_stats_version >= 2:
                            biased_patch_mean = torch.from_numpy(biased_cls_npz["running_patch_mean"]).cuda(gpu)
                            biased_patch_var = torch.from_numpy(biased_cls_npz["running_patch_var"]).cuda(gpu)
                        else:
                            biased_patch_mean = torch.from_numpy(biased_cls_npz["running_patch_var"]).cuda(gpu)
                            biased_patch_var = torch.from_numpy(biased_cls_npz["running_patch_mean"]).cuda(gpu)
                        cls_dd_mean_dirs.append(
                            torch.from_numpy(biased_cls_npz["running_dd_mean"]).cuda(gpu) - _hook.category_running_dd_mean_list[c])
                        cls_dd_var_dirs.append(
                            torch.from_numpy(biased_cls_npz["running_dd_var"]).cuda(gpu) - _hook.category_running_dd_var_list[c])
                        cls_patch_mean_dirs.append(biased_patch_mean - _hook.category_running_patch_mean_list[c])
                        cls_patch_var_dirs.append(biased_patch_var - _hook.category_running_patch_var_list[c])
                    _hook.set_bias_direction(
                        dd_mean_f1, dd_var_f1, patch_mean_f1, patch_var_f1,
                        torch.stack(cls_dd_mean_dirs, 0), torch.stack(cls_dd_var_dirs, 0),
                        torch.stack(cls_patch_mean_dirs, 0), torch.stack(cls_patch_var_dirs, 0))
                    _hook.bdpc_beta = args.bdpc_beta
                    _hook.bdpc_schedule = args.bdpc_schedule
                    hook_idx += 1

    targets_all_all = torch.LongTensor(np.arange(10))[None, ...].expand(len(ipc_id_range), 10).contiguous().view(-1)
    ipc_id_all = torch.LongTensor(ipc_id_range)[..., None].expand(len(ipc_id_range), 10).contiguous().view(-1)

    total_number = 10 * (ipc_id_range[-1] + 1 - ipc_id_range[0])
    turn_index = torch.LongTensor(np.arange(total_number)).view(len(ipc_id_range), 10) \
        .transpose(1, 0).contiguous().view(-1)
    counter = 0
    # Feature bank for cross-class SupCon: shape [10, emb_dim]
    feat_bank = None
    feat_bank_filled = torch.zeros(10, dtype=torch.bool, device=f'cuda:{gpu}')
    _has_proj_any = any(hasattr(m, 'projection_head') for m in model_teacher)
    if args.supcon_weight > 0 and _has_proj_any:
        _ref_teacher_idx = next(i for i, m in enumerate(model_teacher) if hasattr(m, 'projection_head'))
        _ref_teacher = model_teacher[_ref_teacher_idx]
        # 사전 초기화 forward 중에는 hook이 self.targets 없이 실행되므로 set_return()으로 비활성화
        for _mod in loss_r_feature_layers[_ref_teacher_idx]:
            _mod.set_return()
        with torch.no_grad():
            for _c in range(10):
                if args.initial_img_dir is not None:
                    _init_imgs = torch.stack(
                        [initial_img_cache.sequential_img_sample(_c) for _ in range(args.ipc_number)],
                        dim=0).to(f'cuda:{gpu}').float()
                else:
                    _init_imgs = torch.randn(args.ipc_number, 3, 32, 32, device=f'cuda:{gpu}')
                _z_init = _ref_teacher(_init_imgs, train=True)[2]  # [ipc_number, emb_dim]
                if feat_bank is None:
                    feat_bank = torch.zeros(10, _z_init.shape[1], device=f'cuda:{gpu}')
                feat_bank[_c] = _z_init.mean(0)
                feat_bank_filled[_c] = True
        # hook 복원
        for _mod in loss_r_feature_layers[_ref_teacher_idx]:
            _mod.remove_return()
        print(f"[SupCon] feat_bank pre-initialized, emb_dim={feat_bank.shape[1]}")
    for zz in range(0, total_number, batch_size):
        sub_turn_index = turn_index[zz + gpu * sub_batch_size:min(zz + (gpu + 1) * sub_batch_size, total_number)]
        targets = targets_all_all[sub_turn_index].cuda(gpu)
        ipc_ids = ipc_id_all[sub_turn_index].cuda(gpu)

        data_type = torch.float
        sub_batch_size = min(zz + (gpu + 1) * sub_batch_size, total_number) - (zz + gpu * sub_batch_size)
        if sub_batch_size < 0:
            continue
        print(f"In GPU {gpu}, targets is set as: \n{targets}\n, ipc_ids is set as: \n{ipc_ids}")

        if args.initial_img_dir is not None:
            inputs = torch.stack([initial_img_cache.sequential_img_sample(_target) for _target in targets.tolist()], 0).to(f'cuda:{gpu}').to(data_type)
            inputs.requires_grad_(True)
        else:
            inputs = torch.randn((sub_batch_size, 3, 32, 32), requires_grad=True, device=f'cuda:{gpu}',
                             dtype=data_type)
        
        if args.category_aware == "local":
            expand_ratio = int(50000 / (args.ipc_number*10))
            tea_images = torch.stack([original_img_cache.random_img_sample(_target) for _target in (targets.tolist() * expand_ratio)],0).to(f'cuda:{gpu}').to(data_type)
            with torch.no_grad():
                for id in range(len(args.aux_teacher)):
                    for (idx, mod) in enumerate(loss_r_feature_layers[id]):
                        mod.set_tea()
                    sub_outputs = model_teacher[id](tea_images)
        
        iterations_per_layer = args.iteration
        optimizer = optim.Adam([inputs], lr=args.lr, betas=[0.5, 0.9], eps=1e-8)
        lr_scheduler = lr_cosine_policy(args.lr, 0, iterations_per_layer)  # 0 - do not use warmup
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()
        inputs_ema = EMA(alpha=args.ema_alpha, initial_value=inputs)
        _last_tea_z = None  # 마지막 iteration의 teacher 임베딩 저장 (bank 업데이트용)

        for iteration in range(iterations_per_layer):
            # learning rate scheduling
            lr_scheduler(optimizer, iteration, iteration)

            aug_function = ApplyTransformToPair(transforms.Compose([
                transforms.RandomResizedCrop(32, scale=(0.08, 1)),
                transforms.RandomHorizontalFlip(),
            ]))
            
            if args.bdpc_beta > 0:
                for id_ in range(len(loss_r_feature_layers)):
                    for mod in loss_r_feature_layers[id_]:
                        mod.bdpc_cur_iter = iteration
                        mod.bdpc_max_iter = max(iterations_per_layer - 1, 1)
            
            inputs_jit, inputs_ema_jit = aug_function(inputs, inputs_ema.value)
            # forward pass
            id = counter % len(model_teacher)
            for mod in loss_r_feature_layers[id]:
                mod.set_label(targets, ipc_ids=ipc_ids)
            counter += 1
            optimizer.zero_grad()
            for (idx, mod) in enumerate(loss_r_feature_layers[id]):
                mod.set_ori()
            
            clip(inputs.data)
            _has_proj = hasattr(model_teacher[id], 'projection_head')
            if args.supcon_weight > 0 and _has_proj:
                _tea_out = model_teacher[id](inputs_jit, train=True)
                sub_outputs = _tea_out[1]  # out_cb (eval mode와 동일)
                _tea_z = _tea_out[2]       # projection 임베딩 [N, 128]
            else:
                sub_outputs = model_teacher[id](inputs_jit)
                _tea_z = None
            # R_cross classification loss
            loss_ce = criterion(sub_outputs, targets)

            # R_feature loss
            rescale = [args.first_multiplier] + [1. for _ in range(len(loss_r_feature_layers[id]) - 1)]
            loss_r_feature = sum(
                [mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers[id])])
            
            if args.flatness:
                with torch.no_grad():
                    for (idx, mod) in enumerate(loss_r_feature_layers[id]):
                        mod.set_return()
                    ema_sub_outputs = model_teacher[id](inputs_ema_jit)
                    for (idx, mod) in enumerate(loss_r_feature_layers[id]):
                        mod.remove_return()
                    inputs_ema.ema_update(inputs)
                loss_ema_ce = F.kl_div(torch.log_softmax(sub_outputs / 4, dim=1),
                                        torch.softmax(ema_sub_outputs / 4, dim=1))
            else:
                loss_ema_ce = torch.Tensor([0.]).to(inputs_jit.device)

            # combining losses
            loss_aux = args.r_loss * loss_r_feature + loss_ema_ce * args.flatness_weight

            loss = loss_ce + loss_aux
            if args.supcon_weight > 0 and _tea_z is not None:
                _last_tea_z = _tea_z
                # Feature bank에서 현재 클래스를 제외한 다른 클래스 임베딩을 negative로 추가
                current_cls = targets.unique()
                bank_mask = feat_bank_filled.clone()
                for _c in current_cls:
                    bank_mask[_c] = False  # 현재 클래스는 bank에서 제외 (중복 방지)
                bank_cls_idx = bank_mask.nonzero(as_tuple=True)[0]
                bank_feats = feat_bank[bank_cls_idx]
                bank_labels = bank_cls_idx
                _sc_feats = torch.cat([_tea_z, bank_feats], dim=0)
                _sc_labels = torch.cat([targets, bank_labels], dim=0)

                loss_supcon = SupConLoss(_sc_feats, _sc_labels, temperature=args.supcon_temperature)
                loss = loss + args.supcon_weight * loss_supcon
            else:
                loss_supcon = torch.tensor(0., device=inputs_jit.device)

            if args.struct_weight > 0:
                loss_struct = sum(mod.loss_struct for mod in loss_r_feature_layers[id])
                loss = loss + args.struct_weight * loss_struct
            else:
                loss_struct = torch.tensor(0., device=inputs_jit.device)

            if args.gpu == 0:
                total_global_mean = sum(mod.loss_global_mean.item() for mod in loss_r_feature_layers[id])
                total_global_var  = sum(mod.loss_global_var.item()  for mod in loss_r_feature_layers[id])
                total_class_mean  = sum(mod.loss_class_mean.item()  for mod in loss_r_feature_layers[id])
                total_class_var   = sum(mod.loss_class_var.item()   for mod in loss_r_feature_layers[id])
                total_bdpc        = sum(mod.loss_bdpc.item()        for mod in loss_r_feature_layers[id])
                total_cluster_mean = sum(mod.loss_cluster_mean.item() for mod in loss_r_feature_layers[id])
                total_cluster_var  = sum(mod.loss_cluster_var.item()  for mod in loss_r_feature_layers[id])
                print(f"Iter [{iteration+1}/{iterations_per_layer}]  "
                      f"loss={loss.item():.4f}  loss_ce={loss_ce.item():.4f}  "
                      f"loss_r_feature={loss_r_feature.item():.4f}  loss_ema_ce={loss_ema_ce.item():.4f}  "
                      f"loss_supcon={loss_supcon.item():.4f}  loss_struct={loss_struct.item():.4f}  "
                      f"[r_feat breakdown]  global_mean={total_global_mean:.4f}  global_var={total_global_var:.4f}  "
                      f"class_mean={total_class_mean:.4f}  class_var={total_class_var:.4f}  bdpc={total_bdpc:.4f}  "
                      f"cluster_mean={total_cluster_mean:.4f}  cluster_var={total_cluster_var:.4f}")
                if wandb.run is not None:
                    wandb.log({
                        "recover/loss":           loss.item(),
                        "recover/loss_ce":        loss_ce.item(),
                        "recover/loss_r_feature": loss_r_feature.item(),
                        "recover/loss_ema_ce":    loss_ema_ce.item(),
                        "recover/loss_supcon":    loss_supcon.item(),
                        "recover/loss_struct":    loss_struct.item(),
                        "recover/global_mean":    total_global_mean,
                        "recover/global_var":     total_global_var,
                        "recover/class_mean":     total_class_mean,
                        "recover/class_var":      total_class_var,
                        "recover/bdpc":           total_bdpc,
                        "recover/cluster_mean":   total_cluster_mean,
                        "recover/cluster_var":    total_cluster_var,
                    })

            if iteration % save_every == 0 and args.gpu == 0:
                print("------------iteration {}----------".format(iteration))
                print("total loss", loss.item())
                print("loss_r_feature", loss_r_feature.item())
                print("loss_ema_ce", loss_ema_ce.item())
                print("id", id)
                print("main criterion",
                      criterion(sub_outputs, targets).item())
                # comment below line can speed up the training (no validation process)
                if hook_for_display is not None:
                    hook_for_display(inputs, targets)

            # do image update
            loss.backward()
            optimizer.step()

            # clip color outlayers
            # inputs.data = denormalize(inputs.data)
            # normalize = transforms.Normalize([0.5071, 0.4867, 0.4408],
            #                         [0.2675, 0.2565, 0.2761])
            # inputs.data = normalize((inputs.data * 255).int() / 255)
            inputs.data = clip(inputs.data)

            if gpu == 0 and (best_cost > loss.item() or iteration == 1):
                best_inputs = inputs.data.clone()

        # Feature bank EMA 업데이트: 이 클래스 배치의 최종 임베딩을 bank에 반영
        if feat_bank is not None and _last_tea_z is not None:
            with torch.no_grad():
                for _cls in targets.unique().tolist():
                    _cls_mask = (targets == _cls)
                    _cls_z = _last_tea_z[_cls_mask].detach().mean(0)
                    feat_bank[_cls] = 0.9 * feat_bank[_cls] + 0.1 * _cls_z

        if args.store_best_images:
            best_inputs = inputs.data.clone()  # using multicrop, save the last one
            best_inputs = denormalize(best_inputs)
            normalize = transforms.Normalize([0.5071, 0.4867, 0.4408],
                                    [0.2675, 0.2565, 0.2761])
            inputs = normalize((best_inputs * 255).int() / 255)
            print("Testing...")
            hook_for_display(inputs, targets)
            save_images(args, best_inputs, targets, ipc_ids)
        # to reduce memory consumption by states of the optimizer we deallocate memory
        optimizer.state = collections.defaultdict(dict)
        torch.cuda.empty_cache()
        
    if gpu == 0 and wandb.run is not None:
        run_id_path = os.path.join(args.syn_data_path, "wandb_run_id.txt")
        with open(run_id_path, "w") as f:
            f.write(wandb.run.id)
        wandb.finish()

    if args.distributed and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def save_images(args, images, targets, ipc_ids):
    ipc_id_range = ipc_ids
    for id in range(images.shape[0]):
        if targets.ndimension() == 1:
            class_id = targets[id].item()
        else:
            class_id = targets[id].argmax().item()

        if not os.path.exists(args.syn_data_path):
            os.mkdir(args.syn_data_path)

        # save into separate folders
        dir_path = '{}/new{:03d}'.format(args.syn_data_path, class_id)
        place_to_store = dir_path + '/class{:03d}_id{:03d}.jpg'.format(class_id, ipc_id_range[id])
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        pil_image.save(place_to_store)


def validate(input, target, model):
    def accuracy(output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    with torch.no_grad():
        output = model(input)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

    print("Verifier accuracy: ", prec1.item())


def prepare_biased_statistics(args):
    """Collect feature statistics from the biased expert model.

    Mirrors the statistic-collection loop in main_worker() but runs on a
    single GPU (cuda:0) before distributed training starts.  Saves:
      - Per-class BN/Conv statistics  → args.biased_statistic_path
      - BN layer running_mean/var     → .../BNFeatureHook/{name}/bn_running.npz
    """
    gpu = 0
    aux_teacher = [args.arch_name]

    # --- build biased model (same arch as debiased teacher) ---
    biased_model = build_cifar_model(args.arch_name, num_classes=10)

    # --- load biased checkpoint ---
    biased_model.load_state_dict(load_checkpoint_state_dict(args.biased_expert_path))
    biased_model = biased_model.cuda(gpu).eval()
    for p in biased_model.parameters():
        p.requires_grad = False

    # --- register hooks (same structure as main_worker) ---
    biased_hooks = []
    load_tag = True
    for name, module in biased_model.named_modules():
        full_name = str(biased_model.__class__.__name__) + "_" + str(aux_teacher[0]) + "=" + name
        if isinstance(module, nn.BatchNorm2d):
            _hook = BNFeatureHook(module, save_path=args.biased_statistic_path,
                                  name=full_name, gpu=gpu,
                                  training_momentum=args.training_momentum,
                                  flatness_weight=args.flatness_weight,
                                  category_aware=args.category_aware)
            _hook.set_hook(pre=True)
            load_tag = load_tag & _hook.load_tag
            biased_hooks.append(('bn', full_name, _hook, module))
        elif isinstance(module, nn.Conv2d):
            _hook = ConvFeatureHook(module, save_path=args.biased_statistic_path,
                                    name=full_name, gpu=gpu,
                                    training_momentum=args.training_momentum,
                                    drop_rate=args.drop_rate,
                                    flatness_weight=args.flatness_weight,
                                    category_aware=args.category_aware)
            _hook.set_hook(pre=True)
            load_tag = load_tag & _hook.load_tag
            biased_hooks.append(('conv', full_name, _hook, module))

    # --- also check if BN running stats are already saved ---
    bn_running_saved = True
    for hook_type, full_name, _hook, module in biased_hooks:
        if hook_type == 'bn':
            bn_path = os.path.join(args.biased_statistic_path, "BNFeatureHook", full_name, "bn_running.npz")
            if not os.path.exists(bn_path):
                bn_running_saved = False
                break

    if not load_tag:
        # --- collect statistics by running training data ---
        train_dataset = cifar10Imbanlance.Cifar10Imbanlance(
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
            ]),
            imbanlance_rate=args.imbanlance_rate, train=True, file_path=args.train_data_path)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, num_workers=0, batch_size=64, drop_last=False, shuffle=True,
            worker_init_fn=seed_worker, generator=make_generator(args.seed, 5000))

        print("[BDPC] Collecting biased expert statistics...")
        with torch.no_grad():
            total_seen = 0
            for i, (data, targets) in tqdm(enumerate(train_loader)):
                B = data.size(0)
                data = data.cuda(gpu)
                targets = targets.cuda(gpu)
                for _, _, _hook, _ in biased_hooks:
                    _hook.set_label(targets)
                    if hasattr(_hook, "momentum"):
                        _hook.momentum = B / float(total_seen + B + 1e-6)
                _ = biased_model(data)
                total_seen += B

            for _, _, _hook, _ in biased_hooks:
                _hook.save()
        print("[BDPC] Biased expert statistics saved")
    else:
        print("[BDPC] Biased expert statistics already exist, skipping collection")

    # --- save BN running_mean/var for form1 bias direction ---
    if not bn_running_saved:
        for hook_type, full_name, _hook, module in biased_hooks:
            if hook_type == 'bn':
                bn_dir = os.path.join(args.biased_statistic_path, "BNFeatureHook", full_name)
                os.makedirs(bn_dir, exist_ok=True)
                bn_path = os.path.join(bn_dir, "bn_running.npz")
                np.savez(bn_path,
                         running_mean=module.running_mean.data.cpu().numpy(),
                         running_var=module.running_var.data.cpu().numpy())
        print("[BDPC] BN running stats saved")

    # cleanup hooks
    for _, _, _hook, _ in biased_hooks:
        _hook.close()


def main_syn():
    parser = argparse.ArgumentParser(
        "G-VBSM: applying generalized matching for data condensation")
    """Data save flags"""
    parser.add_argument('--flatness', action='store_true', default=False,
                        help='encourage the flatness or not')
    parser.add_argument('--flatness-weight', type=float, default=1.,
                        help='the weight of flatness weight')
    parser.add_argument('--ema_alpha', type=float, default=0.99,
                        help='the weight of EMA learning rate')
    parser.add_argument('--exp-name', type=str, default='test',
                        help='name of the experiment, subfolder under syn_data_path')
    parser.add_argument('--ipc-number', type=int, default=10, help='the number of each ipc')
    parser.add_argument('--syn-data-path', type=str,
                        default='./syn_data_50', help='where to store synthetic data')
    parser.add_argument('--pre-train-path', type=str,
                        default='',
                        help='where to load the pre-trained backbone')
    parser.add_argument('--imbanlance_rate', default=0.1, type=float,
                        help='imbalance')
    parser.add_argument('--adaptive-alpha', action='store_true',
                        help='use class-adaptive alpha weighting: alpha_c=(n_c-n_min)/(n_max-n_min)')
    parser.add_argument('--store-best-images', action='store_true',
                        help='whether to store best images')
    """Optimization related flags"""
    parser.add_argument('--batch-size', type=int,
                        default=100, help='number of images to optimize at the same time')
    parser.add_argument('--gpu-id', type=str, default='0,1')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--initial-img-dir', type=str, default="./exp/../", 
                       help="imgs used for initialization")
    parser.add_argument('--iteration', type=int, default=1000,
                        help='num of iterations to optimize the synthetic data')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate for optimization')
    parser.add_argument('--training-momentum', type=float, default=0.8,
                        help="$\alpha$ in our paper, controls the form of score distillation sampling")
    parser.add_argument('--drop-rate', type=float, default=0.4,
                        help="$\beta_\textrm{dr}$ in our paper, controls the efficiency of GSM")
    parser.add_argument('--jitter', default=32, type=int, help='random shift on the synthetic data')
    parser.add_argument('--category-aware', default="global", type=str, help='category-aware matching (local or global)')
    parser.add_argument('--r-loss', type=float, default=0.05,
                        help='coefficient for BN and Conv feature distribution regularization')
    parser.add_argument('--first-multiplier', type=float, default=10.,
                        help='additional multiplier on first layer of L_bn or L_conv')
    parser.add_argument('--nuc-norm', type=float, default=0.00001,
                        help='coefficient for total variation Nuclear loss')
    """Model related flags"""
    parser.add_argument('--arch-name', type=str, default='resnet18',
                        help='arch name from pretrained torchvision models')
    parser.add_argument('--tau', type=float, default=4.0, help='the temperature of nuc norm')
    parser.add_argument('--verifier', action='store_true',
                        help='whether to evaluate synthetic data with another model')
    parser.add_argument('--verifier-arch', type=str, default='mobilenet_v2',
                        help="arch name from torchvision models to act as a verifier")
    parser.add_argument('--train-data-path', type=str, default='../expert/root',
                        help="the path of the CIFAR-10's training set")
    parser.add_argument('--statistic-path', type=str, default='./statistic',
                        help="the path of the statistic file")
    parser.add_argument('--biased-expert-path', type=str, default=None,
                        help='path to biased expert checkpoint')
    parser.add_argument('--biased-statistic-path', type=str, default='./biased_statistic',
                        help='directory to save/load biased observer statistics')
    parser.add_argument('--bdpc-beta', type=float, default=0.0,
                        help='base BDPC weight; 0 disables BDPC')
    parser.add_argument('--bdpc-schedule', action='store_true', default=False,
                        help='disable quadratic iter scaling of BDPC beta (use constant beta instead)')
    parser.add_argument('--wandb-project', type=str, default=None,
                        help='wandb project name; if set, enables wandb logging')
    parser.add_argument('--wandb-api-key', type=str, default=None,
                        help='wandb API key')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='wandb run name prefix (defaults to --exp-name)')
    parser.add_argument('--supcon-weight', type=float, default=0.0,
                        help='SupCon loss weight on teacher projection embeddings (0 = off)')
    parser.add_argument('--supcon-temperature', type=float, default=0.1,
                        help='temperature for SupCon loss in recovery stage')
    parser.add_argument('--struct-weight', type=float, default=0.0,
                        help='weight lambda for Inter-Class Structure Matching (ICSM) loss (0 = off)')
    parser.add_argument('--cluster-stat-path', type=str, default=None,
                        help='directory containing manifest.json + per-cluster activation stats from initial')
    parser.add_argument('--cluster-loss-weight', type=float, default=0.0,
                        help='weight for cluster mean/var matching loss (0 = off)')
    parser.add_argument('--seed', default=42, type=int,
                        help='seed for Python, NumPy, PyTorch, CUDA, and DataLoader workers')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    seed_everything(args.seed)

    if args.bdpc_beta > 0:
        print(f"BDPC enabled with beta={args.bdpc_beta}, preparing biased statistics...")
        prepare_biased_statistics(args)

    # If --syn-data-path is absolute, treat it as the explicit output dir
    # (experiments/{exp_name}/recovers/{recover_id}/syn_data) and skip the
    # legacy "<base>/<exp_name>" composition.
    if not os.path.isabs(args.syn_data_path):
        args.syn_data_path = os.path.join(args.syn_data_path, args.exp_name)
    if not os.path.exists(args.syn_data_path):
        os.makedirs(args.syn_data_path, exist_ok=True)

    # Dump args.json one level up from syn_data so the artifact dir captures it.
    _args_dir = os.path.dirname(os.path.abspath(args.syn_data_path))
    os.makedirs(_args_dir, exist_ok=True)
    with open(os.path.join(_args_dir, "args.json"), "w") as _f:
        json.dump({k: (v if isinstance(v, (int, float, str, bool, list, dict, type(None))) else str(v))
                   for k, v in vars(args).items()}, _f, indent=2)

    aux_teacher = [args.arch_name]
    args.aux_teacher = aux_teacher
    model_teacher = []
    for name in aux_teacher:
        model = build_cifar_model(name, num_classes=10)
        model.load_state_dict(load_checkpoint_state_dict(args.pre_train_path))
        model_teacher.append(model)

    model_verifier = model_teacher[0]
    ipc_id_range = list(range(0, args.ipc_number))
    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node < 1:
        raise RuntimeError("No CUDA device is visible. Check CUDA_VISIBLE_DEVICES and the NVIDIA driver state.")
    args.world_size = ngpus_per_node * args.world_size
    if ngpus_per_node == 1:
        print("Single visible GPU detected, skipping distributed spawn.")
        args.distributed = False
        main_worker(0, ngpus_per_node, args, model_teacher, model_verifier, ipc_id_range)
        return
    port_id = 10000 + np.random.randint(0, 1000)
    args.dist_url = 'tcp://127.0.0.1:' + str(port_id)
    args.distributed = True
    torch.multiprocessing.set_start_method('spawn')
    mp.spawn(main_worker, nprocs=ngpus_per_node,
             args=(ngpus_per_node, args, model_teacher, model_verifier, ipc_id_range))


if __name__ == '__main__':
    main_syn()
