import torch.nn.functional as F
import torch.nn as nn
from argument import args as sys_args
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torchvision.models as thmodels
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url
from synthesize.models import ConvNet


# use 0 to pad "other three picture"
def pad(input_tensor, target_height, target_width=None):
    if target_width is None:
        target_width = target_height
    vertical_padding = target_height - input_tensor.size(2)
    horizontal_padding = target_width - input_tensor.size(3)

    top_padding = vertical_padding // 2
    bottom_padding = vertical_padding - top_padding
    left_padding = horizontal_padding // 2
    right_padding = horizontal_padding - left_padding

    padded_tensor = F.pad(
        input_tensor, (left_padding, right_padding, top_padding, bottom_padding)
    )

    return padded_tensor


def batched_forward(model, tensor, batch_size):
    total_samples = tensor.size(0)

    all_outputs = []

    model.eval()

    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            batch_data = tensor[i : min(i + batch_size, total_samples)]

            output = model(batch_data)

            all_outputs.append(output)

    final_output = torch.cat(all_outputs, dim=0)

    return final_output


class MultiRandomCrop(torch.nn.Module):
    def __init__(self, num_crop=5, size=224, factor=2):
        super().__init__()
        self.num_crop = num_crop
        self.size = size
        self.factor = factor

    def forward(self, image):
        cropper = transforms.RandomResizedCrop(
            self.size // self.factor,
            ratio=(1, 1),
            antialias=True,
        )
        patches = []
        for _ in range(self.num_crop):
            patches.append(cropper(image))
        return torch.stack(patches, 0)

    def __repr__(self) -> str:
        detail = f"(num_crop={self.num_crop}, size={self.size})"
        return f"{self.__class__.__name__}{detail}"


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

denormalize = transforms.Compose(
    [
        transforms.Normalize(
            mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        ),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
    ]
)


def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)


WeightsEnum.get_state_dict = get_state_dict


def cross_entropy(y_pre, y):
    y_pre = F.softmax(y_pre, dim=1)
    return (-torch.log(y_pre.gather(1, y.view(-1, 1))))[:, 0]


def _score_original(preds, labels, m, keep_limit):
    """Low CE loss = high confidence on correct class. Sort ascending."""
    scores = cross_entropy(preds, labels).reshape(m, keep_limit)
    best_per_img, _ = scores.min(dim=0)
    selected_img_ids = torch.argsort(best_per_img)  # ascending
    return scores, selected_img_ids


def _score_entropy_low(preds, m, keep_limit):
    """Low entropy = high certainty (any class). Sort ascending."""
    probs = F.softmax(preds, dim=1)
    entropy = (-probs * torch.log(probs + 1e-8)).sum(dim=1).reshape(m, keep_limit)
    best_per_img, _ = entropy.min(dim=0)
    selected_img_ids = torch.argsort(best_per_img)  # ascending
    return entropy, selected_img_ids


def _score_entropy_high(preds, m, keep_limit):
    """High entropy = most uncertain (any class). Sort descending."""
    probs = F.softmax(preds, dim=1)
    entropy = (-probs * torch.log(probs + 1e-8)).sum(dim=1).reshape(m, keep_limit)
    best_per_img, _ = entropy.max(dim=0)
    selected_img_ids = torch.argsort(best_per_img, descending=True)  # descending
    return -entropy, selected_img_ids  # negate so selection loop picks argmin consistently


def _extract_features(model, flat_images, keep_limit, m):
    """Extract penultimate features via forward hook on fc_cb. Returns [keep_limit, D]."""
    unwrapped = model.module if isinstance(model, nn.DataParallel) else model
    assert hasattr(unwrapped, 'fc_cb'), "Model has no fc_cb attribute"

    features_list = []
    def hook_fn(module, input, output):
        features_list.append(input[0].detach().cpu())

    handle = unwrapped.fc_cb.register_forward_hook(hook_fn)
    with torch.no_grad():
        batched_forward(model, flat_images, keep_limit)
    handle.remove()

    feats = torch.cat(features_list, dim=0)          # [m*keep_limit, D]
    feats = feats.reshape(m, keep_limit, -1).mean(dim=0)  # [keep_limit, D], avg over crops
    return feats


def _score_center(model, flat_images, m, keep_limit):
    """Distance to class mean in feature space. Sort ascending (closest to center first)."""
    feats = _extract_features(model, flat_images, keep_limit, m)  # [keep_limit, D]
    center = feats.mean(dim=0, keepdim=True)                       # [1, D]
    dists = torch.norm(feats - center, dim=1)                      # [keep_limit]
    selected_img_ids = torch.argsort(dists)                        # ascending
    scores = dists.unsqueeze(0).expand(m, -1).clone()
    return scores, selected_img_ids


def _score_kmeans(model, flat_images, images, m, keep_limit, n):
    """KMeans on penultimate features. Sort by distance to assigned centroid ascending."""
    from sklearn.cluster import KMeans

    feats = _extract_features(model, flat_images, keep_limit, m)  # [keep_limit, D]
    feats_np = feats.float().numpy()

    n_clusters = min(n, keep_limit)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(feats_np)
    centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)  # [n_clusters, D]
    assigned = torch.tensor(kmeans.labels_, dtype=torch.long)             # [keep_limit]

    # Distance of each sample to its assigned centroid
    dists = torch.norm(feats.cpu() - centers[assigned], dim=1)  # [keep_limit]

    selected_img_ids = torch.argsort(dists)  # ascending: closer to centroid first

    # Scores shape [m, keep_limit]: use tiled distances so the selection loop works uniformly
    scores = dists.unsqueeze(0).expand(m, -1).clone()
    return scores, selected_img_ids


def _score_shrinkage_kmeans(model, flat_images, m, keep_limit, n, mipc, imbanlance_rate):
    """
    KMeans centroids shrunk toward global mean based on class size.
      n_max = mipc (head class sample count)
      n_min = mipc * imbanlance_rate (tail class sample count)
      a = (keep_limit - n_min) / (n_max - n_min)
      target = a * centroid + (1-a) * global_mean
    Head classes (keep_limit≈n_max) → a≈1 → pure KMeans.
    Tail classes (keep_limit≈n_min) → a≈0 → pulled toward global mean.
    Each sample is scored by its distance to the nearest target point.
    """
    from sklearn.cluster import KMeans

    feats = _extract_features(model, flat_images, keep_limit, m)  # [keep_limit, D]
    feats_np = feats.float().numpy()

    n_clusters = min(n, keep_limit)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(feats_np)
    centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)  # [n_clusters, D]

    global_mean = feats.mean(dim=0, keepdim=True)  # [1, D]
    n_min = mipc * imbanlance_rate
    a = (keep_limit - n_min) / (mipc - n_min)
    targets = a * centroids + (1 - a) * global_mean  # [n_clusters, D]

    # For each sample, distance to nearest target point
    # feats: [keep_limit, D], targets: [n_clusters, D]
    dists_to_targets = torch.cdist(feats, targets)           # [keep_limit, n_clusters]
    min_dists, _ = dists_to_targets.min(dim=1)               # [keep_limit]

    selected_img_ids = torch.argsort(min_dists)              # ascending
    scores = min_dists.unsqueeze(0).expand(m, -1).clone()    # [m, keep_limit]
    return scores, selected_img_ids


def selector(n, model, images, labels, size, m=3, cls_id=0, method='original', imbanlance_rate=0.01):
    """
    Multi-round selection over keep_limit real images.
    Each (real image, augmentation) pair is selected at most once.

    Args:
        images:  Tensor [mipc, m, 3, H, W]
        labels:  Tensor [mipc]
        method:  'original' | 'entropy_low' | 'kmeans'
    Returns:
        selected_images: [n, 3, H, W]
    """
    with torch.no_grad():
        mipc = images.shape[0]
        device = images.device
        s = images.shape  # [mipc, m, 3, H, W]

        keep_limit = int(5000 * (0.01 ** (cls_id / 9)))
        keep_limit = min(mipc, keep_limit)
        if keep_limit == 0:
            return torch.empty((0, 3, s[3], s[4]), device=device)

        images = images.cuda().permute(1, 0, 2, 3, 4)  # [m, mipc, 3, H, W]
        images = images[:, :keep_limit]                # [m, keep_limit, 3, H, W]
        labels = labels[:keep_limit].repeat(m).cuda()  # [m * keep_limit]

        flat_images = images.reshape(m * keep_limit, s[2], s[3], s[4])

        # ── Phase 1: scoring (method-specific) ────────────────────────────
        if method == 'original':
            preds = batched_forward(model, pad(flat_images, size), keep_limit)
            scores, selected_img_ids = _score_original(preds, labels, m, keep_limit)

        elif method == 'entropy_low':
            preds = batched_forward(model, pad(flat_images, size), keep_limit)
            scores, selected_img_ids = _score_entropy_low(preds, m, keep_limit)

        elif method == 'entropy_high':
            preds = batched_forward(model, pad(flat_images, size), keep_limit)
            scores, selected_img_ids = _score_entropy_high(preds, m, keep_limit)

        elif method == 'center':
            scores, selected_img_ids = _score_center(
                model, pad(flat_images, size), m, keep_limit)

        elif method == 'kmeans':
            scores, selected_img_ids = _score_kmeans(
                model, pad(flat_images, size), images, m, keep_limit, n)

        elif method == 'shrinkage_kmeans':
            scores, selected_img_ids = _score_shrinkage_kmeans(
                model, pad(flat_images, size), m, keep_limit, n, mipc, imbanlance_rate)

        else:
            raise ValueError(f"Unknown selection method: {method}")

        # ── Phase 2: selection loop (shared) ──────────────────────────────
        valid_mask = torch.zeros_like(scores, dtype=torch.bool)
        valid_mask[:, selected_img_ids] = True
        used_mask = torch.zeros_like(scores, dtype=torch.bool)

        selected = []
        selected_pairs = set()

        while len(selected) < n:
            for img_id in selected_img_ids:
                remain_mask = (~used_mask[:, img_id]) & valid_mask[:, img_id]
                if remain_mask.sum() == 0:
                    continue

                remain_aug_ids = remain_mask.nonzero(as_tuple=False).squeeze(1)
                remain_scores = scores[remain_aug_ids, img_id]
                best_aug_idx_in_remain = torch.argmin(remain_scores).item()
                best_aug_idx = remain_aug_ids[best_aug_idx_in_remain].item()

                if (img_id.item(), best_aug_idx) in selected_pairs:
                    continue
                selected_pairs.add((img_id.item(), best_aug_idx))

                used_mask[best_aug_idx, img_id] = True
                selected.append(images[best_aug_idx, img_id])

                if len(selected) == n:
                    break
            
            if len(selected) < n:
                used_mask = torch.zeros_like(scores, dtype=torch.bool)
                selected_pairs = set()

        selected_images = torch.stack(selected, dim=0).detach()

    torch.cuda.empty_cache()
    return selected_images  # [n, 3, H, W]


def mix_images(input_img, out_size, factor, n):
    s = out_size // factor
    remained = out_size % factor
    k = 0
    mixed_images = torch.zeros(
        (n, 3, out_size, out_size),
        requires_grad=False,
        dtype=torch.float,
    )
    h_loc = 0
    for i in range(factor):
        h_r = s + 1 if i < remained else s
        w_loc = 0
        for j in range(factor):
            w_r = s + 1 if j < remained else s
            img_part = F.interpolate(
                input_img.data[k * n : (k + 1) * n], size=(h_r, w_r)
            )
            mixed_images.data[
                0:n,
                :,
                h_loc : h_loc + h_r,
                w_loc : w_loc + w_r,
            ] = img_part
            w_loc += w_r
            k += 1
        h_loc += h_r
    return mixed_images
