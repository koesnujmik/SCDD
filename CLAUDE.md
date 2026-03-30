# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RLDD (Rethinking Long-tailed Dataset Distillation) is an AAAI 2026 oral paper implementing a uni-level framework for dataset distillation on long-tailed (class-imbalanced) datasets. The goal is to distill a large imbalanced training set into a small set of synthetic images that can train a student model effectively.

## Four-Stage Pipeline

The full pipeline runs sequentially via `run.sh`:

```
Stage 1: Expert Model Training (expert/)
    ↓ saves checkpoint to expert/root/model/
Stage 2: Synthetic Data Initialization (initial/)
    ↓ saves synthetic images to initial/exp/{exp_name}/syn_data/
Stage 3: Unbiased Recovery (recover_cifar10/ or recover_cifar100/ etc.)
    ↓ saves optimized images to recover_*/syn_data_50/{exp_name}/
Stage 4: Unbiased Relabeling & Student Training (train_cifar10/ etc.)
    ↓ saves student model to train_*/save/
```

## Running Each Stage

### Stage 1: Expert Training
```bash
cd expert
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 -a convnet --num_classes 10 \
  --imbanlance_rate 0.01 --epochs 200 -b 64 --q 0.8 --gamma1 1
```

### Stage 2: Initialization
```bash
cd initial
sh scripts/cifar10_10ipc_conv3_to_conv3_cr5.sh  # or other dataset scripts
```

### Stage 3: Unbiased Recovery
Adjust paths in `recover_{dataset}/recover.sh` (especially `--pre-train-path` and `--initial-img-dir`), then:
```bash
cd recover_cifar10   # or recover_cifar100, recover_tiny, recover_1k
sh recover.sh
```
For Tiny-ImageNet, also change the imbalance factor in `recover_tiny/tiny_in_dataset.py`.

### Stage 4: Student Training
Adjust paths in `train_{dataset}/train.sh` (especially `--train-dir` and `--pre-train-path`), then:
```bash
cd train_cifar10   # or train_cifar100, train_tiny, train_1k
sh train.sh
```
For Tiny-ImageNet, also change imbalance factor in `train_tiny/tiny_in_dataset.py`.

### Full pipeline (CIFAR-10)
```bash
sh run.sh
```
Note: `run.sh` has a hardcoded expert checkpoint path — update it after Stage 1.

## Architecture

### Dataset-Specific Module Structure
Each dataset has parallel `recover_*` and `train_*` directories with nearly identical code. The core algorithms live in:
- `recover_*/recover.py` — feature-matching optimization loop
- `recover_*/utils.py` — `BNFeatureHook` and `ConvFeatureHook` that capture layer statistics
- `train_*/direct_train.py` — student training with soft-label relabeling

### Key Design Patterns
- **Feature distribution matching**: Recovery optimizes synthetic images so their BatchNorm/Conv statistics match those of real data. Hook classes in `utils.py` register forward hooks on teacher layers.
- **Category-aware matching** (`--category-aware global|local`): `"global"` matches against all-class statistics; `"local"` uses per-class real samples.
- **Soft relabeling**: Student is trained with MSE loss on teacher logits (`--loss-type mse_gt`) rather than hard labels, avoiding bias from the long-tailed label distribution.
- **EMA model**: An exponential moving average of the student is maintained for validation.
- **Multi-teacher**: Recovery can ensemble multiple teacher checkpoints for robustness.

### Expert Model (`expert/`)
- `Trainer.py` implements robust loss training using `--q` (sharpness) and `--gamma1` (loss weight) for long-tail handling
- Imbalanced dataset loaders in `expert/imbalance_data/` create step or exponential imbalance schedules
- Supported architectures: ConvNet (depth 3/4/5), ResNet (18/32/34/50), ResNeXt-50

### Initialization (`initial/`)
- `baseline.py` selects representative crops from training data and blends them using `--num-crop` and `--factor`
- `initial/argument.py` is the canonical config file for this stage
- `--ipc` (images per class) and `--mipc` (multi-IPC buffer size) are the key capacity parameters

## Datasets and Paths
Datasets are expected under `expert/root/`:
- CIFAR-10/100: downloaded automatically by torchvision
- Tiny-ImageNet: `expert/root/tiny-imagenet-200/`
- ImageNet-LT: `expert/root/imagenet/`

Imbalance rate is controlled globally by `--imbanlance_rate` (e.g., `0.01` = 100:1 head-to-tail ratio).

## Experiment Tracking
WandB is used in the student training stage. `train.sh` sets `wandb offline` by default. Project name is set via `--wandb-project`.
