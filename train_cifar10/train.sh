# wandb disabled
wandb enabled
wandb offline

CUDA_VISIBLE_DEVICES=0 python direct_train.py \
    --wandb-project 'final_RN18_fkd' \
    --batch-size 25 --epochs 1000 \
    --model "convnet" \
    --ls-type multisteplr --loss-type "mse_gt" --ce-weight 0.025 \
    -T 20 --sgd-lr 0.1 --adamw-lr 0.001 --gpu-id 0 \
    -j 4 --gradient-accumulation-steps 1  --st 2 --ema-dr 0.99 \
    --mix-type 'cutmix' --adamw-weight-decay 0.01 \
    --output-dir ./save/final_RN18_fkd/ \
    --train-dir "/home/user/junseok/RLDD/recover_cifar10/syn_data_50/test" \
    --val-dir "/home/user/junseok/RLDD/expert/root" \
    --pre-train-path "/home/user/junseok/RLDD/expert/root_model/dataset: cifar10#arch: convnet#imbanlance_rate: 0.01#2026-01-22 13:57:49/ckpt.best.pth.tar" \
