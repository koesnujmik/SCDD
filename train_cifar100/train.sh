# wandb disabled
wandb enabled
wandb offline

CUDA_VISIBLE_DEVICES=6 python direct_train.py \
    --wandb-project 'final_RN18_fkd' \
    --batch-size 100 --epochs 1000 \
    --model "convnet" \
    --ls-type cos2 --loss-type "mse_gt" --ce-weight 0.025 \
    -T 20 --sgd --sgd-lr 0.1 --adamw-lr 0.001 --gpu-id 6 \
    -j 4 --gradient-accumulation-steps 1  --st 2 --ema-dr 0.99 \
    --mix-type 'cutmix' --weight-decay 0.0005 \
    --output-dir ./save/final_RN18_fkd/ \
    --train-dir  \
    --val-dir ''