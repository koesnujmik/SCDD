# wandb disabled
wandb enabled
wandb offline

CUDA_VISIBLE_DEVICES=7 python direct_train.py \
    --wandb-project 'final_conv_fkd' \
    --batch-size 100 --epochs 1000 \
    --model "convnet" \
    --ls-type cos --loss-type "mse_gt" --ce-weight 0.025 \
    -T 20 --sgd --sgd-lr 0.1 --adamw-lr 0.0005 --gpu-id 7 \
    -j 4 --gradient-accumulation-steps 1  --st 2 --ema-dr 0.99 \
    --mix-type 'cutmix' --weight-decay 0.0005 \
    --output-dir ./save/final_conv_fkd/ \
    --train-dir  \
    --val-dir 