# wandb disabled
wandb enabled
wandb offline
python direct_train_1k.py \
    --wandb-project 'final_rn18_fkd' \
    --batch-size 100 \
    --model "resnet" \
    --ls-type cos2 --loss-type "mse_gt" --ce-weight 0.025 \
    -j 4 --gradient-accumulation-steps 1  --st 2 --ema-dr 0.99 \
    -T 20 --gpu-id 7\
    --adamw-lr 0.001 \
    --epoch 1000 \
    --mix-type 'cutmix' \
    --output-dir ./save/final_rn18_fkd/ \
    --train-dir  \
    --val-dir 
 