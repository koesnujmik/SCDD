SELECTION_METHOD=""
RECOVER="harf_no_opt"
IMBALANCE_RATE=0.01
ARCH_NAME="convnet"
EXP_NAME="${SELECTION_METHOD}${RECOVER}"

# cd expert
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 -a $ARCH_NAME --num_classes 10 --imbanlance_rate $IMBALANCE_RATE --epochs 200 -b 64 --q 0.8 --gamma1 1

# cd ..
# cd initial
# python main.py \
# --subset "cifar10" \
# --arch-name "conv3" \
# --factor 1 \
# --num-crop 1 \
# --mipc 5000 \
# --ipc 10 \
# --stud-name "conv3" \
# --re-epochs 300 \
# --selection-method $SELECTION_METHOD \


# cd ..
# cd recover_cifar10
# CUDA_VISIBLE_DEVICES=0 python recover.py \
#     --arch-name $ARCH_NAME \
#     --exp-name $EXP_NAME \
#     --batch-size 100 --category-aware "global" \
#     --lr 0.05 --drop-rate 0.0 \
#     --ipc-number 10 --training-momentum 0.8 \
#     --iteration 75 \
#     --imbanlance_rate $IMBALANCE_RATE \
#     --r-loss 0.01 \
#     --verifier --store-best-images --gpu-id 0 \
#     --pre-train-path "../expert/root/model/dataset: cifar10#arch: convnet#imbanlance_rate: 0.01#2026-03-27 21:25:57/ckpt.best.pth.tar" \
#     --initial-img-dir "../initial/exp/cifar10_conv3_${SELECTION_METHOD}_f1_mipc5000_ipc10_cr1/syn_data"

# cd ..
cd train_cifar10
CUDA_VISIBLE_DEVICES=0 python direct_train.py \
    --wandb-project 'final_RN18_fkd' \
    --batch-size 25 --epochs 1000 \
    --model $ARCH_NAME \
    --ls-type multisteplr --loss-type "mse_gt" --ce-weight 0.025 \
    -T 20 --sgd-lr 0.1 --adamw-lr 0.001 --gpu-id 0 \
    -j 4 --gradient-accumulation-steps 1  --st 2 --ema-dr 0.99 \
    --mix-type 'cutmix' --adamw-weight-decay 0.01 \
    --output-dir "./save/${EXP_NAME}/" \
    --train-dir "../recover_cifar10/syn_data_50/${EXP_NAME}" \
    --pre-train-path "../expert/root/model/dataset: cifar10#arch: convnet#imbanlance_rate: 0.01#2026-03-27 21:25:57/ckpt.best.pth.tar" \
