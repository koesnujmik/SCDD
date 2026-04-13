# export PYTHONNOUSERSITE=1
set -eu

GPU_ID=0
IMBALANCE_RATE=0.1
ARCH_NAME="convnet"

IPC=10
SELECTION_METHOD="original"
PRETRAIN_PATH="../expert/root/model/dataset: cifar100#arch: convnet#imbanlance_rate: 0.1#2026-04-12 18:23:00/ckpt.best.pth.tar"
EXP_NAME="ORG3"

INITIAL_IMAGE_DIR="../initial/exp/${EXP_NAME}cifar100_conv3_${SELECTION_METHOD}_f1_mipc500_ipc${IPC}_cr1/syn_data"



# cd expert
# CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --dataset cifar100 -a $ARCH_NAME --num_classes 100 --imbanlance_rate $IMBALANCE_RATE --epochs 200 -b 64 --q 0.8 --gamma1 1
# CUDA_VISIBLE_DEVICES=$GPU_ID python main_vanilla.py --dataset cifar10 -a $ARCH_NAME --num_classes 10 --imbanlance_rate $IMBALANCE_RATE --epochs 200 -b 64


# cd ..
# cd initial
# CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
# --subset "cifar100" \
# --imbanlance-rate $IMBALANCE_RATE \
# --arch-name "conv3" \
# --factor 1 \
# --num-crop 1 \
# --mipc 500 \
# --ipc $IPC \
# --stud-name "conv3" \
# --re-epochs 300 \
# --selection-method $SELECTION_METHOD \
# --pre-train-path "${PRETRAIN_PATH}" \
# --exp-name $EXP_NAME


# cd ..
# cd recover_cifar100
# CUDA_VISIBLE_DEVICES=$GPU_ID python recover.py \
#     --arch-name $ARCH_NAME \
#     --exp-name $EXP_NAME \
#     --batch-size 100 --category-aware "global" \
#     --lr 0.05 --drop-rate 0.0 \
#     --ipc-number $IPC --training-momentum 0.8 \
#     --iteration 2000 \
#     --imbanlance-rate $IMBALANCE_RATE \
#     --r-loss 0.01 \
#     --verifier --store-best-images --gpu-id $GPU_ID \
#     --pre-train-path "${PRETRAIN_PATH}" \
#     --initial-img-dir "${INITIAL_IMAGE_DIR}" \
#     # --adaptive-alpha \


# cd ..
cd train_cifar100
CUDA_VISIBLE_DEVICES=$GPU_ID python direct_train.py \
    --wandb-project 'final_RN18_fkd' \
    --batch-size 100 --epochs 1000 \
    --model $ARCH_NAME \
    --ls-type cos2 --loss-type "mse_gt" --ce-weight 0.025 \
    -T 20 --sgd-lr 0.1 --adamw-lr 0.001 --gpu-id $GPU_ID \
    -j 4 --gradient-accumulation-steps 1  --st 2 --ema-dr 0.99 \
    --mix-type 'cutmix' --adamw-weight-decay 0.0005 \
    --output-dir "./save/${EXP_NAME}/" \
    --train-dir "../recover_cifar100/syn_data/${EXP_NAME}" \
    --pre-train-path "${PRETRAIN_PATH}"