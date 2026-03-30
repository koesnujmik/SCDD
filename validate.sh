cd train_cifar10
# CUDA_VISIBLE_DEVICES=0 python validate.py \
#     --batch-size 25 \
#     --model "convnet" \
#     --val-dir "../expert/root" \
#     --weight-path "save/entropy_high/model_best.pth.tar"

CUDA_VISIBLE_DEVICES=0 python validate.py \
    --batch-size 25 \
    --model "convnet" \
    --val-dir "../expert/root" \
    --weight-path "../expert/root/model/dataset: cifar10#arch: convnet#imbanlance_rate: 0.01#2026-03-31 02:00:06/ckpt.best.pth.tar"