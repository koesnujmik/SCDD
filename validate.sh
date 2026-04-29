cd train_cifar10


CUDA_VISIBLE_DEVICES=0 python validate.py \
    --batch-size 25 \
    --model "convnet" \
    --val-dir "../expert/root" \
    --weight-path "save/base1/model_best.pth.tar"

# cd train_cifar100

# CUDA_VISIBLE_DEVICES=0 python validate.py \
#     --batch-size 25 \
#     --model "convnet" \
#     --val-dir "../expert/root" \
#     --weight-path "../expert/root/model/dataset: cifar100#arch: convnet#imbanlance_rate: 0.1#2026-04-13 17:00:12/ckpt.best.pth.tar"