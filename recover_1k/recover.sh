CUDA_VISIBLE_DEVICES=5 python recover.py \
    --arch-name "resnet18" \
    --exp-name "" \
    --batch-size 80 \
    --lr 0.05 --category-aware "global" \
    --ipc-number 10 --training-momentum 0.8  \
    --iteration 1000 --drop-rate 0.0 \
    --train-data-path \
    --l2-scale 0 --tv-l2 0 --r-loss 0.1 --nuc-norm 1. \
    --verifier --store-best-images --gpu-id 5 --initial-img-dir 