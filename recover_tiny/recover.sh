CUDA_VISIBLE_DEVICES=1 python recover.py \
    --arch-name "resnet18" \
    --exp-name "test" \
    --batch-size 100 --category-aware "global" \
    --lr 0.05 --drop-rate 0.0 \
    --ipc-number 10 --training-momentum 0.8 \
    --iteration 2000 \
    --train-data-path  \
    --r-loss 0.01 --initial-img-dir  \
    --verifier --store-best-images --gpu-id 1