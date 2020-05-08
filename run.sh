CUDA_VISIBLE_DEVICES=0 python main.py \
    --load "weights/best_exp3_swsl_resnext50_32x4d_fold0.pth" \
    --config "expconfigs/exp3.yaml" \
    --test \

    