import os

for yaml in ["exp1"]:
    for fold in range(0,5):
        os.system(f"CUDA_VISIBLE_DEVICES=0 python main.py \
                    --config expconfigs/{yaml}.yaml \
                    --fold {fold}")