#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

config=configs/pascal.yaml
model_config=configs/segformer_B4.py
cafs_pretrained_path=cafs_pretrained/CAFS_1_4_resnet101_78.32_2.pth


python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    eval.py \
    --config=$config --cafs-pretrained-path $cafs_pretrained_path \
    --model_config=$model_config
