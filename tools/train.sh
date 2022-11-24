#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

config=configs/pascal.yaml
model_config=configs/segformer_B4.py
labeled_id_path=partitions/pascal/1_4/labeled.txt
unlabeled_id_path=partitions/pascal/1_4/unlabeled.txt
save_path=exp/pascal/1_4/1_4_CAFS_V3

mkdir -p $save_path

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    main.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path --model_config=$model_config \
    --save-path $save_path --port $2 2>&1 | tee $save_path/$now.txt
