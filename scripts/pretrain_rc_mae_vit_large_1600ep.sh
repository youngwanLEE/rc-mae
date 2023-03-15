#!/bin/sh
set -e

# It requires two arguments: Model type and experiment name.
if [[ $# -eq 2 ]] ; then
    data_path=$1
    exp=$2
else
    echo 'Experiment name is missing!'
    exit 1
fi

# Create folder.
export OUTPUT_DIR=output/$exp

if [ ! -d "${OUTPUT_DIR}" ]; then
  echo "Folder $output does not exist. Create folder and copy files ..."
  mkdir -p ${OUTPUT_DIR}
  cp $0 ${OUTPUT_DIR}/run.bash

  python -m torch.distributed.launch --nproc_per_node=8 --use_env run_pretrain.py \
    --model rc_vit_large_patch16 \
    --batch_size 256 \
    --accum_iter 2 \
    --epochs 1600 \
    --warmup_epochs 40 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --gamma 1.0 \
    --blr 1.5e-4 \
    --weight_decay 0.05 \
    --data_path $data_path \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} | tee -a ${OUTPUT_DIR}/history.txt

else
    echo "Folder ${OUTPUT_DIR} already exists. Please remove the folder and re-run the script."
fi


OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_finetune.py \
    --batch_size 64 \
    --accum_iter 2 \
    --model vit_large_patch16 \
    --finetune ${OUTPUT_DIR}/checkpoint-1599.pth \
    --epochs 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --checkpoint_key teacher \
    --dist_eval --data_path $data_path \
    --log_dir ${OUTPUT_DIR}/finetune_teacher \
    --output_dir ${OUTPUT_DIR}/finetune_teacher | tee -a ${OUTPUT_DIR}/finetune_teacher/history.txt