set -e

# It requires two arguments: Model type and experiment name.
if [[ $# -eq 3 ]] ; then
    data_path=$1
    pretrained=$2
    exp=$3
else
    echo 'Experiment name is missing!'
    exit 1
fi

# Create folder.
export OUTPUT_DIR=$exp

if [ ! -d "${OUTPUT_DIR}" ]; then
  echo "Folder $output does not exist. Create folder and copy files ..."
  mkdir -p ${OUTPUT_DIR}
  cp $0 ${OUTPUT_DIR}/run.bash


    OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_finetune.py \
      --batch_size 64 \
      --accum_iter 2 \
      --model vit_large_patch16 \
      --finetune $pretrained \
      --epochs 100 \
      --blr 5e-4 --layer_decay 0.65 \
      --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
      --checkpoint_key "teacher" \
      --dist_eval --data_path data_path \
      --log_dir ${OUTPUT_DIR} \
      --output_dir ${OUTPUT_DIR} | tee -a ${OUTPUT_DIR}/history.txt


else
    echo "Folder ${OUTPUT_DIR} already exists. Please remove the folder and re-run the script."
fi
