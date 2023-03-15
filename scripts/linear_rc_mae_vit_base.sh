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


    OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_linprobe.py \
        --batch_size 2048 \
        --model vit_base_patch16 --cls_token \
        --finetune $pretrained \
        --checkpoint_key "teacher" \
        --epochs 90 \
        --blr 0.1 \
        --weight_decay 0.0 \
        --dist_eval \
        --data_path $data_path \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} | tee -a ${OUTPUT_DIR}/history.txt


else
    echo "Folder ${OUTPUT_DIR} already exists. Please remove the folder and re-run the script."
fi
