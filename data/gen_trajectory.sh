#!/bin/bash
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
MASTER_PORT=$((RANDOM % 101 + 20000))

set -x
umask 000

TIME=$(date +%Y%m%d_%H%M)
DATASET=RxR # R2R EnvDrop RxR
# Use different config for RxR vs R2R
if [ "$DATASET" = "RxR" ]; then
    CONFIG_PATH=../config/gen_data_rxr.yaml
else
    CONFIG_PATH=../config/gen_data_r2r.yaml
fi
OUTPUT_PATH=cache/train/${DATASET}
# DATA_PATH=task/envdrop/envdrop.json.gz 
# DATA_PATH=task/r2r/train/train.json.gz 
DATA_PATH=task/rxr/train/train_guide.json.gz

mkdir -p ${OUTPUT_PATH}
torchrun --nproc_per_node=8 --master_port=$MASTER_PORT ./gen_trajectory.py \
    --dataset ${DATASET} \
    --config_path ${CONFIG_PATH} \
    --output_path ${OUTPUT_PATH} \
    --data_path ${DATA_PATH} \
    --render_points
    > ${OUTPUT_PATH}/log.log 2>&1

# mkdir -p ${OUTPUT_PATH}
# torchrun --nproc_per_node=8 --master_port=$MASTER_PORT ./gen_trajectory.py \
#     --dataset ${DATASET} \
#     --config_path ${CONFIG_PATH} \
#     --output_path ${OUTPUT_PATH} \
#     --data_path ${DATA_PATH} \
#     > ${OUTPUT_PATH}/log.log 2>&1