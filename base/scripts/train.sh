#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=30G

export CUDA_VISIBLE_DEVICES=0

MODELS=("DLinear" "FreTS" "iTransformer" "MICN" "OLS" "PatchTST")
DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "exchange_rate" "weather")
PRED_LENS=(96 192 336 720)

for MODEL in "${MODELS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for PRED_LEN in "${PRED_LENS[@]}"; do
      CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${PRED_LEN}/"

      echo "학습: MODEL=${MODEL}, DATASET=${DATASET}, PRED_LEN=${PRED_LEN}"

      python main.py \
        DATA.NAME ${DATASET} \
        DATA.PRED_LEN ${PRED_LEN} \
        MODEL.NAME ${MODEL} \
        MODEL.pred_len ${PRED_LEN} \
        TRAIN.ENABLE True \
        TRAIN.CHECKPOINT_DIR ${CHECKPOINT_DIR}
    done
  done
done