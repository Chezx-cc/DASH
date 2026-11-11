#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=30G

export CUDA_VISIBLE_DEVICES=0

MODELS=("DLinear" "FreTS" "iTransformer" "MICN" "OLS" "PatchTST")
DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "exchange_rate" "weather")
PRED_LENS=(96 192 336 720)

BASE_LR=0.001
WEIGHT_DECAY=0.0001
GATING_INIT=0.01

for MODEL in "${MODELS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for PRED_LEN in "${PRED_LENS[@]}"; do
      CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${PRED_LEN}/"
      RESULT_DIR="./results/TAFAS/"
      OUTPUT="./results/summary/TAFAS/${MODEL}/${DATASET}/${PRED_LEN}.txt"

      echo "실행: MODEL=${MODEL}, DATASET=${DATASET}, PRED_LEN=${PRED_LEN}"

      python main.py \
        DATA.NAME ${DATASET} \
        DATA.PRED_LEN ${PRED_LEN} \
        MODEL.NAME ${MODEL} \
        MODEL.pred_len ${PRED_LEN} \
        TRAIN.ENABLE False \
        TRAIN.CHECKPOINT_DIR ${CHECKPOINT_DIR} \
        TTA.ENABLE True \
        TTA.SOLVER.BASE_LR ${BASE_LR} \
        TTA.SOLVER.WEIGHT_DECAY ${WEIGHT_DECAY} \
        TTA.TAFAS.GATING_INIT ${GATING_INIT} \
        RESULT_DIR ${RESULT_DIR} > ${OUTPUT}
    done
  done
done