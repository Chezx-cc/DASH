#!/bin/bash

MODELS=("DLinear" "FreTS" "iTransformer" "MICN" "OLS" "PatchTST")
DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "exchange_rate" "weather")
PRED_LENS=(96 192 336 720)

# SIMPLE Adapter Basic Configuration
BUFFER_CONTEXT_SIZE=10
STEPS=3
BATCH_SIZE=48

# PAAS  Configuration
PAAS=True
PERIOD_N=1

# Fast Adaptation Optimization Settings
FAST_ADAPTATION=True
ADAPTIVE_LR=True
PER_BATCH_LR_RESET=True
MAX_LR=0.005
MIN_LR=0.0001
 

for MODEL in "${MODELS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for PRED_LEN in "${PRED_LENS[@]}"; do
      CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${PRED_LEN}/"
      RESULT_DIR="./results/SIMPLE/"
      
      OUTPUT="./results/summary/SIMPLE/${MODEL}/${DATASET}/${PRED_LEN}.txt"

      python main.py \
        DATA.NAME ${DATASET} \
        DATA.PRED_LEN ${PRED_LEN} \
        MODEL.NAME ${MODEL} \
        MODEL.pred_len ${PRED_LEN} \
        TRAIN.ENABLE False \
        TRAIN.CHECKPOINT_DIR ${CHECKPOINT_DIR} \
        TTA.ENABLE True \
        TTA.SOLVER.BASE_LR 0.001 \
        TTA.SOLVER.WEIGHT_DECAY 0.0001 \
        TTA.SIMPLE.BATCH_SIZE ${BATCH_SIZE} \
        TTA.SIMPLE.STEPS ${STEPS} \
        TTA.SIMPLE.BUFFER_CONTEXT_SIZE ${BUFFER_CONTEXT_SIZE} \
        TTA.SIMPLE.FAST_ADAPTATION ${FAST_ADAPTATION} \
        TTA.SIMPLE.PER_BATCH_LR_RESET ${PER_BATCH_LR_RESET} \
        TTA.SIMPLE.ADAPTIVE_LR ${ADAPTIVE_LR} \
        TTA.SIMPLE.PAAS ${PAAS} \
        TTA.SIMPLE.PERIOD_N ${PERIOD_N} \
        RESULT_DIR ${RESULT_DIR} > ${OUTPUT}
    done
  done
done