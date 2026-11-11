#!/bin/bash

# DynaTTA: Dynamic Test-Time Adaptation Script
# Dynamic adaptation with metrics-driven learning rate adjustment
# Features: MSE buffer, RTAB/RDB embeddings, calibration modules

export CUDA_VISIBLE_DEVICES=0

MODELS=("DLinear" "FreTS" "iTransformer" "MICN" "OLS" "PatchTST")
DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "exchange_rate" "weather")
PRED_LENS=(96 192 336 720)

# DynaTTA Configuration
MSE_BUFFER_SIZE=256
METRIC_HISTORY_SIZE=256
ALPHA_MIN=1e-4
ALPHA_MAX=1e-3
KAPPA=1.0
ETA=0.1
WARMUP_FACTOR=1
UPDATE_BUFFERS_INTERVAL=1
UPDATE_METRICS_INTERVAL=1
RTAB_SIZE=360
RDB_SIZE=100

# TAFAS Base Configuration (required for DynaTTA)
BATCH_SIZE=64
STEPS=1
GATING_INIT=0.01
HIDDEN_DIM=128
BASE_LR=0.001
WEIGHT_DECAY=0.0001

for MODEL in "${MODELS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for PRED_LEN in "${PRED_LENS[@]}"; do
      CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${PRED_LEN}/"
      RESULT_DIR="./results/DYNATTA/"
      OUTPUT="./results/summary/DYNATTA/${MODEL}/${DATASET}/${PRED_LEN}.txt"

      echo "DynaTTA: MODEL=${MODEL}, DATASET=${DATASET}, PRED_LEN=${PRED_LEN}"
      echo "   Buffer: MSE=${MSE_BUFFER_SIZE}, RTAB=${RTAB_SIZE}, RDB=${RDB_SIZE}"
      echo "   Adaptation: α_min=${ALPHA_MIN}, α_max=${ALPHA_MAX}, κ=${KAPPA}"
      echo "   Updates: Buffer=${UPDATE_BUFFERS_INTERVAL}, Metrics=${UPDATE_METRICS_INTERVAL}"

      python main.py \
        DATA.NAME ${DATASET} \
        DATA.PRED_LEN ${PRED_LEN} \
        MODEL.NAME ${MODEL} \
        MODEL.pred_len ${PRED_LEN} \
        TRAIN.ENABLE False \
        TRAIN.CHECKPOINT_DIR ${CHECKPOINT_DIR} \
        TTA.ENABLE True \
        TTA.MODULE_NAMES_TO_ADAPT all \
        TTA.SOLVER.BASE_LR ${BASE_LR} \
        TTA.SOLVER.WEIGHT_DECAY ${WEIGHT_DECAY} \
        TTA.TAFAS.BATCH_SIZE ${BATCH_SIZE} \
        TTA.TAFAS.STEPS ${STEPS} \
        TTA.TAFAS.GATING_INIT ${GATING_INIT} \
        TTA.TAFAS.HIDDEN_DIM ${HIDDEN_DIM} \
        TTA.DYNATTA.MSE_BUFFER_SIZE ${MSE_BUFFER_SIZE} \
        TTA.DYNATTA.METRIC_HISTORY_SIZE ${METRIC_HISTORY_SIZE} \
        TTA.DYNATTA.ALPHA_MIN ${ALPHA_MIN} \
        TTA.DYNATTA.ALPHA_MAX ${ALPHA_MAX} \
        TTA.DYNATTA.KAPPA ${KAPPA} \
        TTA.DYNATTA.ETA ${ETA} \
        TTA.DYNATTA.WARMUP_FACTOR ${WARMUP_FACTOR} \
        TTA.DYNATTA.UPDATE_BUFFERS_INTERVAL ${UPDATE_BUFFERS_INTERVAL} \
        TTA.DYNATTA.UPDATE_METRICS_INTERVAL ${UPDATE_METRICS_INTERVAL} \
        TTA.DYNATTA.RTAB_SIZE ${RTAB_SIZE} \
        TTA.DYNATTA.RDB_SIZE ${RDB_SIZE} \
        RESULT_DIR ${RESULT_DIR} > ${OUTPUT}
        
      if [ $? -eq 0 ]; then
        echo "Finished: ${MODEL}/${DATASET}/${PRED_LEN}"
      else
        echo "Failed: ${MODEL}/${DATASET}/${PRED_LEN}"
      fi
    done
  done
done