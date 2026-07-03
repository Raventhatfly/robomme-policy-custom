#!/usr/bin/env bash
set -euo pipefail

# Single-GPU smoke training for checking environment, data path, and checkpoint writing.
# This is intentionally short and disables wandb by default.

DATA_ROOT=${DATA_ROOT:-/n/netscratch/hankyang_lab/Lab/felix/dataset/robomme}
DATASET_PATH=${DATASET_PATH:-}
CONFIG_NAME=${CONFIG_NAME:-pi05_baseline}
EXP_NAME=${EXP_NAME:-single_gpu_smoke}
GPU_ID=${GPU_ID:-0}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-0}
NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-2}
SAVE_INTERVAL=${SAVE_INTERVAL:-1}
WANDB_ENABLED=${WANDB_ENABLED:-false}
POLICY_PYTHON=${POLICY_PYTHON:-python}

resolve_dataset_path() {
  local candidates=()

  if [[ -n "${DATASET_PATH}" ]]; then
    candidates+=("${DATASET_PATH}")
  fi

  candidates+=(
    "${DATA_ROOT}"
    "${DATA_ROOT}/robomme_preprocessed_data"
    "${DATA_ROOT}/robomme_preprocessed_data_sample"
    "${DATA_ROOT}/preprocessed"
  )

  for path in "${candidates[@]}"; do
    if [[ -f "${path}/meta/stats.json" && -d "${path}/data" ]]; then
      echo "${path}"
      return 0
    fi
  done

  echo "ERROR: could not find a preprocessed RoboMME dataset." >&2
  echo "Checked DATA_ROOT=${DATA_ROOT}" >&2
  echo "Expected a directory containing meta/stats.json and data/*.pkl." >&2
  echo "Raw h5 data like ${DATA_ROOT}/robomme_data_h5 is not enough for scripts/train.py." >&2
  return 1
}

DATASET_PATH=$(resolve_dataset_path)

echo "[smoke] config: ${CONFIG_NAME}"
echo "[smoke] exp: ${EXP_NAME}"
echo "[smoke] dataset: ${DATASET_PATH}"
echo "[smoke] gpu: ${GPU_ID}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
  "${POLICY_PYTHON}" scripts/train.py "${CONFIG_NAME}" \
    --exp-name="${EXP_NAME}" \
    --batch-size="${BATCH_SIZE}" \
    --num-workers="${NUM_WORKERS}" \
    --num-train-steps="${NUM_TRAIN_STEPS}" \
    --save-interval="${SAVE_INTERVAL}" \
    --fsdp-devices=1 \
    --dataset-path="${DATASET_PATH}" \
    --wandb-enabled="${WANDB_ENABLED}" \
    --overwrite
