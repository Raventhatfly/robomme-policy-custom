#!/usr/bin/env bash
#SBATCH --job-name=robomme-mme-vla-2gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=160G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/n/home04/wfy/repos/robomme_policy_learning}"
mkdir -p logs/slurm

DATA_ROOT=${DATA_ROOT:-/n/netscratch/hankyang_lab/Lab/felix/dataset/robomme}
DATASET_PATH=${DATASET_PATH:-}
MME_VLA_TYPE=${MME_VLA_TYPE:-perceptual-framesamp-modul}
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
EXP_NAME=${EXP_NAME:-${MME_VLA_TYPE}_2gpu_${RUN_TIMESTAMP}}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-4}
NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-80000}
SAVE_INTERVAL=${SAVE_INTERVAL:-10000}
WANDB_ENABLED=${WANDB_ENABLED:-true}
RESUME=${RESUME:-false}
OVERWRITE=${OVERWRITE:-false}
POLICY_ENV=${POLICY_ENV:-robomme-openpi}
POLICY_PYTHON=${POLICY_PYTHON:-}
DEFAULT_POLICY_PYTHON=/n/holylabs/LABS/hankyang_lab/Lab/felix/.conda/envs/robomme-openpi/bin/python

export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-/n/home04/wfy/.cache/openpi}

if [[ -z "${POLICY_PYTHON}" ]]; then
  if [[ -x "${DEFAULT_POLICY_PYTHON}" ]]; then
    POLICY_PYTHON="${DEFAULT_POLICY_PYTHON}"
  elif [[ -n "${POLICY_ENV}" ]] && command -v conda >/dev/null 2>&1; then
    POLICY_PYTHON="conda run -n ${POLICY_ENV} python"
  else
    POLICY_PYTHON="python"
  fi
fi

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

echo "[slurm] job id: ${SLURM_JOB_ID:-none}"
echo "[slurm] node list: ${SLURM_JOB_NODELIST:-none}"
echo "[slurm] dataset: ${DATASET_PATH}"
echo "[slurm] model: ${MME_VLA_TYPE}"
echo "[slurm] exp: ${EXP_NAME}"
echo "[slurm] batch size: ${BATCH_SIZE}"
echo "[slurm] num workers: ${NUM_WORKERS}"
echo "[slurm] robomme shuffle: ${ROBOMME_SHUFFLE:-true}"
echo "[slurm] resume: ${RESUME}"
echo "[slurm] overwrite: ${OVERWRITE}"
echo "[slurm] python: ${POLICY_PYTHON}"

export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}

TRAIN_ARGS=(
  scripts/train.py mme_vla_suite
  --exp-name="${EXP_NAME}"
  --batch-size="${BATCH_SIZE}"
  --num-workers="${NUM_WORKERS}"
  --num-train-steps="${NUM_TRAIN_STEPS}"
  --save-interval="${SAVE_INTERVAL}"
  --fsdp-devices=2
  --dataset-path="${DATASET_PATH}"
  --model.use_history
  --model.history_config="${MME_VLA_TYPE}.yaml"
)

if [[ "${WANDB_ENABLED}" == "true" ]]; then
  TRAIN_ARGS+=(--wandb-enabled)
else
  TRAIN_ARGS+=(--no-wandb-enabled)
fi

if [[ "${RESUME}" == "true" ]]; then
  TRAIN_ARGS+=(--resume)
fi

if [[ "${OVERWRITE}" == "true" ]]; then
  TRAIN_ARGS+=(--overwrite)
fi

${POLICY_PYTHON} "${TRAIN_ARGS[@]}"
