#!/usr/bin/env bash
set -euo pipefail

# Build the preprocessed RoboMME format used by scripts/train.py:
#   data/*.pkl
#   features/episode_*/token_emb_*.npy
#   meta/stats.json

DATA_ROOT=${DATA_ROOT:-/n/netscratch/hankyang_lab/Lab/felix/dataset/robomme}
RAW_DATA_PATH=${RAW_DATA_PATH:-${DATA_ROOT}/robomme_data_h5}
PREPROCESSED_DATA_PATH=${PREPROCESSED_DATA_PATH:-${DATA_ROOT}/robomme_preprocessed_data}
MAX_EPISODES=${MAX_EPISODES:-}
COMPUTE_NORM_STATS=${COMPUTE_NORM_STATS:-true}
POLICY_PYTHON=${POLICY_PYTHON:-python}

if [[ ! -d "${RAW_DATA_PATH}" ]]; then
  echo "ERROR: raw data path does not exist: ${RAW_DATA_PATH}" >&2
  exit 1
fi

if ! find "${RAW_DATA_PATH}" -maxdepth 1 -name "*.h5" -print -quit | grep -q .; then
  echo "ERROR: no .h5 files found under ${RAW_DATA_PATH}" >&2
  echo "If you only have .tar.xz files, decompress them first with scripts/tarxz_h5.py." >&2
  exit 1
fi

BUILD_ARGS=(
  scripts/build_dataset.py
  --dataset_type robomme_pkl
  --raw_data_path "${RAW_DATA_PATH}"
  --preprocessed_data_path "${PREPROCESSED_DATA_PATH}"
)

if [[ -n "${MAX_EPISODES}" ]]; then
  BUILD_ARGS+=(--max_episodes "${MAX_EPISODES}")
fi

echo "[data] raw: ${RAW_DATA_PATH}"
echo "[data] output: ${PREPROCESSED_DATA_PATH}"
echo "[data] python: ${POLICY_PYTHON}"

"${POLICY_PYTHON}" "${BUILD_ARGS[@]}"

echo "[data] built dataset:"
echo "[data]   ${PREPROCESSED_DATA_PATH}/meta/stats.json"
echo "[data]   ${PREPROCESSED_DATA_PATH}/data/*.pkl"
echo "[data]   ${PREPROCESSED_DATA_PATH}/features/episode_*"

if [[ "${COMPUTE_NORM_STATS}" == "true" ]]; then
  echo "[data] computing norm stats for mme_vla_suite"
  "${POLICY_PYTHON}" scripts/compute_norm_stats.py \
    --config-name mme_vla_suite \
    --repo-id robomme \
    --dataset-path "${PREPROCESSED_DATA_PATH}"

  echo "[data] computing norm stats for pi05_baseline"
  "${POLICY_PYTHON}" scripts/compute_norm_stats.py \
    --config-name pi05_baseline \
    --repo-id robomme \
    --dataset-path "${PREPROCESSED_DATA_PATH}"
fi
