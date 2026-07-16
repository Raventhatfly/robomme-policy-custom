#!/usr/bin/env bash
#SBATCH --job-name=robomme-mme-moviechat-sidecar
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/n/home04/wfy/repos/robomme_policy_learning}"
mkdir -p logs/slurm

CONDA_ENV=${CONDA_ENV:-/n/holylabs/LABS/hankyang_lab/Lab/felix/.conda/envs/robomme-openpi}
PYTHON=${PYTHON:-${CONDA_ENV}/bin/python}
ROBOMME_DATA_ROOT=${ROBOMME_DATA_ROOT:-/n/netscratch/hankyang_lab/Lab/felix/dataset/robomme/robomme_data_h5}
ROBOMME_PREPROCESSED_ROOT=${ROBOMME_PREPROCESSED_ROOT:-/n/netscratch/hankyang_lab/Lab/felix/dataset/robomme/robomme_preprocessed_data}
ROBOMME_MEMORY_NAME=${ROBOMME_MEMORY_NAME:-siglip_moviechat_mme_v1}
SOURCE=${SOURCE:-auto}
ENCODER_CHECKPOINT=${ENCODER_CHECKPOINT:-/n/netscratch/hankyang_lab/Lab/felix/ckpts/fluxvla/pi0_base/model.safetensors}
DTYPE=${DTYPE:-bf16}
MAX_MEMORY_TOKENS=${MAX_MEMORY_TOKENS:-128}
FRAME_MEMORY_TOKENS=${FRAME_MEMORY_TOKENS:-32}
SHORT_MEMORY_SIZE=${SHORT_MEMORY_SIZE:-18}
SHORT_MEMORY_MERGE=${SHORT_MEMORY_MERGE:-2}
HIGH_RELEVANCE_KEEP_FRAMES=${HIGH_RELEVANCE_KEEP_FRAMES:-3}
LOW_RELEVANCE_KEEP_FRAMES=${LOW_RELEVANCE_KEEP_FRAMES:-1}
RELEVANCE_THRESHOLD=${RELEVANCE_THRESHOLD:-0.25}
MAX_FILES=${MAX_FILES:-}
MAX_EPISODES_PER_FILE=${MAX_EPISODES_PER_FILE:-}
MAX_EPISODES=${MAX_EPISODES:-}
VERIFY_ONLY=${VERIFY_ONLY:-false}
MAX_VERIFY_FILES=${MAX_VERIFY_FILES:-}
OVERWRITE=${OVERWRITE:-false}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache_${USER}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_${USER}}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"

SIDE_ROOT="${ROBOMME_DATA_ROOT}/memory/${ROBOMME_MEMORY_NAME}"
OLD_SIDE_ROOT="${ROBOMME_DATA_ROOT}/memory/siglip_moviechat_v1"

echo "[mme-sidecar] repo:        ${PWD}"
echo "[mme-sidecar] data root:   ${ROBOMME_DATA_ROOT}"
echo "[mme-sidecar] pkl root:    ${ROBOMME_PREPROCESSED_ROOT}"
echo "[mme-sidecar] output root: ${SIDE_ROOT}"
echo "[mme-sidecar] old root:    ${OLD_SIDE_ROOT} (not touched)"
echo "[mme-sidecar] checkpoint:  ${ENCODER_CHECKPOINT}"
echo "[mme-sidecar] python:      ${PYTHON}"

"${PYTHON}" -c "import sys; print('[mme-sidecar] executable:', sys.executable); import h5py, torch, transformers, safetensors; print('[mme-sidecar] import check: ok')"

CMD=(
  "${PYTHON}" tools/generate_robomme_hdf5_memory_mme.py
  --raw-root "${ROBOMME_DATA_ROOT}"
  --preprocessed-root "${ROBOMME_PREPROCESSED_ROOT}"
  --source "${SOURCE}"
  --memory-name "${ROBOMME_MEMORY_NAME}"
  --encoder-checkpoint "${ENCODER_CHECKPOINT}"
  --dtype "${DTYPE}"
  --max-memory-tokens "${MAX_MEMORY_TOKENS}"
  --frame-memory-tokens "${FRAME_MEMORY_TOKENS}"
  --short-memory-size "${SHORT_MEMORY_SIZE}"
  --short-memory-merge "${SHORT_MEMORY_MERGE}"
  --high-relevance-keep-frames "${HIGH_RELEVANCE_KEEP_FRAMES}"
  --low-relevance-keep-frames "${LOW_RELEVANCE_KEEP_FRAMES}"
  --relevance-threshold "${RELEVANCE_THRESHOLD}"
)

if [[ "${VERIFY_ONLY}" == "true" ]]; then
  CMD+=(--verify-only)
fi
if [[ "${OVERWRITE}" == "true" ]]; then
  CMD+=(--overwrite)
fi
if [[ -n "${MAX_VERIFY_FILES}" ]]; then
  CMD+=(--max-verify-files "${MAX_VERIFY_FILES}")
fi
if [[ -n "${MAX_FILES}" ]]; then
  CMD+=(--max-files "${MAX_FILES}")
fi
if [[ -n "${MAX_EPISODES_PER_FILE}" ]]; then
  CMD+=(--max-episodes-per-file "${MAX_EPISODES_PER_FILE}")
fi
if [[ -n "${MAX_EPISODES}" ]]; then
  CMD+=(--max-episodes "${MAX_EPISODES}")
fi

"${CMD[@]}"
