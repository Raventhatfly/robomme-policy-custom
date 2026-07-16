#!/usr/bin/env bash
#SBATCH --job-name=robomme-moviechat-sidecar
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

FLUXVLA_REPO=${FLUXVLA_REPO:-/n/home04/wfy/repos/FluxVLA}
CONDA_ENV=${CONDA_ENV:-/n/holylabs/LABS/hankyang_lab/Lab/felix/.conda/envs/fluxvla}
ROBOMME_DATA_ROOT=${ROBOMME_DATA_ROOT:-/n/netscratch/hankyang_lab/Lab/felix/dataset/robomme/robomme_data_h5}
ROBOMME_MEMORY_NAME=${ROBOMME_MEMORY_NAME:-siglip_moviechat_v1}
CONFIG=${CONFIG:-${FLUXVLA_REPO}/configs/pi05/pi05_paligemma_robomme_finetune.py}
DEVICE=${DEVICE:-cuda}
DTYPE=${DTYPE:-bf16}
MAX_MEMORY_TOKENS=${MAX_MEMORY_TOKENS:-128}
FRAME_MEMORY_TOKENS=${FRAME_MEMORY_TOKENS:-32}
SHORT_MEMORY_SIZE=${SHORT_MEMORY_SIZE:-18}
HIGH_RELEVANCE_KEEP_FRAMES=${HIGH_RELEVANCE_KEEP_FRAMES:-3}
LOW_RELEVANCE_KEEP_FRAMES=${LOW_RELEVANCE_KEEP_FRAMES:-1}
RELEVANCE_THRESHOLD=${RELEVANCE_THRESHOLD:-0.25}
MAX_FILES=${MAX_FILES:-}
MAX_EPISODES_PER_FILE=${MAX_EPISODES_PER_FILE:-}
VERIFY_ONLY=${VERIFY_ONLY:-false}
MAX_VERIFY_FILES=${MAX_VERIFY_FILES:-}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export PYTHONPATH="${PWD}/tools:${FLUXVLA_REPO}:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache_${USER}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_${USER}}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"

test -d "${ROBOMME_DATA_ROOT}" || {
  echo "Missing RoboMME data root: ${ROBOMME_DATA_ROOT}"
  exit 2
}

SIDE_ROOT="${ROBOMME_DATA_ROOT}/memory/${ROBOMME_MEMORY_NAME}"
BEFORE_COUNT=$(find "${SIDE_ROOT}" -name '*.npz' 2>/dev/null | wc -l || true)
H5_COUNT=$(find "${ROBOMME_DATA_ROOT}" -name '*.h5' 2>/dev/null | wc -l || true)

echo "[sidecar] repo:             ${PWD}"
echo "[sidecar] fluxvla repo:      ${FLUXVLA_REPO}"
echo "[sidecar] data root:         ${ROBOMME_DATA_ROOT}"
echo "[sidecar] memory root:       ${SIDE_ROOT}"
echo "[sidecar] config:            ${CONFIG}"
echo "[sidecar] existing sidecars: ${BEFORE_COUNT}"
echo "[sidecar] h5 files found:    ${H5_COUNT}"

CMD=(
  python tools/generate_robomme_hdf5_memory.py
  --raw-root "${ROBOMME_DATA_ROOT}"
  --memory-name "${ROBOMME_MEMORY_NAME}"
  --config "${CONFIG}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --max-memory-tokens "${MAX_MEMORY_TOKENS}"
  --frame-memory-tokens "${FRAME_MEMORY_TOKENS}"
  --short-memory-size "${SHORT_MEMORY_SIZE}"
  --high-relevance-keep-frames "${HIGH_RELEVANCE_KEEP_FRAMES}"
  --low-relevance-keep-frames "${LOW_RELEVANCE_KEEP_FRAMES}"
  --relevance-threshold "${RELEVANCE_THRESHOLD}"
)

if [[ "${VERIFY_ONLY}" == "true" ]]; then
  CMD+=(--verify-only)
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

"${CMD[@]}"
