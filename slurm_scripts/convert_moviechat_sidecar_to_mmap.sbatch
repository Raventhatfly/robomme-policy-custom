#!/usr/bin/env bash
#SBATCH --job-name=moviechat-mmap
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/n/home04/wfy/repos/robomme_policy_learning}"
mkdir -p logs/slurm

POLICY_PYTHON=${POLICY_PYTHON:-/n/holylabs/LABS/hankyang_lab/Lab/felix/.conda/envs/robomme-openpi/bin/python}
SRC_ROOT=${SRC_ROOT:-/n/netscratch/hankyang_lab/Lab/felix/dataset/robomme/robomme_data_h5/memory/siglip_moviechat_v1}
DST_ROOT=${DST_ROOT:-/n/netscratch/hankyang_lab/Lab/felix/dataset/robomme/robomme_data_h5/memory/siglip_moviechat_v1_mmap}
MAX_FILES=${MAX_FILES:-}
OVERWRITE=${OVERWRITE:-false}

ARGS=(
  tools/convert_moviechat_npz_to_mmap.py
  --src-root "${SRC_ROOT}"
  --dst-root "${DST_ROOT}"
)

if [[ -n "${MAX_FILES}" ]]; then
  ARGS+=(--max-files "${MAX_FILES}")
fi

if [[ "${OVERWRITE}" == "true" ]]; then
  ARGS+=(--overwrite)
fi

echo "[slurm] src: ${SRC_ROOT}"
echo "[slurm] dst: ${DST_ROOT}"
echo "[slurm] max files: ${MAX_FILES:-all}"
echo "[slurm] overwrite: ${OVERWRITE}"
echo "[slurm] python: ${POLICY_PYTHON}"

"${POLICY_PYTHON}" "${ARGS[@]}"
