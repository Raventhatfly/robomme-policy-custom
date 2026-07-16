#!/usr/bin/env bash
#SBATCH --job-name=eval-moviechat-context
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/n/home04/wfy/repos/robomme_policy_learning}"
mkdir -p logs/slurm

export MODEL_TYPE=${MODEL_TYPE:-moviechat-memory-context_2gpu_80k_REPLACE_ME}
export CKPT_ID=${CKPT_ID:-79999}
export CONFIG_TYPE=${CONFIG_TYPE:-mme_vla_suite}
export EVAL_SAVE_DIR=${EVAL_SAVE_DIR:-/n/netscratch/hankyang_lab/Lab/felix/ckpts/robomme_policy_ckpt/evaluation_moviechat_memory_context}
export MODE=${MODE:-full}
export OVERWRITE=${OVERWRITE:-true}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.70}

if [[ "${MODEL_TYPE}" == *REPLACE_ME* ]]; then
  echo "ERROR: set MODEL_TYPE to the trained moviechat-memory-context checkpoint directory." >&2
  echo "Example:" >&2
  echo "  MODEL_TYPE=moviechat-memory-context_2gpu_80k_YYYYMMDD_HHMMSS sbatch slurm_scripts/eval_moviechat_memory_context_80k.sbatch" >&2
  exit 1
fi

bash slurm_scripts/eval_memory_checkpoint.sbatch
