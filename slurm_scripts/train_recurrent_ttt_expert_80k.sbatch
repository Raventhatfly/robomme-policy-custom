#!/usr/bin/env bash
#SBATCH --job-name=robomme-ttt-80k
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

# Default behavior: start a fresh recurrent TTT expert 80k run.
export MME_VLA_TYPE=${MME_VLA_TYPE:-recurrent-ttt-expert}
export RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
export EXP_NAME=${EXP_NAME:-${MME_VLA_TYPE}_2gpu_80k_${RUN_TIMESTAMP}}
export BATCH_SIZE=${BATCH_SIZE:-16}
export NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-80000}
export SAVE_INTERVAL=${SAVE_INTERVAL:-10000}
export RESUME=${RESUME:-false}
export WANDB_ENABLED=${WANDB_ENABLED:-true}

bash slurm_scripts/train_2gpu_mme_vla.sbatch
