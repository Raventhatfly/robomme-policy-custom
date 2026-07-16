#!/usr/bin/env bash
#SBATCH --job-name=eval-ttt-80k
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

export MODEL_TYPE=${MODEL_TYPE:-recurrent-ttt-expert_2gpu_20260705_104256}
export CKPT_ID=${CKPT_ID:-79999}
export CONFIG_TYPE=${CONFIG_TYPE:-mme_vla_suite}
export EVAL_SAVE_DIR=${EVAL_SAVE_DIR:-/n/netscratch/hankyang_lab/Lab/felix/ckpts/robomme_policy_ckpt/evaluation_memory}
export MODE=${MODE:-full}
export OVERWRITE=${OVERWRITE:-true}

bash slurm_scripts/eval_memory_checkpoint.sbatch
