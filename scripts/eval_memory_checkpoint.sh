#!/usr/bin/env bash
set -euo pipefail

# Evaluate an existing MME-VLA memory checkpoint in the foreground.
# Defaults to the interrupted recurrent TTT expert run at ckpt 30000.

MODEL_TYPE=${MODEL_TYPE:-recurrent-ttt-expert_2gpu_20260703_143622}
CKPT_ID=${CKPT_ID:-30000}
SEED=${SEED:-7}
MODE=${MODE:-full}
ONLY_TASKS=${ONLY_TASKS:-BinFill}
OVERWRITE=${OVERWRITE:-true}

GPU_ID_SERVER=${GPU_ID_SERVER:-0}
GPU_ID_CLIENT=${GPU_ID_CLIENT:-1}
PORT=${PORT:-auto}
SERVER_STARTUP_WAIT=${SERVER_STARTUP_WAIT:-60}
EVAL_SAVE_DIR=${EVAL_SAVE_DIR:-/n/netscratch/hankyang_lab/Lab/felix/ckpts/robomme_policy_ckpt/evaluation_memory}
POLICY_PYTHON=${POLICY_PYTHON:-/n/holylabs/LABS/hankyang_lab/Lab/felix/.conda/envs/robomme-openpi/bin/python}
EVAL_ENV=${EVAL_ENV:-robomme}

EXTRA_ARGS_LIST=()

if [[ "${MODE}" == "smoke" ]]; then
  EXTRA_ARGS_LIST+=(--args.only_tasks="${ONLY_TASKS}")
elif [[ "${MODE}" != "full" ]]; then
  echo "ERROR: MODE must be smoke or full, got: ${MODE}" >&2
  exit 1
fi

if [[ "${OVERWRITE}" == "true" ]]; then
  EXTRA_ARGS_LIST+=(--args.overwrite)
fi

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  USER_EXTRA_ARGS=(${EXTRA_ARGS})
  EXTRA_ARGS_LIST+=("${USER_EXTRA_ARGS[@]}")
fi

POLICY_DIR="runs/ckpts/mme_vla_suite/${MODEL_TYPE}/${CKPT_ID}"
if [[ ! -d "${POLICY_DIR}/params" ]]; then
  echo "ERROR: checkpoint params not found: ${POLICY_DIR}/params" >&2
  echo "Available checkpoints:" >&2
  find -L "runs/ckpts/mme_vla_suite/${MODEL_TYPE}" -maxdepth 2 -mindepth 1 -type d -printf "  %p\n" 2>/dev/null >&2 || true
  exit 1
fi

echo "[memory-eval] model: ${MODEL_TYPE}"
echo "[memory-eval] ckpt: ${CKPT_ID}"
echo "[memory-eval] mode: ${MODE}"
echo "[memory-eval] save dir: ${EVAL_SAVE_DIR}"
echo "[memory-eval] extra args: ${EXTRA_ARGS_LIST[*]}"

MODEL_TYPE="${MODEL_TYPE}" \
CKPT_ID="${CKPT_ID}" \
SEED="${SEED}" \
GPU_ID_SERVER="${GPU_ID_SERVER}" \
GPU_ID_CLIENT="${GPU_ID_CLIENT}" \
PORT="${PORT}" \
SERVER_STARTUP_WAIT="${SERVER_STARTUP_WAIT}" \
EVAL_SAVE_DIR="${EVAL_SAVE_DIR}" \
POLICY_PYTHON="${POLICY_PYTHON}" \
EVAL_ENV="${EVAL_ENV}" \
EXTRA_ARGS="${EXTRA_ARGS_LIST[*]}" \
bash scripts/eval_foreground.sh
