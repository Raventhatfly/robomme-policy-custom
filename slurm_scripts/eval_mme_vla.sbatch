#!/usr/bin/env bash
#SBATCH --job-name=robomme-mme-vla-eval
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

MODEL_TYPE=${MODEL_TYPE:-pi05_baseline_2gpu_80k_20260705_212855}
REQUESTED_MODEL_TYPE=${MODEL_TYPE}
CKPT_ID=${CKPT_ID:-79999}
SEED=${SEED:-7}
PORT=${PORT:-8000}
CONFIG_TYPE=${CONFIG_TYPE:-auto}
GPU_ID_SERVER=${GPU_ID_SERVER:-0}
GPU_ID_CLIENT=${GPU_ID_CLIENT:-1}
SERVER_STARTUP_WAIT=${SERVER_STARTUP_WAIT:-60}
EVAL_SAVE_DIR=${EVAL_SAVE_DIR:-/n/netscratch/hankyang_lab/Lab/felix/ckpts/robomme_policy_ckpt/evaluation}
POLICY_PYTHON=${POLICY_PYTHON:-/n/holylabs/LABS/hankyang_lab/Lab/felix/.conda/envs/robomme-openpi/bin/python}
EVAL_PYTHON=${EVAL_PYTHON:-/n/holylabs/LABS/hankyang_lab/Lab/felix/.conda/envs/robomme/bin/python}
MODE=${MODE:-full}
ONLY_TASKS=${ONLY_TASKS:-BinFill}
OVERWRITE=${OVERWRITE:-true}
MODEL_EXTRA_EVAL_ARGS_LIST=()

if [[ "${MODEL_TYPE}" == pi05_baseline* ]]; then
  CONFIG_TYPE="pi05_baseline"
  MODEL_EXTRA_EVAL_ARGS_LIST+=(--args.no-use-history)
else
  if [[ "${CONFIG_TYPE}" == "auto" ]]; then
    CONFIG_TYPE="mme_vla_suite"
  fi
  case "${MODEL_TYPE}" in
    symbolic_simpleSG_oracle)
      MODEL_EXTRA_EVAL_ARGS_LIST+=(--args.use-oracle --args.subgoal-type=simple_subgoal)
      MODEL_TYPE="symbolic-simple-subgoal"
      ;;
    symbolic_groundedSG_oracle)
      MODEL_EXTRA_EVAL_ARGS_LIST+=(--args.use-oracle --args.subgoal-type=grounded_subgoal)
      MODEL_TYPE="symbolic-grounded-subgoal"
      ;;
    symbolic_simpleSG_qwenvl)
      MODEL_EXTRA_EVAL_ARGS_LIST+=(--args.use-qwenvl --args.subgoal-type=simple_subgoal)
      MODEL_TYPE="symbolic-simple-subgoal"
      ;;
    symbolic_groundedSG_qwenvl)
      MODEL_EXTRA_EVAL_ARGS_LIST+=(--args.use-qwenvl --args.subgoal-type=grounded_subgoal)
      MODEL_TYPE="symbolic-grounded-subgoal"
      ;;
    symbolic_simpleSG_gemini)
      MODEL_EXTRA_EVAL_ARGS_LIST+=(--args.use-gemini --args.subgoal-type=simple_subgoal)
      MODEL_TYPE="symbolic-simple-subgoal"
      ;;
    symbolic_groundedSG_gemini)
      MODEL_EXTRA_EVAL_ARGS_LIST+=(--args.use-gemini --args.subgoal-type=grounded_subgoal)
      MODEL_TYPE="symbolic-grounded-subgoal"
      ;;
    MemER)
      MODEL_EXTRA_EVAL_ARGS_LIST+=(--args.use-memer --args.subgoal-type=grounded_subgoal)
      MODEL_TYPE="symbolic-grounded-subgoal"
      ;;
  esac
fi

if [[ "${MODE}" == "smoke" ]]; then
  MODEL_EXTRA_EVAL_ARGS_LIST+=(--args.only_tasks="${ONLY_TASKS}")
elif [[ "${MODE}" != "full" ]]; then
  echo "ERROR: MODE must be smoke or full, got: ${MODE}" >&2
  exit 1
fi

if [[ "${OVERWRITE}" == "true" ]]; then
  MODEL_EXTRA_EVAL_ARGS_LIST+=(--args.overwrite)
fi

if [[ -n "${EXTRA_EVAL_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  USER_EXTRA_EVAL_ARGS=(${EXTRA_EVAL_ARGS})
  MODEL_EXTRA_EVAL_ARGS_LIST+=("${USER_EXTRA_EVAL_ARGS[@]}")
fi

POLICY_DIR=${POLICY_DIR:-runs/ckpts/${CONFIG_TYPE}/${MODEL_TYPE}/${CKPT_ID}}
SERVER_LOG=${SERVER_LOG:-logs/slurm/${SLURM_JOB_NAME:-robomme-mme-vla-eval}-${SLURM_JOB_ID:-manual}-server.log}
RESULT_DIR=${EVAL_SAVE_DIR}/${MODEL_TYPE}/ckpt${CKPT_ID}

if [[ ! -d "${POLICY_DIR}/params" ]]; then
  echo "ERROR: checkpoint params not found: ${POLICY_DIR}/params" >&2
  echo "Available checkpoints:" >&2
  find -L "runs/ckpts/${CONFIG_TYPE}/${MODEL_TYPE}" -maxdepth 2 -mindepth 1 -type d -printf "  %p\n" 2>/dev/null >&2 || true
  exit 1
fi

mkdir -p "${EVAL_SAVE_DIR}"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[eval-slurm] stopping policy server pid ${SERVER_PID}"
    kill -- -"${SERVER_PID}" 2>/dev/null || kill "${SERVER_PID}" 2>/dev/null || true
    sleep 2
    kill -9 -- -"${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[eval-slurm] node: ${SLURM_JOB_NODELIST:-unknown}"
echo "[eval-slurm] requested model: ${REQUESTED_MODEL_TYPE}"
echo "[eval-slurm] model: ${MODEL_TYPE}"
echo "[eval-slurm] config: ${CONFIG_TYPE}"
echo "[eval-slurm] ckpt: ${CKPT_ID}"
echo "[eval-slurm] mode: ${MODE}"
echo "[eval-slurm] policy dir: ${POLICY_DIR}"
echo "[eval-slurm] eval save dir: ${EVAL_SAVE_DIR}"
echo "[eval-slurm] result dir: ${RESULT_DIR}"
echo "[eval-slurm] model eval args: ${MODEL_EXTRA_EVAL_ARGS_LIST[*]}"
echo "[eval-slurm] policy python: ${POLICY_PYTHON}"
echo "[eval-slurm] eval python: ${EVAL_PYTHON}"

setsid bash -lc "CUDA_VISIBLE_DEVICES=${GPU_ID_SERVER} ${POLICY_PYTHON} scripts/serve_policy.py --seed=${SEED} --port=${PORT} policy:checkpoint --policy.dir=${POLICY_DIR} --policy.config=${CONFIG_TYPE}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

echo "[eval-slurm] policy server pid: ${SERVER_PID}"
echo "[eval-slurm] policy server log: ${SERVER_LOG}"
echo "[eval-slurm] waiting ${SERVER_STARTUP_WAIT}s"
sleep "${SERVER_STARTUP_WAIT}"

if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
  echo "[eval-slurm] policy server exited before eval started"
  cat "${SERVER_LOG}" >&2 || true
  exit 1
fi

SAPIEN_RENDER_DEVICE=${SAPIEN_RENDER_DEVICE:-cuda} \
MUJOCO_GL=${MUJOCO_GL:-egl} \
CUDA_VISIBLE_DEVICES=${GPU_ID_CLIENT} \
"${EVAL_PYTHON}" examples/robomme/eval.py \
  --args.model_seed="${SEED}" \
  --args.port="${PORT}" \
  --args.policy_name="${MODEL_TYPE}" \
  --args.model_ckpt_id="${CKPT_ID}" \
  --args.save_dir="${EVAL_SAVE_DIR}" \
  "${MODEL_EXTRA_EVAL_ARGS_LIST[@]}"
