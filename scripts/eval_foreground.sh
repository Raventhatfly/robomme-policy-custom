#!/usr/bin/env bash
set -euo pipefail

# Foreground rollout runner for VS Code/debugging.
# Unlike scripts/eval.sh, this does not use tmux, so logs stay in the current terminal.

MODEL_TYPE=${MODEL_TYPE:-symbolic_groundedSG_oracle}
SEED=${SEED:-7}
CKPT_ID=${CKPT_ID:-79999}
GPU_ID_SERVER=${GPU_ID_SERVER:-0}
GPU_ID_CLIENT=${GPU_ID_CLIENT:-1}
PORT=${PORT:-8000}
SERVER_STARTUP_WAIT=${SERVER_STARTUP_WAIT:-30}
EVAL_ENV=${EVAL_ENV:-robomme}
POLICY_PYTHON=${POLICY_PYTHON:-/n/holylabs/LABS/hankyang_lab/Lab/felix/.conda/envs/robomme-openpi/bin/python}
EVAL_SAVE_DIR=${EVAL_SAVE_DIR:-/n/netscratch/hankyang_lab/Lab/felix/ckpts/robomme_policy_ckpt/evaluation}
USER_EXTRA_ARGS=${EXTRA_ARGS:-}

find_free_port() {
  local min=${1:-2000}
  local max=${2:-30000}
  local port
  local tries=5000

  for ((i = 0; i < tries; i++)); do
    port=$(shuf -i"${min}"-"${max}" -n1)
    if ! lsof -iTCP:"${port}" -sTCP:LISTEN &>/dev/null; then
      echo "${port}"
      return 0
    fi
  done

  echo "ERROR: not found free port in range ${min}-${max}" >&2
  return 1
}

if [[ "${PORT}" == "auto" ]]; then
  PORT=$(find_free_port)
fi

ORIGINAL_MODEL_TYPE=${MODEL_TYPE}
MODEL_EXTRA_ARGS=""

if [[ "${MODEL_TYPE}" == "pi05_baseline" ]]; then
  CONFIG_TYPE="pi05_baseline"
  MODEL_EXTRA_ARGS="--args.no-use-history"
else
  CONFIG_TYPE="mme_vla_suite"
  case "${MODEL_TYPE}" in
    symbolic_simpleSG_oracle)
      MODEL_EXTRA_ARGS="--args.use-oracle --args.subgoal-type=simple_subgoal"
      MODEL_TYPE="symbolic-simple-subgoal"
      ;;
    symbolic_groundedSG_oracle)
      MODEL_EXTRA_ARGS="--args.use-oracle --args.subgoal-type=grounded_subgoal"
      MODEL_TYPE="symbolic-grounded-subgoal"
      ;;
    symbolic_simpleSG_qwenvl)
      MODEL_EXTRA_ARGS="--args.use-qwenvl --args.subgoal-type=simple_subgoal"
      MODEL_TYPE="symbolic-simple-subgoal"
      ;;
    symbolic_groundedSG_qwenvl)
      MODEL_EXTRA_ARGS="--args.use-qwenvl --args.subgoal-type=grounded_subgoal"
      MODEL_TYPE="symbolic-grounded-subgoal"
      ;;
    symbolic_simpleSG_gemini)
      MODEL_EXTRA_ARGS="--args.use-gemini --args.subgoal-type=simple_subgoal"
      MODEL_TYPE="symbolic-simple-subgoal"
      ;;
    symbolic_groundedSG_gemini)
      MODEL_EXTRA_ARGS="--args.use-gemini --args.subgoal-type=grounded_subgoal"
      MODEL_TYPE="symbolic-grounded-subgoal"
      ;;
    MemER)
      MODEL_EXTRA_ARGS="--args.use-memer --args.subgoal-type=grounded_subgoal"
      MODEL_TYPE="symbolic-grounded-subgoal"
      ;;
  esac
fi

SERVER_LOG=$(mktemp "/tmp/robomme_policy_server.${PORT}.XXXXXX.log")
SERVER_PID=""
TAIL_PID=""

cleanup() {
  local status=$?

  if [[ -n "${TAIL_PID}" ]] && kill -0 "${TAIL_PID}" 2>/dev/null; then
    kill "${TAIL_PID}" 2>/dev/null || true
  fi

  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[main] stopping policy server pid ${SERVER_PID}"
    kill -- -"${SERVER_PID}" 2>/dev/null || kill "${SERVER_PID}" 2>/dev/null || true
    sleep 2
    kill -9 -- -"${SERVER_PID}" 2>/dev/null || true
  fi

  echo "[main] server log: ${SERVER_LOG}"
  exit "${status}"
}
trap cleanup EXIT INT TERM

SERVER_CMD="CUDA_VISIBLE_DEVICES=${GPU_ID_SERVER} ${POLICY_PYTHON} scripts/serve_policy.py --seed=${SEED} --port=${PORT} policy:checkpoint --policy.dir=runs/ckpts/${CONFIG_TYPE}/${MODEL_TYPE}/${CKPT_ID} --policy.config=${CONFIG_TYPE}"
EVAL_PY_CMD="python examples/robomme/eval.py --args.model_seed=${SEED} --args.port=${PORT} --args.policy_name=${MODEL_TYPE} --args.model_ckpt_id=${CKPT_ID} --args.save_dir=${EVAL_SAVE_DIR} ${MODEL_EXTRA_ARGS} ${USER_EXTRA_ARGS}"
EVAL_CMD="CUDA_VISIBLE_DEVICES=${GPU_ID_CLIENT} ${EVAL_PY_CMD}"

if [[ -n "${EVAL_ENV}" ]]; then
  if command -v micromamba >/dev/null 2>&1; then
    EVAL_CMD="CUDA_VISIBLE_DEVICES=${GPU_ID_CLIENT} micromamba run -n ${EVAL_ENV} ${EVAL_PY_CMD}"
  elif command -v conda >/dev/null 2>&1; then
    EVAL_CMD="CUDA_VISIBLE_DEVICES=${GPU_ID_CLIENT} conda run -n ${EVAL_ENV} ${EVAL_PY_CMD}"
  else
    echo "[main] WARNING: neither micromamba nor conda was found; running eval in the current environment"
  fi
fi

echo "[main] requested model: ${ORIGINAL_MODEL_TYPE}"
echo "[main] normalized model: ${MODEL_TYPE}"
echo "[main] config: ${CONFIG_TYPE}, seed: ${SEED}, ckpt: ${CKPT_ID}, port: ${PORT}"
echo "[main] eval save dir: ${EVAL_SAVE_DIR}"
echo "[main] starting policy server on GPU ${GPU_ID_SERVER}"
echo "[main] ${SERVER_CMD}"

setsid bash -lc "${SERVER_CMD}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
tail -n +1 -F "${SERVER_LOG}" 2>/dev/null | sed -u 's/^/[server] /' &
TAIL_PID=$!

echo "[main] waiting ${SERVER_STARTUP_WAIT}s for server startup"
sleep "${SERVER_STARTUP_WAIT}"

if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
  echo "[main] policy server exited before eval started"
  exit 1
fi

echo "[main] starting eval client on GPU ${GPU_ID_CLIENT}"
echo "[main] ${EVAL_CMD}"

set +e
bash -lc "${EVAL_CMD}" 2>&1 | sed -u 's/^/[eval] /'
EVAL_STATUS=${PIPESTATUS[0]}
set -e

if [[ "${EVAL_STATUS}" -ne 0 ]]; then
  echo "[main] eval failed with status ${EVAL_STATUS}"
  exit "${EVAL_STATUS}"
fi

echo "[main] eval finished successfully"
