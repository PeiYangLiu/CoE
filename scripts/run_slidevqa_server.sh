#!/usr/bin/env bash
# Start the CoE SlideVQA web/API server.

set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export COE_HOST="${COE_HOST:-0.0.0.0}"
export COE_PORT="${COE_PORT:-7860}"
export COE_MODEL_PATH="${COE_MODEL_PATH:-PeiyangLiu/CoE-SlideVQA-8B}"
export COE_PROCESSOR_PATH="${COE_PROCESSOR_PATH:-${COE_MODEL_PATH}}"
export COE_TOP_K="${COE_TOP_K:-20}"
export COE_MAX_TOP_K="${COE_MAX_TOP_K:-20}"
export COE_PRELOAD_MODEL="${COE_PRELOAD_MODEL:-0}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export COE_SERVER_MODE="${COE_SERVER_MODE:-single}"
export COE_RUNTIME_DIR="${COE_RUNTIME_DIR:-$(pwd)/service/runtime}"
unset WEB_CONCURRENCY

if [ -z "${COE_API_TOKEN:-}" ]; then
  echo "WARNING: COE_API_TOKEN is not set; the upload/inference API will be unauthenticated." >&2
fi

mkdir -p "${COE_RUNTIME_DIR}/logs" "${COE_RUNTIME_DIR}/pids"

if [ "${COE_SERVER_MODE}" = "multi" ]; then
  export COE_CUDA_DEVICES="${COE_CUDA_DEVICES:-4,5,6,7}"
  export COE_BACKEND_START_PORT="${COE_BACKEND_START_PORT:-7861}"

  IFS=',' read -r -a GPU_IDS <<< "${COE_CUDA_DEVICES}"
  BACKEND_URLS=()
  CHILD_PIDS=()

  cleanup() {
    for pid in "${CHILD_PIDS[@]:-}"; do
      if kill -0 "${pid}" >/dev/null 2>&1; then
        kill "${pid}" >/dev/null 2>&1 || true
      fi
    done
  }
  trap cleanup EXIT INT TERM

  for i in "${!GPU_IDS[@]}"; do
    gpu="${GPU_IDS[$i]}"
    port=$((COE_BACKEND_START_PORT + i))
    url="http://127.0.0.1:${port}"
    BACKEND_URLS+=("${url}")
    log="${COE_RUNTIME_DIR}/logs/backend_gpu${gpu}_port${port}.log"
    echo "Starting backend ${i} on GPU ${gpu}, port ${port}; log=${log}"
    env \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      COE_HOST="127.0.0.1" \
      COE_PORT="${port}" \
      COE_PRELOAD_MODEL=1 \
      COE_DISABLE_CLEANUP=1 \
      COE_RUNTIME_DIR="${COE_RUNTIME_DIR}" \
      "${PYTHON_BIN}" -m uvicorn service.app:app \
        --host 127.0.0.1 \
        --port "${port}" \
        --workers 1 \
        > "${log}" 2>&1 &
    pid=$!
    CHILD_PIDS+=("${pid}")
    echo "${pid}" > "${COE_RUNTIME_DIR}/pids/backend_${i}.pid"
  done

  backend_csv=$(IFS=','; echo "${BACKEND_URLS[*]}")
  gateway_log="${COE_RUNTIME_DIR}/logs/gateway_port${COE_PORT}.log"
  echo "Starting gateway on ${COE_HOST}:${COE_PORT}; backends=${backend_csv}; log=${gateway_log}"
  env \
    COE_BACKEND_URLS="${backend_csv}" \
    COE_PRELOAD_MODEL=0 \
    COE_RUNTIME_DIR="${COE_RUNTIME_DIR}" \
    "${PYTHON_BIN}" -m uvicorn service.gateway:gateway \
      --host "${COE_HOST}" \
      --port "${COE_PORT}" \
      --workers 1 \
      > "${gateway_log}" 2>&1 &
  gateway_pid=$!
  CHILD_PIDS+=("${gateway_pid}")
  echo "${gateway_pid}" > "${COE_RUNTIME_DIR}/pids/gateway.pid"

  wait -n "${CHILD_PIDS[@]}"
  exit $?
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
exec "${PYTHON_BIN}" -m uvicorn service.app:app \
  --host "${COE_HOST}" \
  --port "${COE_PORT}" \
  --workers 1
