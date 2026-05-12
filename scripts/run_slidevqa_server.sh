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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHON_BIN="${PYTHON_BIN:-python}"

if [ -z "${COE_API_TOKEN:-}" ]; then
  echo "WARNING: COE_API_TOKEN is not set; the upload/inference API will be unauthenticated." >&2
fi

mkdir -p service/runtime/logs

exec "${PYTHON_BIN}" -m uvicorn service.app:app \
  --host "${COE_HOST}" \
  --port "${COE_PORT}" \
  --workers 1
