#!/usr/bin/env bash
# CoE top-5 candidate evaluation entrypoint.
# Usage: bash scripts/run_eval.sh [wiki_coe|slidevqa] [model_path] [num_gpus] [out_dir]

set -euo pipefail

DATASET=${1:-"wiki_coe"}
MODEL_PATH=${2:-""}
NUM_GPUS=${3:-8}
OUT_DIR=${4:-""}
BATCH_SIZE=${BATCH_SIZE:-4}

echo "=========================================="
echo " CoE Evaluation"
echo " Dataset: ${DATASET}"
echo "=========================================="

if [ "$DATASET" == "wiki_coe" ]; then
    MODEL_PATH=${MODEL_PATH:-"checkpoints/coe_phase2/best"}
    DATA_FILE="data/wiki_coe/release_strict_chainsplit_20260428_1338/test.json"
    IMAGE_DIR="data/wiki_coe/screenshots_trimmed"
    DISTRACTOR_STRATEGY="global"
    IMAGE_MAX_PIXELS=1048576
    EVAL_RESOLUTION=1024
    OUT_DIR=${OUT_DIR:-"eval_out/wiki_coe_top5"}
elif [ "$DATASET" == "slidevqa" ]; then
    MODEL_PATH=${MODEL_PATH:-"checkpoints/slidevqa_phase2/best"}
    DATA_FILE="data/slidevqa/test_answerable_full.json"
    IMAGE_DIR="data/slidevqa/slides"
    DISTRACTOR_STRATEGY="same_deck"
    IMAGE_MAX_PIXELS=1048576
    EVAL_RESOLUTION=1024
    OUT_DIR=${OUT_DIR:-"eval_out/slidevqa_top5"}
else
    echo "Error: Unknown dataset $DATASET. Use 'wiki_coe' or 'slidevqa'"
    exit 1
fi

echo " Model: ${MODEL_PATH}"
echo " Output: ${OUT_DIR}"
mkdir -p "$OUT_DIR"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

torchrun --nproc_per_node="${NUM_GPUS}" scripts/offline_full_eval.py \
    --model_path "${MODEL_PATH}" \
    --val_file "${DATA_FILE}" \
    --image_dir "${IMAGE_DIR}" \
    --mode multi_hop \
    --image_max_pixels "${IMAGE_MAX_PIXELS}" \
    --eval_resolution "${EVAL_RESOLUTION}" \
    --max_new_tokens 512 \
    --candidate_top_k 5 \
    --candidate_distractor_strategy "${DISTRACTOR_STRATEGY}" \
    --candidate_seed 42 \
    --batch_size "${BATCH_SIZE}" \
    --resume \
    --out_dir "${OUT_DIR}"

echo "Evaluation complete. Summary saved to ${OUT_DIR}/summary.json"
