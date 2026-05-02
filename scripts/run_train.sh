#!/usr/bin/env bash
# CoE training entrypoint.
# Usage: bash scripts/run_train.sh [wiki_phase1|wiki_phase2|slidevqa_phase1|slidevqa_phase2] [num_gpus]

set -euo pipefail

TARGET=${1:-"wiki_phase1"}
NUM_GPUS=${2:-8}

case "$TARGET" in
    phase1|wiki_phase1)
        NAME="wiki_phase1"
        CONFIG="configs/train_phase1.yaml"
        OUTPUT_DIR="checkpoints/coe_phase1"
        REQUIRED_CKPT=""
        ;;
    phase2|wiki_phase2)
        NAME="wiki_phase2"
        CONFIG="configs/train_phase2.yaml"
        OUTPUT_DIR="checkpoints/coe_phase2"
        REQUIRED_CKPT="checkpoints/coe_phase1/best"
        ;;
    slidevqa_phase1)
        NAME="slidevqa_phase1"
        CONFIG="configs/train_slidevqa_phase1.yaml"
        OUTPUT_DIR="checkpoints/slidevqa_phase1"
        REQUIRED_CKPT=""
        ;;
    slidevqa_phase2)
        NAME="slidevqa_phase2"
        CONFIG="configs/train_slidevqa_phase2.yaml"
        OUTPUT_DIR="checkpoints/slidevqa_phase2"
        REQUIRED_CKPT="checkpoints/slidevqa_phase1/best"
        ;;
    *)
        echo "Error: unknown target '$TARGET'"
        echo "Valid targets: wiki_phase1, wiki_phase2, slidevqa_phase1, slidevqa_phase2"
        exit 1
        ;;
esac

if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file $CONFIG not found"
    exit 1
fi

if [ -n "$REQUIRED_CKPT" ] && [ ! -d "$REQUIRED_CKPT" ]; then
    echo "Error: required Phase I checkpoint not found at $REQUIRED_CKPT"
    exit 1
fi

echo "=========================================="
echo " CoE Training - ${NAME}"
echo " Config: ${CONFIG}"
echo " GPUs: ${NUM_GPUS}"
echo "=========================================="

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export LOCAL_CKPT_DIR="$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR" logs

deepspeed --num_gpus ${NUM_GPUS} \
    training/train.py \
    --config ${CONFIG} \
    --output_dir "$OUTPUT_DIR"

echo "Training complete. Checkpoints saved to ${OUTPUT_DIR}/"
