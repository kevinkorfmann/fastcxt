#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUS="${GPUS:-0}"
EPOCHS="${EPOCHS:-30}"

fastcxt-train \
    --model base \
    --dataset-path "$SCRIPT_DIR/sims/processed" \
    --gpus $GPUS \
    --epochs "$EPOCHS" \
    --batch-size 128 \
    --grad-accum 2 \
    --workers 8 \
    --log-dir "$SCRIPT_DIR/outputs/pairwise" \
    --csv-log \
    --save-every-epoch
