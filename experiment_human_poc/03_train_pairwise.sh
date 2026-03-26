#!/usr/bin/env bash
set -euo pipefail
# Train pairwise FastCxtModel (base preset, SFS-only)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUS="${GPUS:-0}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-128}"

fastcxt-train \
    --model base \
    --dataset-path "$SCRIPT_DIR/sims/processed" \
    --gpus $GPUS \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum 2 \
    --workers 8 \
    --log-dir "$SCRIPT_DIR/outputs/pairwise" \
    --csv-log \
    --save-every-epoch

echo "Pairwise training complete. Checkpoints: outputs/pairwise/"
