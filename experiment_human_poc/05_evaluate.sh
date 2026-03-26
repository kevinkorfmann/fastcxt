#!/usr/bin/env bash
set -euo pipefail
# Evaluate pairwise model: scatter, genome profile, residuals

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CKPT=$(find "$SCRIPT_DIR/outputs/pairwise" -name "*.ckpt" -type f 2>/dev/null | sort -t= -k2 -n | tail -1)
if [ -z "$CKPT" ]; then
    echo "ERROR: No checkpoint found in outputs/pairwise/"
    exit 1
fi

echo "Evaluating checkpoint: $CKPT"
python "$SCRIPT_DIR/evaluate_pairwise.py" \
    --checkpoint "$CKPT" \
    --out-dir "$SCRIPT_DIR/figures/pairwise"
