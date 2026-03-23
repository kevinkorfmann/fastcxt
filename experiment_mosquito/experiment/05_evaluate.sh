#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CKPT=$(find "$SCRIPT_DIR/outputs/pairwise" -name "*.ckpt" -type f 2>/dev/null | sort | tail -1)
if [ -z "$CKPT" ]; then echo "ERROR: No checkpoint found"; exit 1; fi

python "$SCRIPT_DIR/evaluate_pairwise.py" \
    --checkpoint "$CKPT" \
    --out-dir "$SCRIPT_DIR/figures/pairwise"
