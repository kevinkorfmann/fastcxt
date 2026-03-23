#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CKPT=$(find "$SCRIPT_DIR/outputs/pairwise" -name "*.ckpt" -type f 2>/dev/null | sort | tail -1)
if [ -z "$CKPT" ]; then echo "ERROR: No checkpoint found"; exit 1; fi

ARGS=(
    --pairwise-ckpt "$CKPT"
    --node-weights-true "$SCRIPT_DIR/outputs/node_time_true.pt"
    --out-csv "$SCRIPT_DIR/outputs/benchmark_results.csv"
)
[ -f "$SCRIPT_DIR/outputs/node_time_tsinfer.pt" ] && \
    ARGS+=(--node-weights-tsinfer "$SCRIPT_DIR/outputs/node_time_tsinfer.pt")

python "$SCRIPT_DIR/benchmark_scaling.py" "${ARGS[@]}"
