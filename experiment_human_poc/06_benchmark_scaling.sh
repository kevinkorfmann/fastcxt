#!/usr/bin/env bash
set -euo pipefail
# Benchmark pairwise vs node-time scaling

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CKPT=$(find "$SCRIPT_DIR/outputs/pairwise" -name "*.ckpt" -type f 2>/dev/null | sort -t= -k2 -n | tail -1)
if [ -z "$CKPT" ]; then
    echo "ERROR: No pairwise checkpoint found"
    exit 1
fi

ARGS=(
    --pairwise-ckpt "$CKPT"
    --node-weights-true "$SCRIPT_DIR/outputs/node_time_true.pt"
    --out-csv "$SCRIPT_DIR/outputs/benchmark_results.csv"
)

if [ -f "$SCRIPT_DIR/outputs/node_time_tsinfer.pt" ]; then
    ARGS+=(--node-weights-tsinfer "$SCRIPT_DIR/outputs/node_time_tsinfer.pt")
fi

python "$SCRIPT_DIR/benchmark_scaling.py" "${ARGS[@]}"
