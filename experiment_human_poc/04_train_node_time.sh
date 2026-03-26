#!/usr/bin/env bash
set -euo pipefail
# Train NodeTimeModel: true trees and tsinfer trees

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EPOCHS="${EPOCHS:-30}"

echo "=== Training NodeTimeModel (true trees) ==="
python "$SCRIPT_DIR/train_node_time.py" \
    --features-dir "$SCRIPT_DIR/outputs/node_features" \
    --out "$SCRIPT_DIR/outputs/node_time_true.pt" \
    --epochs "$EPOCHS"

echo ""
echo "=== Training NodeTimeModel (tsinfer trees) ==="
python "$SCRIPT_DIR/train_node_time.py" \
    --features-dir "$SCRIPT_DIR/outputs/node_features_tsinfer" \
    --out "$SCRIPT_DIR/outputs/node_time_tsinfer.pt" \
    --epochs "$EPOCHS"

echo "Node-time training complete."
