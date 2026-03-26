#!/usr/bin/env bash
set -euo pipefail
# Preprocess: SFS features + node-time features for AnoGam (100kb, 200bp windows)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIMS_DIR="$SCRIPT_DIR/sims"
NUM_WORKERS=${NUM_WORKERS:-8}

echo "=== Phase 1: SFS preprocessing ==="
fastcxt-preprocess \
    --base-dir "$SIMS_DIR" \
    --out-subdir processed \
    --window-size 200 \
    --sequence-length 100000 \
    --num-pairs 200 \
    --num-workers "$NUM_WORKERS"

echo ""
echo "=== Phase 2: Node-time features (true trees) ==="
python "$SCRIPT_DIR/preprocess_node_times.py" \
    --sims-dir "$SIMS_DIR" \
    --out-dir "$SCRIPT_DIR/outputs/node_features"

echo ""
echo "=== Phase 3: Node-time features (tsinfer trees) ==="
python "$SCRIPT_DIR/preprocess_node_times.py" \
    --sims-dir "$SIMS_DIR" \
    --out-dir "$SCRIPT_DIR/outputs/node_features_tsinfer" \
    --use-tsinfer

echo "Preprocessing complete."
