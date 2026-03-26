#!/usr/bin/env bash
# Run population dating for a given chromosome arm on poppy.
#
# Usage:
#   bash run.sh <checkpoint_path>               # date 2L (default)
#   ARM=2R bash run.sh <checkpoint_path>        # date 2R
#   bash run.sh <checkpoint_path> --populations "Burkina Faso" "Cameroon"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ARM="${ARM:-2L}"
CHECKPOINT="${1:?Usage: bash run.sh <checkpoint_path> [extra args...]}"
shift

echo "=== Mosquito Population Dating ==="
echo "Arm: $ARM"
echo "Checkpoint: $CHECKPOINT"
echo "Device: $(python -c 'import torch; print("cuda:0" if torch.cuda.is_available() else "cpu")')"
echo ""

python "$SCRIPT_DIR/run_population_dating.py" \
    --arm "$ARM" \
    --checkpoint "$CHECKPOINT" \
    --out-dir "$SCRIPT_DIR/results/$ARM" \
    "$@"
