#!/usr/bin/env bash
# Quick launcher for 2L population dating on poppy.
#
# Usage:
#   bash launch_2L.sh <checkpoint_path>
#   bash launch_2L.sh <checkpoint_path> --populations "Burkina Faso"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ARM=2L bash "$SCRIPT_DIR/run.sh" "$@"
