#!/usr/bin/env bash
# Run the AnoGam experiment in the background with nohup.
# Usage from repo root:  bash experiment_mossies/run_nohup.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
LOG="$SCRIPT_DIR/nohup.out"

cd "$REPO_DIR"
nohup bash -c 'cd experiment_mossies && bash run_experiment.sh' > "$LOG" 2>&1 &
echo "PID $! — log: $LOG"
