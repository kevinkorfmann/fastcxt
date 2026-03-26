#!/usr/bin/env bash
set -euo pipefail
# AnoGam (mosquito) experiment pipeline — 100kb, 200bp windows, 500 windows
#
# Usage:
#   bash run_all.sh
#   GPUS="0 1" bash run_all.sh
#   bash run_all.sh --from 03

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FROM="${1:-01}"
FROM="${FROM#--from=}"
[[ "$FROM" == "--from" ]] && FROM="${2:-01}"

elapsed() { printf '%dh %dm %ds' $(($1/3600)) $(($1%3600/60)) $(($1%60)); }

run_step() {
    local script="$1"
    local step="${script%%_*}"
    if [[ "$step" < "$FROM" ]]; then
        echo "Skipping $script (--from $FROM)"
        return
    fi
    echo ""
    echo "======== $script ========"
    local t0=$SECONDS
    bash "$SCRIPT_DIR/$script"
    echo "  Done in $(elapsed $((SECONDS - t0)))"
}

TOTAL_START=$SECONDS

run_step 01_simulate.sh
run_step 02_preprocess.sh
run_step 03_train_pairwise.sh
run_step 04_train_node_time.sh
run_step 05_evaluate.sh
run_step 06_benchmark_scaling.sh

echo ""
echo "======== ALL COMPLETE ========"
echo "  Total time: $(elapsed $((SECONDS - TOTAL_START)))"
