#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Whole HomSap Chromosome Decoding — Launch Full Curriculum Pipeline
# ============================================================================
# Submits all stages with SLURM dependency chaining:
#   Stage 1 sim → Stage 1 train → Stage 2 sim → Stage 2 train → ...
#
# Usage:
#   bash launch_all.sh                    # Start from stage 1
#   bash launch_all.sh --start-stage 2    # Resume from stage 2
#   bash launch_all.sh --dry-run          # Show commands without submitting
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
START_STAGE=1
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --start-stage)  START_STAGE="$2"; shift 2;;
        --dry-run)      DRY_RUN=true; shift;;
        -h|--help)
            echo "Usage: bash launch_all.sh [--start-stage N] [--dry-run]"
            exit 0;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

submit() {
    local cmd="$1"
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY-RUN] $cmd" >&2
        echo "FAKE_JOBID_$$"
    else
        local output
        output=$(eval "$cmd")
        local jobid
        jobid=$(echo "$output" | grep -o '[0-9]*' | tail -1)
        echo "  Submitted job $jobid" >&2
        echo "$jobid"
    fi
}

# Time limits per stage for simulation
sim_time() {
    case $1 in
        1|2) echo "2-00:00:00";;
        3|4) echo "4-00:00:00";;
    esac
}

PREV_JOBID=""

for STAGE in 1 2 3 4; do
    if [ "$STAGE" -lt "$START_STAGE" ]; then
        continue
    fi

    echo ""
    echo "=== Stage $STAGE ==="

    # -- Simulation job --
    SIM_CMD="sbatch --export=STAGE=$STAGE --time=$(sim_time $STAGE) --job-name=homsap-sim-s${STAGE}"
    if [ -n "$PREV_JOBID" ]; then
        SIM_CMD="$SIM_CMD --dependency=afterok:$PREV_JOBID"
    fi
    SIM_CMD="$SIM_CMD $SCRIPT_DIR/slurm_sim.sh"

    echo "  Sim:"
    SIM_JOBID=$(submit "$SIM_CMD")

    # -- Training job (depends on sim) --
    TRAIN_CMD="sbatch --export=STAGE=$STAGE --job-name=homsap-train-s${STAGE} --dependency=afterok:$SIM_JOBID"
    TRAIN_CMD="$TRAIN_CMD $SCRIPT_DIR/slurm_train.sh"

    echo "  Train:"
    PREV_JOBID=$(submit "$TRAIN_CMD")
done

echo ""
echo "All stages submitted. Monitor with: squeue -u \$USER"
