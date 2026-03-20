#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Whole HomSap Chromosome Decoding — Simulation & Preprocessing
# ============================================================================
# Usage: bash run_sim.sh --stage N
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
WORK_DIR="/vast/projects/smathi/cohort/kkor/homsap_curriculum"

WINDOW_SIZE=2000
NUM_SIM_PROCS=50
MAX_SAMPLES=200
SAMPLE_SIZES="10 25 50 100 200"
NUM_WORKERS=8
STAGE=""

# -- Parse arguments ---------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --stage)          STAGE="$2"; shift 2;;
        --num-sim-procs)  NUM_SIM_PROCS="$2"; shift 2;;
        -h|--help)
            echo "Usage: bash run_sim.sh --stage N [--num-sim-procs P]"
            echo "  --stage N          Stage 1-4"
            echo "  --num-sim-procs P  Number of simulation processes (default: 50)"
            exit 0;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

if [ -z "$STAGE" ]; then
    echo "ERROR: --stage is required (1-4)"
    exit 1
fi

# -- Stage lookup (bash 3.2 compatible) --------------------------------------
case $STAGE in
    1) SEQ_LEN=50000000;  NUM_TS_PER_SIZE=100; N_WIN=25000;;
    2) SEQ_LEN=100000000; NUM_TS_PER_SIZE=50;  N_WIN=50000;;
    3) SEQ_LEN=150000000; NUM_TS_PER_SIZE=25;  N_WIN=75000;;
    4) SEQ_LEN=250000000; NUM_TS_PER_SIZE=10;  N_WIN=125000;;
    *) echo "ERROR: Invalid stage $STAGE (must be 1-4)"; exit 1;;
esac

STAGE_DIR="$WORK_DIR/stage${STAGE}"
SIMS_DIR="$STAGE_DIR/sims"

echo "========================================================================"
echo "  Stage $STAGE: ${SEQ_LEN} bp, ${NUM_TS_PER_SIZE} tree sequences/size"
echo "  n_windows=${N_WIN}, window_size=${WINDOW_SIZE}"
echo "========================================================================"

step() {
    echo ""
    echo "-------- $1 --------"
    echo ""
}

elapsed() {
    local t=$1
    printf '%dh %dm %ds' $((t/3600)) $((t%3600/60)) $((t%60))
}

TOTAL_START=$SECONDS
mkdir -p "$SIMS_DIR"

# -- Redirect caches to project storage (home quota is full) ----------------
export TRITON_CACHE_DIR="$WORK_DIR/.triton_cache"
export XDG_CACHE_HOME="$WORK_DIR/.cache"
mkdir -p "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

# -- Activate venv -----------------------------------------------------------
if [ -f "$REPO_DIR/.venv/bin/activate" ]; then
    source "$REPO_DIR/.venv/bin/activate"
fi

# -- Simulate HomSap --------------------------------------------------------
step "Simulating HomSap (stage $STAGE)"
T0=$SECONDS
for N in $SAMPLE_SIZES; do
    echo "  Simulating n_samples=$N ($NUM_TS_PER_SIZE tree sequences) ..."
    fastcxt-simulate \
        --scenario HomSap \
        --data-dir "$SIMS_DIR/n${N}" \
        --num-ts "$NUM_TS_PER_SIZE" \
        --n-samples "$N" \
        --sequence-length "$SEQ_LEN" \
        --num-processes "$NUM_SIM_PROCS"
done
echo "  Simulation done in $(elapsed $((SECONDS - T0)))"

# -- Preprocess with lazy SFS (no tsinfer) ----------------------------------
step "Preprocessing (lazy SFS, no trees)"
T0=$SECONDS
fastcxt-preprocess \
    --base-dir "$SIMS_DIR" \
    --out-subdir processed_lazy \
    --window-size "$WINDOW_SIZE" \
    --sequence-length "$SEQ_LEN" \
    --num-pairs 200 \
    --max-samples "$MAX_SAMPLES" \
    --num-workers "$NUM_WORKERS" \
    --lazy-sfs
echo "  Preprocessing done in $(elapsed $((SECONDS - T0)))"

step "COMPLETE (stage $STAGE)"
echo "  Total time: $(elapsed $((SECONDS - TOTAL_START)))"
echo "  Simulations: $SIMS_DIR/"
echo "  Processed:   $SIMS_DIR/processed_lazy/"
