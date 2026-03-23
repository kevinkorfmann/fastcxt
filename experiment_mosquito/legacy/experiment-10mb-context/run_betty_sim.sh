#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# fastcxt — 10 Mb AnoGam simulation + preprocess (single sample size)
# ============================================================================
# 50000 windows (10 Mb / 200 bp) — 100x context of the original 100kb model
# Run one sample size per SLURM job for parallelism
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
echo "REPO_DIR=$REPO_DIR"
echo "SCRIPT_DIR=$SCRIPT_DIR"
WORK_DIR="/vast/projects/smathi/cohort/fastcxt_10mb_context"
SIMS_DIR="$WORK_DIR/sims"

# -- Defaults ----------------------------------------------------------------
NUM_TS_PER_SIZE=50
NUM_SIM_PROCS=50
NUM_WORKERS=50
MAX_SAMPLES=200

SEQ_LEN=10000000      # 10 Mb
WINDOW_SIZE=200       # 200 bp -> 50000 windows

SKIP_INSTALL=false
SKIP_SIM=false
SKIP_PREPROCESS=false
N_SAMPLES=""

# -- Parse arguments ---------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --n-samples)      N_SAMPLES="$2"; shift 2;;
        --skip-install)   SKIP_INSTALL=true; shift;;
        --skip-sim)       SKIP_SIM=true; shift;;
        --skip-preprocess) SKIP_PREPROCESS=true; shift;;
        -h|--help)
            echo "Usage: bash run_betty_sim.sh --n-samples N [OPTIONS]"
            echo "  --n-samples N       Sample size to simulate (required)"
            echo "  --skip-install      Skip pip install"
            echo "  --skip-sim          Skip simulation step"
            echo "  --skip-preprocess   Skip preprocessing step"
            exit 0;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

if [ -z "$N_SAMPLES" ]; then
    echo "ERROR: --n-samples is required"
    exit 1
fi

step() {
    echo ""
    echo "========================================================================"
    echo "  $1"
    echo "========================================================================"
    echo ""
}

elapsed() {
    local t=$1
    printf '%dh %dm %ds' $((t/3600)) $((t%3600/60)) $((t%60))
}

TOTAL_START=$SECONDS

mkdir -p "$WORK_DIR" "$SIMS_DIR"

# -- Redirect caches to project storage (home quota is full) ----------------
export TRITON_CACHE_DIR="$WORK_DIR/.triton_cache"
export XDG_CACHE_HOME="$WORK_DIR/.cache"
mkdir -p "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

# -- Step 0: Install --------------------------------------------------------
if [ "$SKIP_INSTALL" = false ]; then
    step "Step 0: Install fastcxt (with stdpopsim)"
    cd "$REPO_DIR"
    if [ ! -d ".venv" ]; then
        uv venv
    fi
    source .venv/bin/activate
    uv pip install -e ".[all]"
fi

if [ -f "$REPO_DIR/.venv/bin/activate" ]; then
    source "$REPO_DIR/.venv/bin/activate"
fi

# -- Step 1: Simulate AnoGam (single sample size) ---------------------------
if [ "$SKIP_SIM" = false ]; then
    step "Step 1: Simulate AnoGam (10 Mb sequences) n_samples=$N_SAMPLES"
    T0=$SECONDS
    echo "  Simulating n_samples=$N_SAMPLES ($NUM_TS_PER_SIZE tree sequences) ..."
    fastcxt-simulate \
        --scenario AnoGam \
        --data-dir "$SIMS_DIR/n${N_SAMPLES}" \
        --num-ts "$NUM_TS_PER_SIZE" \
        --n-samples "$N_SAMPLES" \
        --sequence-length "$SEQ_LEN" \
        --num-processes "$NUM_SIM_PROCS"
    echo "  Simulation done in $(elapsed $((SECONDS - T0)))"
fi

# -- Step 2: Preprocess with lazy SFS + tsinfer trees -----------------------
if [ "$SKIP_PREPROCESS" = false ]; then
    step "Step 2: Preprocess (lazy SFS + tsinfer tree topology features)"
    T0=$SECONDS
    fastcxt-preprocess \
        --base-dir "$SIMS_DIR" \
        --out-subdir processed_lazy_tsinfer \
        --window-size "$WINDOW_SIZE" \
        --sequence-length "$SEQ_LEN" \
        --num-pairs 200 \
        --extract-tsinfer-trees \
        --max-samples "$MAX_SAMPLES" \
        --num-workers "$NUM_WORKERS" \
        --lazy-sfs
    echo "  Preprocessing done in $(elapsed $((SECONDS - T0)))"
fi

# -- Done --------------------------------------------------------------------
step "COMPLETE (n_samples=$N_SAMPLES)"
echo "  Total time: $(elapsed $((SECONDS - TOTAL_START)))"
echo ""
echo "  Outputs:"
echo "    Simulations:  $SIMS_DIR/n${N_SAMPLES}/"
echo ""
