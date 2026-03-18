#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# fastcxt — Large-context AnoGam experiment (2000 windows) on Betty cluster
# ============================================================================
# Adapted from run_experiment.sh:
#   - 2000 windows (400 kb / 200 bp) instead of 4000
#   - model: base_trees_2k (d_model=256)
#   - 200 epochs
#   - CSV metrics logging instead of TensorBoard
#   - Storage on /vast/projects/smathi/cohort/
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
echo "REPO_DIR=$REPO_DIR"
echo "SCRIPT_DIR=$SCRIPT_DIR"
WORK_DIR="/vast/projects/smathi/cohort/fastcxt_large_context"
SIMS_DIR="$WORK_DIR/sims"

# -- Defaults ----------------------------------------------------------------
GPUS="0"
EPOCHS=200
NUM_TS_PER_SIZE=200
SAMPLE_SIZES="10 25 50 100 200"
MAX_SAMPLES=200
BATCH_SIZE=16
GRAD_ACCUM=16
NUM_WORKERS=8
NUM_SIM_PROCS=8

SEQ_LEN=400000        # 400 kb
WINDOW_SIZE=200       # 200 bp -> 2000 windows

SKIP_INSTALL=false
SKIP_SIM=false
SKIP_PREPROCESS=false
SKIP_TRAIN=false

# -- Parse arguments ---------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)           GPUS="$2"; shift 2;;
        --epochs)         EPOCHS="$2"; shift 2;;
        --num-ts)         NUM_TS_PER_SIZE="$2"; shift 2;;
        --batch-size)     BATCH_SIZE="$2"; shift 2;;
        --skip-install)   SKIP_INSTALL=true; shift;;
        --skip-sim)       SKIP_SIM=true; shift;;
        --skip-preprocess) SKIP_PREPROCESS=true; shift;;
        --skip-train)     SKIP_TRAIN=true; shift;;
        -h|--help)
            echo "Usage: bash run_betty.sh [OPTIONS]"
            exit 0;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

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

# -- Auto-detect CUDA toolkit -----------------------------------------------
if [ -z "${CUDA_HOME:-}" ]; then
    for candidate in \
        /usr/local/cuda \
        /usr/local/cuda-12 \
        /usr/local/cuda-12.* \
        "$HOME/miniconda/envs/"*/; do
        if [ -x "${candidate}/bin/nvcc" ]; then
            export CUDA_HOME="$candidate"
            break
        fi
    done
fi
if [ -n "${CUDA_HOME:-}" ]; then
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
    echo "CUDA_HOME=$CUDA_HOME"
fi

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

# -- Step 1: Simulate AnoGam ------------------------------------------------
if [ "$SKIP_SIM" = false ]; then
    step "Step 1: Simulate AnoGam (400 kb sequences, 2000 windows) training data"
    T0=$SECONDS
    for N in $SAMPLE_SIZES; do
        echo "  Simulating n_samples=$N ($NUM_TS_PER_SIZE tree sequences) ..."
        fastcxt-simulate \
            --scenario AnoGam \
            --data-dir "$SIMS_DIR/n${N}" \
            --num-ts "$NUM_TS_PER_SIZE" \
            --n-samples "$N" \
            --sequence-length "$SEQ_LEN" \
            --num-processes "$NUM_SIM_PROCS"
    done
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

# -- Step 3: Train base_trees_2k model with CSV logging --------------------
if [ "$SKIP_TRAIN" = false ]; then
    step "Step 3: Train base_trees_2k model (${EPOCHS} epochs, lazy SFS, CSV metrics)"
    T0=$SECONDS
    fastcxt-train \
        --model base_trees_2k \
        --dataset-path "$SIMS_DIR/processed_lazy_tsinfer" \
        --gpus $GPUS \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --grad-accum "$GRAD_ACCUM" \
        --workers "$NUM_WORKERS" \
        --log-dir "$WORK_DIR/tsinfer_2k_logs" \
        --dataset-mode lazy \
        --csv-log
    echo "  Training done in $(elapsed $((SECONDS - T0)))"
fi

# -- Done --------------------------------------------------------------------
step "COMPLETE"
echo "  Total time: $(elapsed $((SECONDS - TOTAL_START)))"
echo ""
echo "  Outputs:"
echo "    Logs/Metrics: $WORK_DIR/tsinfer_2k_logs/"
echo "    Simulations:  $SIMS_DIR/"
echo ""
