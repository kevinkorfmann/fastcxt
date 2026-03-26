#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# fastcxt — Anopheles gambiae (AnoGam) experiment
# ============================================================================
# Runs the full pipeline: simulate AnoGam via stdpopsim → preprocess with
# tsinfer tree features → train tree-augmented model → evaluate + figures.
#
# Differences from the default experiment/:
#   - stdpopsim species:  AnoGam (Anopheles gambiae)
#   - sequence length:    100 kb  (vs 1 Mb)
#   - window size:        200 bp  (vs 2 kb)  → still 500 windows
#   - single model:       tree-augmented (tsinfer) only
#
# Usage:
#   bash run_experiment.sh                  # defaults
#   bash run_experiment.sh --gpus 0         # single GPU
#   bash run_experiment.sh --skip-sim       # skip simulation
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
SIMS_DIR="$SCRIPT_DIR/sims"

# ── Defaults ────────────────────────────────────────────────────────────
GPUS="0"
EPOCHS=30
NUM_TS_PER_SIZE=200
SAMPLE_SIZES="10 25 50 100 200"
MAX_SAMPLES=200
BATCH_SIZE=128
GRAD_ACCUM=2
NUM_WORKERS=8
NUM_SIM_PROCS=8

SEQ_LEN=100000       # 100 kb
WINDOW_SIZE=200       # 200 bp → 500 windows

SKIP_INSTALL=false
SKIP_SIM=false
SKIP_PREPROCESS=false
SKIP_TRAIN=false
SKIP_FIGURES=false

# ── Parse arguments ─────────────────────────────────────────────────────
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
        --skip-figures)   SKIP_FIGURES=true; shift;;
        -h|--help)
            echo "Usage: bash run_experiment.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gpus '0'           GPU IDs (default: 0)"
            echo "  --epochs N           Training epochs (default: 30)"
            echo "  --num-ts N           Tree sequences per sample size (default: 200)"
            echo "  --batch-size N       Training batch size (default: 128)"
            echo "  --skip-install       Skip installation step"
            echo "  --skip-sim           Skip simulation step"
            echo "  --skip-preprocess    Skip preprocessing step"
            echo "  --skip-train         Skip training step"
            echo "  --skip-figures       Skip figure generation"
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

# ── Auto-detect CUDA toolkit ────────────────────────────────────────────
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

# ── Step 0: Install ─────────────────────────────────────────────────────
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

# ── Step 1: Simulate AnoGam ─────────────────────────────────────────────
if [ "$SKIP_SIM" = false ]; then
    step "Step 1: Simulate AnoGam (Anopheles gambiae) training data"
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

# ── Step 2: Preprocess with tsinfer trees ───────────────────────────────
if [ "$SKIP_PREPROCESS" = false ]; then
    step "Step 2: Preprocess (SFS + tsinfer tree topology features)"
    T0=$SECONDS
    fastcxt-preprocess \
        --base-dir "$SIMS_DIR" \
        --out-subdir processed_tsinfer \
        --window-size "$WINDOW_SIZE" \
        --sequence-length "$SEQ_LEN" \
        --num-pairs 200 \
        --extract-tsinfer-trees \
        --max-samples "$MAX_SAMPLES" \
        --num-workers "$NUM_WORKERS"
    echo "  Preprocessing done in $(elapsed $((SECONDS - T0)))"
fi

# ── Step 3: Train tree-augmented (tsinfer) model ────────────────────────
if [ "$SKIP_TRAIN" = false ]; then
    step "Step 3: Train tree-augmented (tsinfer) model (${EPOCHS} epochs)"
    T0=$SECONDS
    fastcxt-train \
        --model base_trees \
        --dataset-path "$SIMS_DIR/processed_tsinfer" \
        --gpus $GPUS \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --grad-accum "$GRAD_ACCUM" \
        --workers "$NUM_WORKERS" \
        --log-dir "$SCRIPT_DIR/tsinfer_logs"
    echo "  Training done in $(elapsed $((SECONDS - T0)))"
fi

# ── Step 4: Generate evaluation figures ─────────────────────────────────
if [ "$SKIP_FIGURES" = false ]; then
    step "Step 4: Generate evaluation figures"
    cd "$SCRIPT_DIR"

    CKPT=$(find tsinfer_logs -name "*.ckpt" -type f 2>/dev/null | sort | tail -1 || true)
    if [ -n "$CKPT" ]; then
        python generate_figures.py \
            --checkpoint "$CKPT" \
            --sims-dir "$SIMS_DIR" \
            --out-dir paper_figures
    else
        echo "  WARNING: No checkpoint found, skipping figures"
    fi
fi

# ── Done ────────────────────────────────────────────────────────────────
step "COMPLETE"
echo "  Total time: $(elapsed $((SECONDS - TOTAL_START)))"
echo ""
echo "  Outputs:"
echo "    Checkpoint:  $SCRIPT_DIR/tsinfer_logs/"
echo "    Figures:     $SCRIPT_DIR/paper_figures/"
echo ""
