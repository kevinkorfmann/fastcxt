#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# fastcxt end-to-end experiment
# ============================================================================
# Runs the full pipeline from installation to scaling comparison figures.
#
# Usage:
#   bash run_experiment.sh                  # defaults: 3 GPUs, 20+50 epochs
#   bash run_experiment.sh --gpus 0         # single GPU
#   bash run_experiment.sh --skip-sim       # skip simulation (already done)
#   bash run_experiment.sh --skip-train     # skip training (already done)
#
# Requirements:
#   - CUDA-capable GPU(s)
#   - uv (https://docs.astral.sh/uv/)
#   - ~10 GB disk for simulations + checkpoints
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
SIMS_DIR="$SCRIPT_DIR/sims"

# ── Defaults ────────────────────────────────────────────────────────────
GPUS="0 1 2"
PAIRWISE_EPOCHS=30
NODE_EPOCHS=30
NUM_TS_PER_SIZE=200
SAMPLE_SIZES="10 25 50 100 200"
MAX_SAMPLES=200
BATCH_SIZE=128
GRAD_ACCUM=2
NUM_WORKERS=8
NUM_SIM_PROCS=8
BENCHMARK_GPU=0

SKIP_INSTALL=false
SKIP_SIM=false
SKIP_PREPROCESS=false
SKIP_TRAIN=false
SKIP_BENCHMARK=false

# ── Parse arguments ─────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)           GPUS="$2"; shift 2;;
        --pairwise-epochs) PAIRWISE_EPOCHS="$2"; shift 2;;
        --node-epochs)    NODE_EPOCHS="$2"; shift 2;;
        --num-ts)         NUM_TS_PER_SIZE="$2"; shift 2;;
        --batch-size)     BATCH_SIZE="$2"; shift 2;;
        --benchmark-gpu)  BENCHMARK_GPU="$2"; shift 2;;
        --skip-install)   SKIP_INSTALL=true; shift;;
        --skip-sim)       SKIP_SIM=true; shift;;
        --skip-preprocess) SKIP_PREPROCESS=true; shift;;
        --skip-train)     SKIP_TRAIN=true; shift;;
        --skip-benchmark) SKIP_BENCHMARK=true; shift;;
        -h|--help)
            echo "Usage: bash run_experiment.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gpus '0 1 2'       GPU IDs for pairwise training (default: 0 1 2)"
            echo "  --pairwise-epochs N  Pairwise model epochs (default: 20)"
            echo "  --node-epochs N      Node-time model epochs (default: 50)"
            echo "  --num-ts N           Tree sequences per sample size (default: 200)"
            echo "  --batch-size N       Training batch size (default: 128)"
            echo "  --benchmark-gpu N    Single GPU for benchmark (default: 0)"
            echo "  --skip-install       Skip installation step"
            echo "  --skip-sim           Skip simulation step"
            echo "  --skip-preprocess    Skip preprocessing step"
            echo "  --skip-train         Skip both training steps"
            echo "  --skip-benchmark     Skip benchmark + plotting"
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

# ── Step 0: Install ─────────────────────────────────────────────────────
if [ "$SKIP_INSTALL" = false ]; then
    step "Step 0: Install fastcxt"
    cd "$REPO_DIR"
    if [ ! -d ".venv" ]; then
        uv venv
    fi
    source .venv/bin/activate
    uv pip install -e ".[all]"
fi

# Ensure venv is active for subsequent steps
if [ -f "$REPO_DIR/.venv/bin/activate" ]; then
    source "$REPO_DIR/.venv/bin/activate"
fi

# ── Step 1: Simulate ────────────────────────────────────────────────────
if [ "$SKIP_SIM" = false ]; then
    step "Step 1: Simulate training data (variable sample sizes)"
    T0=$SECONDS
    for N in $SAMPLE_SIZES; do
        echo "  Simulating n_samples=$N ($NUM_TS_PER_SIZE tree sequences) ..."
        fastcxt-simulate \
            --scenario constant \
            --data-dir "$SIMS_DIR/n${N}" \
            --num-ts "$NUM_TS_PER_SIZE" \
            --n-samples "$N" \
            --sequence-length 1e6 \
            --num-processes "$NUM_SIM_PROCS"
    done
    echo "  Simulation done in $(elapsed $((SECONDS - T0)))"
fi

# ── Step 2: Preprocess ──────────────────────────────────────────────────
if [ "$SKIP_PREPROCESS" = false ]; then
    step "Step 2: Preprocess (SFS + tree topology features)"
    T0=$SECONDS
    fastcxt-preprocess \
        --base-dir "$SIMS_DIR" \
        --out-subdir processed \
        --window-size 2000 \
        --sequence-length 1000000 \
        --num-pairs 200 \
        --extract-trees \
        --max-samples "$MAX_SAMPLES" \
        --num-workers "$NUM_WORKERS"
    echo "  Preprocessing done in $(elapsed $((SECONDS - T0)))"
fi

# ── Step 3a: Train pairwise model ───────────────────────────────────────
if [ "$SKIP_TRAIN" = false ]; then
    step "Step 3a: Train pairwise model (base, ${PAIRWISE_EPOCHS} epochs)"
    T0=$SECONDS
    fastcxt-train \
        --model base \
        --dataset-path "$SIMS_DIR/processed" \
        --gpus $GPUS \
        --epochs "$PAIRWISE_EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --grad-accum "$GRAD_ACCUM" \
        --workers "$NUM_WORKERS" \
        --log-dir "$SCRIPT_DIR"
    echo "  Pairwise training done in $(elapsed $((SECONDS - T0)))"

# ── Step 3b: Train node-time model ──────────────────────────────────────
    step "Step 3b: Train node-time model (${NODE_EPOCHS} epochs)"
    T0=$SECONDS
    cd "$SCRIPT_DIR"
    CUDA_VISIBLE_DEVICES="$BENCHMARK_GPU" python train_and_benchmark_node_times.py \
        --sims-dir "$SIMS_DIR" \
        --max-samples "$MAX_SAMPLES" \
        --epochs "$NODE_EPOCHS" \
        --skip-benchmark
    echo "  Node-time training done in $(elapsed $((SECONDS - T0)))"
fi

# ── Step 4: Plot pairwise model figures ─────────────────────────────────
if [ "$SKIP_TRAIN" = false ]; then
    step "Step 4: Plot pairwise model evaluation figures"
    cd "$SCRIPT_DIR"

    CKPT=$(find lightning_logs -name "*.ckpt" -type f 2>/dev/null | sort | tail -1 || true)
    if [ -n "$CKPT" ]; then
        python plot_experiment_figures.py \
            --checkpoint "$CKPT" \
            --out-dir figures
    else
        echo "  WARNING: No checkpoint found, skipping pairwise figures"
    fi
fi

# ── Step 5: Benchmark scaling + figures ─────────────────────────────────
if [ "$SKIP_BENCHMARK" = false ]; then
    step "Step 5: Benchmark pairwise vs node-time scaling"
    cd "$SCRIPT_DIR"
    T0=$SECONDS

    CKPT=$(find lightning_logs -name "*.ckpt" -type f 2>/dev/null | sort | tail -1 || true)
    if [ -z "$CKPT" ]; then
        echo "  ERROR: No pairwise checkpoint found. Cannot benchmark."
        exit 1
    fi

    CUDA_VISIBLE_DEVICES="$BENCHMARK_GPU" python benchmark_scaling.py \
        --base-checkpoint "$CKPT" \
        --node-weights node_time_model.pt \
        --max-samples "$MAX_SAMPLES" \
        --out-csv benchmark_results.csv
    echo "  Benchmark done in $(elapsed $((SECONDS - T0)))"

    step "Step 6: Generate scaling comparison figures"
    python plot_scaling.py \
        --csv benchmark_results.csv \
        --out-dir figures
fi

# ── Done ────────────────────────────────────────────────────────────────
step "COMPLETE"
echo "  Total time: $(elapsed $((SECONDS - TOTAL_START)))"
echo ""
echo "  Outputs:"
echo "    Pairwise checkpoint:  $SCRIPT_DIR/lightning_logs/"
echo "    Node-time weights:    $SCRIPT_DIR/node_time_model.pt"
echo "    Benchmark CSV:        $SCRIPT_DIR/benchmark_results.csv"
echo "    Figures:              $SCRIPT_DIR/figures/"
echo ""
echo "  Key figures:"
echo "    figures/01_true_vs_predicted.png   True vs predicted scatter"
echo "    figures/02_tmrca_along_genome.png  TMRCA along the genome"
echo "    figures/scaling_combined.png       Runtime + speedup comparison"
echo ""
