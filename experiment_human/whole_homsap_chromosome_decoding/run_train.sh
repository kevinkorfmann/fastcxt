#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Whole HomSap Chromosome Decoding — Training
# ============================================================================
# Usage: bash run_train.sh --stage N [--checkpoint PATH]
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
WORK_DIR="/vast/projects/smathi/cohort/kkor/homsap_curriculum"

GPUS="0"
EPOCHS=200
BATCH_SIZE=4
GRAD_ACCUM=64
NUM_WORKERS=8
STAGE=""
CHECKPOINT=""

# -- Parse arguments ---------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --stage)       STAGE="$2"; shift 2;;
        --checkpoint)  CHECKPOINT="$2"; shift 2;;
        --gpus)        GPUS="$2"; shift 2;;
        --epochs)      EPOCHS="$2"; shift 2;;
        --batch-size)  BATCH_SIZE="$2"; shift 2;;
        --grad-accum)  GRAD_ACCUM="$2"; shift 2;;
        -h|--help)
            echo "Usage: bash run_train.sh --stage N [--checkpoint PATH] [--gpus G] [--epochs E]"
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
    1) PRESET=base_25k;;
    2) PRESET=base_50k;;
    3) PRESET=base_75k;;
    4) PRESET=base_125k;;
    *) echo "ERROR: Invalid stage $STAGE (must be 1-4)"; exit 1;;
esac

STAGE_DIR="$WORK_DIR/stage${STAGE}"
DATASET_PATH="$STAGE_DIR/sims/processed_lazy"
LOG_DIR="$STAGE_DIR/logs"

# -- Auto-find previous stage checkpoint if not given ------------------------
if [ -z "$CHECKPOINT" ] && [ "$STAGE" -gt 1 ]; then
    PREV_STAGE=$((STAGE - 1))
    PREV_CKPT_DIR="$WORK_DIR/stage${PREV_STAGE}/logs"
    if [ -d "$PREV_CKPT_DIR" ]; then
        CHECKPOINT=$(find "$PREV_CKPT_DIR" -name "*.ckpt" -type f | sort -t= -k2 -n | tail -1 || true)
        if [ -n "$CHECKPOINT" ]; then
            echo "Auto-found previous stage checkpoint: $CHECKPOINT"
        else
            echo "WARNING: No checkpoint found in $PREV_CKPT_DIR"
        fi
    fi
fi

echo "========================================================================"
echo "  Stage $STAGE: model=$PRESET"
echo "  Dataset: $DATASET_PATH"
echo "  Checkpoint: ${CHECKPOINT:-none (training from scratch)}"
echo "========================================================================"

# -- Redirect caches to project storage (home quota is full) ----------------
export TRITON_CACHE_DIR="$WORK_DIR/.triton_cache"
export XDG_CACHE_HOME="$WORK_DIR/.cache"
mkdir -p "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

# -- Auto-detect CUDA toolkit -----------------------------------------------
if [ -z "${CUDA_HOME:-}" ]; then
    for candidate in \
        /usr/local/cuda \
        /usr/local/cuda-12 \
        /usr/local/cuda-12.*; do
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

# -- Activate venv -----------------------------------------------------------
if [ -f "$REPO_DIR/.venv/bin/activate" ]; then
    source "$REPO_DIR/.venv/bin/activate"
fi

mkdir -p "$LOG_DIR"

# -- Build training command --------------------------------------------------
TRAIN_CMD=(
    fastcxt-train
    --model "$PRESET"
    --dataset-path "$DATASET_PATH"
    --gpus $GPUS
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --grad-accum "$GRAD_ACCUM"
    --workers "$NUM_WORKERS"
    --log-dir "$LOG_DIR"
    --dataset-mode lazy
    --csv-log
    --save-every-epoch
    --early-stopping
)

if [ -n "$CHECKPOINT" ]; then
    TRAIN_CMD+=(--checkpoint "$CHECKPOINT")
fi

echo "Running: ${TRAIN_CMD[*]}"
"${TRAIN_CMD[@]}"

echo ""
echo "======== TRAINING COMPLETE (stage $STAGE) ========"
echo "  Logs: $LOG_DIR/"
