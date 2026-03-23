#!/usr/bin/env bash
#SBATCH --job-name=fastcxt-1mb-figs
#SBATCH --partition=b200-mig90
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/vast/projects/smathi/cohort/fastcxt_1mb_context/figures_%j.log
set -euo pipefail

REPO=/vast/projects/smathi/cohort/fastcxt_repo
WORK=/vast/projects/smathi/cohort/fastcxt_1mb_context

# Cache redirects (home quota full)
export CACHE_BASE=$WORK/.cache
export UV_CACHE_DIR=$CACHE_BASE/uv
export PIP_CACHE_DIR=$CACHE_BASE/pip
export TRITON_HOME=$CACHE_BASE/triton_home
export TORCHINDUCTOR_CACHE_DIR=$CACHE_BASE/inductor
export HF_HOME=$CACHE_BASE/hf
export XDG_CACHE_HOME=$CACHE_BASE/xdg
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TRITON_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$HF_HOME" "$XDG_CACHE_HOME"

source "$REPO/.venv/bin/activate"

CKPT="$WORK/tsinfer_5k_logs/csv_metrics/version_0/checkpoints/epoch=50-step=35547.ckpt"
SIMS="$WORK/sims"
OUT="$WORK/paper_figures"

python "$REPO/experiment-1mb-context/generate_figures.py" \
    --checkpoint "$CKPT" \
    --sims-dir "$SIMS" \
    --out-dir "$OUT"
