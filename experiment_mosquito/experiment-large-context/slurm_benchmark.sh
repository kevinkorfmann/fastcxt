#!/usr/bin/env bash
#SBATCH --job-name=fastcxt-bench-ctx
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=benchmark_context_%j.log
set -euo pipefail

# ============================================================================
# Benchmark AnoGam tsinfer models: 500 vs 2000 windows
# ============================================================================
# Usage:
#   sbatch experiment-large-context/slurm_benchmark.sh
#   # OR directly:
#   bash experiment-large-context/slurm_benchmark.sh

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Activate venv
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

# --- Checkpoint paths (adjust as needed) ---
CKPT_500="${CKPT_500:-experiment_mossies/tsinfer_logs/lightning_logs/version_0/checkpoints/epoch=29-step=20880.ckpt}"
CKPT_2000="${CKPT_2000:-/vast/projects/smathi/cohort/fastcxt_large_context/tsinfer_2k_logs/lightning_logs/version_0/checkpoints/epoch=29-step=20880.ckpt}"

echo "500-win checkpoint:  $CKPT_500"
echo "2000-win checkpoint: $CKPT_2000"
echo "Device: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")')"
echo ""

python experiment/benchmark_context_length.py \
    --ckpt-500  "$CKPT_500" \
    --ckpt-2000 "$CKPT_2000" \
    --n-samples 50 \
    --n-pairs 3 \
    --repeats 5 \
    --warmup 2

echo ""
echo "Done. Results in experiment/paper_figures/table_context_benchmark.csv"
