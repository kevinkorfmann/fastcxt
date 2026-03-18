#!/bin/bash
#SBATCH --job-name=fastcxt-2k
#SBATCH --partition=b200-mig90
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:90gb:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/vast/projects/smathi/cohort/fastcxt_large_context/slurm_%j.out
#SBATCH --error=/vast/projects/smathi/cohort/fastcxt_large_context/slurm_%j.err

# ============================================================================
# SLURM job script for fastcxt large-context experiment (2000 windows)
# Submit from the repo root: sbatch experiment-large-context/slurm_betty.sh
# ============================================================================

set -euo pipefail

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Load modules
module load cuda/12.8.1

# Redirect caches away from home (home quota exceeded)
export CACHE_BASE=/vast/projects/smathi/cohort/fastcxt_large_context/.cache
export UV_CACHE_DIR=$CACHE_BASE/uv
export PIP_CACHE_DIR=$CACHE_BASE/pip
export TRITON_HOME=$CACHE_BASE/triton_home
export TORCHINDUCTOR_CACHE_DIR=$CACHE_BASE/inductor
export HF_HOME=$CACHE_BASE/hf
export XDG_CACHE_HOME=$CACHE_BASE/xdg
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TRITON_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$HF_HOME" "$XDG_CACHE_HOME"

# Create output directory
mkdir -p /vast/projects/smathi/cohort/fastcxt_large_context

# Run experiment
REPO_DIR="/vast/projects/smathi/cohort/fastcxt_repo"
bash "$REPO_DIR/experiment-large-context/run_betty.sh" --gpus 0 --skip-sim --skip-preprocess --skip-install

echo "Job $SLURM_JOB_ID finished at $(date)"
