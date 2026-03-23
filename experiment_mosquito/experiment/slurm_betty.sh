#!/usr/bin/env bash
#SBATCH --job-name=fastcxt-mosquito-exp
#SBATCH --partition=dgx-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/vast/projects/smathi/cohort/kkor/fastcxt_repo/experiment_mosquito/experiment/slurm_%j.log
set -euo pipefail

# Redirect caches
CACHE_BASE="/vast/projects/smathi/cohort/kkor/homsap_curriculum/.cache"
export UV_CACHE_DIR="$CACHE_BASE/uv"
export PIP_CACHE_DIR="$CACHE_BASE/pip"
export TRITON_HOME="$CACHE_BASE/triton_home"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_BASE/inductor"
export HF_HOME="$CACHE_BASE/hf"
export XDG_CACHE_HOME="$CACHE_BASE/xdg"
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TRITON_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$HF_HOME" "$XDG_CACHE_HOME"

module load cuda/12.8.1 2>/dev/null || true

REPO_DIR="/vast/projects/smathi/cohort/kkor/fastcxt_repo"
VENV_DIR="$REPO_DIR/.venv"
export PATH="$VENV_DIR/bin:$PATH"

# Install CUDA packages if missing
if ! "$VENV_DIR/bin/python" -c "import mamba_ssm" 2>/dev/null; then
    echo "Installing CUDA packages..."
    uv pip install causal-conv1d mamba-ssm tsinfer -e "$REPO_DIR[sim,trees]" --cache-dir "$UV_CACHE_DIR"
fi

FROM="${FROM:-01}"
GPUS=0 bash "$REPO_DIR/experiment_mosquito/experiment/run_all.sh" --from "$FROM"
