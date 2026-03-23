#!/usr/bin/env bash
#SBATCH --job-name=homsap-curriculum-train
#SBATCH --partition=dgx-b200
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=4-00:00:00
#SBATCH --output=/vast/projects/smathi/cohort/kkor/homsap_curriculum/train_stage%x_%j.log
set -euo pipefail

# Usage:
#   sbatch --export=STAGE=1 slurm_train.sh
#   sbatch --export=STAGE=2,CHECKPOINT=/path/to/ckpt slurm_train.sh

if [ -z "${STAGE:-}" ]; then
    echo "ERROR: STAGE not set. Use: sbatch --export=STAGE=1 slurm_train.sh"
    exit 1
fi

# Redirect ALL caches to project storage (home quota is full)
CACHE_BASE="/vast/projects/smathi/cohort/kkor/homsap_curriculum/.cache"
export UV_CACHE_DIR="$CACHE_BASE/uv"
export PIP_CACHE_DIR="$CACHE_BASE/pip"
export TRITON_HOME="$CACHE_BASE/triton_home"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_BASE/inductor"
export HF_HOME="$CACHE_BASE/hf"
export XDG_CACHE_HOME="$CACHE_BASE/xdg"
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TRITON_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$HF_HOME" "$XDG_CACHE_HOME"

# Hardcoded paths (BASH_SOURCE is unreliable in SLURM spool)
REPO_DIR="/vast/projects/smathi/cohort/kkor/fastcxt_repo"
SCRIPT_DIR="$REPO_DIR/experiment_human/whole_homsap_chromosome_decoding"

export VENV_DIR="/vast/projects/smathi/cohort/kkor/fastcxt_repo_old/.venv"

TRAIN_ARGS=(--stage "$STAGE" --gpus "0 1 2 3")
if [ -n "${CHECKPOINT:-}" ]; then
    TRAIN_ARGS+=(--checkpoint "$CHECKPOINT")
fi

bash "$SCRIPT_DIR/run_train.sh" "${TRAIN_ARGS[@]}"
