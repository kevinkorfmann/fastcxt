#!/usr/bin/env bash
#SBATCH --job-name=homsap-curriculum-train
#SBATCH --partition=dgx-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --output=/vast/projects/smathi/cohort/kkor/homsap_curriculum/train_stage%x_%j.log
set -euo pipefail

# Usage:
#   sbatch --export=STAGE=1 slurm_train.sh
#   sbatch --export=STAGE=2,CHECKPOINT=/path/to/ckpt slurm_train.sh

if [ -z "${STAGE:-}" ]; then
    echo "ERROR: STAGE not set. Use: sbatch --export=STAGE=1 slurm_train.sh"
    exit 1
fi

# Redirect caches to project storage (home quota is full)
export TRITON_CACHE_DIR="/vast/projects/smathi/cohort/kkor/homsap_curriculum/.triton_cache"
export XDG_CACHE_HOME="/vast/projects/smathi/cohort/kkor/homsap_curriculum/.cache"
mkdir -p "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRAIN_ARGS=(--stage "$STAGE")
if [ -n "${CHECKPOINT:-}" ]; then
    TRAIN_ARGS+=(--checkpoint "$CHECKPOINT")
fi

bash "$SCRIPT_DIR/run_train.sh" "${TRAIN_ARGS[@]}"
