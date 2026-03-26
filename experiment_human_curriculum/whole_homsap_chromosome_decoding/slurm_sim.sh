#!/usr/bin/env bash
#SBATCH --job-name=homsap-curriculum-sim
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=50
#SBATCH --mem=128G
#SBATCH --output=/vast/projects/smathi/cohort/kkor/homsap_curriculum/sim_stage%x_%j.log
set -euo pipefail

# Usage:
#   sbatch --export=STAGE=1 slurm_sim.sh
#   sbatch --export=STAGE=3 --time=4-00:00:00 slurm_sim.sh

if [ -z "${STAGE:-}" ]; then
    echo "ERROR: STAGE not set. Use: sbatch --export=STAGE=1 slurm_sim.sh"
    exit 1
fi

# Stage 1-2: 2 days, Stage 3-4: 4 days (set via --time on sbatch if needed)
# Default: #SBATCH --time=2-00:00:00

# Redirect caches to project storage (home quota is full)
export TRITON_CACHE_DIR="/vast/projects/smathi/cohort/kkor/homsap_curriculum/.triton_cache"
export XDG_CACHE_HOME="/vast/projects/smathi/cohort/kkor/homsap_curriculum/.cache"
mkdir -p "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/run_sim.sh" --stage "$STAGE"
