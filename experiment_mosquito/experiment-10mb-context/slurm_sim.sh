#!/usr/bin/env bash
#SBATCH --job-name=fastcxt-10mb-sim
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=50
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/vast/projects/smathi/cohort/fastcxt_10mb_context/sim_%j.log
set -euo pipefail

# Usage: sbatch --export=N_SAMPLES=10 slurm_sim.sh
# Or submit all 5 at once:
#   for N in 10 25 50 100 200; do
#     sbatch --export=N_SAMPLES=$N --job-name=fastcxt-10mb-n${N} slurm_sim.sh
#   done

if [ -z "${N_SAMPLES:-}" ]; then
    echo "ERROR: N_SAMPLES not set. Use: sbatch --export=N_SAMPLES=10 slurm_sim.sh"
    exit 1
fi

# Redirect caches to project storage (home quota is full)
export TRITON_CACHE_DIR="/vast/projects/smathi/cohort/fastcxt_10mb_context/.triton_cache"
export XDG_CACHE_HOME="/vast/projects/smathi/cohort/fastcxt_10mb_context/.cache"
mkdir -p "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

bash /vast/projects/smathi/cohort/fastcxt_repo/experiment-10mb-context/run_betty_sim.sh \
    --n-samples "$N_SAMPLES"
