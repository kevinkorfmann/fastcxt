#!/usr/bin/env bash
#SBATCH --job-name=fastcxt-1mb-anogam
#SBATCH --partition=dgx-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=/vast/projects/smathi/cohort/fastcxt_1mb_context/train_%j.log
set -euo pipefail

bash /vast/projects/smathi/cohort/fastcxt_repo/experiment-1mb-context/run_betty.sh "$@"
