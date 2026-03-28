#!/bin/bash
# Run tsinfer+tsdate validation on poppy
# Usage: screen -dmS tsdate_val bash experiment_burkina_faso/validation/run_validation.sh

set -euo pipefail
cd /sietch_colab/kkor/fastcxt

echo "=== Installing latest tsinfer + tsdate ==="
/home/kkor/.pixi/bin/pixi run pip install --upgrade tsinfer tsdate cyvcf2

echo "=== Running validation ==="
/home/kkor/.pixi/bin/pixi run python -u experiment_burkina_faso/validation/run_tsdate_comparison.py \
    --vcf-dir /sietch_colab/data_share/Ag1000G/Ag3.0/vcf/phased_vcf/gamb \
    --meta /sietch_colab/data_share/Ag1000G/Ag3.0/args_trees/gamb.meta.tsinfer.csv \
    --accessibility /sietch_colab/data_share/Ag1000G/Ag3.0/vcf/agp3.is_accessible.txt.npz \
    --fastcxt-results /sietch_colab/kkor/fastcxt/results/het_allpop \
    --out-dir /sietch_colab/kkor/fastcxt/results/tsdate_validation \
    2>&1 | tee /sietch_colab/kkor/fastcxt/results/tsdate_validation/run.log

echo "=== Done ==="
