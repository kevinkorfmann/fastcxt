#!/bin/bash
# Re-infer focal regions from VCF with latest tsinfer + tsdate
# Usage: screen -dmS tsdate_val bash experiment_burkina_faso/validation/run_validation.sh

set -euo pipefail
cd /sietch_colab/kkor/fastcxt

echo "=== Installing latest tsinfer + tsdate ==="
/home/kkor/.pixi/bin/pixi run pip install --upgrade tsinfer tsdate

echo "=== Subsetting VCFs with tabix ==="
VCFDIR=/sietch_colab/data_share/Ag1000G/Ag3.0/vcf/phased_vcf/gamb
OUTDIR=/sietch_colab/kkor/fastcxt/results/tsdate_validation
mkdir -p $OUTDIR/vcf_subsets

# Vgsc/kdr: 2L:1.9-2.9 Mb
tabix -h $VCFDIR/gamb.2L.phased.n1470.derived.vcf.gz 2L:1900000-2900000 | bgzip > $OUTDIR/vcf_subsets/2L_Vgsc_kdr.vcf.gz && tabix -p vcf $OUTDIR/vcf_subsets/2L_Vgsc_kdr.vcf.gz
echo "  Vgsc_kdr subset done"

# Rdl: 2L:24.9-25.9 Mb
tabix -h $VCFDIR/gamb.2L.phased.n1470.derived.vcf.gz 2L:24900000-25900000 | bgzip > $OUTDIR/vcf_subsets/2L_Rdl.vcf.gz && tabix -p vcf $OUTDIR/vcf_subsets/2L_Rdl.vcf.gz
echo "  Rdl subset done"

# Neutral: 3L:20-21 Mb
tabix -h $VCFDIR/gamb.3L.phased.n1470.derived.vcf.gz 3L:20000000-21000000 | bgzip > $OUTDIR/vcf_subsets/3L_Neutral.vcf.gz && tabix -p vcf $OUTDIR/vcf_subsets/3L_Neutral.vcf.gz
echo "  Neutral subset done"

echo "=== Running validation ==="
/home/kkor/.pixi/bin/pixi run python -u experiment_burkina_faso/validation/run_tsdate_comparison.py \
    --vcf-dir $OUTDIR/vcf_subsets \
    --meta /sietch_colab/data_share/Ag1000G/Ag3.0/args_trees/gamb.meta.tsinfer.csv \
    --fastcxt-results /sietch_colab/kkor/fastcxt/results/het_allpop \
    --out-dir $OUTDIR \
    2>&1 | tee $OUTDIR/run.log

echo "=== Done ==="
