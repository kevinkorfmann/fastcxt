#!/bin/bash
# Run SMC++ on Ag3 phased VCF for Burkina Faso — chromosome 3L
# Uses apptainer with the smcpp container
# Compares to fastcxt IICR estimates

set -euo pipefail

WORKDIR=/sietch_colab/kkor/fastcxt/results/smcpp_burkina_faso
VCF=/sietch_colab/data_share/Ag1000G/Ag3.0/vcf/phased_vcf/gamb/gamb.3L.phased.n1470.derived.vcf.gz
META=/sietch_colab/data_share/Ag1000G/Ag3.0/args_trees/gamb.meta.tsinfer.csv
CHROM=3L
MU=3.5e-9
CORES=32

mkdir -p $WORKDIR/smc_input $WORKDIR/output

# Pull SMC++ container if not present
SIF=$WORKDIR/smcpp_latest.sif
if [ ! -f "$SIF" ]; then
    echo "Pulling SMC++ container..."
    apptainer pull $SIF docker://terhorst/smcpp:latest
fi

# Get Burkina Faso sample IDs from metadata
echo "Extracting Burkina Faso samples..."
python3 << 'PYEOF'
import pandas as pd
meta = pd.read_csv("/sietch_colab/data_share/Ag1000G/Ag3.0/args_trees/gamb.meta.tsinfer.csv")
bf = meta[meta["country"] == "Burkina Faso"]
# Use first 20 samples (40 haploids) to keep runtime reasonable
samples = bf["sample_id"].values[:20]
with open("/sietch_colab/kkor/fastcxt/results/smcpp_burkina_faso/samples.txt", "w") as f:
    f.write(",".join(samples))
print(f"Selected {len(samples)} Burkina Faso samples")
for s in samples[:5]:
    print(f"  {s}")
print(f"  ...")
PYEOF

SAMPLES=$(cat $WORKDIR/samples.txt)
echo "Samples: $SAMPLES"

# Generate accessibility mask BED for SMC++ if not present
MASK=$WORKDIR/accessibility_${CHROM}.bed.gz
if [ ! -f "$MASK" ]; then
    echo "Generating accessibility mask BED for $CHROM..."
    python3 << MASKEOF
import numpy as np
acc = np.load("/sietch_colab/data_share/Ag1000G/Ag3.0/vcf/agp3.is_accessible.txt.npz")
mask = acc[f"access_${CHROM}"]
# Find contiguous accessible regions using diff
changes = np.diff(mask.astype(np.int8))
starts = np.where(changes == 1)[0] + 1  # 0->1 transitions
ends = np.where(changes == -1)[0] + 1    # 1->0 transitions
# Handle edge cases
if mask[0]:
    starts = np.concatenate([[0], starts])
if mask[-1]:
    ends = np.concatenate([ends, [len(mask)]])
with open("${MASK%.gz}", "w") as f:
    for s, e in zip(starts, ends):
        f.write(f"${CHROM}\t{s}\t{e}\n")
print(f"Written {len(starts)} accessible regions covering {sum(mask)} bases")
MASKEOF
    # Compress with bgzip and index with tabix
    BGZIP=/usr/local/bin/bgzip
    TABIX=/usr/local/bin/tabix
    $BGZIP -f ${MASK%.gz}
    $TABIX -p bed $MASK
    echo "Mask indexed: $MASK"
fi

# Step 1: vcf2smc — create one file per distinguished pair
# Use first 5 individuals as distinguished lineages for better resolution
echo "Running vcf2smc..."
IFS=',' read -ra SAMPLE_ARR <<< "$SAMPLES"
for i in 0 1 2; do
    if [ -f "$WORKDIR/smc_input/pair_${i}.smc.gz" ]; then
        echo "  Pair $i already done, skipping"
        continue
    fi
    DIST=${SAMPLE_ARR[$i]}
    echo "  Distinguished individual: $DIST"
    apptainer run --bind /sietch_colab $SIF vcf2smc \
        -d $DIST $DIST \
        --mask $MASK \
        $VCF \
        $WORKDIR/smc_input/pair_${i}.smc.gz \
        $CHROM \
        "BF:$SAMPLES"
done

# Step 2: estimate — fit demographic model
echo "Running SMC++ estimate (this takes a while)..."
apptainer run --bind /sietch_colab $SIF estimate \
    --cores $CORES \
    --knots 24 \
    --timepoints 1e2 1e7 \
    -o $WORKDIR/output \
    $MU \
    $WORKDIR/smc_input/pair_*.smc.gz

echo "SMC++ estimation complete!"
echo "Model: $WORKDIR/output/model.final.json"

# Step 3: Plot (SMC++ built-in)
apptainer run --bind /sietch_colab $SIF plot \
    --csv \
    $WORKDIR/output/smcpp_plot.png \
    $WORKDIR/output/model.final.json

# Step 4: Extract Ne(t) curve for comparison
echo "Extracting Ne(t) curve..."
python3 << 'PYEOF2'
import json, numpy as np

with open("/sietch_colab/kkor/fastcxt/results/smcpp_burkina_faso/output/model.final.json") as f:
    model = json.load(f)

# SMC++ model.final.json contains 'model' with 'knots' (time, size pairs)
# The CSV from smc++ plot is easier to parse
import csv
csv_path = "/sietch_colab/kkor/fastcxt/results/smcpp_burkina_faso/output/smcpp_plot.csv"
try:
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    x = np.array([float(r["x"]) for r in rows])  # generations
    y = np.array([float(r["y"]) for r in rows])  # Ne
    np.savez("/sietch_colab/kkor/fastcxt/results/smcpp_burkina_faso/smcpp_ne.npz",
             generations=x, ne=y)
    print(f"Saved Ne(t) curve: {len(x)} points, Ne range [{y.min():.0f}, {y.max():.0f}]")
except Exception as e:
    print(f"CSV extraction failed: {e}, trying JSON model directly")
    # Fallback: parse the JSON model
    knots = model.get("model", {})
    print(f"Model keys: {list(model.keys())}")
    print(json.dumps(model, indent=2)[:500])

echo "Done! Results in $WORKDIR/"
PYEOF2
