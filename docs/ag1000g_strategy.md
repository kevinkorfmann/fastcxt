# Ag1000G Analysis Strategy

**End-to-end protocol for analysing ~1000 genomes of *Anopheles gambiae*
with fastcxt: from simulation-based training to genome-wide TimeAtlas
construction and figure generation.**

---

## Overview

| Stage | What | Wall-time estimate | Hardware |
|-------|------|--------------------|----------|
| 0 | Environment setup | 5 min | any |
| 1 | Simulate training data | 2--4 h | 80 CPUs |
| 2 | Preprocess (SFS + targets) | 1--2 h | 80 CPUs |
| 3 | Train fastcxt model | 6--12 h | 3 GPUs |
| 4 | Load Ag1000G real data | 10 min | 1 GPU node |
| 5 | Whole-genome inference | 2--6 h | 1--3 GPUs |
| 6 | Build TimeAtlas | 5 min | CPU |
| 7 | Analysis & plotting | 30 min | CPU + 1 GPU |

Total: **~12--24 hours** on a typical cluster node (80 CPU, 3 A100s).

---

## Data paths (sietch)

All paths are configurable via environment variables.  Defaults point to the
sietch cluster.

```bash
# These are the defaults -- override if your data lives elsewhere
export FASTCXT_BASE_DIR="/sietch_colab/data_share/cxt_scratch"
export AG1000G_DATA_DIR="/sietch_colab/data_share/Ag1000G/Ag3.0/args_trees/tsinfer_data_v2"
export AG1000G_ACCESSIBILITY="/sietch_colab/data_share/Ag1000G/Ag3.0/args_trees/singer/agp3.is_accessible.txt.npz"
```

Verify paths exist:

```bash
python -c "from fastcxt.paths import PATHS; print(PATHS.exists_report())"
```

Expected Ag1000G data layout:

```
$AG1000G_DATA_DIR/
├── 2L.trees        # tsinfer tree sequence, ~1000 haploids, 49.4 Mb
├── 2R.trees        # 61.5 Mb
├── 3L.trees        # 42.0 Mb
├── 3R.trees        # 53.2 Mb
└── X.trees         # 24.4 Mb

$AG1000G_ACCESSIBILITY
└── agp3.is_accessible.txt.npz   # per-bp boolean mask
```

---

## Stage 0: Environment setup

```bash
cd /home/kkor/fastcxt

# Create isolated venv with uv
uv venv .venv --python 3.12
source .venv/bin/activate

# Install fastcxt with all optional deps
uv pip install -e ".[all]"

# Verify
python -c "import fastcxt; print(fastcxt.__version__)"
python -c "from fastcxt.paths import PATHS; print(PATHS.exists_report())"
```

---

## Stage 1: Simulate training data

Train on a diverse mix: AnoGam-specific stdpopsim models, plus broader
demographic scenarios to improve generalisation.  Variable sample sizes
teach the model to handle the ~1000 haploids in Ag1000G.

```bash
BASE="$FASTCXT_BASE_DIR/data"
WORKERS=80

# --- AnoGam (primary): 3000 tree sequences at variable sample sizes ---
for N in 25 50 100; do
    fastcxt-simulate \
        --scenario AnoGam \
        --data-dir "$BASE/anogam_n${N}" \
        --num-ts 1000 \
        --n-samples $N \
        --sequence-length 1000000 \
        --num-processes $WORKERS
done

# --- Constant demography (diverse Ne): 2000 TS ---
fastcxt-simulate \
    --scenario constant \
    --data-dir "$BASE/constant" \
    --num-ts 2000 \
    --n-samples 50 \
    --num-processes $WORKERS

# --- Sawtooth (population size oscillations): 1000 TS ---
fastcxt-simulate \
    --scenario sawtooth \
    --data-dir "$BASE/sawtooth" \
    --num-ts 1000 \
    --n-samples 50 \
    --num-processes $WORKERS

# --- Island model (structure): 1000 TS ---
fastcxt-simulate \
    --scenario island \
    --data-dir "$BASE/island" \
    --num-ts 1000 \
    --n-samples 50 \
    --num-processes $WORKERS

# --- Other stdpopsim species (generalisation): 500 each ---
for SPECIES in HomSap DroMel BosTau PanTro; do
    fastcxt-simulate \
        --scenario $SPECIES \
        --data-dir "$BASE/stdpopsim_${SPECIES}" \
        --num-ts 500 \
        --n-samples 25 \
        --num-processes $WORKERS
done
```

**Total**: ~10,000 tree sequences across 10 scenarios.

---

## Stage 2: Preprocess

Convert simulated `.trees` files into training tensors (SFS features +
log-TMRCA targets), applying the Ag1000G accessibility mask so the model
learns to handle missing data from the start.

```bash
MASK="$AG1000G_ACCESSIBILITY"

# Preprocess all scenarios together (the directory walker finds all .trees)
fastcxt-preprocess \
    --base-dir "$BASE" \
    --out-subdir processed \
    --window-size 2000 \
    --sequence-length 1000000 \
    --num-pairs 200 \
    --train-ratio 0.9 \
    --global-seed 42 \
    --num-workers $WORKERS \
    --accessibility-mask "$MASK" \
    --skip-existing
```

Verify output:

```bash
# Check train/test split counts
find "$BASE/processed/train" -name "X.npy" | wc -l
find "$BASE/processed/test"  -name "X.npy" | wc -l

# Spot-check one sample
python -c "
import numpy as np, json
X = np.load('$BASE/processed/train/default/ts_00000000_i0/X.npy', mmap_mode='r')
y = np.load('$BASE/processed/train/default/ts_00000000_i0/y.npy', mmap_mode='r')
meta = json.load(open('$BASE/processed/train/default/ts_00000000_i0/meta.json'))
print(f'X: {X.shape}, y: {y.shape}, mu_rate: {meta[\"mutation_rate\"]:.2e}')
print(f'Accessibility mask used: {meta[\"has_accessibility_mask\"]}')
"
```

---

## Stage 3: Train the model

```bash
PROCESSED="$BASE/processed"
GPUS="0 1 2"
LOG_DIR="$FASTCXT_BASE_DIR/lightning_logs"

# Train base model (6 enc + 4 dec layers, d_model=256)
fastcxt-train \
    --model base \
    --dataset-path "$PROCESSED" \
    --gpus $GPUS \
    --epochs 10 \
    --lr 3e-4 \
    --batch-size 128 \
    --grad-accum 4 \
    --workers 16 \
    --log-dir "$LOG_DIR"
```

Monitor training:

```bash
# TensorBoard
tensorboard --logdir "$LOG_DIR" --port 6006

# Check latest checkpoint
ls -lhrt "$LOG_DIR"/*/checkpoints/*.ckpt | tail -5
```

Save the best checkpoint path for Stage 5:

```bash
CKPT=$(ls -t "$LOG_DIR"/*/checkpoints/*.ckpt | head -1)
echo "Best checkpoint: $CKPT"
```

---

## Stage 4: Load Ag1000G real data

Load all five autosomal/X chromosome arms from the tsinfer tree sequences.

```python
#!/usr/bin/env python3
"""stage4_load_ag1000g.py -- Load Ag1000G data for all arms."""

import tskit
import numpy as np
from fastcxt.paths import PATHS
from fastcxt.mosquito import AccessibilityMask, ANOGAM_CHROMOSOME_ARMS

ARMS = ["2L", "2R", "3L", "3R", "X"]
OUT_DIR = PATHS.base_dir / "ag1000g_loaded"
OUT_DIR.mkdir(parents=True, exist_ok=True)

for arm in ARMS:
    print(f"\n--- Loading {arm} ---")

    # Load tree sequence
    ts_path = PATHS.ag1000g_arm_trees(arm)
    ts = tskit.load(str(ts_path))
    print(f"  Samples: {ts.num_samples}, Sites: {ts.num_sites}, Trees: {ts.num_trees}")

    # Extract genotype matrix and positions
    gm = ts.genotype_matrix().T   # (n_haploids, n_sites)
    positions = ts.tables.sites.position

    # Save for reuse (avoid re-extracting the genotype matrix each time)
    np.save(OUT_DIR / f"{arm}_gm.npy", gm.astype(np.int8))
    np.save(OUT_DIR / f"{arm}_pos.npy", positions)
    print(f"  Saved: gm {gm.shape}, positions {positions.shape}")

    # Load and report accessibility
    mask = AccessibilityMask.from_npz(PATHS.ag1000g_accessibility_mask, arm)
    print(f"  Accessibility: {mask.accessible_fraction:.1%} of {ANOGAM_CHROMOSOME_ARMS[arm]:,} bp")

print("\nDone. Genotype matrices saved to:", OUT_DIR)
```

```bash
python stage4_load_ag1000g.py
```

---

## Stage 5: Whole-genome inference

Run fastcxt across all chromosome arms for all pairwise combinations.
With ~1000 haploids the number of pairs is ~500,000, so we process in
batches and store results incrementally.

```python
#!/usr/bin/env python3
"""stage5_inference.py -- Whole-genome Ag1000G inference."""

import sys
import time
import numpy as np
import torch
from pathlib import Path

from fastcxt.paths import PATHS
from fastcxt.config import PRESETS
from fastcxt.model import FastCxtModel
from fastcxt.train import LitFastCxt
from fastcxt.mosquito import (
    MosquitoAnalysis, AccessibilityMask,
    ANOGAM_CHROMOSOME_ARMS, generate_blocks,
)
from fastcxt.atlas import TimeAtlas
from fastcxt.preprocess import choose_pairs

# ---- Configuration ----
CKPT = sys.argv[1] if len(sys.argv) > 1 else None
DEVICE = "cuda:0"
BATCH_SIZE = 256
MAX_PAIRS_PER_BATCH = 5000       # process pairs in chunks to fit in GPU memory
MUTATION_RATE = 3.5e-9           # Anopheles gambiae estimate
ARMS = ["2L", "2R", "3L", "3R", "X"]
LOADED_DIR = PATHS.base_dir / "ag1000g_loaded"
ATLAS_DIR = PATHS.base_dir / "ag1000g_atlas"

# ---- Load model ----
config = PRESETS["base"]
if CKPT:
    lit = LitFastCxt.load_from_checkpoint(CKPT, model_config=config)
    model = lit.model
else:
    model = FastCxtModel(config)
model.eval().to(DEVICE)
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

# ---- Build pair list ----
# Load sample count from first arm
gm_2L = np.load(LOADED_DIR / "2L_gm.npy", mmap_mode="r")
n_haploids = gm_2L.shape[0]
print(f"Haploids: {n_haploids}")

# All pairs or a large subsample
n_all_pairs = n_haploids * (n_haploids - 1) // 2
print(f"Total possible pairs: {n_all_pairs:,}")

if n_all_pairs > 50_000:
    # Subsample to keep runtime manageable (50k pairs ~ 6h for full genome)
    NUM_PAIRS = 50_000
    pairs = choose_pairs(n_haploids, NUM_PAIRS, seed=42)
    print(f"Subsampled to {NUM_PAIRS:,} pairs")
else:
    pairs = np.array(
        [(i, j) for i in range(n_haploids) for j in range(i + 1, n_haploids)],
        dtype=np.int32,
    )
    print(f"Using all {len(pairs):,} pairs")

# ---- Run inference per arm ----
atlas = TimeAtlas()
atlas.metadata = {
    "species": "Anopheles gambiae",
    "dataset": "Ag1000G v3.0",
    "n_haploids": int(n_haploids),
    "n_pairs": int(len(pairs)),
    "mutation_rate": MUTATION_RATE,
}

analysis = MosquitoAnalysis(
    model=model,
    device=DEVICE,
    block_size=1_000_000,
    batch_size=BATCH_SIZE,
    build_workers=8,
)

for arm in ARMS:
    print(f"\n{'='*60}")
    print(f"ARM: {arm} ({ANOGAM_CHROMOSOME_ARMS[arm]:,} bp)")
    print(f"{'='*60}")
    t0 = time.perf_counter()

    gm = np.load(LOADED_DIR / f"{arm}_gm.npy", mmap_mode="r")
    positions = np.load(LOADED_DIR / f"{arm}_pos.npy")
    mask = AccessibilityMask.from_npz(PATHS.ag1000g_accessibility_mask, arm)
    print(f"  Sites: {len(positions):,}, Accessible: {mask.accessible_fraction:.1%}")

    # Process pairs in chunks to manage memory
    all_means = []
    all_vars = []
    pair_list = pairs.tolist()

    for chunk_start in range(0, len(pair_list), MAX_PAIRS_PER_BATCH):
        chunk_end = min(chunk_start + MAX_PAIRS_PER_BATCH, len(pair_list))
        chunk_pairs = [tuple(p) for p in pair_list[chunk_start:chunk_end]]
        print(f"  Pairs {chunk_start:,}--{chunk_end:,} / {len(pair_list):,} ...")

        result = analysis.run_chromosome_arm(
            genotype_matrix=np.array(gm),
            positions=positions,
            arm=arm,
            pivot_pairs=chunk_pairs,
            mutation_rate=MUTATION_RATE,
            accessibility_mask=mask,
            progress=True,
        )
        all_means.append(result["means"])
        all_vars.append(result["variances"])

        # Free GPU memory between chunks
        torch.cuda.empty_cache()

    means = np.concatenate(all_means, axis=0)
    variances = np.concatenate(all_vars, axis=0)

    atlas.add_arm(arm, means, variances, pairs,
                  window_size=2000, mutation_rate=MUTATION_RATE)

    elapsed = time.perf_counter() - t0
    print(f"  Done: {means.shape[0]:,} pairs x {means.shape[1]:,} windows in {elapsed:.0f}s")

# ---- Save atlas ----
atlas.save(str(ATLAS_DIR))
print(f"\nAtlas saved to {ATLAS_DIR}")
print(atlas)
print(atlas.summary())
```

```bash
# Run with checkpoint from Stage 3
python stage5_inference.py "$CKPT"
```

---

## Stage 6: Build and verify the TimeAtlas

```python
#!/usr/bin/env python3
"""stage6_verify_atlas.py -- Load and inspect the Ag1000G TimeAtlas."""

from fastcxt.atlas import TimeAtlas
from fastcxt.paths import PATHS
import numpy as np

ATLAS_DIR = PATHS.base_dir / "ag1000g_atlas"
atlas = TimeAtlas.load(str(ATLAS_DIR))

print(atlas)
print()

# Summary
summary = atlas.summary()
for arm, info in summary["per_arm"].items():
    print(f"  {arm}: {info['n_pairs']:>6,} pairs x {info['n_windows']:>6,} windows "
          f"({info['arm_length_bp']:>12,} bp)  mean_log_tmrca={info['mean_log_tmrca']:.2f}")

print(f"\n  Total: {summary['total_pairs']:,} pair-arm entries, "
      f"{summary['total_windows']:,} windows")

# Spot-check queries
print("\n--- Query examples ---")
m, v = atlas.query_pair("2L", 0, 1)
print(f"Pair (0,1) on 2L: mean TMRCA range [{np.exp(m).min():.0f}, {np.exp(m).max():.0f}] gen")

# Deepest pairs near Rdl locus (2L:25,363,652)
RDL_POS = 25_363_652
deep = atlas.deepest_pairs("2L", RDL_POS, k=10)
print(f"\nDeepest 10 pairs at Rdl ({RDL_POS:,}):")
for a, b in deep:
    r = atlas.query_pair("2L", int(a), int(b))
    if r is not None:
        w = atlas.arms["2L"].window_at(RDL_POS)
        print(f"  ({a:>4}, {b:>4})  log-TMRCA={r[0][w]:.2f} +/- {np.sqrt(r[1][w]):.2f}")
```

```bash
python stage6_verify_atlas.py
```

---

## Stage 7: Analysis and plotting

### 7a. Genome-wide TMRCA landscape (per arm)

```python
#!/usr/bin/env python3
"""stage7a_landscape.py -- Plot genome-wide TMRCA landscape."""

import numpy as np
import matplotlib.pyplot as plt
from fastcxt.atlas import TimeAtlas
from fastcxt.paths import PATHS
from fastcxt.mosquito import ANOGAM_CHROMOSOME_ARMS

atlas = TimeAtlas.load(str(PATHS.base_dir / "ag1000g_atlas"))
ARMS = ["2L", "2R", "3L", "3R", "X"]

fig, axes = plt.subplots(len(ARMS), 1, figsize=(18, 3 * len(ARMS)), sharex=False)

for ax, arm in zip(axes, ARMS):
    ad = atlas.arms[arm]

    # Median and IQR across all pairs at each window
    median_tmrca = np.exp(np.median(ad.means, axis=0))
    q25 = np.exp(np.percentile(ad.means, 25, axis=0))
    q75 = np.exp(np.percentile(ad.means, 75, axis=0))

    x_mb = ad.window_starts / 1e6

    ax.fill_between(x_mb, q25, q75, alpha=0.3, color="steelblue")
    ax.plot(x_mb, median_tmrca, linewidth=0.5, color="steelblue")
    ax.set_ylabel("TMRCA (gen)")
    ax.set_title(f"Chromosome {arm} ({ANOGAM_CHROMOSOME_ARMS[arm]/1e6:.1f} Mb)")
    ax.set_yscale("log")
    ax.set_xlim(0, ANOGAM_CHROMOSOME_ARMS[arm] / 1e6)

axes[-1].set_xlabel("Position (Mb)")
plt.tight_layout()
plt.savefig(str(PATHS.figures_dir / "ag1000g_tmrca_landscape.pdf"), dpi=150)
plt.savefig(str(PATHS.figures_dir / "ag1000g_tmrca_landscape.png"), dpi=150)
print("Saved: ag1000g_tmrca_landscape.pdf")
plt.show()
```

### 7b. Rdl locus sweep signal (2L:25.3 Mb)

The *Rdl* (resistance to dieldrin) locus on 2L is a known insecticide
resistance gene with a strong selective sweep signature.

```python
#!/usr/bin/env python3
"""stage7b_rdl_sweep.py -- Zoom into the Rdl locus on 2L."""

import numpy as np
import matplotlib.pyplot as plt
from fastcxt.atlas import TimeAtlas
from fastcxt.paths import PATHS

atlas = TimeAtlas.load(str(PATHS.base_dir / "ag1000g_atlas"))
ad = atlas.arms["2L"]

# Rdl region: 24--27 Mb
REGION_START = 24_000_000
REGION_END = 27_000_000
RDL_GENE = 25_363_652

pairs, means_region, vars_region = atlas.query_region("2L", REGION_START, REGION_END)
w_start = ad.window_at(REGION_START)
w_end = ad.window_at(REGION_END) + 1
x_mb = ad.window_starts[w_start:w_end] / 1e6

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Top: median TMRCA with IQR
median = np.exp(np.median(means_region, axis=0))
q10 = np.exp(np.percentile(means_region, 10, axis=0))
q90 = np.exp(np.percentile(means_region, 90, axis=0))

ax1.fill_between(x_mb, q10, q90, alpha=0.2, color="firebrick")
ax1.plot(x_mb, median, color="firebrick", linewidth=1)
ax1.axvline(RDL_GENE / 1e6, color="black", linestyle="--", alpha=0.5, label="Rdl gene")
ax1.set_ylabel("TMRCA (generations)")
ax1.set_yscale("log")
ax1.set_title("Pairwise TMRCA near the Rdl locus (2L)")
ax1.legend()

# Bottom: prediction uncertainty (mean variance across pairs)
mean_var = np.mean(vars_region, axis=0)
ax2.plot(x_mb, mean_var, color="darkorange", linewidth=1)
ax2.axvline(RDL_GENE / 1e6, color="black", linestyle="--", alpha=0.5)
ax2.set_ylabel("Mean prediction variance")
ax2.set_xlabel("Position on 2L (Mb)")

plt.tight_layout()
plt.savefig(str(PATHS.figures_dir / "rdl_sweep_tmrca.pdf"), dpi=150)
plt.savefig(str(PATHS.figures_dir / "rdl_sweep_tmrca.png"), dpi=150)
print("Saved: rdl_sweep_tmrca.pdf")
plt.show()
```

### 7c. Pairwise TMRCA heatmap at a locus

```python
#!/usr/bin/env python3
"""stage7c_heatmap.py -- Pairwise TMRCA heatmap at a position."""

import numpy as np
import matplotlib.pyplot as plt
from fastcxt.atlas import TimeAtlas
from fastcxt.paths import PATHS

atlas = TimeAtlas.load(str(PATHS.base_dir / "ag1000g_atlas"))

POSITION = 25_363_652  # Rdl
pairs, means_at_pos, _ = atlas.query_window("2L", POSITION)

# Build a dense matrix from the pair predictions
sample_ids = sorted(set(pairs[:, 0].tolist() + pairs[:, 1].tolist()))
n = len(sample_ids)
id_to_idx = {s: i for i, s in enumerate(sample_ids)}

matrix = np.full((n, n), np.nan)
for (a, b), val in zip(pairs, means_at_pos):
    i, j = id_to_idx[a], id_to_idx[b]
    matrix[i, j] = val
    matrix[j, i] = val
np.fill_diagonal(matrix, 0)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(np.exp(matrix), cmap="magma", aspect="auto")
ax.set_xlabel("Sample index")
ax.set_ylabel("Sample index")
ax.set_title(f"Pairwise TMRCA at 2L:{POSITION:,} (Rdl locus)")
plt.colorbar(im, ax=ax, label="TMRCA (generations)")
plt.tight_layout()
plt.savefig(str(PATHS.figures_dir / "rdl_pairwise_heatmap.pdf"), dpi=150)
plt.savefig(str(PATHS.figures_dir / "rdl_pairwise_heatmap.png"), dpi=150)
print("Saved: rdl_pairwise_heatmap.pdf")
plt.show()
```

### 7d. Cross-arm summary: per-pair mean TMRCA

```python
#!/usr/bin/env python3
"""stage7d_cross_arm.py -- Per-pair genome-wide mean TMRCA."""

import numpy as np
import matplotlib.pyplot as plt
from fastcxt.atlas import TimeAtlas
from fastcxt.paths import PATHS

atlas = TimeAtlas.load(str(PATHS.base_dir / "ag1000g_atlas"))
ARMS = ["2L", "2R", "3L", "3R", "X"]

# Collect per-pair mean TMRCA across arms
per_arm_means = {}
for arm in ARMS:
    per_arm_means[arm] = atlas.mean_tmrca(arm)

fig, ax = plt.subplots(figsize=(10, 5))
positions = np.arange(len(ARMS))
bps = [per_arm_means[arm] for arm in ARMS]

parts = ax.violinplot(bps, positions, showmedians=True, showextrema=False)
for pc in parts["bodies"]:
    pc.set_facecolor("steelblue")
    pc.set_alpha(0.6)

ax.set_xticks(positions)
ax.set_xticklabels(ARMS)
ax.set_ylabel("Mean TMRCA (generations)")
ax.set_title("Distribution of per-pair genome-wide mean TMRCA by chromosome arm")
plt.tight_layout()
plt.savefig(str(PATHS.figures_dir / "ag1000g_cross_arm_tmrca.pdf"), dpi=150)
plt.savefig(str(PATHS.figures_dir / "ag1000g_cross_arm_tmrca.png"), dpi=150)
print("Saved: ag1000g_cross_arm_tmrca.pdf")
plt.show()
```

---

## Quick-reference: all shell commands in order

```bash
# ---- Stage 0: Setup ----
cd /home/kkor/fastcxt
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[all]"

# ---- Stage 1: Simulate ----
BASE="$FASTCXT_BASE_DIR/data"
for N in 25 50 100; do
    fastcxt-simulate --scenario AnoGam --data-dir "$BASE/anogam_n${N}" --num-ts 1000 --n-samples $N --num-processes 80
done
fastcxt-simulate --scenario constant  --data-dir "$BASE/constant"  --num-ts 2000 --n-samples 50 --num-processes 80
fastcxt-simulate --scenario sawtooth  --data-dir "$BASE/sawtooth"  --num-ts 1000 --n-samples 50 --num-processes 80
fastcxt-simulate --scenario island    --data-dir "$BASE/island"    --num-ts 1000 --n-samples 50 --num-processes 80
for SP in HomSap DroMel BosTau PanTro; do
    fastcxt-simulate --scenario $SP --data-dir "$BASE/stdpopsim_${SP}" --num-ts 500 --n-samples 25 --num-processes 80
done

# ---- Stage 2: Preprocess ----
fastcxt-preprocess --base-dir "$BASE" --out-subdir processed --num-pairs 200 --num-workers 80 \
    --accessibility-mask "$AG1000G_ACCESSIBILITY" --skip-existing

# ---- Stage 3: Train ----
fastcxt-train --model base --dataset-path "$BASE/processed" --gpus 0 1 2 --epochs 10 \
    --batch-size 128 --grad-accum 4 --workers 16 --log-dir "$FASTCXT_BASE_DIR/lightning_logs"

CKPT=$(ls -t "$FASTCXT_BASE_DIR"/lightning_logs/*/checkpoints/*.ckpt | head -1)

# ---- Stage 4: Load Ag1000G ----
python stage4_load_ag1000g.py

# ---- Stage 5: Inference ----
python stage5_inference.py "$CKPT"

# ---- Stage 6: Verify ----
python stage6_verify_atlas.py

# ---- Stage 7: Plots ----
mkdir -p "$FASTCXT_BASE_DIR/figures/output"
python stage7a_landscape.py
python stage7b_rdl_sweep.py
python stage7c_heatmap.py
python stage7d_cross_arm.py
```

---

## Notes

- **Memory**: loading the full Ag1000G genotype matrix for 2R (~1000
  haploids x millions of sites) requires ~30--60 GB RAM.  Use `mmap_mode="r"`
  when possible.
- **GPU memory**: with `batch_size=256` and `max_samples=200`, each forward
  pass uses ~2 GB VRAM.  Reduce `MAX_PAIRS_PER_BATCH` if you hit OOM.
- **Pair subsampling**: with ~1000 haploids there are ~500k pairs.  50k is
  a good default for a first pass; increase to cover all pairs if runtime
  permits.
- **Inversions**: chromosomes 2La and 2Rb carry common polymorphic
  inversions in *A. gambiae*.  Consider analysing standard and inverted
  karyotype samples separately for those arms.
- **Population labels**: the Ag1000G metadata includes species, country,
  and collection site.  Use these to group pairs for between-population vs
  within-population comparisons in the atlas queries.
