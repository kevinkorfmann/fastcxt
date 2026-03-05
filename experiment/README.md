# fastcxt experiment

End-to-end experiment comparing two TMRCA inference approaches:

1. **Pairwise model** (`FastCxtModel`) -- predicts TMRCA for one sample pair per forward pass using SFS features. Accurate but O(n²) in the number of samples.
2. **Node-time model** (`NodeTimeModel`) -- predicts all internal node times from tree topology in a single forward pass, then recovers any pairwise TMRCA via O(log n) LCA lookups. O(1) neural network cost regardless of the number of pairs.

## Setup

```bash
git clone https://github.com/kevinkorfmann/fastcxt.git
cd fastcxt
uv pip install -e ".[all]"
```

## 1. Simulate training data

```bash
fastcxt-simulate \
    --scenario constant \
    --data-dir ./experiment/sims/constant \
    --num-ts 1000 \
    --n-samples 50 \
    --sequence-length 1e6 \
    --num-processes 8
```

**Simulation parameters:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| `--scenario` | `constant` | Constant-size population (Ne=10,000) |
| `--n-samples` | `50` | 50 haploid samples (25 diploid) |
| `--num-ts` | `1000` | 1000 independent tree sequences |
| `--sequence-length` | `1e6` | 1 Mb per tree sequence |
| `--mutation-rate` | `1e-8` | (default) per-bp per-generation |
| `--recombination-rate` | `1e-8` | (default) per-bp per-generation |

## 2. Preprocess

Preprocess SFS features + TMRCA targets for the pairwise model.
Add `--extract-trees` to also extract tree topology features (needed for both the
`base_trees` pairwise variant and the node-time model).

```bash
fastcxt-preprocess \
    --base-dir ./experiment/sims/constant \
    --out-subdir processed \
    --window-size 2000 \
    --sequence-length 1000000 \
    --num-pairs 200 \
    --extract-trees \
    --num-workers 16
```

**Preprocessing parameters:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| `--window-size` | `2000` | 2 kb genomic windows → 500 windows per 1 Mb |
| `--num-pairs` | `200` | Random sample pairs per tree sequence |
| `--extract-trees` | flag | Extract coalescence topology features (5 values/node) |

## 3. Train

### Pairwise model (base)

Predicts log(TMRCA) per pair from SFS. One forward pass per pair.
Uses Beta-NLL loss with cosine LR schedule.

```bash
fastcxt-train \
    --model base \
    --dataset-path ./experiment/sims/constant/processed \
    --gpus 0 1 2 \
    --epochs 20 \
    --batch-size 128 \
    --grad-accum 2 \
    --workers 8
```

**Model: `base` preset**

| Parameter | Value |
|-----------|-------|
| `d_model` | 256 |
| `n_enc_layers` | 6 |
| `n_dec_layers` | 4 |
| Parameters | ~16 M |
| Input | SFS (2 channels × 500 windows × n_samples) |
| Output | (μ, log σ²) per window |
| Loss | Beta-NLL (β=0.5) |

### Node-time model

Predicts log(time) for all internal nodes from tree topology in one forward pass.
Pairwise TMRCA for any pair is then a cheap LCA lookup.

```bash
cd experiment
CUDA_VISIBLE_DEVICES=0 python train_and_benchmark_node_times.py
```

**Model: `NodeTimeModel`**

| Parameter | Value |
|-----------|-------|
| `d_model` | 256 |
| `n_layers` | 4 (BiMamba blocks) |
| Parameters | ~14 M |
| Input | Tree topology features (500 windows × 245 features) |
| Conditioning | Mutation rate via FiLM layer |
| Output | log(time) per internal node per window |

The topology features encode 5 values per internal node per window:

| Feature | Description |
|---------|-------------|
| `rank` | Coalescence order (0 = first merge, normalized) |
| `min_leaf_left` | Smallest leaf index in left subtree (normalized) |
| `min_leaf_right` | Smallest leaf index in right subtree (normalized) |
| `subtree_size_left` | Number of leaves below left child (normalized) |
| `subtree_size_right` | Number of leaves below right child (normalized) |

## 4. Evaluate

### Plot pairwise model figures

```bash
cd experiment
python plot_experiment_figures.py \
    --checkpoint lightning_logs/<version>/checkpoints/<ckpt>.ckpt
```

Generates `figures/`: true-vs-predicted scatter, TMRCA along genome,
residuals, and training summary.

### Benchmark scaling

Compare inference time for varying numbers of pairs:

```bash
cd experiment
CUDA_VISIBLE_DEVICES=0 python benchmark_scaling.py
```

This simulates tree sequences with 50, 100, and 200 samples and benchmarks
both approaches end-to-end. Expected results:

| Pairs | Pairwise | Node-time | Speedup |
|-------|----------|-----------|---------|
| 50 | ~800 ms | ~80 ms | ~10x |
| 500 | ~4 s | ~250 ms | ~16x |
| 5000 | ~800 s | ~5 s | ~160x |

## Conceptual comparison

```
                    Pairwise model                  Node-time model
                    ──────────────                  ───────────────
Input:              SFS (pair-specific)             Tree topology (shared)
Forward passes:     1 per pair → O(n²)              1 total → O(1)
Pair recovery:      Direct prediction               LCA lookup → O(log n)
Uncertainty:        Yes (μ, σ² per window)          Not yet (point estimate)
Accuracy:           Higher (sees mutations)          Lower (topology only)
```

The pairwise model sees the actual mutation patterns between two samples (SFS),
which directly reflect divergence time. The node-time model only sees tree
structure (which lineages merge in what order), so it has less signal per
prediction but amortizes the cost across all O(n²) pairs.
