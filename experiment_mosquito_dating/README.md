# Mosquito Population Dating

Pairwise TMRCA dating of AG1000G populations using the trained `base_anogam`
FastCxtModel on tsinfer tree sequences.

## What it does

For each chromosome arm:
1. Loads the tsinfer+tsdate tree sequence (`gamb.{arm}.gff.dated.ne.trees`)
2. Extracts populations from individual metadata (country field)
3. For each population: simplifies the tree, extracts the genotype matrix,
   applies the accessibility mask, and runs pairwise TMRCA inference
4. Saves per-population results (means, variances, pairs) and a combined summary

## Variable sample sizes

Populations have different numbers of individuals. The model was trained with
`max_samples=200` haploids. Two cases:

- **<=200 haploids** (<=100 individuals): The `MultiScaleInputProjection`
  zero-pads the SFS frequency axis to 200. All pairs run in a single batch.

- **>200 haploids** (>100 individuals): The genotype matrix is subsampled
  to 200 haploids **per pair** — the two pivot haploids are always kept, and
  198 context haploids are randomly drawn from the remaining population. This
  ensures the SFS frequency spectrum matches what the model was trained on.
  Each pair is processed individually with its own subsample. This is slower
  but produces correct frequency-calibrated predictions.

## Prerequisites

- Trained `base_anogam` checkpoint (from `experiment_mosquito_poc`)
- tsinfer trees on sietch: `/sietch_colab/data_share/Ag1000G/Ag3.0/args_trees/tsinfer_data_v2/`
- Accessibility mask: `/sietch_colab/data_share/Ag1000G/Ag3.0/args_trees/singer/agp3.is_accessible.txt.npz`
- Python environment with fastcxt installed (`pip install -e .[sim,trees]`)

## Usage on poppy

```bash
cd experiment_mosquito_dating

# List available populations (no GPU needed)
python run_population_dating.py --arm 2L --list-populations

# Date all populations on 2L
bash run.sh /path/to/best.ckpt

# Date a specific population
bash run.sh /path/to/best.ckpt --populations "Burkina Faso"

# Date other chromosome arms
ARM=2R bash run.sh /path/to/best.ckpt
ARM=3L bash run.sh /path/to/best.ckpt
ARM=3R bash run.sh /path/to/best.ckpt
ARM=X  bash run.sh /path/to/best.ckpt

# Increase max pairs per population (default 200)
bash run.sh /path/to/best.ckpt --max-pairs 500

# Re-read cached results without re-running inference
python run_population_dating.py --arm 2L --from-cache
```

## Output structure

```
results/
├── 2L/
│   ├── all_populations_summary.json
│   ├── burkina_faso/
│   │   ├── results.npz          # means, variances, index_map, pivot_pairs
│   │   └── summary.json         # n_individuals, mean_log_tmrca, etc.
│   ├── cameroon/
│   │   ├── results.npz
│   │   └── summary.json
│   └── .../
├── 2R/
│   └── ...
└── ...
```

### results.npz contents

| Array         | Shape          | Description                                  |
|---------------|----------------|----------------------------------------------|
| `means`       | `(N, 500)`     | Predicted log-TMRCA means per window         |
| `variances`   | `(N, 500)`     | Predicted log-TMRCA variances per window     |
| `index_map`   | `(N, 2)`       | Maps each row to `[block_idx, pair_idx]`     |
| `pivot_pairs` | `(n_pairs, 2)` | Haploid index pairs (local, post-simplify)   |
| `block_starts`| `(n_blocks,)`  | Block start positions (bp)                   |
| `block_ends`  | `(n_blocks,)`  | Block end positions (bp)                     |

## Configuration

Edit `config.py` to change:
- `BLOCK_SIZE` / `WINDOW_SIZE` — must match the trained model (100kb / 200bp for `base_anogam`)
- `MAX_PAIRS_PER_POP` — cap on pairwise pairs per population (default 200)
- `MUTATION_RATE` — 3.5e-9 for An. gambiae
- `TREES_DIR` / `ACCESSIBILITY_MASK` — data paths on sietch
