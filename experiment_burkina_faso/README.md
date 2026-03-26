# Burkina Faso population inference

Full pairwise TMRCA inference on all pairs of An. gambiae individuals from
Burkina Faso using the trained fastcxt pairwise model.

## Data

- **157 individuals** (314 haplotypes) from 4 locations
- **12,246 pairs** (all pairwise combinations)
- **5 chromosome arms** (2L, 2R, 3L, 3R, X)
- Source: Ag1000G/Ag3.0 tsinfer+tsdate trees on sietch

## Usage

```bash
# Full run (all pairs, all arms)
python run_inference.py --out-dir /sietch_colab/kkor/fastcxt/results/burkina_faso

# Test run (100 pairs only)
python run_inference.py --out-dir ./results/bf_test --max-pairs 100

# Different population
python run_inference.py --out-dir ./results/uganda --population Uganda

# Single arm
python run_inference.py --out-dir ./results/bf_2L --arms 2L
```

## Output structure

```
results/burkina_faso/
├── config.json              # run parameters (population, model, etc.)
├── population_metadata.csv  # sample info (country, location, year, etc.)
├── pair_index.csv           # pair_idx -> (sample_a, sample_b)
├── haploid_pairs.npy        # (n_pairs, 2) int32 haploid node indices
├── 2L/
│   ├── means.npz            # (n_rows, n_windows) float32 predicted log(TMRCA)
│   ├── variances.npz        # (n_rows, n_windows) float32 predicted variance
│   ├── index_map.npy        # (n_rows, 2) int32 [block_idx, pair_idx]
│   └── blocks.json          # block coordinates [{start, end}, ...]
├── 2R/
│   └── ...
├── 3L/
│   └── ...
├── 3R/
│   └── ...
└── X/
    └── ...
```

## Output dimensions

Per chromosome arm:

| File | Shape | dtype | Description |
|------|-------|-------|-------------|
| `means.npz["means"]` | `(n_rows, 500)` | float32 | Predicted log(TMRCA) per window |
| `variances.npz["variances"]` | `(n_rows, 500)` | float32 | Predicted variance per window |
| `index_map.npy` | `(n_rows, 2)` | int32 | `[block_idx, pair_idx]` per row |
| `blocks.json` | list of dicts | — | `{idx, start, end}` genomic coords |

- Each row is one **(block, pair)** combination
- 500 values per row = 500 output windows × 200 bp = 100 kb per block
- `n_rows = n_blocks × n_pairs` (e.g. 2L: ~49 blocks × 12,246 pairs ≈ 600k rows)
- `block_idx` indexes into `blocks.json` for genomic position
- `pair_idx` indexes into `pair_index.csv` for sample identity
- Values in `means` are **log(TMRCA)** in generations

## Querying results

```python
import numpy as np, pandas as pd, json

out = "/sietch_colab/kkor/fastcxt/results/burkina_faso"

# --- Load metadata ---
pairs = pd.read_csv(f"{out}/pair_index.csv")            # pair_idx -> sample_a, sample_b
pop = pd.read_csv(f"{out}/population_metadata.csv")      # full sample info
config = json.load(open(f"{out}/config.json"))            # run parameters

# --- Load per-arm results ---
arm = "2L"
means = np.load(f"{out}/{arm}/means.npz")["means"]       # (n_rows, 500)
variances = np.load(f"{out}/{arm}/variances.npz")["variances"]
index_map = np.load(f"{out}/{arm}/index_map.npy")         # (n_rows, 2)
with open(f"{out}/{arm}/blocks.json") as f:
    blocks = json.load(f)

# --- Get TMRCA for a single pair across the whole chromosome ---
pair_idx = 42
row_mask = index_map[:, 1] == pair_idx
pair_means = means[row_mask]                              # (n_blocks, 500)
pair_blocks = index_map[row_mask, 0]                      # which block each row is

# Stitch blocks into a full chromosome profile (ordered by block)
full_tmrca = pair_means[np.argsort(pair_blocks)].ravel()  # (n_blocks * 500,)

# Genomic positions (200 bp windows)
window_size = 200
positions_bp = np.arange(len(full_tmrca)) * window_size

# Who is pair 42?
print(pairs.iloc[pair_idx])  # sample_a, sample_b

# --- Get TMRCA for a single block, all pairs ---
block_idx = 0
block_mask = index_map[:, 0] == block_idx
block_means = means[block_mask]                           # (n_pairs, 500)
block_pairs = index_map[block_mask, 1]                    # pair indices
# block_means[i] corresponds to pairs.iloc[block_pairs[i]]

# --- Find specific pair by sample name ---
sa, sb = "AB0085-Cx", "AB0086-Cx"
match = pairs[(pairs["sample_a"] == sa) & (pairs["sample_b"] == sb)]
if len(match) == 0:
    match = pairs[(pairs["sample_a"] == sb) & (pairs["sample_b"] == sa)]
target_pair_idx = match.index[0]
```
