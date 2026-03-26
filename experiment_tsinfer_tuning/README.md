# tsinfer parameter tuning experiment

## Motivation

The existing AG3 tsinfer trees on sietch were inferred with tsinfer 0.2.3 using
default parameters — no `mismatch_ratio` set, and tsdate used `mutation_rate=1e-9`
(should be 3.5e-9 for AnoGam). The default mismatch ratio in 0.2.x produces
overly flat trees with too few recombination breakpoints.

This experiment tests different `mismatch_ratio` values to find the best setting
for AnoGam data before reinferring the full chromosome arms.

## Design

Two evaluation modes:

### 1. Simulated (ground truth available)

- Simulate AnoGam-like data with msprime (1Mb, n=50, Ne=1e6, mu=3.5e-9, r=1e-8)
- Extract genotypes, run tsinfer with each mismatch_ratio
- Compare inferred vs true trees using:
  - **KC distance** (topology λ=0, branch lengths λ=1) — lower is better
  - **Tree count** — should be close to true
  - **Breakpoint recall** — fraction of true recombination breakpoints recovered

### 2. Real AG3 data

- Subset 1Mb region from `gamb.2L.phased.n1470.derived.vcf.gz` (polarized VCF)
- Run tsinfer with each mismatch_ratio
- Measure: tree count, mean tree span, runtime
- No ground truth, but tree count and span reveal over/under-fragmentation

### Mismatch ratios tested

| mismatch_ratio | Expected behavior |
|---|---|
| 0.1 | Very few mismatches allowed → many trees, fragmented |
| 0.5 | Conservative |
| 1.0 | Balanced |
| 2.0 | More sharing allowed |
| 5.0 | Aggressive sharing |
| 10.0 | Very flat trees |
| default | tsinfer's built-in default |

## Usage

```bash
# Simulated only (fast, ~5 min)
python run_experiment.py --mode sim --out-dir results

# Real data only (needs sietch VCFs, ~30 min)
python run_experiment.py --mode real --out-dir results --region 2L:5000000-6000000

# Both
python run_experiment.py --mode both --out-dir results
```

## Output

```
results/
├── sim/
│   ├── true.trees                          # ground truth
│   ├── inferred_mr0.1.trees                # inferred per mismatch_ratio
│   ├── ...
│   ├── results.json                        # all metrics
│   └── mismatch_ratio_comparison.png       # summary plot
└── real/
    ├── region.samples                      # tsinfer SampleData
    ├── inferred_mr0.1.trees
    ├── ...
    ├── results.json
    └── mismatch_ratio_comparison.png
```

## Next steps

Once the best mismatch_ratio is identified, use `scripts/reinfer_ag3_trees.py`
to reinfer all 5 chromosome arms with the optimal setting:

```bash
python scripts/reinfer_ag3_trees.py \
    --out-dir /sietch_colab/kkor/fastcxt/data/ag3_trees \
    --mismatch-ratio <best_value>
```
