# Repository Organization

## Top-level structure

```
fastcxt/                    # Core Python package
experiment/                 # Human baseline experiment (constant Ne, 1Mb, 500 windows)
experiment_human/           # Whole human chromosome decoding (curriculum learning)
experiment_mosquito/        # AnoGam (mosquito) experiments
paper/                      # LaTeX paper + figures
docs/                       # Sphinx documentation source
scripts/                    # Misc utility scripts
tests/                      # Unit tests
```

## Core package (`fastcxt/`)

The neural network library for pairwise TMRCA inference.

```
fastcxt/
├── config.py               # FastCxtConfig, PRESETS, TrainingConfig
├── model.py                # FastCxtModel (BiMamba encoder-decoder)
├── modules.py              # BiMambaBlock, MultiScaleInputProjection, FiLMLayer, UncertaintyHead
├── train.py                # LitFastCxt Lightning module + CLI (fastcxt-train)
├── translate.py            # Inference API (predict, translate_from_ts, translate_from_genotype_matrix)
├── dataset.py              # PairDataset, LazyPairDataset
├── simulate.py             # Simulation via msprime/stdpopsim (fastcxt-simulate)
├── preprocess.py           # SFS preprocessing pipeline (fastcxt-preprocess)
├── sfs.py                  # SFS computation from genotype matrices
├── node_time_model.py      # NodeTimeModel — predicts node times for O(n log n) scaling
├── hybrid_node_time_model.py # HybridNodeTimeModel variant
├── tree_utils.py           # Tree feature extraction + LCA lookup for node-time approach
├── atlas.py                # TimeAtlas data structure
├── benchmark.py            # Benchmarking utilities (fastcxt-benchmark)
├── mosquito.py             # Anopheles-specific helpers
└── paths.py                # Cluster path configuration
```

### Two inference approaches

1. **Pairwise (FastCxtModel)**: Takes SFS features → predicts TMRCA per pair. Simple but O(n²) in pairs.
2. **Node-time (NodeTimeModel)**: Takes tree topology features → predicts internal node times → LCA lookup for any pair. O(n log n) scaling.

## Human baseline experiment (`experiment/`)

Reproducible pipeline comparing pairwise vs node-time inference on simulated human-like data.

**Parameters**: constant Ne=2e4, 1Mb sequence, 2000bp windows, 500 windows, sample sizes 10/25/50/100/200.

```
experiment/
├── config.py               # Shared constants (single source of truth)
├── run_all.sh              # Master pipeline (supports --from 03 to restart)
├── 01_simulate.sh          # Simulate tree sequences (constant scenario)
├── 02_preprocess.sh        # SFS + node-time features (true + tsinfer)
├── 03_train_pairwise.sh    # Train FastCxtModel (base preset)
├── 04_train_node_time.sh   # Train NodeTimeModel (true trees + tsinfer trees)
├── 05_evaluate.sh          # Scatter, genome profile, residual figures
├── 06_benchmark_scaling.sh # Pairwise O(n²) vs node-time O(1) benchmark
├── slurm_betty.sh          # SLURM wrapper for betty cluster
├── preprocess_node_times.py    # Extract node features (--use-tsinfer flag)
├── train_node_time.py          # NodeTimeModel training
├── evaluate_pairwise.py        # Pairwise model evaluation figures
├── benchmark_scaling.py        # Scaling benchmark (3 models compared)
├── _scratch/                   # Legacy scripts and artifacts
│   └── ...
└── outputs/                    # Generated at runtime
    ├── pairwise/               # FastCxtModel checkpoints + CSV metrics
    ├── node_features/          # True tree features (numpy)
    ├── node_features_tsinfer/  # Tsinfer tree features (numpy)
    ├── node_time_true.pt       # NodeTimeModel weights (true trees)
    ├── node_time_tsinfer.pt    # NodeTimeModel weights (tsinfer trees)
    └── benchmark_results.csv   # Scaling benchmark results
```

**Three models compared**:
1. Pairwise FastCxtModel (SFS-only)
2. NodeTimeModel with ground-truth tree topology + LCA lookup
3. NodeTimeModel with tsinfer-inferred topology + LCA lookup

## Whole human chromosome decoding (`experiment_human/`)

Curriculum learning for full-chromosome TMRCA inference (50Mb → 250Mb).

```
experiment_human/
├── whole_homsap_chromosome_decoding/
│   ├── run_sim.sh          # Simulate HomSap at each stage length
│   ├── run_train.sh        # Train with curriculum stage presets
│   ├── slurm_sim.sh        # SLURM wrapper (CPU)
│   ├── slurm_train.sh      # SLURM wrapper (GPU, auto-installs CUDA deps)
│   ├── launch_all.sh       # Submit full pipeline with SLURM dependencies
│   ├── STRATEGY.md         # Curriculum design rationale
│   ├── strategy_paper/     # Strategy paper (LaTeX + PDF)
│   └── eval_tree_ablation.py
└── legacy_experiment4/     # Earlier human experiment figures
```

**Curriculum stages** (SFS-only, `base` family, 2000bp windows):

| Stage | Seq Length | Windows | Sims | Preset    |
|-------|-----------|---------|------|-----------|
| 1     | 50 Mb     | 25,000  | 100  | base_25k  |
| 2     | 100 Mb    | 50,000  | 50   | base_50k  |
| 3     | 150 Mb    | 75,000  | 25   | base_75k  |
| 4     | 250 Mb    | 125,000 | 10   | base_125k |

Each stage fine-tunes from the previous checkpoint. Positional embeddings are extended for longer sequences.

**Work directory on betty**: `/vast/projects/smathi/cohort/kkor/homsap_curriculum/`

## AnoGam (mosquito) experiments (`experiment_mosquito/`)

```
experiment_mosquito/
├── experiment/             # Clean pipeline (mirrors experiment/ but for AnoGam)
│   ├── config.py           # AnoGam params: 100kb, 200bp windows, 500 windows
│   ├── run_all.sh
│   ├── 01_simulate.sh      # stdpopsim AnoGam scenario
│   ├── 02_preprocess.sh
│   ├── 03_train_pairwise.sh
│   ├── 04_train_node_time.sh
│   ├── 05_evaluate.sh
│   ├── 06_benchmark_scaling.sh
│   ├── slurm_betty.sh
│   └── *.py                # Same scripts as experiment/, import from local config.py
├── experiment_mossies/     # Legacy 100kb AnoGam experiment
├── experiment-1mb-context/ # 1Mb context extension
├── experiment-10mb-context/# 10Mb context extension
└── experiment-large-context/# 400kb (2000 windows) context experiment
```

## Cluster (betty) details

- **Repo**: `/vast/projects/smathi/cohort/kkor/fastcxt_repo/`
- **Curriculum data**: `/vast/projects/smathi/cohort/kkor/homsap_curriculum/`
- **venv**: `/vast/projects/smathi/cohort/kkor/fastcxt_repo/.venv/` (uv, Python 3.11)
- **CUDA packages** (mamba-ssm, causal-conv1d): auto-installed on first GPU job via SLURM scripts
- **Cache redirects**: all caches → `/vast/projects/smathi/cohort/kkor/homsap_curriculum/.cache/` (home quota exceeded)
- **Partitions used**: `b200-mig45` (250 billing weight, 45GB VRAM), `dgx-b200` (1000 billing weight, 183GB VRAM)
