# Repository Organization

## Top-level structure

```
fastcxt/                    # Core Python package
experiment_human_poc/       # Human baseline PoC (constant Ne, 1Mb, 500 windows)
experiment_human_curriculum/# Whole human chromosome decoding (curriculum learning)
experiment_mosquito_poc/    # AnoGam (mosquito) PoC (100kb, 200bp windows)
experiment_mosquito_legacy/ # Legacy mosquito experiments (1mb, 10mb, large context)
paper/                      # LaTeX paper + figures
docs/                       # Sphinx documentation source
scripts/                    # Misc utility scripts + AG3 data pipeline
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

## Human baseline PoC (`experiment_human_poc/`)

Reproducible pipeline comparing pairwise vs node-time inference on simulated human-like data.

**Parameters**: constant Ne=2e4, 1Mb sequence, 2000bp windows, 500 windows, sample sizes 10/25/50/100/200.

```
experiment_human_poc/
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

## Whole human chromosome decoding (`experiment_human_curriculum/`)

Curriculum learning for full-chromosome TMRCA inference (50Mb → 250Mb).

```
experiment_human_curriculum/
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

## AnoGam (mosquito) PoC (`experiment_mosquito_poc/`)

```
experiment_mosquito_poc/
├── config.py               # AnoGam params: 100kb, 200bp windows, 500 windows
├── run_all.sh
├── 01_simulate.sh          # stdpopsim AnoGam scenario
├── 02_preprocess.sh
├── 03_train_pairwise.sh    # Uses base_anogam preset (window_size=200)
├── 04_train_node_time.sh
├── 05_evaluate.sh
├── 06_benchmark_scaling.sh
├── slurm_betty.sh
└── *.py                    # Same scripts as experiment_human_poc/, import from local config.py
```

## Real data: Ag1000G / AG3 pipeline (`scripts/`)

Scripts for downloading and preparing real *Anopheles gambiae* data from MalariaGEN's AG3 release for use with fastcxt.

### Prerequisites

```bash
# 1. Request data access (one-time, approval usually within a day):
#    https://forms.gle/d1NV3aL3EoVQGSHYA

# 2. Install gcloud CLI and authenticate:
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
tar xf google-cloud-cli-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh --quiet
source ~/.bashrc
gcloud auth application-default login  # interactive, opens browser URL

# 3. Python dependencies (already in pixi env)
pip install malariagen_data pysam
```

### `scripts/ag3_pipeline.py` — Download + polarize AG3 data

End-to-end pipeline: downloads phased haplotypes from MalariaGEN, polarizes using
An. arabiensis (368 samples in AG3) as outgroup, outputs `0=ancestral, 1=derived`
genotype matrices ready for fastcxt.

```bash
# Full pipeline — all 5 chromosome arms, all gambiae/coluzzii samples
python scripts/ag3_pipeline.py --out-dir /sietch_colab/kkor/fastcxt/data/ag3

# Single arm
python scripts/ag3_pipeline.py --out-dir ./data/ag3 --arms 2L

# Subset to one country
python scripts/ag3_pipeline.py --out-dir ./data/ag3 \
    --sample-query "country == 'Burkina Faso'"

# High-confidence ancestral calls only (outgroup freq >= 0.95)
python scripts/ag3_pipeline.py --out-dir ./data/ag3 --min-aa-prob 0.95

# Skip download (use cached data from previous run)
python scripts/ag3_pipeline.py --out-dir ./data/ag3 --skip-download

# Validate against existing polarized VCFs on sietch
python scripts/ag3_pipeline.py --out-dir ./data/ag3 --arms 2L \
    --test-region 0:100000 --validate
```

**Output format** (per chromosome arm `.npz`):
| Array | Shape | Description |
|-------|-------|-------------|
| `haplotypes` | `(n_haplotypes, n_sites)` int8 | `0`=ancestral, `1`=derived |
| `positions` | `(n_sites,)` int32 | Genomic position (bp) |
| `aa_prob` | `(n_sites,)` float32 | Confidence of ancestral call |
| `samples` | `(n_samples,)` | Sample IDs |

### `scripts/polarize_ag3.py` — Polarize from local zarr (sietch only)

Faster alternative that reads from the pre-existing zarr store on sietch
(`/sietch_colab/data_share/Ag1000G/Ag3.0/vcf/AgamP3.phased.zarr`), which
already contains AA/AAProb fields from Scott Small's est-sfs pipeline.

```bash
# Polarize from local zarr
python scripts/polarize_ag3.py --out-dir ./data/ag3_polarized --arms 2L

# Test on small region with validation
python scripts/polarize_ag3.py --out-dir ./data/ag3_test --arms 2L --test-region 0:100000
```

### Data lineage

```
MalariaGEN AG3 (GCS)
  └─ ag3.haplotypes()          # phased by SHAPEIT4
  └─ ag3.snp_calls(arabiensis) # 368 outgroup samples
       │
       ▼
  Parsimony polarization       # arabiensis allele freq → ancestral call
       │
       ▼
  Polarized .npz               # 0=ancestral, 1=derived → fastcxt input
```

### Existing data on sietch (`/sietch_colab/data_share/Ag1000G/Ag3.0/`)

| Path | Contents |
|------|----------|
| `vcf/AgamP3.phased.zarr/` | Phased zarr with AA/AAProb (est-sfs polarized) |
| `vcf/phased_vcf/gamb/` | Polarized VCFs: `gamb.{arm}.phased.n1470.derived.vcf.gz` |
| `args_trees/tsinfer_data_v2/` | tsinfer+tsdate trees: `gamb.{arm}.gff.dated.ne.trees` |
| `vcf/agp3.is_accessible.txt.npz` | Accessibility mask per arm |
| `args_trees/gamb.meta.tsinfer.csv` | 1,470 An. gambiae sample metadata |

## Cluster (betty) details

- **Repo**: `/vast/projects/smathi/cohort/kkor/fastcxt_repo/`
- **Curriculum data**: `/vast/projects/smathi/cohort/kkor/homsap_curriculum/`
- **venv**: `/vast/projects/smathi/cohort/kkor/fastcxt_repo/.venv/` (uv, Python 3.11)
- **CUDA packages** (mamba-ssm, causal-conv1d): auto-installed on first GPU job via SLURM scripts
- **Cache redirects**: all caches → `/vast/projects/smathi/cohort/kkor/homsap_curriculum/.cache/` (home quota exceeded)
- **Partitions used**: `b200-mig45` (250 billing weight, 45GB VRAM), `dgx-b200` (1000 billing weight, 183GB VRAM)
