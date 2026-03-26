# Whole HomSap Chromosome Decoding — Curriculum Learning

## Goal

Decode TMRCA across all human chromosomes (47–249 Mb) at 2000 bp resolution
using SFS-only features (no tsinfer). The Mamba architecture scales linearly
with sequence length, enabling whole-chromosome inference.

## Architecture Notes

- **Positional embeddings** are learnable: `pos_embed = (1, n_windows, d_model)`,
  sliced to `[:, :W, :]` at forward time — supports variable-length input natively.
- **SFS-only** (`use_trees=False`, `base` family) — no tsinfer dependency at inference.
- **Window size**: 2000 bp (default), so `n_windows = seq_len / 2000`.
- **Checkpoint resume**: `--checkpoint` passes new `model_config` to
  `load_from_checkpoint`. pos_embed is extended (new tail initialized from
  truncated normal, existing positions preserved) when resuming with larger
  `n_windows`.

## Why Curriculum Learning?

The largest chromosome (chr1, 249 Mb) requires 124,500 windows. Simulating many
replicates at this length is prohibitively slow. Curriculum learning solves this:

1. Many samples at shorter lengths to learn base features
2. Progressively fewer samples at longer lengths to extend context
3. Each stage fine-tunes from the previous checkpoint

## Curriculum Stages

| Stage | Seq Length | n_windows | num_ts | Preset    | Purpose                          |
|-------|-----------|-----------|--------|-----------|----------------------------------|
| 1     | 50 Mb     | 25,000    | 100    | base_25k  | Learn base features at chr21-scale |
| 2     | 100 Mb    | 50,000    | 50     | base_50k  | Extend to mid-size chromosomes   |
| 3     | 150 Mb    | 75,000    | 25     | base_75k  | Cover most chromosomes           |
| 4     | 250 Mb    | 125,000   | 10     | base_125k | Full chr1-scale                  |

All stages use:
- `window_size=2000` bp
- `sample_sizes="10 25 50 100 200"`
- Species: `HomSap`
- Early stopping: patience=20 on val_loss

## Training Configuration

- `--epochs 200` (early stopping will trigger before this)
- `--batch-size 4` (small actual batch for GPU memory)
- `--grad-accum 64` (effective batch size = 256)
- `--csv-log --save-every-epoch`

## Positional Embedding Extension

When loading a stage-N checkpoint into stage-(N+1):
1. Existing positions are preserved (physical genomic positions at fixed spacing)
2. New tail positions are initialized with `trunc_normal_(std=0.02)`
3. No interpolation needed — positions represent absolute genomic locations

## Scripts

- `run_sim.sh --stage N` — simulate + preprocess for one stage
- `run_train.sh --stage N [--checkpoint PATH]` — train one stage
- `slurm_sim.sh` — SLURM wrapper for simulation (CPU)
- `slurm_train.sh` — SLURM wrapper for training (GPU)
- `launch_all.sh [--start-stage N]` — submit full pipeline with dependencies

## Work Directory

All outputs go to `/vast/projects/smathi/cohort/kkor/homsap_curriculum/`.
