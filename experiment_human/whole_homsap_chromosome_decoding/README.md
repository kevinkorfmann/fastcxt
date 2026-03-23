# Whole HomSap Chromosome Decoding — Instructions

## Quick Start

On Betty, from the repo root:

```bash
# Full pipeline (stage 1 → 2 → 3 → 4, auto-chained with SLURM dependencies)
bash whole_homsap_chromosome_decoding/launch_all.sh

# Preview commands without submitting
bash whole_homsap_chromosome_decoding/launch_all.sh --dry-run

# Resume from a specific stage (e.g. stage 2)
bash whole_homsap_chromosome_decoding/launch_all.sh --start-stage 2
```

## Running Stages Individually

### Simulation (CPU)

```bash
# Submit via SLURM
sbatch --export=STAGE=1 whole_homsap_chromosome_decoding/slurm_sim.sh

# Stages 3-4 need more time
sbatch --export=STAGE=3 --time=4-00:00:00 whole_homsap_chromosome_decoding/slurm_sim.sh
```

### Training (GPU)

```bash
# Stage 1 (from scratch)
sbatch --export=STAGE=1 whole_homsap_chromosome_decoding/slurm_train.sh

# Stage 2+ (auto-finds previous stage checkpoint)
sbatch --export=STAGE=2 whole_homsap_chromosome_decoding/slurm_train.sh

# With explicit checkpoint
sbatch --export=STAGE=2,CHECKPOINT=/path/to/ckpt whole_homsap_chromosome_decoding/slurm_train.sh
```

### Running Directly (without SLURM)

```bash
bash whole_homsap_chromosome_decoding/run_sim.sh --stage 1
bash whole_homsap_chromosome_decoding/run_train.sh --stage 1
bash whole_homsap_chromosome_decoding/run_train.sh --stage 2 --checkpoint /path/to/ckpt
```

## Monitoring

```bash
# Check job queue
squeue -u $USER

# Watch training logs
tail -f /vast/projects/smathi/cohort/kkor/homsap_curriculum/stage1/logs/csv_metrics/version_*/metrics.csv
```

## Curriculum Stages

| Stage | Seq Length | n_windows | Replicates/size | Model Preset | Sim Time Limit |
|-------|-----------|-----------|-----------------|--------------|----------------|
| 1     | 50 Mb     | 25,000    | 100             | base_25k     | 2 days         |
| 2     | 100 Mb    | 50,000    | 50              | base_50k     | 2 days         |
| 3     | 150 Mb    | 75,000    | 25              | base_75k     | 4 days         |
| 4     | 250 Mb    | 125,000   | 10              | base_125k    | 4 days         |

Each stage fine-tunes from the previous checkpoint. Early stopping (patience=20 on val_loss) triggers progression to the next stage.

## Output Directory

```
/vast/projects/smathi/cohort/kkor/homsap_curriculum/
├── stage1/
│   ├── sims/              # Raw simulations + processed_lazy/
│   └── logs/              # Checkpoints + CSV metrics
├── stage2/
│   ├── sims/
│   └── logs/
├── stage3/
│   └── ...
└── stage4/
    └── ...
```

## Simulation Details

- Species: `HomSap` (stdpopsim)
- Demographic models: randomly sampled from stdpopsim catalog per replicate
- Mutation/recombination rates: stdpopsim species defaults (~1.29e-8 / ~1.25e-8)
- Sample sizes: 10, 25, 50, 100, 200 (all simulated per stage)
- SFS-only (no tsinfer trees)
- Window size: 2000 bp

See [STRATEGY.md](STRATEGY.md) for the full rationale.
