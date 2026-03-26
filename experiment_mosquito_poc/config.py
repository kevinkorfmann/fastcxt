"""Shared constants for the AnoGam (mosquito) experiment pipeline."""

from pathlib import Path

# Simulation parameters (Anopheles gambiae via stdpopsim)
SCENARIO = "AnoGam"
SAMPLE_SIZES = [10, 25, 50, 100, 200]
MAX_SAMPLES = 200
MAX_INTERNAL = MAX_SAMPLES - 1
SEQ_LEN = 100_000          # 100 kb
WINDOW_SIZE = 200           # 200 bp
N_WINDOWS = SEQ_LEN // WINDOW_SIZE  # 500
MUTATION_RATE = 3.5e-9      # AnoGam default
RECOMBINATION_RATE = 1e-8
NE = 1e6                   # approximate effective pop size for AnoGam (used in benchmarks)
NUM_TS_PER_SIZE = 200

# Paths (relative to experiment_mosquito/experiment/)
EXPERIMENT_DIR = Path(__file__).resolve().parent
SIMS_DIR = EXPERIMENT_DIR / "sims"
PROCESSED_DIR = SIMS_DIR / "processed"
OUTPUTS_DIR = EXPERIMENT_DIR / "outputs"
FIGURES_DIR = EXPERIMENT_DIR / "figures"

# NodeTimeModel hyperparameters
NODE_D_MODEL = 256
NODE_N_LAYERS = 4

# Plot colors
BLUE = "#2166ac"
GREEN = "#1b7837"
RED = "#b2182b"
ORANGE = "#e08214"
GREY = "#636363"
BLACK = "#252525"
