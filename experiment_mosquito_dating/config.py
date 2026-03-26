"""Shared constants for the mosquito population dating experiment.

Pairwise TMRCA dating of AG1000G populations using tsinfer tree sequences
and the trained base_anogam FastCxtModel.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (sietch defaults, override via env vars or CLI)
# ---------------------------------------------------------------------------

EXPERIMENT_DIR = Path(__file__).resolve().parent

# tsinfer+tsdate tree sequences (per arm)
# File pattern: gamb.{arm}.gff.dated.ne.trees
TREES_DIR = Path("/sietch_colab/data_share/Ag1000G/Ag3.0/args_trees/tsinfer_data_v2")
TREE_FILE_TEMPLATE = "gamb.{arm}.gff.dated.ne.trees"

# Accessibility mask
ACCESSIBILITY_MASK = Path(
    "/sietch_colab/data_share/Ag1000G/Ag3.0/args_trees/singer/agp3.is_accessible.txt.npz"
)

# Sample metadata
METADATA_CSV = Path(
    "/sietch_colab/data_share/Ag1000G/Ag3.0/args_trees/gamb.meta.tsinfer.csv"
)

# Trained pairwise checkpoint (set via --checkpoint or update this default)
# This should point to the best base_anogam checkpoint from experiment_mosquito_poc
DEFAULT_CHECKPOINT = None  # must be provided via CLI

# ---------------------------------------------------------------------------
# Model / inference constants (must match training config: base_anogam)
# ---------------------------------------------------------------------------

BLOCK_SIZE = 100_000       # 100 kb blocks (matches base_anogam training)
WINDOW_SIZE = 200          # 200 bp windows
N_WINDOWS = 500            # BLOCK_SIZE / WINDOW_SIZE
MUTATION_RATE = 3.5e-9     # Anopheles gambiae (stdpopsim)

# ---------------------------------------------------------------------------
# Chromosome arms
# ---------------------------------------------------------------------------

CHROMOSOME_ARMS = {
    "2L": 49_364_325,
    "2R": 61_545_105,
    "3L": 41_963_435,
    "3R": 53_200_684,
    "X":  24_393_108,
}

# ---------------------------------------------------------------------------
# Inference settings
# ---------------------------------------------------------------------------

BATCH_SIZE = 64
BUILD_WORKERS = 4
MAX_PAIRS_PER_POP = 200    # cap pairwise pairs per population (None = all)
