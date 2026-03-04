End-to-End Tutorial
===================

This tutorial walks through the complete fastcxt workflow: simulating training
data, preprocessing, training a model, running inference on new data, building
a TimeAtlas, and generating publication-quality visualizations.

Each step includes the exact commands to run and explanations of what's
happening under the hood.


Prerequisites
-------------

.. code-block:: bash

   # Install fastcxt with all optional dependencies
   uv pip install -e ".[all]"

   # Or with pip
   pip install -e ".[all]"

   # Verify installation
   python -c "import fastcxt; print(fastcxt.__version__)"


Step 1 — Simulate training data
--------------------------------

fastcxt trains on coalescent simulations.  The simulation pipeline uses
a **scenario registry** — each scenario maps to either a ``stdpopsim``
species or a custom ``msprime`` demographic model.

.. code-block:: bash

   # Simulate 500 tree sequences under constant demography
   fastcxt-simulate \
       --scenario constant \
       --data-dir ./tutorial/sims/constant \
       --num-ts 500 \
       --n-samples 25 \
       --sequence-length 1000000

   # Add diversity with different demographic histories
   fastcxt-simulate \
       --scenario sawtooth \
       --data-dir ./tutorial/sims/sawtooth \
       --num-ts 300

   # Simulate Anopheles gambiae with stdpopsim
   fastcxt-simulate \
       --scenario AnoGam \
       --data-dir ./tutorial/sims/anogam \
       --num-ts 200

This creates ``.trees`` files in each output directory.  Each tree sequence
contains a simulated genome segment with mutations, from which we extract
training features and targets.

**What's generated:**

.. code-block:: text

   tutorial/sims/
   ├── constant/
   │   ├── ts_00000000.trees
   │   ├── ts_00000001.trees
   │   └── ... (500 files)
   ├── sawtooth/
   │   └── ... (300 files)
   └── anogam/
       └── ... (200 files)

**Available scenarios:**

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Scenario
     - Type
     - Description
   * - ``constant``
     - msprime
     - Constant population size (Ne=10,000)
   * - ``sawtooth``
     - msprime
     - Oscillating population sizes
   * - ``island``
     - msprime
     - 3-deme island model with migration
   * - ``AnoGam``
     - stdpopsim
     - *Anopheles gambiae* (mosquito)
   * - ``HomSap``
     - stdpopsim
     - *Homo sapiens*
   * - ``DroMel``
     - stdpopsim
     - *Drosophila melanogaster*


Step 2 — Preprocess into training data
----------------------------------------

Preprocessing converts tree sequences into SFS feature tensors and log-TMRCA
target vectors.

.. code-block:: bash

   # Preprocess all simulated data
   fastcxt-preprocess \
       --base-dir ./tutorial/sims \
       --out-subdir processed \
       --window-size 2000 \
       --num-pairs 200 \
       --global-seed 42 \
       --num-workers 8

   # With tree topology features (for the tree-aware model)
   fastcxt-preprocess \
       --base-dir ./tutorial/sims \
       --out-subdir processed_trees \
       --extract-trees \
       --num-pairs 200

**What preprocessing does for each tree sequence:**

1. Extracts the genotype matrix and site positions
2. Applies biallelic filtering (removes non-biallelic sites)
3. Randomly samples ``num-pairs`` sample pairs
4. For each pair: computes the multi-scale SFS (4 window scales × 2 channels)
5. Computes the exact span-weighted TMRCA per window (continuous target)
6. Estimates the per-site mutation rate from the data
7. Saves ``X.npy``, ``y.npy``, ``pairs.npy``, ``meta.json``

**Output per simulation:**

.. code-block:: text

   processed/train/default/ts_00000042_i0/
   ├── X.npy        # (P, 2, 4, 500, N) SFS features, float16
   ├── y.npy        # (P, 500) log-TMRCA targets, float16
   ├── pairs.npy    # (P, 2) sample pair indices, int32
   └── meta.json    # {"mutation_rate": 1.2e-8, "num_samples": 50, ...}


Step 3 — Train a model
-----------------------

Training uses PyTorch Lightning with Gaussian NLL loss.

.. code-block:: bash

   # Train a base model (requires GPU)
   fastcxt-train \
       --model base \
       --dataset-path ./tutorial/sims/processed \
       --gpus 0 \
       --epochs 10 \
       --batch-size 128 \
       --lr 3e-4

   # Train a tree-aware model
   fastcxt-train \
       --model base_trees \
       --dataset-path ./tutorial/sims/processed_trees \
       --gpus 0 \
       --epochs 10

**What happens during training:**

- The model receives batches of ``(X, mutation_rate)`` → predicts ``(μ, log σ²)``
- Loss is Gaussian NLL: penalizes both mean error and miscalibrated variance
- Metrics tracked: ``train_loss``, ``val_loss``, ``val_rmse``, ``val_coverage_95``
- The ``val_coverage_95`` metric measures what fraction of true TMRCAs fall within
  the predicted 95% confidence interval — a well-calibrated model should score ~0.95

**Model presets:**

.. list-table::
   :header-rows: 1
   :widths: 15 15 15 15 15

   * - Preset
     - d_model
     - Encoder
     - Decoder
     - Trees
   * - ``small``
     - 128
     - 4 layers
     - 2 layers
     - no
   * - ``base``
     - 256
     - 6 layers
     - 4 layers
     - no
   * - ``large``
     - 512
     - 8 layers
     - 6 layers
     - no
   * - ``base_trees``
     - 256
     - 6 layers
     - 4 layers
     - yes


Step 4 — Run inference
-----------------------

Inference is a single forward pass per pair.

**From a tree sequence:**

.. code-block:: python

   import torch
   from fastcxt.config import PRESETS
   from fastcxt.model import FastCxtModel
   from fastcxt.translate import translate_from_ts
   import tskit

   # Load model
   config = PRESETS["base"]
   model = FastCxtModel(config)
   model.load_state_dict(torch.load("checkpoint.pt", map_location="cpu"))

   # Load data
   ts = tskit.load("path/to/data.trees")

   # Run inference
   means, variances, index_map = translate_from_ts(
       ts, model,
       pivot_pairs=[(0, 1), (0, 2), (1, 2)],
       mutation_rate=1e-8,
       device="cuda:0",
       batch_size=256,
   )

   # means: (N, W) predicted log-TMRCA
   # variances: (N, W) predicted variance
   import numpy as np
   tmrca = np.exp(means)                                   # natural scale
   ci_lo = np.exp(means - 1.96 * np.sqrt(variances))       # 95% CI lower
   ci_hi = np.exp(means + 1.96 * np.sqrt(variances))       # 95% CI upper

**From a genotype matrix (e.g. VCF data):**

.. code-block:: python

   from fastcxt.translate import translate_from_genotype_matrix

   means, variances, index_map = translate_from_genotype_matrix(
       gm=genotype_matrix,         # (n_haploids, n_sites) int8
       positions=site_positions,    # (n_sites,) float64 in bp
       model=model,
       blocks=[(0, 1_000_000), (1_000_000, 2_000_000)],
       pivot_pairs=[(0, 1)],
       mutation_rate=3.5e-9,
       device="cuda:0",
   )


Step 5 — Build a TimeAtlas
---------------------------

For genome-wide analysis across multiple chromosome arms, collect results
into a TimeAtlas.

.. code-block:: python

   from fastcxt.atlas import TimeAtlas
   import numpy as np

   atlas = TimeAtlas()
   atlas.metadata = {
       "species": "Anopheles gambiae",
       "description": "Tutorial example",
   }

   # Add results for each chromosome arm
   for arm in ["2L", "2R", "3L", "3R", "X"]:
       # (run inference for this arm ...)
       atlas.add_arm(
           arm,
           means=means_dict[arm],
           variances=variances_dict[arm],
           pairs=pairs_array,
           window_size=2000,
           mutation_rate=3.5e-9,
       )

   # Save for later use
   atlas.save("tutorial/my_atlas/")
   print(atlas.summary())

**Query the atlas:**

.. code-block:: python

   atlas = TimeAtlas.load("tutorial/my_atlas/")

   # TMRCA profile for one pair across a chromosome arm
   m, v = atlas.query_pair("2L", sample_a=0, sample_b=5)

   # All pairs at one genomic position
   pairs, means_at_pos, vars_at_pos = atlas.query_window("2L", position_bp=20_000_000)

   # Deepest-coalescing pairs at a position
   deep = atlas.deepest_pairs("2L", position_bp=20_700_000, k=10)
   print("Pairs with deepest TMRCA at Rdl:", deep)

   # Summary statistics
   print(f"Total: {atlas.total_pairs} pairs × {atlas.total_windows} windows")


Step 6 — Mosquito protocol (Ag1000G)
--------------------------------------

For *Anopheles gambiae* data specifically, the ``MosquitoAnalysis`` class
handles chromosome-arm tiling and accessibility masks:

.. code-block:: python

   from fastcxt.mosquito import MosquitoAnalysis, AccessibilityMask

   # Set up the analysis
   analysis = MosquitoAnalysis(
       model=model,
       device="cuda:0",
       block_size=1_000_000,
       batch_size=256,
   )

   # Load accessibility mask (for missing data regions)
   mask = AccessibilityMask.from_npz("ag1000g_masks.npz", arm="2L")
   print(f"chr2L accessible fraction: {mask.accessible_fraction:.1%}")

   # Run inference on one arm
   result = analysis.run_chromosome_arm(
       genotype_matrix, positions, "2L",
       pivot_pairs=pairs,
       mutation_rate=3.5e-9,
       accessibility_mask=mask,
   )

   # Add to atlas
   atlas.add_arm("2L", result["means"], result["variances"], pairs)

See :doc:`mosquito_protocol` for the complete Ag1000G protocol.


Step 7 — Visualize results
---------------------------

Generate the full suite of publication-quality figures:

.. code-block:: bash

   # Generate showcase figures with simulated placeholder data
   python scripts/plot_atlas_showcase.py --outdir figures/

This produces 8 figures:

1. **Collection sites** — geographic map with TMRCA-colored markers
2. **Connectivity map** — great-circle arcs between populations
3. **Genome landscape** — TMRCA ribbons across all 5 chromosome arms
4. **Population heatmap** — hierarchically clustered TMRCA matrix
5. **Sweep panel** — 5-panel Rdl sweep analysis
6. **TMRCA raster** — dense pairwise heatmap across windows
7. **Composite dashboard** — all-in-one summary figure
8. **Scaling comparison** — cxt vs fastcxt vs fastcxt+tsinfer

See :doc:`visualization` for the full gallery and customization guide.

To visualize a real TimeAtlas:

.. code-block:: python

   from scripts.plot_atlas_showcase import (
       plot_genome_landscape,
       plot_sweep_panel,
       plot_composite_dashboard,
   )
   from fastcxt.atlas import TimeAtlas
   from pathlib import Path

   atlas = TimeAtlas.load("tutorial/my_atlas/")
   outdir = Path("tutorial/figures")
   outdir.mkdir(exist_ok=True)

   # pop_sample_map: dict mapping population code → list of sample indices
   # sample_pop: dict mapping sample index → population code
   plot_genome_landscape(atlas, pop_sample_map, sample_pop, outdir)
   plot_sweep_panel(atlas, pop_sample_map, sample_pop, outdir)
   plot_composite_dashboard(atlas, pop_sample_map, sample_pop, outdir)


Step 8 — Benchmark scaling
---------------------------

Verify the scaling behavior on your hardware:

.. code-block:: bash

   # Run all benchmark modes
   fastcxt-benchmark --mode all \
       --sample-sizes 5 10 25 50 100 \
       --device cuda:0 \
       --output tutorial/benchmarks.json

See :doc:`scaling` for detailed benchmark instructions and interpretation.


Complete command reference
--------------------------

.. code-block:: bash

   # ─── Step 1: Simulate ───
   fastcxt-simulate --scenario constant --data-dir ./sims/constant --num-ts 500
   fastcxt-simulate --scenario sawtooth --data-dir ./sims/sawtooth --num-ts 300
   fastcxt-simulate --scenario AnoGam  --data-dir ./sims/anogam   --num-ts 200

   # ─── Step 2: Preprocess ───
   fastcxt-preprocess --base-dir ./sims --out-subdir processed --num-pairs 200

   # ─── Step 3: Train ───
   fastcxt-train --model base --dataset-path ./sims/processed --gpus 0 --epochs 10

   # ─── Step 4–5: Inference + atlas (Python) ───
   python -c "
   from fastcxt.translate import translate_from_ts
   from fastcxt.atlas import TimeAtlas
   # ... see Step 4–5 above
   "

   # ─── Step 7: Visualize ───
   python scripts/plot_atlas_showcase.py --outdir figures/

   # ─── Step 8: Benchmark ───
   fastcxt-benchmark --mode all --device cuda:0


One-command reproduction
------------------------

Instead of running each step manually, use the reproduction script to
execute everything end-to-end:

.. code-block:: bash

   # Full pipeline: simulate → preprocess → train → benchmark → infer → atlas → figures
   ./scripts/reproduce.sh

   # Run only specific stages
   ./scripts/reproduce.sh simulate preprocess train
   ./scripts/reproduce.sh infer atlas figures

   # Override GPU list and output directory
   GPUS="0 1" BASE_DIR=/scratch/repro ./scripts/reproduce.sh

The ``figures`` stage automatically detects whether a real TimeAtlas exists
and replaces all simulated placeholders with real inference results.
See ``docs/ag1000g_strategy.md`` for the full Ag1000G analysis protocol.


Next steps
----------

- :doc:`comparison` — how fastcxt differs from cxt in detail
- :doc:`mosquito_protocol` — full Ag1000G analysis protocol
- :doc:`time_atlas` — advanced TimeAtlas queries and analytics
- :doc:`scaling` — runtime benchmarks and scaling analysis
- :doc:`visualization` — customizing the showcase figures
