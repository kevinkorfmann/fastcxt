TimeAtlas
=========

The ``TimeAtlas`` is a purpose-built data structure for storing, querying, and
analyzing genome-wide pairwise TMRCA predictions at scale.


Motivation
----------

When running fastcxt on thousands of sample pairs across entire chromosomes,
the output is a large collection of per-block, per-pair (mean, variance)
vectors. The TimeAtlas organizes these into a queryable structure indexed
by chromosome arm, sample pair, and genomic position.


Creating an atlas
-----------------

.. code-block:: python

   from fastcxt.atlas import TimeAtlas
   import numpy as np

   atlas = TimeAtlas()

   # Add results for each chromosome arm
   atlas.add_arm(
       "2L",
       means=means_2L,           # (n_pairs, n_windows)
       variances=variances_2L,   # (n_pairs, n_windows)
       pairs=pairs_array,        # (n_pairs, 2)
       window_size=2000,
       mutation_rate=3.5e-9,
   )


Querying pairs
--------------

.. code-block:: python

   # Get TMRCA profile for one pair across a chromosome arm
   m, v = atlas.query_pair("2L", sample_a=0, sample_b=42)

   # m: (n_windows,) log-TMRCA means
   # v: (n_windows,) log-TMRCA variances


Querying positions
------------------

.. code-block:: python

   # Get all pairwise TMRCAs at a specific genomic position
   pairs, means_at_pos, vars_at_pos = atlas.query_window("2L", position_bp=5_000_000)

   # Get TMRCAs across a region
   pairs, means_region, vars_region = atlas.query_region("2L", 5_000_000, 6_000_000)


Finding extreme pairs
---------------------

.. code-block:: python

   # Which pairs have the deepest coalescence at a position?
   deep = atlas.deepest_pairs("2L", position_bp=5_000_000, k=10)

   # Which pairs are most closely related?
   shallow = atlas.shallowest_pairs("2L", position_bp=5_000_000, k=10)


Summary statistics
------------------

.. code-block:: python

   print(atlas.summary())
   # {
   #   "n_arms": 5,
   #   "total_pairs": 4950,
   #   "total_windows": 115000,
   #   "per_arm": {
   #     "2L": {"n_pairs": 4950, "n_windows": 24682, ...},
   #     ...
   #   }
   # }

   # Per-pair genome-wide mean TMRCA
   mean_tmrca_per_pair = atlas.mean_tmrca("2L")


Serialization
-------------

.. code-block:: python

   # Save
   atlas.save("my_atlas/")

   # Load
   atlas = TimeAtlas.load("my_atlas/")

Storage format:

.. code-block:: text

   my_atlas/
   ├── manifest.json       # metadata, arm list, parameters
   ├── 2L.npz              # means, variances, pairs, window_starts
   ├── 2R.npz
   ├── 3L.npz
   ├── 3R.npz
   └── X.npz


Iterating over pairs
--------------------

.. code-block:: python

   for sample_a, sample_b, means, variances in atlas.iter_pairs("2L"):
       # Process each pair's TMRCA profile
       avg_tmrca = np.exp(means).mean()
       print(f"Pair ({sample_a}, {sample_b}): mean TMRCA = {avg_tmrca:.0f}")


Visualization
-------------

The TimeAtlas integrates directly with the showcase visualization script
to produce publication-quality figures.  See :doc:`visualization` for the
full gallery and usage guide.

.. code-block:: bash

   # Generate all figures from simulated data
   python scripts/plot_atlas_showcase.py --outdir figures/

The script generates geographic maps of collection sites, TMRCA landscapes
across chromosome arms, population heatmaps, selective sweep panels, dense
raster heatmaps, and a composite dashboard — all powered by the TimeAtlas
query API.

.. image:: ../../figures/03_genome_landscape.png
   :width: 100%
   :alt: Genome-wide TMRCA landscape

*Genome-wide TMRCA landscape across 5 chromosome arms with per-population
ribbons and a selective sweep dip visible on chr2L.*
