Mosquito Analysis Protocol
==========================

fastcxt includes a dedicated analysis module for *Anopheles gambiae* population
genomics, designed around the Ag1000G project's data characteristics:
high missing-data rates, chromosome arm organization, and large sample sizes.


Chromosome arms
---------------

The *A. gambiae* genome is organized into five major autosomal/X arms:

.. code-block:: text

   2L:  49.4 Mb
   2R:  61.5 Mb
   3L:  42.0 Mb
   3R:  53.2 Mb
   X:   24.4 Mb

fastcxt tiles each arm into 1 Mb analysis blocks and runs inference across
all blocks for a set of sample pairs.


Accessibility masks
-------------------

Ag1000G provides accessibility masks indicating callable regions. fastcxt
integrates these directly into the SFS computation:

.. code-block:: python

   from fastcxt.mosquito import AccessibilityMask

   mask = AccessibilityMask.from_npz("ag1000g_masks.npz", arm="2L")
   print(f"Accessible fraction: {mask.accessible_fraction:.1%}")

   # Check a specific region
   accessible_bp = mask.accessible_bp(start=5_000_000, end=6_000_000)


Running the protocol
--------------------

.. code-block:: python

   from fastcxt.mosquito import MosquitoAnalysis, AccessibilityMask
   from fastcxt.atlas import TimeAtlas

   model = ...  # loaded FastCxtModel

   analysis = MosquitoAnalysis(
       model=model,
       device="cuda:0",
       block_size=1_000_000,
       batch_size=256,
   )

   # Load data for one arm
   gm_2L = ...       # (n_haploids, n_sites) genotype matrix
   pos_2L = ...       # (n_sites,) positions
   pairs = [(i, j) for i in range(100) for j in range(i+1, 100)]
   mask_2L = AccessibilityMask.from_npz("masks.npz", "2L")

   result = analysis.run_chromosome_arm(
       gm_2L, pos_2L, "2L",
       pivot_pairs=pairs,
       mutation_rate=3.5e-9,
       accessibility_mask=mask_2L,
   )

   # Build an atlas for the whole genome
   atlas = TimeAtlas()
   for arm in ["2L", "2R", "3L", "3R", "X"]:
       res = analysis.run_chromosome_arm(...)
       atlas.add_arm(arm, res["means"], res["variances"], pairs)

   atlas.save("anogam_atlas/")


Multi-arm analysis
------------------

.. code-block:: python

   genotype_data = {
       "2L": (gm_2L, pos_2L),
       "2R": (gm_2R, pos_2R),
       # ...
   }
   masks = {
       "2L": AccessibilityMask.from_npz("masks.npz", "2L"),
       "2R": AccessibilityMask.from_npz("masks.npz", "2R"),
   }

   results = analysis.run_all_arms(
       genotype_data, pairs,
       mutation_rate=3.5e-9,
       accessibility_masks=masks,
   )


Simulating mosquito-like data
-----------------------------

For testing and development:

.. code-block:: python

   from fastcxt.mosquito import simulate_anogam

   ts = simulate_anogam(seed=42, n_samples=50, segment_length=1e6)


Visualizing results
-------------------

After inference, the TimeAtlas can be visualized with geographic context
using the showcase plotting script.  See :doc:`visualization` for the
full gallery including:

- Collection site maps across sub-Saharan Africa
- Connectivity arcs colored by between-population TMRCA
- Genome-wide TMRCA landscapes across all chromosome arms
- Multi-panel selective sweep analysis (Rdl locus on chr2L)
- Dense pairwise TMRCA rasters grouped by population

.. code-block:: bash

   python scripts/plot_atlas_showcase.py --outdir figures/

.. image:: ../../figures/02_connectivity_map.png
   :width: 100%
   :alt: Ag1000G population connectivity

*Population connectivity arcs across Africa.  Thicker, cooler arcs indicate
more recent coalescence between populations.*
