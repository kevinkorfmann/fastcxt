fastcxt
=======

.. raw:: html

   <div style="background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%); border-radius: 12px; padding: 2.5rem; margin-bottom: 2rem; border: 1px solid #334155; text-align: center;">
     <p style="color: #94a3b8; font-size: 1.15rem; margin-top: 0.5rem;">
       Fast pairwise coalescence time inference with Mamba state-space models
     </p>
     <p style="color: #64748b; font-size: 0.9rem;">
       Single-pass · Built-in uncertainty · Mutation-rate conditioned · O(n) with tree topology
     </p>
   </div>

fastcxt predicts pairwise time to most recent common ancestor (TMRCA) from
genotype data using a **bidirectional Mamba encoder-decoder**. It replaces the
autoregressive transformer from `cxt <https://github.com/kevinkorfmann/cxt>`_
with a single-pass architecture that produces means and calibrated variances
for all genomic windows in one forward pass — no stochastic sampling, no
post-hoc correction.


.. grid:: 3
   :gutter: 3

   .. grid-item-card:: Quick Start
      :link: quickstart
      :link-type: doc

      Install and run inference in five minutes.

   .. grid-item-card:: Tutorial
      :link: tutorial
      :link-type: doc

      Full pipeline: simulate, train, infer, visualize.

   .. grid-item-card:: Algorithm
      :link: architecture
      :link-type: doc

      Bidirectional Mamba, FiLM conditioning, Beta-NLL loss.

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: cxt vs fastcxt
      :link: comparison
      :link-type: doc

      Architecture comparison and migration guide.

   .. grid-item-card:: Mosquito Protocol
      :link: mosquito_protocol
      :link-type: doc

      Ag1000G analysis: inversions, karyotypes, selection scans.

   .. grid-item-card:: Figure Gallery
      :link: gallery
      :link-type: doc

      Publication-quality plots from the Ag1000G analysis.

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: Demography
      :link: demography
      :link-type: doc

      IICR estimation and Ne(t) from TMRCA distributions.

   .. grid-item-card:: Geographic
      :link: visualization
      :link-type: doc

      Maps, sparklines, and spatial TMRCA patterns.

   .. grid-item-card:: API Reference
      :link: api/index
      :link-type: doc

      Full Python API for all modules.


How it works
------------

.. raw:: html

   <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin: 1.5rem 0;">
     <div style="border-left: 3px solid #60a5fa; padding-left: 1rem;">
       <strong style="color: #60a5fa;">1. Build SFS features</strong><br/>
       <span style="color: #94a3b8;">Site frequency spectrum in XOR/XNOR channels from a genotype matrix — same representation as cxt.</span>
     </div>
     <div style="border-left: 3px solid #60a5fa; padding-left: 1rem;">
       <strong style="color: #60a5fa;">2. Single forward pass</strong><br/>
       <span style="color: #94a3b8;">Bidirectional Mamba encoder reads the full sequence, decoder outputs (μ, log σ²) for every window.</span>
     </div>
     <div style="border-left: 3px solid #60a5fa; padding-left: 1rem;">
       <strong style="color: #60a5fa;">3. FiLM conditioning</strong><br/>
       <span style="color: #94a3b8;">Mutation rate injected via learned scale/shift at each encoder layer — no post-hoc correction needed.</span>
     </div>
     <div style="border-left: 3px solid #60a5fa; padding-left: 1rem;">
       <strong style="color: #60a5fa;">4. Calibrated uncertainty</strong><br/>
       <span style="color: #94a3b8;">Beta-NLL loss directly models variance alongside the mean — 95% CI = exp(μ ± 1.96√σ²).</span>
     </div>
   </div>


Minimal example
---------------

.. code-block:: python

   import tskit
   from fastcxt.translate import translate_from_genotype_matrix

   ts = tskit.load("data.trees")
   gm = ts.genotype_matrix().T
   positions = ts.tables.sites.position

   pairs = [(0, 1), (0, 2), (1, 2)]
   blocks = [(i, i + 100_000) for i in range(0, 1_000_000, 100_000)]

   means, variances, index_map = translate_from_genotype_matrix(
       gm, positions, model,
       blocks=blocks, pivot_pairs=pairs,
       mutation_rate=3.5e-9, device="cuda:0",
       batch_size=128, build_workers=64,
   )


.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   quickstart
   tutorial
   comparison
   architecture
   simulation
   preprocessing
   training
   inference

.. toctree::
   :maxdepth: 2
   :caption: Applications
   :hidden:

   mosquito_protocol
   demography
   time_atlas
   scaling
   visualization
   gallery

.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:

   api/index
   cli
   changelog
