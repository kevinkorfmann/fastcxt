fastcxt
=======

**Fast pairwise coalescence time inference with Mamba state-space models.**

fastcxt predicts pairwise time to most recent common ancestor (TMRCA) from
genotype data using a bidirectional Mamba encoder-decoder. It replaces the
autoregressive transformer from cxt with a single-pass architecture that
produces means and calibrated variances for all genomic windows in one
forward pass — no stochastic sampling, no post-hoc correction.

.. image:: ../../figures/07_composite_dashboard.png
   :width: 100%
   :alt: fastcxt TimeAtlas dashboard

*Composite dashboard with geographic context, TMRCA landscapes, population
heatmap, uncertainty, and per-arm summary statistics (simulated Ag1000G data).*


.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Getting Started
      :link: quickstart
      :link-type: doc

      Installation, first simulation, and inference in five minutes.

   .. grid-item-card:: End-to-End Tutorial
      :link: tutorial
      :link-type: doc

      Full walkthrough from simulation → preprocessing → training →
      inference → TimeAtlas → visualization.

   .. grid-item-card:: cxt vs fastcxt
      :link: comparison
      :link-type: doc

      Detailed comparison of the old transformer and new Mamba
      architecture, with migration guide.

   .. grid-item-card:: Architecture
      :link: architecture
      :link-type: doc

      How the bidirectional Mamba encoder-decoder works, FiLM conditioning,
      and Gaussian NLL loss.

   .. grid-item-card:: Mosquito Protocol
      :link: mosquito_protocol
      :link-type: doc

      *Anopheles gambiae* analysis at scale with accessibility mask support
      for missing data.

   .. grid-item-card:: TimeAtlas
      :link: time_atlas
      :link-type: doc

      Purpose-built data structure for genome-wide TMRCA storage, queries,
      and analytics.

   .. grid-item-card:: Scaling & Benchmarks
      :link: scaling
      :link-type: doc

      Runtime comparison of cxt vs fastcxt vs fastcxt+tsinfer across
      sample sizes, with instructions for running benchmarks.

   .. grid-item-card:: Visualization
      :link: visualization
      :link-type: doc

      Publication-quality geographic maps, TMRCA landscapes, sweep panels,
      connectivity arcs, and composite dashboards.

   .. grid-item-card:: API Reference
      :link: api/index
      :link-type: doc

      Full Python API documentation for all modules.


Key features
------------

- **Single-pass inference**: one forward pass per pair produces means and
  variances for all genomic windows — no autoregressive sampling.
- **Built-in uncertainty**: Gaussian NLL loss directly models prediction
  variance alongside the mean.
- **Mutation-rate conditioning**: FiLM layers inject mutation rate as a
  model input, replacing post-hoc correction.
- **Variable sample sizes**: ``InputProjection`` handles any sample count
  up to ``max_samples`` without adapter modules.
- **Tree topology integration**: optional ``--use-trees`` flag exploits
  tsinfer coalescence order for O(n log n) scaling.
- **TimeAtlas**: purpose-built data structure for storing and querying
  at-scale TMRCA predictions across entire genomes.
- **Geographic visualization**: Cartopy-powered maps, TMRCA landscapes,
  connectivity arcs, sweep panels, and composite dashboards.


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
   time_atlas
   scaling
   visualization

.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:

   api/index
   cli
   changelog
