Visualization
=============

fastcxt ships with a showcase plotting script that generates publication-quality
figures combining geographic maps of collection sites with genome-wide TMRCA
results.  The script works with both **real** TimeAtlas data and **simulated**
placeholder data for prototyping.


Quick start
-----------

.. code-block:: bash

   # Generate all 7 showcase figures with simulated Ag1000G-style data
   python scripts/plot_atlas_showcase.py --outdir figures/

   # Customize the simulation
   python scripts/plot_atlas_showcase.py \
       --outdir figures/ \
       --samples-per-pop 6 \
       --window-size 25000 \
       --seed 42

This produces 7 PNG files in the output directory.


Dependencies
------------

The visualization script requires the ``vis`` optional dependencies:

.. code-block:: bash

   pip install -e ".[vis]"

This installs:

- `matplotlib <https://matplotlib.org>`_ — core plotting
- `seaborn <https://seaborn.pydata.org>`_ — statistical visualizations
- `cartopy <https://scitools.org.uk/cartopy>`_ — geographic projections and basemaps


Figure catalog
--------------


01 — Collection sites
^^^^^^^^^^^^^^^^^^^^^

A Cartopy Mercator projection of sub-Saharan Africa with 10 Ag1000G population
sites.  Circle size scales with the real sample count from the Ag1000G phase 3
release.  Color maps within-population mean TMRCA on chr2L through a custom
deep-ocean-to-fire colormap.

.. image:: ../../figures/01_collection_sites.png
   :width: 100%
   :alt: Ag1000G collection sites

**Key insight**: East African populations (UGS, KES, TZS) tend to have different
within-population diversity compared to West African populations (GWA, GNS, BFM).


02 — Connectivity map
^^^^^^^^^^^^^^^^^^^^^

Great-circle arcs between all population pairs, rendered with Cartopy's geodetic
transform for correct curvature.  Arc color encodes between-population mean
TMRCA, width inversely encodes it: thicker cooler arcs = more recent
coalescence; thinner warmer arcs = deeper divergence.

.. image:: ../../figures/02_connectivity_map.png
   :width: 100%
   :alt: Population connectivity arcs

**Key insight**: Nearby populations (BFM ↔ BFS, UGS ↔ KES) show the thickest,
coolest arcs, while cross-continental pairs (GWA ↔ TZS) show thin warm arcs.


03 — Genome-wide TMRCA landscape
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All 5 chromosome arms (2L, 2R, 3L, 3R, X) displayed side-by-side with
widths proportional to arm length.  Each population's median within-population
TMRCA is shown as a line with the IQR shaded as a ribbon.

.. image:: ../../figures/03_genome_landscape.png
   :width: 100%
   :alt: Genome-wide TMRCA landscape

**Key features visible**:

- Selective sweep dip on chr2L near ~21 Mb (Rdl insecticide-resistance locus)
- Centromeric diversity bumps at the midpoint of each arm
- Population-level stratification of coalescence depth


04 — Population TMRCA heatmap
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A symmetric 10 × 10 matrix of mean pairwise TMRCA values between populations
on chr2L, with exact values annotated in each cell.  Paired with a geographic
inset showing collection locations.

.. image:: ../../figures/04_population_heatmap.png
   :width: 100%
   :alt: Population heatmap

The diagonal (within-population) cells show the shallowest coalescence.
Off-diagonal patterns reflect isolation-by-distance and demographic structure.


05 — Selective sweep panel
^^^^^^^^^^^^^^^^^^^^^^^^^^

A 5-panel deep dive into the Rdl sweep region on chr2L:

- **Panel A**: Full-arm mean TMRCA with sweep region highlighted
- **Panel B**: Zoomed per-population TMRCA traces through the sweep
- **Panel C**: Model uncertainty (log-variance) spiking at the sweep
- **Panel D**: Waterfall of all 780 pairs sorted by TMRCA at the Rdl locus
- **Panel E**: Geographic distance vs TMRCA at the sweep (isolation-by-distance)

.. image:: ../../figures/05_sweep_panel.png
   :width: 100%
   :alt: Selective sweep panel

The uncertainty spike (Panel C) is a natural model behavior — the sweep
creates an unusual pattern that the model is less confident about.


06 — Dense TMRCA raster
^^^^^^^^^^^^^^^^^^^^^^^^

A heatmap image with 780 pairs as rows (grouped by population) and 987 windows
as columns.  A color sidebar on the right identifies population membership.

.. image:: ../../figures/06_tmrca_raster.png
   :width: 100%
   :alt: Dense TMRCA raster

The sweep is strikingly visible as a dark vertical stripe.  Between-population
pair rows (bottom section) show deeper overall coalescence.


07 — Composite dashboard
^^^^^^^^^^^^^^^^^^^^^^^^^

All panels combined into a single showcase figure: geographic map, two arm
landscapes (2L and 2R), the population matrix, within-population violin
distributions, per-arm uncertainty bars, and a summary statistics table.

.. image:: ../../figures/07_composite_dashboard.png
   :width: 100%
   :alt: Composite dashboard


Using real data
---------------

The plotting functions accept any ``TimeAtlas`` object.  To use real inference
results instead of simulated data:

.. code-block:: python

   from fastcxt.atlas import TimeAtlas
   from scripts.plot_atlas_showcase import (
       plot_collection_sites,
       plot_connectivity_map,
       plot_genome_landscape,
       plot_population_heatmap,
       plot_sweep_panel,
       plot_tmrca_heatmap_raster,
       plot_composite_dashboard,
   )

   atlas = TimeAtlas.load("path/to/real_atlas/")

   # You'll also need to provide:
   #   pop_sample_map: dict[str, list[int]]  — population code → sample indices
   #   sample_pop: dict[int, str]            — sample index → population code

   from pathlib import Path
   outdir = Path("real_figures")
   outdir.mkdir(exist_ok=True)

   plot_collection_sites(atlas, pop_sample_map, sample_pop, outdir)
   plot_connectivity_map(atlas, pop_sample_map, sample_pop, outdir)
   plot_genome_landscape(atlas, pop_sample_map, sample_pop, outdir)
   plot_population_heatmap(atlas, pop_sample_map, sample_pop, outdir)
   plot_sweep_panel(atlas, pop_sample_map, sample_pop, outdir)
   plot_tmrca_heatmap_raster(atlas, pop_sample_map, sample_pop, outdir)
   plot_composite_dashboard(atlas, pop_sample_map, sample_pop, outdir)


Customizing the style
---------------------

The global color palette is defined at the top of
``scripts/plot_atlas_showcase.py``:

.. code-block:: python

   PALETTE = {
       "bg": "#0d1117",        # GitHub-dark background
       "panel": "#161b22",     # Panel fill
       "grid": "#21262d",      # Grid and borders
       "text": "#c9d1d9",      # Text color
       "accent": "#58a6ff",    # Primary accent
       "highlight": "#f0883e", # Secondary accent
       "sweep": "#da3633",     # Sweep marker red
   }

Two custom colormaps are used:

- ``tmrca_deep``: deep ocean → teal → lime → amber → red (for TMRCA values)
- ``uncertainty``: dark → navy → blue → purple → pink (for variance values)

Each population has a fixed color in the ``POP_COLORS`` dictionary for
consistent identification across all figures.


Population metadata
-------------------

The script includes real geographic coordinates and sample counts for 10
Ag1000G phase 3 populations:

.. list-table::
   :header-rows: 1
   :widths: 10 30 10 10 10

   * - Code
     - Location
     - Latitude
     - Longitude
     - n (Ag1000G)
   * - BFM
     - Burkina Faso (Mopti)
     - 14.49
     - −4.20
     - 81
   * - BFS
     - Burkina Faso (Savanna)
     - 11.17
     - −1.52
     - 82
   * - GNS
     - Guinea-Bissau
     - 12.10
     - −14.95
     - 12
   * - CMS
     - Cameroon
     - 3.85
     - 11.50
     - 79
   * - GAS
     - Gabon
     - −0.39
     - 9.45
     - 69
   * - UGS
     - Uganda
     - 0.35
     - 32.58
     - 112
   * - KES
     - Kenya
     - −0.09
     - 34.77
     - 48
   * - TZS
     - Tanzania
     - −6.17
     - 35.74
     - 29
   * - GWA
     - Gambia
     - 13.45
     - −16.58
     - 73
   * - AOM
     - Angola
     - −8.84
     - 13.23
     - 78
