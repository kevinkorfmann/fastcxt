Quickstart
==========

Installation
------------

fastcxt requires Python 3.10+ and a CUDA-capable GPU for the Mamba kernels.

.. tab-set::

   .. tab-item:: uv (recommended)

      .. code-block:: bash

         uv pip install -e ".[all]"

   .. tab-item:: pip

      .. code-block:: bash

         pip install -e ".[all]"


For CPU-only development (e.g. preprocessing, simulation), install without
the GPU dependencies:

.. code-block:: bash

   uv pip install -e ".[sim,docs,dev]"


End-to-end example
------------------

1. **Simulate** training data (1000 tree sequences, constant demography):

.. code-block:: bash

   fastcxt-simulate --scenario constant --data-dir ./sims/constant --num-ts 1000

2. **Preprocess** into SFS features and TMRCA targets:

.. code-block:: bash

   fastcxt-preprocess --base-dir ./sims/constant --out-subdir processed

3. **Train** a model:

.. code-block:: bash

   fastcxt-train --model base --dataset-path ./sims/constant/processed --gpus 0

4. **Run inference** from Python:

.. code-block:: python

   import fastcxt
   from fastcxt.config import PRESETS
   from fastcxt.model import FastCxtModel

   config = PRESETS["base"]
   model = FastCxtModel(config)
   # model.load_state_dict(torch.load("checkpoint.pt"))

   from fastcxt.translate import translate_from_ts
   import tskit

   ts = tskit.load("my_data.trees")
   means, variances, index_map = translate_from_ts(
       ts, model,
       pivot_pairs=[(0, 1), (0, 2)],
       mutation_rate=1e-8,
       device="cuda:0",
   )

5. **Build a TimeAtlas** for genome-wide results:

.. code-block:: python

   from fastcxt.atlas import TimeAtlas

   atlas = TimeAtlas()
   atlas.add_arm("2L", means, variances, pairs, window_size=2000)
   atlas.save("my_atlas/")

   # Query later
   atlas = TimeAtlas.load("my_atlas/")
   m, v = atlas.query_pair("2L", sample_a=0, sample_b=5)

6. **Visualize** with geographic context:

.. code-block:: bash

   # Generate showcase figures with simulated placeholder data
   python scripts/plot_atlas_showcase.py --outdir figures/

This generates publication-quality geographic maps, TMRCA landscapes, population
heatmaps, selective sweep panels, and a composite dashboard.  See
:doc:`visualization` for the full gallery.

.. image:: ../../figures/07_composite_dashboard.png
   :width: 100%
   :alt: Composite dashboard
