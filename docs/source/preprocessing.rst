Preprocessing
=============

The preprocessing pipeline converts simulated tree sequences into training
data: SFS feature tensors, log-TMRCA target vectors, and metadata.


Pipeline overview
-----------------

.. code-block:: text

   .trees files
       │
       ├──→ genotype matrix extraction
       ├──→ biallelic filtering
       ├──→ (optional) accessibility mask application
       ├──→ multi-scale SFS computation (xor/xnor × 4 window scales)
       ├──→ windowed TMRCA computation (exact span-weighted averages)
       ├──→ (optional) tree topology feature extraction
       │
       └──→ output per simulation:
               X.npy           (P, 2, 4, W, N)  float16
               y.npy           (P, W)            float16  log-TMRCA
               pairs.npy       (P, 2)            int32
               meta.json       { mutation_rate, num_samples, ... }
               tree_feats.npy  (P, W, feat_dim)  float32  (optional)


CLI usage
---------

.. code-block:: bash

   # Basic preprocessing
   fastcxt-preprocess --base-dir ./sims/anogam --out-subdir processed

   # With accessibility mask (for real data with missing regions)
   fastcxt-preprocess --base-dir ./sims/anogam \
       --accessibility-mask masks/ag1000g_accessible.npz \
       --out-subdir processed

   # With tree topology features
   fastcxt-preprocess --base-dir ./sims/anogam \
       --extract-trees \
       --out-subdir processed

   # Customize pair sampling
   fastcxt-preprocess --base-dir ./sims/anogam \
       --num-pairs 500 \
       --global-seed 42 \
       --out-subdir processed


Accessibility masks
-------------------

For species with high missing-data rates (e.g. *Anopheles gambiae* from
Ag1000G), accessibility masks ensure the SFS is computed only over callable
regions:

.. code-block:: python

   from fastcxt.preprocess import apply_accessibility_mask
   import numpy as np

   mask = np.load("ag1000g_accessible_2L.npz")["is_accessible"]
   gm_filtered, pos_filtered = apply_accessibility_mask(gm, positions, mask, seq_len)


Output layout
-------------

.. code-block:: text

   processed/
   ├── train/
   │   ├── default/
   │   │   ├── ts_00000000_i0/
   │   │   │   ├── X.npy
   │   │   │   ├── y.npy
   │   │   │   ├── pairs.npy
   │   │   │   └── meta.json
   │   │   └── ts_00000001_i1/
   │   │       └── ...
   │   └── scenario_name/
   │       └── ...
   └── test/
       └── ...
