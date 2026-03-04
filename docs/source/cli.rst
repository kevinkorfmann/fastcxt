CLI Reference
=============


fastcxt-simulate
----------------

.. code-block:: text

   usage: fastcxt-simulate [--scenario NAME] [--data-dir DIR] [options]

   Simulate tree sequences for fastcxt training.

   --scenario          Scenario name: species code (AnoGam, HomSap, ...) or
                       msprime scenario (constant, sawtooth, island)
   --data-dir          Output directory for .trees files
   --num-ts            Number of tree sequences (default: 1000)
   --n-samples         Diploid individuals (default: 25)
   --sequence-length   Segment length in bp (default: 1000000)
   --mutation-rate     Override mutation rate
   --recombination-rate  Override recombination rate
   --genetic-map       stdpopsim genetic map name
   --num-processes     Parallel workers (default: 8)


fastcxt-preprocess
------------------

.. code-block:: text

   usage: fastcxt-preprocess [--base-dir DIR] [options]

   Preprocess tree sequences into training data.

   --base-dir          Root directory with .trees files
   --out-subdir        Output subdirectory name (default: processed)
   --window-size       Base window size in bp (default: 2000)
   --sequence-length   Expected sequence length (default: 1000000)
   --num-pairs         Pairs to sample per tree sequence (default: 200)
   --simplify-n        Simplify to first N samples (default: 0 = keep all)
   --train-ratio       Train/test split ratio (default: 0.9)
   --global-seed       Random seed (default: 12345)
   --skip-existing     Skip already-processed files
   --num-workers       Parallel workers (default: CPU count)
   --extract-trees     Compute tree topology features
   --mutation-rate     Override mutation rate
   --accessibility-mask  Path to .npz accessibility mask


fastcxt-train
-------------

.. code-block:: text

   usage: fastcxt-train [--model PRESET] [--dataset-path DIR] [options]

   Train a fastcxt model.

   --model             Model preset: small, base, large, base_trees
   --dataset-path      Path to preprocessed data
   --gpus              GPU device IDs (default: 0)
   --epochs            Training epochs (default: 10)
   --lr                Maximum learning rate (default: 3e-4)
   --batch-size        Batch size (default: 128)
   --grad-accum        Gradient accumulation steps (default: 4)
   --workers           Data loading workers (default: 8)
   --checkpoint        Resume from checkpoint
   --log-dir           Logging directory


fastcxt-benchmark
-----------------

.. code-block:: text

   usage: fastcxt-benchmark [--mode MODE] [options]

   Run scaling benchmarks.

   --mode              Modes: fastcxt_notree, fastcxt_tree, all
   --sample-sizes      Sample sizes to benchmark (default: 5 10 25 50 100)
   --device            Device (default: cuda:0)
   --batch-size        Batch size (default: 64)
   --output            JSON output path for results
