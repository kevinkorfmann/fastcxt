Inference
=========

fastcxt inference is a single forward pass per pair -- no autoregressive
sampling, no stochastic averaging.


From a tree sequence
--------------------

.. code-block:: python

   from fastcxt.translate import translate_from_ts

   means, variances, index_map = translate_from_ts(
       ts, model,
       pivot_pairs=[(0, 1), (2, 3), (0, 4)],
       mutation_rate=1e-8,
       device="cuda:0",
       batch_size=256,
   )

   # means: (N, W) log-TMRCA predictions
   # variances: (N, W) prediction variances
   # index_map: (N, 2) mapping to [block_idx, pair_idx]


From a genotype matrix
----------------------

.. code-block:: python

   from fastcxt.translate import translate_from_genotype_matrix

   means, variances, index_map = translate_from_genotype_matrix(
       gm=genotype_matrix,         # (n_haploids, n_sites)
       positions=site_positions,    # (n_sites,) in bp
       model=model,
       blocks=[(0, 1_000_000), (1_000_000, 2_000_000)],
       pivot_pairs=[(0, 1)],
       mutation_rate=3.5e-9,
   )


Universal entry point
---------------------

The ``translate`` function auto-detects input type:

.. code-block:: python

   from fastcxt.translate import translate

   # Accepts tree sequences or (gm, positions) tuples
   means, variances, index_map = translate(ts, model, pivot_pairs=[(0, 1)])
   means, variances, index_map = translate((gm, pos), model, pivot_pairs=[(0, 1)])


Understanding the output
------------------------

- **means**: predicted log-TMRCA per window. Exponentiate for natural scale:
  ``np.exp(means)``
- **variances**: predicted variance of log-TMRCA. Use for confidence intervals:
  ``np.exp(means ± 1.96 * np.sqrt(variances))``
- **index_map**: maps each output row to ``[block_index, pair_index]``


Scaling to whole genomes
------------------------

For whole-genome analysis, use the ``TimeAtlas``:

.. code-block:: python

   from fastcxt.atlas import TimeAtlas

   atlas = TimeAtlas()
   for arm in ["2L", "2R", "3L", "3R", "X"]:
       result = analysis.run_chromosome_arm(gm, pos, arm, pairs, mutation_rate)
       atlas.add_arm(arm, result["means"], result["variances"], pairs)
   atlas.save("genome_atlas/")

See :doc:`time_atlas` for the full query API and :doc:`visualization` for
generating publication-quality figures from atlas data.
