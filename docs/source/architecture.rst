Architecture
============

fastcxt replaces the autoregressive decoder-only transformer from cxt with a
**bidirectional Mamba encoder-decoder** that predicts all window TMRCAs in a
single forward pass.


Overview
--------

.. code-block:: text

   Input                     Encoder                         Decoder              Output
   ─────                     ───────                         ───────              ──────
   SFS features         ┌──→ BiMambaBlock ──→ FiLM  ──┐    BiMambaBlock ──┐
   (2, 4, 500, N)  ──→  │   BiMambaBlock ──→ FiLM  ──┤    + skip conns   ├──→  (μ, log σ²)
   + InputProjection    │   BiMambaBlock ──→ FiLM  ──┤    BiMambaBlock ──┤      per window
                        │   ...                      └──→ ...            ──┘
   Mutation rate ────────────────────────→ FiLM (γ, β)
   Tree topology ──────(optional)──→ TreeEncoder ──→ add to embedding


Bidirectional Mamba blocks
--------------------------

Each ``BiMambaBlock`` runs a Mamba-2 SSM in both the forward and backward
directions over the sequence of genomic windows, then merges the two hidden
states through a learned linear projection:

.. code-block:: text

   x ──→ LayerNorm ──┬──→ Mamba(forward)  ──┐
                     └──→ Mamba(backward) ──┤──→ Linear(2D → D) + Dropout ──→ x + residual

This is critical for TMRCA inference because the coalescence time at window *t*
depends on mutations both upstream and downstream.


FiLM conditioning
-----------------

The mutation rate (log-scaled) is projected to per-layer scale (γ) and shift (β)
parameters via ``FiLMLayer``:

.. math::

   h' = \gamma \odot h + \beta

where γ, β are generated from the log mutation rate by a small MLP. This
completely replaces the post-hoc bias correction from cxt.


Gaussian NLL loss
-----------------

The output head produces two values per window: predicted mean μ and
log-variance log σ². Training uses the heteroscedastic Gaussian negative
log-likelihood:

.. math::

   \mathcal{L} = \frac{1}{2} \left( \log \sigma^2 + \frac{(y - \mu)^2}{\sigma^2} \right)

At inference time, the variance is a direct model output -- no Monte Carlo
sampling needed.


Variable sample sizes
---------------------

``InputProjection`` handles arbitrary sample counts by zero-padding (or
truncating) the SFS sample dimension to ``max_samples`` before projecting
into the model's latent space. Any sample size from 4 to ``max_samples``
works without architecture changes.


Tree topology integration
-------------------------

When the ``--use-trees`` flag is enabled:

1. ``extract_topology_features`` extracts the coalescence order from each
   local tree in the tree sequence (rank, left-child-hash, right-child-hash).
2. ``TreeEncoder`` projects these into the model's latent dimension.
3. The tree embedding is added to the SFS embedding before the encoder.

The key insight: instead of predicting all n(n−1)/2 pairs independently,
predict the O(n) internal node times. All pairwise TMRCAs are then O(log n)
LCA lookups, giving O(n log n) total cost vs O(n²) without trees.

See :doc:`scaling` for quantitative benchmarks comparing all three modes
(cxt, fastcxt pairwise, fastcxt+tsinfer) across sample sizes.
