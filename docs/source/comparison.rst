cxt vs fastcxt
==============

fastcxt is a ground-up redesign of `cxt <https://github.com/kevinkorfmann/cxt>`_,
the original transformer-based pairwise TMRCA inference tool.  This chapter
details every architectural, algorithmic, and API difference between the two.


Architecture at a glance
------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * -
     - **cxt**
     - **fastcxt**
   * - Core architecture
     - Decoder-only transformer (GPT-style)
     - Bidirectional Mamba encoder-decoder
   * - Sequence model
     - Causal self-attention + RoPE
     - Mamba-2 SSM (forward + backward)
   * - Directionality
     - Unidirectional (left-to-right)
     - Bidirectional (both directions merged)
   * - Inference mode
     - Autoregressive token generation
     - Single forward pass
   * - Output representation
     - 324 discrete TMRCA bins (classification)
     - Continuous (μ, log σ²) per window (regression)
   * - Uncertainty
     - Monte Carlo: 15 stochastic samples, average
     - Direct: Gaussian NLL loss outputs variance
   * - Mutation rate
     - Post-hoc bias correction (``_apply_bias_correction``)
     - FiLM conditioning (learned γ, β per layer)
   * - Sample size handling
     - Fixed at 50 haploids; adapter module for others
     - ``InputProjection`` pads/truncates to ``max_samples``
   * - Tree topology
     - Not supported
     - Not supported (pairwise mode); supported via ``HybridNodeTimeModel`` (node mode)
   * - Node time model
     - Not available
     - ``HybridNodeTimeModel``: predicts all internal node times in one pass; pairwise TMRCA via O(log n) LCA lookup
   * - tsinfer integration
     - Not available
     - Can use `tsinfer <https://tskit.dev/tsinfer/>`_ to infer tree topology from genotype data, then predict node times — reduces O(n²) pair passes to O(n) node passes
   * - KV cache
     - Yes (for autoregressive decoding)
     - Not needed (single pass)


Inference pipeline comparison
-----------------------------

**cxt — 7,500 forward passes per pair:**

.. code-block:: text

   For each (pair, block):
     1. Build SFS features (500 source tokens)
     2. Feed source → transformer (with KV cache for 500 tokens)
     3. Autoregressively decode 500 target tokens, one at a time
        └── Each token: forward pass → sample from 324-bin softmax
     4. Repeat 15× with different random seeds
     5. Average the 15 discrete distributions → expected log-TMRCA
     6. Apply post-hoc mutation rate bias correction

   Total per pair: 15 reps × (500 prefill + 500 decode) = ~15,000 forward passes
   For n samples: n(n−1)/2 pairs × 15,000 passes

**fastcxt — 1 forward pass per pair:**

.. code-block:: text

   For each (pair, block):
     1. Build SFS features (same multi-scale xor/xnor as cxt)
     2. Forward pass: SFS → encoder → decoder → (μ, log σ²) for all 500 windows
     3. Done. Variance is a direct output.

   Total per pair: 1 forward pass
   For n samples: n(n−1)/2 pairs × 1 pass

**fastcxt + tsinfer — 1 forward pass per *node*:**

.. code-block:: text

   1. Run tsinfer on genotype data → inferred tree sequence
   2. Extract coalescence order (topology only, no times)
   3. For each of the O(n) internal nodes:
      Forward pass: SFS + tree features → (μ, log σ²) for node time
   4. For any pairwise TMRCA: LCA lookup in the tree → O(log n)

   Total: O(n) forward passes + O(n² log n) table lookups


Output format
-------------

**cxt** outputs discrete token IDs that must be converted:

.. code-block:: python

   # cxt: 324 bins spanning log-TMRCA ∈ [3, 17]
   GRID_SIZE = 324
   TIMES = np.linspace(3, 17, GRID_SIZE)  # log-scale TMRCA values

   # After autoregressive generation: shape (n_pairs, 500) of token indices
   tokens = generate(model, src, B=20, device="cuda")

   # Convert to log-times: take softmax → expected value over grid
   log_tmrca = to_log_times(tokens)  # still needs bias correction

**fastcxt** directly outputs continuous predictions with uncertainty:

.. code-block:: python

   # fastcxt: continuous (mean, log-variance) per window
   means, variances, index_map = translate_from_ts(
       ts, model, pivot_pairs=[(0, 1)], mutation_rate=1e-8)

   # means: (N, W) predicted log-TMRCA — ready to use
   # variances: (N, W) predicted variance of log-TMRCA
   # 95% CI: np.exp(means ± 1.96 * np.sqrt(variances))


Mutation rate handling
----------------------

**cxt: post-hoc correction**

cxt trains without mutation rate awareness.  After inference, the predicted
log-TMRCAs are corrected by subtracting a bias term estimated from the
expected diversity at a known mutation rate:

.. code-block:: python

   # cxt/correction.py
   corrected = predicted_log_tmrca - stochastic_diversity_bias_correction_v2(
       genotype_matrix, positions, mutation_rate)

This is fragile — the correction depends on data quality, and errors
compound across windows.

**fastcxt: FiLM conditioning**

fastcxt injects the mutation rate directly into the model via Feature-wise
Linear Modulation (FiLM).  Each encoder layer applies learned scale (γ) and
shift (β) parameters derived from the log mutation rate:

.. math::

   h' = \gamma(log\, \mu) \odot h + \beta(log\, \mu)

The model learns how mutation rate affects the SFS → TMRCA mapping during
training, producing correctly calibrated outputs for any mutation rate
without post-hoc adjustment.


Variable sample sizes
---------------------

**cxt: fixed at 50 haploids**

cxt's ``MutationsToLatentSpace`` module has a hardcoded ``num_samples=50``
dimension.  To handle different sample sizes, a separate ``IEAdapter``
module was trained that transforms the SFS from ``n`` samples to the
50-sample representation:

.. code-block:: python

   # cxt: requires an adapter for n ≠ 50
   adapter = IEAdapter(ie_in=n_samples, ie_out=50)
   # adapter.load_state_dict(torch.load("adapter_n30.pt"))
   result = cxt.translate(ts, model, adapter=adapter, ...)

**fastcxt: any sample size up to max_samples**

``InputProjection`` in fastcxt zero-pads the SFS sample dimension to
``max_samples`` (default 200), so any sample count from 2 to 200 works
without any adapter:

.. code-block:: python

   # fastcxt: just works for any sample size
   result = translate_from_ts(ts, model, pivot_pairs=pairs,
                               mutation_rate=1e-8)


Multi-GPU support
-----------------

**cxt**: distributes pairs across GPUs using ``deepcopy`` model replicas
and Python threads:

.. code-block:: python

   cxt.translate(ts, model, devices=["cuda:0", "cuda:1", "cuda:2"],
                 B_per_device=512)

**fastcxt**: uses PyTorch Lightning's built-in distributed training and
standard ``DataParallel`` / ``DistributedDataParallel`` for inference.
Pairs are batched and sent to a single device:

.. code-block:: python

   translate_from_ts(ts, model, device="cuda:0", batch_size=256)


Configuration comparison
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 35

   * - Parameter
     - cxt (``ModelConfig``)
     - fastcxt (``FastCxtConfig``)
   * - Model dimension
     - ``n_embd = 400``
     - ``d_model = 256``
   * - Layers
     - ``n_layer = 10`` (decoder blocks)
     - ``n_enc_layers = 6``, ``n_dec_layers = 4``
   * - Attention heads
     - ``n_head = 4``
     - N/A (Mamba uses no attention)
   * - SSM state dimension
     - N/A
     - ``d_state = 64``
   * - Output dimension
     - ``output_dim = 326`` (324 bins + 2 special)
     - ``output_dim = 2`` (μ, log σ²)
   * - Sample size
     - ``num_samples = 50`` (fixed)
     - ``max_samples = 200`` (variable)
   * - Window size
     - ``window_size = 2000``
     - ``window_size = 2000``
   * - Loss function
     - Cross-entropy (classification)
     - Beta-NLL (β=0.5, regression)


API migration guide
-------------------

**Loading a model:**

.. code-block:: python

   # cxt
   import cxt
   model = cxt.load_model("broad", device="cuda:0")

   # fastcxt
   from fastcxt.config import PRESETS
   from fastcxt.model import FastCxtModel
   config = PRESETS["base"]
   model = FastCxtModel(config)
   model.load_state_dict(torch.load("checkpoint.pt"))

**Running inference:**

.. code-block:: python

   # cxt — returns discrete tokens, needs 15× sampling + correction
   output = cxt.translate(
       ts, model,
       pivot_pairs=[(0, 1)],
       blocks=[(0, 1_000_000)],
       devices=["cuda:0"],
       B=512,
       build_workers=36,
       mutation_rate=1.5e-8,  # for bias correction
   )

   # fastcxt — returns (means, variances) directly
   from fastcxt.translate import translate_from_ts
   means, variances, index_map = translate_from_ts(
       ts, model,
       pivot_pairs=[(0, 1)],
       mutation_rate=1.5e-8,  # FiLM conditioning, not correction
       device="cuda:0",
       batch_size=256,
   )

**Genome-wide results:**

.. code-block:: python

   # cxt — manual aggregation of per-block results
   all_results = []
   for block in blocks:
       out = cxt.translate(ts, model, blocks=[block], ...)
       all_results.append(out)
   # ... manually combine into arrays

   # fastcxt — TimeAtlas handles everything
   from fastcxt.atlas import TimeAtlas
   atlas = TimeAtlas()
   atlas.add_arm("2L", means, variances, pairs, window_size=2000)
   atlas.save("results/")
   m, v = atlas.query_pair("2L", sample_a=0, sample_b=5)


What's preserved from cxt
--------------------------

- **SFS features**: the multi-scale xor/xnor SFS computation is the same
  (``calculate_window_sfs``, ``build_sfs_tensor``)
- **Window-averaged TMRCA targets**: span-weighted interpolation of true
  TMRCA values (``windowed_tmrca``, formerly ``interpolate_tmrcas``)
- **Simulation pipeline**: ``msprime`` + ``stdpopsim`` based, same scenarios
  available (constant, sawtooth, island, AnoGam, HomSap, ...)
- **Train/test splitting**: same deterministic hashing strategy


See also
--------

- :doc:`architecture` — detailed architecture of the Mamba encoder-decoder
- :doc:`scaling` — quantitative runtime benchmarks comparing all three modes
