Simulation
==========

fastcxt uses a **scenario registry** pattern for simulations, replacing the
large ``if/elif`` chains from cxt.


Available scenarios
-------------------

**stdpopsim species** (recommended for training):

- ``AnoGam`` -- *Anopheles gambiae* (mosquito)
- ``HomSap`` -- *Homo sapiens*
- ``DroMel`` -- *Drosophila melanogaster*
- ``BosTau`` -- *Bos taurus* (cattle)
- ``CanFam`` -- *Canis familiaris* (dog)
- ``PanTro`` -- *Pan troglodytes* (chimpanzee)
- ``PapAnu`` -- *Papio anubis* (baboon)
- ``PonAbe`` -- *Pongo abelii* (orangutan)
- ``AraTha`` -- *Arabidopsis thaliana*
- ``CaeEle`` -- *Caenorhabditis elegans*
- ``AedAeg`` -- *Aedes aegypti* (yellow fever mosquito)
- ``HelAnn`` -- *Helianthus annuus* (sunflower)

**Custom msprime scenarios**:

- ``constant`` -- constant population size
- ``sawtooth`` -- oscillating population sizes
- ``island`` -- 3-deme island model with migration


CLI usage
---------

.. code-block:: bash

   # Mosquito simulations (500 tree sequences)
   fastcxt-simulate --scenario AnoGam --data-dir ./sims/anogam --num-ts 500

   # Human with genetic map
   fastcxt-simulate --scenario HomSap --genetic-map HapMapII_GRCh38 --data-dir ./sims/homsap

   # Variable sample sizes
   fastcxt-simulate --scenario constant --n-samples 100 --data-dir ./sims/constant_n100

   # Custom parameters
   fastcxt-simulate --scenario constant \
       --mutation-rate 3.5e-9 \
       --recombination-rate 1.5e-8 \
       --sequence-length 2000000 \
       --data-dir ./sims/custom


Python API
----------

.. code-block:: python

   from fastcxt.simulate import resolve_scenario, generate_tree_sequences

   sim_func, cfg = resolve_scenario("AnoGam", n_samples=50, sequence_length=2e6)
   generate_tree_sequences(
       num_ts=1000,
       output_dir="./sims/anogam",
       sim_func=sim_func,
       cfg=cfg,
       num_processes=16,
   )

   # Or simulate a single tree sequence
   ts = sim_func(seed=42, cfg=cfg)


Adding a new scenario
---------------------

For stdpopsim species, just add an entry to ``STDPOPSIM_DEFAULTS``:

.. code-block:: python

   STDPOPSIM_DEFAULTS["NewSpp"] = {"population_size": 10_000}

For custom msprime scenarios, register a simulation function:

.. code-block:: python

   def simulate_bottleneck(seed, cfg):
       ...
       return ts

   SCENARIO_DISPATCH["bottleneck"] = simulate_bottleneck
