"""Simulation pipeline for fastcxt.

stdpopsim-first design: every species/model is a config entry, not an if/elif
branch.  Custom msprime scenarios (constant, sawtooth, island) are thin
wrappers that produce a tree sequence -- they share the same
``generate_tree_sequences`` backend.

Usage:
    python -m fastcxt.simulate --scenario AnoGam --data-dir ./sims --num-ts 500
    python -m fastcxt.simulate --scenario HomSap --genetic-map HapMapII_GRCh38
    python -m fastcxt.simulate --scenario constant --n-samples 50 --num-ts 1000
"""

from __future__ import annotations

import os
import argparse
import warnings
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import get_context
from typing import Callable

import numpy as np
import msprime
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Simulation config dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimConfig:
    """Immutable description of one simulation scenario."""
    name: str
    kind: str                            # "stdpopsim" | "msprime"
    species: str | None = None           # stdpopsim species code
    genetic_map: str | None = None
    population_size: float | None = None
    mutation_rate: float = 1e-8
    recombination_rate: float = 1e-8
    sequence_length: float = 1e6
    n_samples: int = 25                  # diploid individuals
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# stdpopsim simulation (unified for all species)
# ---------------------------------------------------------------------------

def _sampling_populations(model):
    pops = []
    for pop in model.populations:
        dst = getattr(pop, "default_sampling_time", None)
        if isinstance(dst, (int, float)) and dst > 0:
            continue
        if getattr(pop, "allow_samples", True):
            pops.append(pop)
    return pops


def _pick_autosomal_chromosome(species):
    chroms = [
        c for c in species.genome.chromosomes
        if c.id not in ("Mt", "Pt", "X", "Y", "Z", "W")
        and any(ch.isdigit() for ch in c.id)
    ]
    if not chroms:
        chroms = [c for c in species.genome.chromosomes if c.id not in ("Mt", "Pt")]
    return chroms[np.random.randint(len(chroms))]


def simulate_stdpopsim(seed: int, cfg: SimConfig) -> msprime.TreeSequence:
    """Generic stdpopsim simulation for any species."""
    import stdpopsim
    import random as _random

    rng_np = np.random.RandomState(seed)
    seed_val = int(rng_np.randint(1, 2**31))

    species = stdpopsim.get_species(cfg.species)

    models = species.demographic_models
    excluded = {
        "Multi-population model of ancient Eurasia",
        "Out-of-Africa with archaic admixture into Papuans",
        "Multi-population model of ancient Europe",
    }
    if models:
        valid = [m for m in models if m.description not in excluded]
    else:
        Ne = cfg.population_size or 1e4
        valid = [stdpopsim.PiecewiseConstantSize(Ne)]
    model = valid[rng_np.randint(len(valid))]

    pops = _sampling_populations(model)
    rng_py = _random.Random(seed_val)
    counts = {p.name: 0 for p in pops}
    for p in rng_py.choices(pops, k=cfg.n_samples):
        counts[p.name] += 1

    engine = stdpopsim.get_engine("msprime")

    chrom = _pick_autosomal_chromosome(species)
    seg = cfg.sequence_length
    left = rng_np.uniform(0, max(chrom.length - seg, 1))
    right = left + seg

    if cfg.genetic_map is None:
        contig = species.get_contig(
            chrom.id, left=left, right=right,
            mutation_rate=model.mutation_rate,
        )
    else:
        for _ in range(100):
            chrom = _pick_autosomal_chromosome(species)
            left = rng_np.uniform(0, max(chrom.length - seg, 1))
            right = left + seg
            try:
                contig = species.get_contig(
                    chrom.id, left=left, right=right,
                    mutation_rate=model.mutation_rate,
                    genetic_map=cfg.genetic_map,
                )
                rates = contig.recombination_map.rate[1:-1]
                if len(rates) == 0 or np.isfinite(rates).all():
                    break
            except ValueError:
                continue
        else:
            raise RuntimeError(f"No valid segment found for {cfg.species}")

    ts = engine.simulate(model, contig, counts, seed=seed_val)
    return ts.trim()


# ---------------------------------------------------------------------------
# Pure msprime scenarios
# ---------------------------------------------------------------------------

def simulate_constant(seed: int, cfg: SimConfig) -> msprime.TreeSequence:
    rng = np.random.default_rng(seed)
    ts = msprime.sim_ancestry(
        samples=cfg.n_samples,
        population_size=cfg.population_size or 2e4,
        recombination_rate=cfg.recombination_rate,
        sequence_length=cfg.sequence_length,
        random_seed=int(rng.integers(1, 2**31)),
    )
    return msprime.sim_mutations(
        ts, rate=cfg.mutation_rate, random_seed=int(rng.integers(1, 2**31)),
    )


def _sawtooth_demography(Ne: float = 2e4, magnitude: int = 3) -> msprime.Demography:
    d = msprime.Demography()
    d.add_population(initial_size=Ne)
    scale = magnitude * 1e4
    growth_schedule = [
        (20, 6437.7516497364 / (4 * 1e4), None),
        (30, None, -378.691273513906 / scale),
        (200, None, -643.77516497364 / scale),
        (300, None, 37.8691273513906 / scale),
        (2_000, None, 64.377516497364 / scale),
        (3_000, None, -3.78691273513906 / scale),
        (20_000, None, -6.4377516497364 / scale),
        (30_000, None, 0.378691273513906 / scale),
        (200_000, None, 0.64377516497364 / scale),
        (300_000, None, -0.0378691273513906 / scale),
        (2_000_000, None, -0.064377516497364 / scale),
        (3_000_000, None, 0.00378691273513906 / scale),
        (20_000_000, Ne, 0),
    ]
    for t, init_size, gr in growth_schedule:
        kw: dict = {"time": t, "population": None}
        if gr is not None:
            kw["growth_rate"] = gr
        if init_size is not None:
            kw["initial_size"] = init_size
        d.add_population_parameters_change(**kw)
    return d


def simulate_sawtooth(seed: int, cfg: SimConfig) -> msprime.TreeSequence:
    Ne = cfg.population_size or 2e4
    mag = cfg.extra.get("magnitude", 3)
    dem = _sawtooth_demography(Ne, mag)
    rng = np.random.default_rng(seed)
    ts = msprime.sim_ancestry(
        samples=cfg.n_samples, demography=dem,
        recombination_rate=cfg.recombination_rate,
        sequence_length=cfg.sequence_length,
        random_seed=int(rng.integers(1, 2**31)),
    )
    return msprime.sim_mutations(
        ts, rate=cfg.mutation_rate, random_seed=int(rng.integers(1, 2**31)),
    )


def simulate_island(seed: int, cfg: SimConfig) -> msprime.TreeSequence:
    Ne = cfg.population_size or 1e4
    mig = cfg.extra.get("migration_rate", 0.1)
    dem = msprime.Demography.island_model(
        [Ne, Ne / 2, Ne / 2], migration_rate=mig,
    )
    n = cfg.n_samples
    samples = {0: int(n * 0.6), 1: int(n * 0.2), 2: n - int(n * 0.6) - int(n * 0.2)}
    rng = np.random.default_rng(seed)
    ts = msprime.sim_ancestry(
        samples=samples, demography=dem,
        recombination_rate=cfg.recombination_rate,
        sequence_length=cfg.sequence_length,
        random_seed=int(rng.integers(1, 2**31)),
    )
    return msprime.sim_mutations(
        ts, rate=cfg.mutation_rate, random_seed=int(rng.integers(1, 2**31)),
    )


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

SCENARIO_DISPATCH: dict[str, Callable] = {
    "constant": simulate_constant,
    "sawtooth": simulate_sawtooth,
    "island": simulate_island,
}

STDPOPSIM_DEFAULTS: dict[str, dict] = {
    "HomSap":  {},
    "AnoGam":  {},
    "DroMel":  {},
    "BosTau":  {},
    "CanFam":  {"population_size": 13_000},
    "PanTro":  {},
    "PapAnu":  {},
    "PonAbe":  {},
    "AraTha":  {},
    "CaeEle":  {"population_size": 10_000},
    "GorGor":  {},
    "MusMus":  {},
    "AedAeg":  {"population_size": 1_000_000},
    "HelAnn":  {"population_size": 673_968},
}


def resolve_scenario(name: str, **overrides) -> tuple[Callable, SimConfig]:
    """Return (sim_func, config) for a scenario name.

    The name can be a stdpopsim species code (e.g. "AnoGam") or a custom
    msprime scenario ("constant", "sawtooth", "island").
    """
    if name in SCENARIO_DISPATCH:
        cfg = SimConfig(name=name, kind="msprime", **overrides)
        return SCENARIO_DISPATCH[name], cfg

    if name in STDPOPSIM_DEFAULTS:
        defaults = STDPOPSIM_DEFAULTS[name]
        merged = {**defaults, **overrides}
        cfg = SimConfig(name=name, kind="stdpopsim", species=name, **merged)
        return simulate_stdpopsim, cfg

    raise ValueError(
        f"Unknown scenario {name!r}. "
        f"Available: {sorted(list(SCENARIO_DISPATCH) + list(STDPOPSIM_DEFAULTS))}"
    )


# ---------------------------------------------------------------------------
# Parallel tree-sequence generation
# ---------------------------------------------------------------------------

def _dump_one(i: int, outdir: str, sim_func: Callable, cfg: SimConfig) -> str:
    ts = sim_func(i, cfg)
    path = os.path.join(outdir, f"ts_{i:08d}.trees")
    ts.dump(path)
    return path


def generate_tree_sequences(
    num_ts: int,
    output_dir: str,
    sim_func: Callable,
    cfg: SimConfig,
    num_processes: int = 8,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    worker = partial(_dump_one, outdir=output_dir, sim_func=sim_func, cfg=cfg)
    ctx = get_context("fork")
    with ctx.Pool(num_processes) as pool:
        for _ in tqdm(pool.imap_unordered(worker, range(num_ts)),
                      total=num_ts, desc=f"Simulating {cfg.name}"):
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ALL_SCENARIOS = sorted(list(SCENARIO_DISPATCH) + list(STDPOPSIM_DEFAULTS))


def main():
    ap = argparse.ArgumentParser(
        description="Simulate tree sequences for fastcxt training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available scenarios:\n  " + "\n  ".join(ALL_SCENARIOS),
    )
    ap.add_argument("--scenario", required=True,
                    help="Scenario name (species code or msprime scenario)")
    ap.add_argument("--data-dir", required=True,
                    help="Output directory for .trees files")
    ap.add_argument("--num-ts", type=int, default=1000,
                    help="Number of tree sequences to generate")
    ap.add_argument("--n-samples", type=int, default=25,
                    help="Number of diploid individuals")
    ap.add_argument("--sequence-length", type=float, default=1e6)
    ap.add_argument("--mutation-rate", type=float, default=None,
                    help="Override mutation rate (stdpopsim uses species default)")
    ap.add_argument("--recombination-rate", type=float, default=None,
                    help="Override recombination rate")
    ap.add_argument("--genetic-map", type=str, default=None,
                    help="stdpopsim genetic map name")
    ap.add_argument("--num-processes", type=int, default=8)
    args = ap.parse_args()

    overrides: dict = {"n_samples": args.n_samples, "sequence_length": args.sequence_length}
    if args.mutation_rate is not None:
        overrides["mutation_rate"] = args.mutation_rate
    if args.recombination_rate is not None:
        overrides["recombination_rate"] = args.recombination_rate
    if args.genetic_map is not None:
        overrides["genetic_map"] = args.genetic_map

    sim_func, cfg = resolve_scenario(args.scenario, **overrides)
    print(f"Scenario: {cfg.name} ({cfg.kind}), n_samples={cfg.n_samples}, "
          f"seq_len={cfg.sequence_length:.0f}")

    generate_tree_sequences(
        args.num_ts, args.data_dir, sim_func, cfg,
        num_processes=args.num_processes,
    )


if __name__ == "__main__":
    main()
