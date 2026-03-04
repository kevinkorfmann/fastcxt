"""Tests for fastcxt.simulate -- new scenario-registry simulation pipeline."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from fastcxt.simulate import (
    SimConfig,
    resolve_scenario,
    simulate_constant,
    simulate_sawtooth,
    simulate_island,
    ALL_SCENARIOS,
    SCENARIO_DISPATCH,
    STDPOPSIM_DEFAULTS,
)


# ---------------------------------------------------------------------------
# SimConfig
# ---------------------------------------------------------------------------

class TestSimConfig:
    def test_frozen(self):
        cfg = SimConfig(name="test", kind="msprime")
        with pytest.raises(AttributeError):
            cfg.name = "other"

    def test_defaults(self):
        cfg = SimConfig(name="t", kind="msprime")
        assert cfg.mutation_rate == 1e-8
        assert cfg.recombination_rate == 1e-8
        assert cfg.sequence_length == 1e6
        assert cfg.n_samples == 25

    def test_extra_dict(self):
        cfg = SimConfig(name="t", kind="msprime", extra={"magnitude": 5})
        assert cfg.extra["magnitude"] == 5


# ---------------------------------------------------------------------------
# resolve_scenario
# ---------------------------------------------------------------------------

class TestResolveScenario:
    def test_constant(self):
        func, cfg = resolve_scenario("constant")
        assert cfg.kind == "msprime"
        assert cfg.name == "constant"
        assert func is simulate_constant

    def test_sawtooth(self):
        func, cfg = resolve_scenario("sawtooth")
        assert func is simulate_sawtooth

    def test_island(self):
        func, cfg = resolve_scenario("island")
        assert func is simulate_island

    def test_stdpopsim_species(self):
        func, cfg = resolve_scenario("AnoGam")
        assert cfg.kind == "stdpopsim"
        assert cfg.species == "AnoGam"

    def test_overrides(self):
        _, cfg = resolve_scenario("constant", n_samples=100, sequence_length=2e6)
        assert cfg.n_samples == 100
        assert cfg.sequence_length == 2e6

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown scenario"):
            resolve_scenario("nonexistent_scenario")

    def test_all_scenarios_consistent(self):
        for name in ALL_SCENARIOS:
            func, cfg = resolve_scenario(name)
            assert cfg.name == name
            assert callable(func)


# ---------------------------------------------------------------------------
# Msprime scenarios
# ---------------------------------------------------------------------------

class TestMsprimeScenarios:
    @pytest.fixture
    def short_cfg(self):
        return SimConfig(name="test", kind="msprime", n_samples=10,
                         sequence_length=1e5)

    def test_constant_produces_ts(self, short_cfg):
        ts = simulate_constant(42, short_cfg)
        assert ts.num_samples == 20  # diploid * 2
        assert ts.num_mutations > 0
        assert ts.sequence_length == 1e5

    def test_constant_deterministic(self, short_cfg):
        ts1 = simulate_constant(42, short_cfg)
        ts2 = simulate_constant(42, short_cfg)
        assert ts1.num_mutations == ts2.num_mutations
        assert ts1.num_trees == ts2.num_trees

    def test_constant_different_seeds(self, short_cfg):
        ts1 = simulate_constant(1, short_cfg)
        ts2 = simulate_constant(2, short_cfg)
        assert ts1.num_mutations != ts2.num_mutations or ts1.num_trees != ts2.num_trees

    def test_sawtooth_produces_ts(self, short_cfg):
        ts = simulate_sawtooth(42, short_cfg)
        assert ts.num_samples == 20

    def test_island_produces_ts(self, short_cfg):
        ts = simulate_island(42, short_cfg)
        assert ts.num_samples == 20

    def test_island_sample_distribution(self):
        cfg = SimConfig(name="island", kind="msprime", n_samples=100,
                        sequence_length=1e5)
        ts = simulate_island(42, cfg)
        assert ts.num_samples == 200

    def test_custom_rates(self):
        cfg = SimConfig(name="constant", kind="msprime", n_samples=5,
                        sequence_length=1e5, mutation_rate=5e-8,
                        recombination_rate=5e-8)
        ts = simulate_constant(42, cfg)
        assert ts.num_mutations > 0


# ---------------------------------------------------------------------------
# stdpopsim scenarios (require stdpopsim)
# ---------------------------------------------------------------------------

class TestStdpopsimScenarios:
    @pytest.mark.slow
    def test_anogam(self):
        from fastcxt.simulate import simulate_stdpopsim
        cfg = SimConfig(name="AnoGam", kind="stdpopsim", species="AnoGam",
                        n_samples=5, sequence_length=1e5)
        ts = simulate_stdpopsim(42, cfg)
        assert ts.num_samples > 0
        assert ts.num_mutations > 0

    @pytest.mark.slow
    def test_homsap(self):
        from fastcxt.simulate import simulate_stdpopsim
        cfg = SimConfig(name="HomSap", kind="stdpopsim", species="HomSap",
                        n_samples=5, sequence_length=1e5)
        ts = simulate_stdpopsim(42, cfg)
        assert ts.num_samples > 0

    @pytest.mark.slow
    def test_canfam_with_default_ne(self):
        _, cfg = resolve_scenario("CanFam", n_samples=5, sequence_length=1e5)
        assert cfg.population_size == 13_000


# ---------------------------------------------------------------------------
# Scenario registry completeness
# ---------------------------------------------------------------------------

class TestScenarioRegistry:
    def test_msprime_scenarios(self):
        assert "constant" in SCENARIO_DISPATCH
        assert "sawtooth" in SCENARIO_DISPATCH
        assert "island" in SCENARIO_DISPATCH

    def test_stdpopsim_species(self):
        expected = {"HomSap", "AnoGam", "DroMel", "BosTau", "CanFam",
                    "PanTro", "PapAnu", "PonAbe", "AraTha", "CaeEle"}
        assert expected.issubset(set(STDPOPSIM_DEFAULTS.keys()))

    def test_all_scenarios_non_empty(self):
        assert len(ALL_SCENARIOS) >= 14

    def test_no_overlap(self):
        assert set(SCENARIO_DISPATCH.keys()).isdisjoint(set(STDPOPSIM_DEFAULTS.keys()))
