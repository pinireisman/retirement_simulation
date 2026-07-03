import numpy as np
import pytest

from engine.guardrails import (
    GuardrailConfig,
    VolatilityDiscretionaryScaling,
    build_handlers,
    compute_guardrail_stats,
)
from engine.params import Band, Lump, Property, SimulationParams
from engine.simulation import run_simulation
from cli import read_scenario_data

G1_DEFAULTS = dict(drop_threshold=0.20, rise_threshold=0.20, cut_pct=0.15, raise_pct=0.10)


def make_params(**over):
    base = dict(
        start_age=60, end_age=95, initial_portfolio=3_000_000,
        real_return_mean=0.042, real_return_sd=0.13, fat_tails_df=5,
        annual=True, n_paths=10_000, random_seed=42,
        spending_bands=[Band(60, 95, 300_000, "base", "strict"),
                        Band(60, 80, 60_000, "travel", "lifestyle")],
        income_bands=[Band(60, 67, 144_000, "consulting")],
        lumps=[Lump(70, -400_000, "gift", "gifts")],
        properties=[Property(60, 2_500_000, 72_000, 0.018, 0.08, "apt")],
    )
    base.update(over)
    return SimulationParams(**base)


def test_multiplier_cuts_only_the_crashed_path_the_following_year():
    """3 paths, hand-crafted returns: path 0 crashes -25% in year index 3 ->
    year-4 discretionary multiplier is x0.85 for path 0 only."""
    n_paths, n_years = 3, 6
    port_ret = np.zeros((n_paths, n_years))
    port_ret[0, 3] = -0.25
    h = VolatilityDiscretionaryScaling(G1_DEFAULTS)
    for year_idx in range(n_years):
        mult = h.multiplier(year_idx, port_ret, n_paths)
        expected = [0.85, 1.0, 1.0] if year_idx == 4 else [1.0, 1.0, 1.0]
        np.testing.assert_allclose(mult, expected)
    assert h.triggered_down.tolist() == [True, False, False]
    assert h.triggered_up.tolist() == [False, False, False]


def test_multiplier_raises_symmetrically_after_a_rally():
    n_paths, n_years = 3, 6
    port_ret = np.zeros((n_paths, n_years))
    port_ret[1, 2] = 0.25
    h = VolatilityDiscretionaryScaling(G1_DEFAULTS)
    for year_idx in range(n_years):
        mult = h.multiplier(year_idx, port_ret, n_paths)
        expected = [1.0, 1.10, 1.0] if year_idx == 3 else [1.0, 1.0, 1.0]
        np.testing.assert_allclose(mult, expected)
    assert h.triggered_up.tolist() == [False, True, False]
    assert h.triggered_down.tolist() == [False, False, False]


def test_zero_cut_and_raise_is_noop():
    baseline = run_simulation(make_params())
    guarded = run_simulation(make_params(), guardrails=[
        GuardrailConfig(type="volatility_discretionary_scaling",
                         options={**G1_DEFAULTS, "cut_pct": 0.0, "raise_pct": 0.0}),
    ])
    assert baseline["summary"] == guarded["summary"]
    np.testing.assert_array_equal(baseline["bal_over_time"], guarded["bal_over_time"])


def test_frac_paths_cut_matches_hand_countable_fraction():
    n_paths, n_years = 4, 5
    port_ret = np.zeros((n_paths, n_years))
    port_ret[0, 1] = -0.30
    port_ret[2, 1] = -0.25
    h = VolatilityDiscretionaryScaling(G1_DEFAULTS)
    for year_idx in range(n_years):
        mult = h.multiplier(year_idx, port_ret, n_paths)
        h.adjustments.append(np.full(n_paths, 1000.0) * (mult - 1.0))
    stats = compute_guardrail_stats([h])
    assert stats["frac_paths_cut"] == 2 / 4
    assert stats["frac_paths_raised"] == 0.0
    assert stats["frac_paths_triggered"] == 2 / 4


def test_g1_defaults_do_not_increase_ruin_on_example_scenario():
    """PRD §8 Phase-5 acceptance: on scenario_data_example.xlsx (a realistic,
    near-zero-ruin scenario) enabling G1 with defaults must not raise ruin %.
    `make_params()`'s baseline is deliberately near-total-failure (~93.5%,
    pinned for regression in test_engine_baseline.py) and isn't representative
    here — most of its paths are already ruined, so the symmetric raise-after-
    rally leg can tip marginal survivors into ruin, which is a real property
    of G1, not a bug."""
    scenario_data = read_scenario_data("scenario_data_example.xlsx")
    params = SimulationParams.from_legacy_scenario_data(scenario_data)
    baseline = run_simulation(params)
    guarded = run_simulation(params, guardrails=[
        GuardrailConfig(type="volatility_discretionary_scaling", options=G1_DEFAULTS),
    ])
    assert guarded["summary"]["ruin_probability"] <= baseline["summary"]["ruin_probability"]
    assert guarded["guardrail_stats"]["type"] == "volatility_discretionary_scaling"
    assert 0.0 <= guarded["guardrail_stats"]["frac_paths_triggered"] <= 1.0


def test_unknown_guardrail_type_raises_before_the_loop():
    with pytest.raises(ValueError):
        build_handlers([GuardrailConfig(type="not_a_real_guardrail", options={})])
