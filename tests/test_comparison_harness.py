"""Acceptance tests for the two-bucket vs single-portfolio comparison harness
(PRD §11). `build_comparison` and its wiring into `execute_run` live in
webapp/callbacks.py.
"""

import copy
import os
import sys

import dataclasses

import numpy as np
import pytest

_tests_dir = os.path.dirname(__file__)
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)
import golden_scenarios  # noqa: E402

from engine.simulation import run_simulation
from engine.withdrawal_strategies import ReserveConfig, ReturnModelConfig, WithdrawalStrategyConfig
from webapp.callbacks import build_comparison


def _two_bucket_cfg(target_years=3.0, trigger_years=2.0):
    return WithdrawalStrategyConfig(
        type="two_bucket",
        reserve=ReserveConfig(target_years=target_years, refill_trigger_years=trigger_years,
                               coverage_scope="recurring_gap_only",
                               return_model=ReturnModelConfig(distribution="normal",
                                                               mean_real=0.01, std_real=0.03)),
    )


def _two_bucket_params(**over):
    wcfg = over.pop("withdrawal_strategy", _two_bucket_cfg())
    return golden_scenarios._base_params(withdrawal_strategy=wcfg, **over)


def test_build_comparison_returns_compact_dict_with_required_keys():
    params = _two_bucket_params(random_seed=42)
    comp_params = dataclasses.replace(params, withdrawal_strategy=None)

    results = run_simulation(params, guardrails=None)
    comp = run_simulation(comp_params, guardrails=None)

    comparison = build_comparison(results, comp)

    assert "summary" in comparison
    assert comparison["summary"]["ruin_probability"] == comp["summary"]["ruin_probability"]
    assert "guardrail_stats" in comparison
    for key in ("p10", "p50", "p90"):
        assert key in comparison["bal_percentiles"]
        assert comparison["bal_percentiles"][key].shape == (comp_params.n_periods
                                                              if hasattr(comp_params, "n_periods")
                                                              else comp["bal_over_time"].shape[0],)


def test_build_comparison_percentiles_match_comparator_bal_over_time():
    params = _two_bucket_params(random_seed=7)
    comp_params = dataclasses.replace(params, withdrawal_strategy=None)
    results = run_simulation(params, guardrails=None)
    comp = run_simulation(comp_params, guardrails=None)

    comparison = build_comparison(results, comp)

    np.testing.assert_allclose(comparison["bal_percentiles"]["p50"],
                                np.percentile(comp["bal_over_time"], 50, axis=1))
    np.testing.assert_allclose(comparison["bal_percentiles"]["p10"],
                                np.percentile(comp["bal_over_time"], 10, axis=1))
    np.testing.assert_allclose(comparison["bal_percentiles"]["p90"],
                                np.percentile(comp["bal_over_time"], 90, axis=1))


def test_build_comparison_is_compact_no_full_bal_over_time_array():
    """The comparator's own bal_over_time (n_periods x n_paths) must not be
    stored whole -- only the 3 x n_periods percentile summary."""
    params = _two_bucket_params(random_seed=7)
    comp_params = dataclasses.replace(params, withdrawal_strategy=None)
    results = run_simulation(params, guardrails=None)
    comp = run_simulation(comp_params, guardrails=None)

    comparison = build_comparison(results, comp)

    for v in comparison.values():
        if isinstance(v, np.ndarray):
            assert v.ndim == 1, "comparison must not embed a full 2D (period x path) array"
        if isinstance(v, dict):
            for vv in v.values():
                if isinstance(vv, np.ndarray):
                    assert vv.ndim == 1


def test_same_seed_same_shocks_target_years_zero_comparator_matches_main():
    """With target_years=0 two-bucket degenerates to single mode (existing
    killer-RNG guarantee, tests/test_two_bucket_rng.py). So for that
    degenerate config, the comparator run must be bit-identical to the main
    run's own combined balance -- the cheapest possible sanity check that
    the comparator truly reuses the same seed / same shocks."""
    wcfg = _two_bucket_cfg(target_years=0.0, trigger_years=0.0)
    params = _two_bucket_params(withdrawal_strategy=wcfg, random_seed=42)
    comp_params = dataclasses.replace(params, withdrawal_strategy=None)

    results = run_simulation(params, guardrails=None)
    comp = run_simulation(comp_params, guardrails=None)

    np.testing.assert_array_equal(results["bal_over_time"], comp["bal_over_time"])


_SCENARIO = {
    "$schema": "scenario.v1",
    "name": "comparison-harness-test",
    "portfolio": {
        "initial_portfolio": 3_000_000,
        "start_age": 60,
        "end_age": 75,
        "market": "IL",
        "fat_tails_enabled": False,
        "fat_tails_df": 5,
        "mode": "annual",
        "n_paths": 100,
        "random_seed": 42,
        "mu": 0.05,
        "sigma": 0.15,
        "real_discount_rate": 0.01,
    },
    "spending_bands": [
        {"age_from": 60, "age_to": 75, "amount_monthly": 15000,
         "label": "base", "category": "strict"},
    ],
    "income_bands": [],
    "lumps": [],
    "properties": [],
    "withdrawal_strategy": {
        "type": "two_bucket",
        "reserve": {
            "target_years": 3.0,
            "refill_trigger_years": 2.0,
            "coverage_scope": "recurring_gap_only",
            "return_model": {"distribution": "normal", "mean_real": 0.01, "std_real": 0.03},
        },
    },
}


def test_execute_run_attaches_comparison_for_two_bucket_scenario():
    """execute_run(scenario, ...) must populate results["comparison"] when
    the scenario carries a two_bucket withdrawal_strategy block and
    compare_enabled defaults True; single-portfolio scenarios must not pay
    for or carry a comparison at all."""
    from webapp.callbacks import execute_run, RESULTS_CACHE

    two_bucket_scenario = copy.deepcopy(_SCENARIO)
    single_scenario = copy.deepcopy(_SCENARIO)
    del single_scenario["withdrawal_strategy"]

    run_id_two = execute_run(two_bucket_scenario, guardrail_cfg=None, playground_events=[],
                              include_historic=False)[0]
    run_id_single = execute_run(single_scenario, guardrail_cfg=None, playground_events=[],
                                 include_historic=False)[0]

    assert "comparison" in RESULTS_CACHE[run_id_two]
    assert "comparison" not in RESULTS_CACHE[run_id_single]


def test_comparison_not_computed_for_single_portfolio_scenario():
    params, guardrails = golden_scenarios.annual_plain()
    assert params.withdrawal_strategy is None
    results = run_simulation(params, guardrails=guardrails)
    assert "comparison" not in results
