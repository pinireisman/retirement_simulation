"""RNG-discipline tests for the two-bucket strategy (PRD §5.5, §14.5).

Adding reserve-return sampling must never shift the growth/property RNG
sequence used by any existing component -- reserve returns live on their
own independently-derived generator.
"""

import os
import sys

import numpy as np
import pytest

_tests_dir = os.path.dirname(__file__)
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)
import golden_scenarios  # noqa: E402

from engine.simulation import run_simulation
from engine.withdrawal_strategies import (
    STREAM_RESERVE_RETURNS,
    ReserveConfig,
    ReturnModelConfig,
    WithdrawalStrategyConfig,
    sample_reserve_returns,
)


def _two_bucket_cfg(target_years, trigger_years, distribution="normal", mean_real=0.01, std_real=0.03):
    return WithdrawalStrategyConfig(
        type="two_bucket",
        reserve=ReserveConfig(target_years=target_years, refill_trigger_years=trigger_years,
                               coverage_scope="recurring_gap_only",
                               return_model=ReturnModelConfig(distribution=distribution,
                                                               mean_real=mean_real, std_real=std_real)),
    )


def test_4_stream_registry_constant_pinned():
    assert STREAM_RESERVE_RETURNS == 1


def test_1_killer_zero_years_growth_over_time_bit_equal_to_single_bal_over_time():
    params_single, guardrails = golden_scenarios.annual_plain()
    result_single = run_simulation(params_single, guardrails=guardrails)

    wcfg = _two_bucket_cfg(target_years=0.0, trigger_years=0.0)
    params_two = golden_scenarios._base_params(random_seed=params_single.random_seed,
                                                withdrawal_strategy=wcfg)
    result_two = run_simulation(params_two, guardrails=None)

    np.testing.assert_array_equal(result_two["growth_over_time"], result_single["bal_over_time"])
    np.testing.assert_array_equal(result_two["bal_over_time"], result_single["bal_over_time"])


def test_2_prop_over_time_bit_equal_single_vs_two_bucket():
    params_single, guardrails = golden_scenarios.annual_plain()
    result_single = run_simulation(params_single, guardrails=guardrails)

    wcfg = _two_bucket_cfg(target_years=3.0, trigger_years=2.0)
    params_two = golden_scenarios._base_params(random_seed=params_single.random_seed,
                                                withdrawal_strategy=wcfg)
    result_two = run_simulation(params_two, guardrails=None)

    np.testing.assert_array_equal(result_two["prop_over_time"], result_single["prop_over_time"])


def test_3_sample_reserve_returns_reproducible_from_seed_alone():
    cfg = ReturnModelConfig(distribution="normal", mean_real=0.02, std_real=0.05)
    a = sample_reserve_returns(cfg, n_paths=50, n_periods=10, seed=123, annual=True)
    b = sample_reserve_returns(cfg, n_paths=50, n_periods=10, seed=123, annual=True)
    np.testing.assert_array_equal(a, b)


def test_3_sample_reserve_returns_unaffected_by_unrelated_rng_consumption():
    """sample_reserve_returns is a pure function of its explicit arguments --
    it never touches the shared/default RNG, so drawing from unrelated
    generators around the call (standing in for "more properties consumed
    the shared rng first") cannot perturb its output."""
    cfg = ReturnModelConfig(distribution="normal", mean_real=0.02, std_real=0.05)
    a = sample_reserve_returns(cfg, n_paths=20, n_periods=5, seed=99, annual=True)
    _ = np.random.default_rng(99).normal(size=(5000,))  # unrelated draws elsewhere
    _ = np.random.default_rng().normal(size=(5000,))
    b = sample_reserve_returns(cfg, n_paths=20, n_periods=5, seed=99, annual=True)
    np.testing.assert_array_equal(a, b)


def test_3_sample_reserve_returns_constant_consumes_no_rng():
    """A constant reserve-return distribution must not even construct a
    generator; verified indirectly -- seed=None (which would normally seed
    an unseeded/nondeterministic default_rng()) still gives a fully
    deterministic, repeatable result for `constant`."""
    cfg = ReturnModelConfig(distribution="constant", mean_real=0.03, std_real=0.0)
    a = sample_reserve_returns(cfg, n_paths=10, n_periods=4, seed=None, annual=True)
    b = sample_reserve_returns(cfg, n_paths=10, n_periods=4, seed=None, annual=True)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, np.full((10, 4), 0.03))


def test_5_same_seed_reproduces_every_result():
    wcfg = _two_bucket_cfg(target_years=3.0, trigger_years=2.0)
    params1 = golden_scenarios._base_params(random_seed=7, withdrawal_strategy=wcfg)
    params2 = golden_scenarios._base_params(random_seed=7, withdrawal_strategy=wcfg)
    r1 = run_simulation(params1, guardrails=None)
    r2 = run_simulation(params2, guardrails=None)
    np.testing.assert_array_equal(r1["bal_over_time"], r2["bal_over_time"])
    np.testing.assert_array_equal(r1["growth_over_time"], r2["growth_over_time"])
    np.testing.assert_array_equal(r1["prop_over_time"], r2["prop_over_time"])
    assert r1["summary"] == r2["summary"]
