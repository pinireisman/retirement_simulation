"""Statistical sanity tests for the two-bucket strategy (PRD §14.6)."""

import os
import sys

import numpy as np
import pytest

_tests_dir = os.path.dirname(__file__)
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)
import golden_scenarios  # noqa: E402

from engine.params import Band, SimulationParams
from engine.simulation import run_simulation
from engine.withdrawal_strategies import (
    RefillPolicyConfig,
    ReserveConfig,
    ReturnModelConfig,
    WithdrawalStrategyConfig,
    sample_reserve_returns,
)


def _wcfg(target_years, trigger_years, amount_rule="to_target", distribution="normal",
          mean_real=0.01, std_real=0.03):
    return WithdrawalStrategyConfig(
        type="two_bucket",
        reserve=ReserveConfig(target_years=target_years, refill_trigger_years=trigger_years,
                               coverage_scope="recurring_gap_only",
                               return_model=ReturnModelConfig(distribution=distribution,
                                                               mean_real=mean_real, std_real=std_real)),
        refill_policy=RefillPolicyConfig(amount_rule=amount_rule),
    )


def test_1_zero_reserve_years_matches_single_mode_summary():
    params_single, guardrails = golden_scenarios.annual_plain()
    result_single = run_simulation(params_single, guardrails=guardrails)

    wcfg = _wcfg(target_years=0.0, trigger_years=0.0)
    params_two = golden_scenarios._base_params(random_seed=params_single.random_seed,
                                                withdrawal_strategy=wcfg)
    result_two = run_simulation(params_two, guardrails=None)

    assert result_two["summary"] == result_single["summary"]


def test_2_transfers_conserve_wealth_to_target_vs_none():
    """With identical growth/reserve returns every period, combined wealth
    is bucket-split-invariant: growth_gain+reserve_gain == combined*(1+r)
    regardless of how much sits in each bucket, and internal transfers
    cancel by construction. So refilling (to_target) vs never refilling
    (none) must produce bit-for-bit-equal combined balances."""

    def _run(amount_rule):
        # trigger_years close to target_years: a single period's drawdown
        # already dips the reserve below trigger, so a refill actually fires
        # under to_target (and is skipped under none) -- otherwise both runs
        # would trivially never refill and the test wouldn't exercise anything.
        wcfg = _wcfg(target_years=3.0, trigger_years=2.99, amount_rule=amount_rule,
                      distribution="constant", mean_real=0.042, std_real=0.0)
        params = SimulationParams(
            start_age=60, end_age=75, initial_portfolio=1_000_000,
            real_return_mean=0.042, real_return_sd=0.0, fat_tails_df=None,
            annual=True, n_paths=50, random_seed=11,
            spending_bands=[Band(60, 75, 80_000, "base", "strict")],
            withdrawal_strategy=wcfg,
        )
        return run_simulation(params, guardrails=None)

    r_target = _run("to_target")
    r_none = _run("none")
    np.testing.assert_allclose(r_target["bal_over_time"], r_none["bal_over_time"])
    # sanity: the two runs actually differ internally (refills did happen)
    assert not np.allclose(r_target["growth_over_time"], r_none["growth_over_time"])


def test_3_zero_volatility_no_gaps_is_deterministic_compounding():
    wcfg = _wcfg(target_years=2.0, trigger_years=1.0, distribution="constant",
                  mean_real=0.03, std_real=0.0)
    params = SimulationParams(
        start_age=60, end_age=64, initial_portfolio=100_000,
        real_return_mean=0.05, real_return_sd=0.0, fat_tails_df=None,
        annual=True, n_paths=3, random_seed=5,
        withdrawal_strategy=wcfg,  # no spending/income bands at all -> zero planned gap
    )
    result = run_simulation(params, guardrails=None)
    expected = 100_000 * (1.05 ** np.arange(1, 6))
    for path in range(3):
        np.testing.assert_allclose(result["bal_over_time"][:, path], expected)


def test_4_reserve_return_sample_moments():
    cfg = ReturnModelConfig(distribution="normal", mean_real=0.02, std_real=0.06)
    samples = sample_reserve_returns(cfg, n_paths=20_000, n_periods=1, seed=1, annual=True)
    assert abs(samples.mean() - 0.02) < 0.005
    assert abs(samples.std() - 0.06) < 0.005


def test_5_larger_target_years_gives_weakly_higher_average_reserve_share():
    def _avg_reserve_share(target_years):
        wcfg = _wcfg(target_years, max(0.0, target_years - 1.0))
        params = golden_scenarios._base_params(n_paths=200, withdrawal_strategy=wcfg)
        result = run_simulation(params, guardrails=None)
        reserve_over_time = result["bal_over_time"] - result["growth_over_time"]
        share = reserve_over_time / np.maximum(result["bal_over_time"], 1e-9)
        return share.mean()

    small = _avg_reserve_share(1.0)
    large = _avg_reserve_share(6.0)
    assert large >= small
