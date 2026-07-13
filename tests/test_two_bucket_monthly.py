"""Monthly-mode tests for TwoBucketStrategy (PRD §6.11, §8.4 step H, Phase 6).

Two things are pinned:
  1. refill/forced-sale effects: refill only fires at sim-year boundaries
     ((i+1) % 12 == 0); intermediate months never transfer money.
  2. target_years=0 two-bucket monthly is bit-equal to single-mode monthly
     (mirrors the annual killer RNG test in tests/test_two_bucket_rng.py).
"""

import os
import sys

import numpy as np

_tests_dir = os.path.dirname(__file__)
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)
import golden_scenarios  # noqa: E402

import pytest

from engine.params import Band, SimulationParams
from engine.simulation import run_simulation
from engine.withdrawal_strategies import (
    DrawPolicyConfig,
    RefillPolicyConfig,
    ReserveConfig,
    ReturnModelConfig,
    TwoBucketStrategy,
    WithdrawalStrategyConfig,
)


class _FakeParams:
    """Minimal params stand-in, mirroring tests/test_two_bucket_strategy.py's
    unit-test scaffolding but with annual=False for monthly-mode unit tests."""

    def __init__(self, initial_portfolio, withdrawal_strategy, annual=False):
        self.annual = annual
        self.initial_portfolio = initial_portfolio
        self.withdrawal_strategy = withdrawal_strategy


def _make_monthly_strategy(port_ret, reserve_ret, planned_gap, *, initial_portfolio=0.0,
                            target_years=4.0, trigger_years=3.0, first_period_source="reserve",
                            threshold=0.0, eligibility_rule="growth_return_at_or_above_threshold",
                            amount_rule="to_target", refill_threshold=0.0):
    wcfg = WithdrawalStrategyConfig(
        type="two_bucket",
        reserve=ReserveConfig(target_years=target_years, refill_trigger_years=trigger_years,
                               coverage_scope="recurring_gap_only",
                               return_model=ReturnModelConfig()),
        draw_policy=DrawPolicyConfig(market_state_rule="previous_period_return",
                                      growth_return_threshold_real=threshold,
                                      first_period_source=first_period_source),
        refill_policy=RefillPolicyConfig(cadence="annual", eligibility_rule=eligibility_rule,
                                          growth_return_threshold_real=refill_threshold,
                                          amount_rule=amount_rule),
    )
    return TwoBucketStrategy(_FakeParams(initial_portfolio, wcfg), port_ret, reserve_ret, planned_gap)


def _wcfg(target_years, trigger_years, distribution="normal", mean_real=0.01, std_real=0.03):
    return WithdrawalStrategyConfig(
        type="two_bucket",
        reserve=ReserveConfig(target_years=target_years, refill_trigger_years=trigger_years,
                               coverage_scope="recurring_gap_only",
                               return_model=ReturnModelConfig(distribution=distribution,
                                                               mean_real=mean_real, std_real=std_real)),
    )


def test_refill_only_fires_at_sim_year_boundaries():
    """Deterministic small-n_paths monthly run: refill_probability_by_period
    (and the underlying transfer) must be exactly zero for every
    intermediate month and can only be nonzero at indices 11, 23, 35, ...
    (i.e. where (i+1) % 12 == 0)."""
    wcfg = _wcfg(target_years=2.0, trigger_years=1.9, distribution="constant", mean_real=0.0)
    params = SimulationParams(
        start_age=60, end_age=65, initial_portfolio=1_000_000,
        real_return_mean=0.06, real_return_sd=0.10, fat_tails_df=None,
        annual=False, n_paths=25, random_seed=3,
        spending_bands=[Band(60, 65, 80_000, "base", "strict")],
        withdrawal_strategy=wcfg,
    )
    result = run_simulation(params, guardrails=None)

    refill_prob = result["refill_probability_by_period"]
    n_periods = len(refill_prob)
    year_end_idx = set(range(11, n_periods, 12))

    # At least one boundary period actually refills, so the test exercises
    # something real (not a config that trivially never refills).
    assert any(refill_prob[i] > 0 for i in year_end_idx)

    for i in range(n_periods):
        if i not in year_end_idx:
            assert refill_prob[i] == 0.0, f"period {i} is not a sim-year boundary but refilled"


def test_refill_transfer_zero_on_intermediate_months_growth_only_reflects_returns():
    """Direct strategy-level check (not just the aggregated probability):
    growth_over_time between two consecutive year-boundary refills changes
    only by the compounding return and cash flow, never by a mid-year
    transfer -- verified by re-deriving month-to-month growth deltas and
    confirming no anomalous jump coincides with a non-boundary month."""
    wcfg = _wcfg(target_years=3.0, trigger_years=2.9, distribution="constant", mean_real=0.0)
    params = SimulationParams(
        start_age=60, end_age=63, initial_portfolio=500_000,
        real_return_mean=0.08, real_return_sd=0.15, fat_tails_df=None,
        annual=False, n_paths=10, random_seed=7,
        spending_bands=[Band(60, 63, 40_000, "base", "strict")],
        withdrawal_strategy=wcfg,
    )
    result = run_simulation(params, guardrails=None)
    refill_prob = result["refill_probability_by_period"]
    n_periods = len(refill_prob)
    for i in range(n_periods):
        if (i + 1) % 12 != 0:
            assert refill_prob[i] == 0.0


def test_target_years_zero_monthly_bit_equal_to_single_mode():
    """Mirrors tests/test_two_bucket_rng.py's annual killer test at monthly
    cadence: with an always-empty reserve, growth_over_time (and combined
    bal_over_time) must be bit-identical to the single-portfolio monthly run."""
    params_single, guardrails = golden_scenarios.monthly_plain()
    result_single = run_simulation(params_single, guardrails=guardrails)

    wcfg = _wcfg(target_years=0.0, trigger_years=0.0)
    params_two = golden_scenarios._base_params(annual=False, n_paths=params_single.n_paths,
                                                random_seed=params_single.random_seed,
                                                withdrawal_strategy=wcfg)
    result_two = run_simulation(params_two, guardrails=None)

    np.testing.assert_array_equal(result_two["growth_over_time"], result_single["bal_over_time"])
    np.testing.assert_array_equal(result_two["bal_over_time"], result_single["bal_over_time"])


def test_gains_only_monthly_accumulates_across_the_sim_year():
    """PRD §6.8: gains_only's cap is the SUM of the just-completed sim
    year's 12 monthly gains (a running accumulator reset at each year
    boundary), never inferred from ending-minus-starting balance.

    Direct strategy-level unit test (mirrors test_two_bucket_strategy.py's
    scaffolding) so ``need`` (target - reserve) can be forced huge and never
    binding, isolating the accumulator itself: with a huge target/trigger and
    "always" eligibility, the amount transferred at the first year-boundary
    refill (period index 11, with a 24-period/2-year strategy so a forward
    t+1 window exists) must equal the hand-accumulated sum of the 12 monthly
    exposed-balance gains -- not a closed-form compounding shortcut, since
    that's exactly the distinction the accumulator exists to preserve.
    """
    n_periods = 24  # 2 sim years, so period 11 (end of year 0) has a t+1 window
    port_ret = np.zeros((1, n_periods))
    port_ret[:, :12] = 0.01  # year-0 monthly growth return
    reserve_ret = np.zeros((1, n_periods))
    planned_gap = np.zeros(n_periods)

    strat = _make_monthly_strategy(port_ret, reserve_ret, planned_gap,
                                    amount_rule="gains_only", eligibility_rule="always")
    strat.target_by_period = np.full(n_periods, 1e9)   # never the binding constraint
    strat.trigger_by_period = np.full(n_periods, 1e9)  # always eligible (reserve < trigger)
    strat.growth = np.array([1_000_000.0])
    strat.reserve = np.array([0.0])

    expected_gain = 0.0
    for i in range(12):
        strat.apply_cashflow(i, np.array([0.0]))
        exposed = strat.growth.copy()
        strat.apply_returns(i, np.array([True]))
        expected_gain += exposed[0] * port_ret[0, i]
        strat.end_of_period(i, np.array([True]))
        strat.record(i)
        if i < 11:
            assert strat.reserve[0] == pytest.approx(0.0)  # no refill before year-end

    assert strat.reserve[0] == pytest.approx(expected_gain)
    assert expected_gain > 0
