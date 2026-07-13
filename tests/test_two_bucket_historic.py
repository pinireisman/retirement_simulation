"""Historic two-bucket test (PRD §6.13, Phase 6).

run_historic_scenario ships two-bucket support by reusing TwoBucketStrategy
with n_paths=1, always annual: the growth bucket is driven by the historic
real_factors, the reserve earns its configured mean_real deterministically
(no sampling, regardless of the configured distribution).
"""

import numpy as np
import pytest

from engine.params import Band, Lump, Property, SimulationParams
from engine.simulation import run_historic_scenario
from engine.withdrawal_strategies import ReserveConfig, ReturnModelConfig, WithdrawalStrategyConfig

REAL_FACTORS = [1.10, 0.85, 1.05, 1.20, 0.95, 1.02, 1.15, 0.90, 1.08]


def _base_params(**over):
    d = dict(
        start_age=60, end_age=68, initial_portfolio=1_000_000,
        real_return_mean=0.05, real_return_sd=0.12, fat_tails_df=None,
        annual=True, n_paths=1, random_seed=1,
        spending_bands=[Band(60, 68, 70_000, "base", "strict")],
        income_bands=[Band(60, 63, 20_000, "work")],
        lumps=[Lump(65, -50_000, "roof", "strict")],
        properties=[Property(60, 500_000, 20_000, 0.02, 0.05, "apt")],
    )
    d.update(over)
    return SimulationParams(**d)


def test_historic_two_bucket_runs_and_produces_sane_balances():
    wcfg = WithdrawalStrategyConfig(
        type="two_bucket",
        reserve=ReserveConfig(target_years=3.0, refill_trigger_years=2.0,
                               coverage_scope="recurring_gap_only",
                               return_model=ReturnModelConfig(distribution="normal",
                                                               mean_real=0.015, std_real=0.02)),
    )
    params = _base_params(withdrawal_strategy=wcfg)
    result = run_historic_scenario(params, REAL_FACTORS)

    assert result["withdrawal_strategy_type"] == "two_bucket"
    n = len(result["ages"])
    assert len(result["growth_over_time"]) == n
    assert len(result["reserve_over_time"]) == n

    # No crash, no NaNs/negatives sneaking in.
    assert all(np.isfinite(v) for v in result["growth_over_time"])
    assert all(np.isfinite(v) for v in result["reserve_over_time"])
    assert all(g >= -1e-6 for g in result["growth_over_time"])
    assert all(r >= -1e-6 for r in result["reserve_over_time"])

    # Combined balance reconciles with the reported growth+reserve split
    # at every recorded period.
    for g, r, combined in zip(result["growth_over_time"], result["reserve_over_time"],
                               result["portfolio_over_time"]):
        assert abs(combined - (g + r)) < 1e-6

    assert result["final_growth"] + result["final_reserve"] == pytest.approx(result["terminal_portfolio"])


def test_historic_reserve_tracks_configured_mean_real_not_historical_returns():
    """The reserve must compound by the configured mean_real EVERY period,
    fully decoupled from the historic real_factors driving growth.

    Setup isolates the reserve from any cash-flow draws or refills for the
    first 5 periods (ages 60-64: income exactly matches spending, so net
    external cash flow is zero and apply_cashflow performs no transfer;
    refill_trigger_years=0 means refill never fires at all), so those
    periods' recorded reserve balances must equal a pure compounding
    formula driven ONLY by mean_real -- with zero appearance of
    real_factors in the expected value, this is a direct, not just
    statistical, decoupling check.
    """
    mean_real = 0.02
    wcfg = WithdrawalStrategyConfig(
        type="two_bucket",
        # target_years >= plan length pulls the ENTIRE future gap (ages 65-68,
        # the only years without matching income) into the t=0 rolling window,
        # giving a nonzero initial reserve split.
        reserve=ReserveConfig(target_years=10.0, refill_trigger_years=0.0,
                               coverage_scope="recurring_gap_only",
                               return_model=ReturnModelConfig(distribution="normal",
                                                               mean_real=mean_real, std_real=0.10)),
    )
    params = _base_params(
        withdrawal_strategy=wcfg,
        spending_bands=[Band(60, 68, 70_000, "base", "strict")],
        income_bands=[Band(60, 64, 70_000, "work")],  # exactly covers spend through age 64
        lumps=[], properties=[],
    )
    result = run_historic_scenario(params, REAL_FACTORS)

    reserve0 = 4 * 70_000  # planned gap only in ages 65-68 (4 periods), fully inside the window
    assert result["reserve_over_time"][0] == pytest.approx(reserve0 * (1 + mean_real))
    for i in range(5):  # ages 60-64: zero cash flow, zero refill -> pure compounding
        expected = reserve0 * (1 + mean_real) ** (i + 1)
        assert result["reserve_over_time"][i] == pytest.approx(expected), f"period {i}"


def test_historic_two_bucket_zero_target_years_matches_single_portfolio():
    """PRD §14.5/§14.6-style equivalence, ported to historic mode: with the
    reserve permanently empty (target_years=0), the two-bucket path must
    reproduce the single-portfolio historic path bit-for-bit."""
    single_params = _base_params()
    result_single = run_historic_scenario(single_params, REAL_FACTORS)

    wcfg = WithdrawalStrategyConfig(
        type="two_bucket",
        reserve=ReserveConfig(target_years=0.0, refill_trigger_years=0.0,
                               coverage_scope="recurring_gap_only",
                               return_model=ReturnModelConfig(distribution="constant", mean_real=0.01)),
    )
    two_params = _base_params(withdrawal_strategy=wcfg)
    result_two = run_historic_scenario(two_params, REAL_FACTORS)

    np.testing.assert_allclose(result_two["portfolio_over_time"], result_single["portfolio_over_time"])
    np.testing.assert_allclose(result_two["growth_over_time"], result_single["portfolio_over_time"])
    assert result_two["ruined"] == result_single["ruined"]
    assert result_two["terminal_portfolio"] == pytest.approx(result_single["terminal_portfolio"])
