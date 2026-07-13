"""Guardrail integration tests for the two-bucket strategy (PRD §6.12, §14.4).

Verified in the PRD: "no changes to engine/guardrails.py are needed" --
the funded-ratio guardrail reads strategy.combined_balance(), and the
volatility guardrail already reads port_ret[:, i-1] directly (the
growth-bucket stream in two-bucket mode). These tests prove both claims by
running the SAME guardrail config against single-mode and
two_bucket(target_years=0) params (reserve always empty, so growth ==
combined) with common random numbers, and confirming bit-identical output.
"""

import os
import sys

import numpy as np

_tests_dir = os.path.dirname(__file__)
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)
import golden_scenarios  # noqa: E402

from engine.simulation import run_simulation
from engine.withdrawal_strategies import (
    ReserveConfig,
    ReturnModelConfig,
    WithdrawalStrategyConfig,
)


def _zero_years_cfg(distribution="normal", mean_real=0.01, std_real=0.03):
    return WithdrawalStrategyConfig(
        type="two_bucket",
        reserve=ReserveConfig(target_years=0.0, refill_trigger_years=0.0,
                               coverage_scope="recurring_gap_only",
                               return_model=ReturnModelConfig(distribution=distribution,
                                                               mean_real=mean_real, std_real=std_real)),
    )


def test_funded_ratio_confidence_guardrail_equivalent_single_vs_two_bucket_zero_years():
    """§14.4 item 1: funded-ratio guardrail (confidence mode, which also
    exercises the inner same-seed baseline recursion) uses combined_balance();
    with target_years=0 the reserve is always empty, so multipliers and the
    resulting balances must be bit-identical to single mode."""
    params_single, guardrails = golden_scenarios.annual_confidence()
    result_single = run_simulation(params_single, guardrails=guardrails)

    wcfg = _zero_years_cfg()
    params_two = golden_scenarios._base_params(random_seed=params_single.random_seed,
                                                withdrawal_strategy=wcfg)
    result_two = run_simulation(params_two, guardrails=guardrails)

    np.testing.assert_array_equal(result_two["bal_over_time"], result_single["bal_over_time"])
    np.testing.assert_array_equal(result_two["growth_over_time"], result_single["bal_over_time"])
    assert result_two["summary"] == result_single["summary"]

    gp_single = result_single["guardrail_mult_percentiles"]
    gp_two = result_two["guardrail_mult_percentiles"]
    for key in ("p10", "p50", "p90"):
        np.testing.assert_array_equal(gp_two[key], gp_single[key])


def test_volatility_guardrail_scaled_spending_identical_single_vs_two_bucket_zero_years():
    """§14.4 item 2/§6.12: the volatility-discretionary-scaling guardrail
    reads port_ret[:, i-1] directly (unaffected by strategy choice) and
    needs zero changes for two-bucket -- confirm the guardrail-scaled
    spending path is identical between single and two_bucket(target_years=0)."""
    params_single, guardrails = golden_scenarios.annual_volatility()
    result_single = run_simulation(params_single, guardrails=guardrails)

    wcfg = _zero_years_cfg()
    params_two = golden_scenarios._base_params(random_seed=params_single.random_seed,
                                                withdrawal_strategy=wcfg)
    result_two = run_simulation(params_two, guardrails=guardrails)

    np.testing.assert_array_equal(result_two["bal_over_time"], result_single["bal_over_time"])
    assert result_two["guardrail_stats"] == result_single["guardrail_stats"]


def test_confidence_mode_calibration_runs_with_two_bucket_strategy():
    """§14.4 item 5/§6.12: confidence-mode funded-ratio calibration must run
    to completion with a real (non-degenerate) two-bucket strategy and
    produce sane calibrated multipliers -- smoke test, not a value pin."""
    from engine.guardrails import GuardrailConfig

    wcfg = WithdrawalStrategyConfig(
        type="two_bucket",
        reserve=ReserveConfig(target_years=4.0, refill_trigger_years=3.0,
                               coverage_scope="recurring_gap_only",
                               return_model=ReturnModelConfig(distribution="normal",
                                                               mean_real=0.01, std_real=0.03)),
    )
    params = golden_scenarios._base_params(withdrawal_strategy=wcfg, n_paths=300)
    guardrails = [GuardrailConfig(
        type="funded_ratio_guardrail",
        options=dict(mode="confidence", c_cut=0.85, c_target=0.95, c_raise=0.99, c_severe=0.80),
    )]

    result = run_simulation(params, guardrails=guardrails)

    assert result["baseline_summary"] is not None
    mult = result["guardrail_mult_percentiles"]
    assert mult is not None
    for key in ("p10", "p50", "p90"):
        arr = mult[key]
        assert np.all(np.isfinite(arr))
        assert np.all(arr > 0)
    assert result["withdrawal_strategy_type"] == "two_bucket"
    assert np.all(np.isfinite(result["bal_over_time"]))
