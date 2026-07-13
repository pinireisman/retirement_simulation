"""Smoke tests for the two-bucket chart functions (PRD §12.3). These are not
golden/pixel tests -- just "does it build a valid go.Figure without crashing,
for the shapes the engine actually produces" (annual, monthly, with/without
a comparison attached).
"""

import dataclasses
import os
import sys

import plotly.graph_objects as go
import pytest

_tests_dir = os.path.dirname(__file__)
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)
import golden_scenarios  # noqa: E402

from engine.figures import (fig_bucket_balances, fig_funding_source,
                            fig_strategy_events, fig_terminal_comparison)
from engine.simulation import run_simulation
from engine.withdrawal_strategies import ReserveConfig, ReturnModelConfig, WithdrawalStrategyConfig
from webapp.callbacks import build_comparison


def _wcfg():
    return WithdrawalStrategyConfig(
        type="two_bucket",
        reserve=ReserveConfig(target_years=3.0, refill_trigger_years=2.0,
                               coverage_scope="recurring_gap_only",
                               return_model=ReturnModelConfig(distribution="normal",
                                                               mean_real=0.01, std_real=0.03)),
    )


@pytest.fixture(scope="module")
def annual_results():
    params = golden_scenarios._base_params(withdrawal_strategy=_wcfg(), random_seed=42)
    return run_simulation(params, guardrails=None)


@pytest.fixture(scope="module")
def monthly_results():
    params = golden_scenarios._base_params(withdrawal_strategy=_wcfg(), random_seed=42,
                                            annual=False, n_paths=200, end_age=65)
    return run_simulation(params, guardrails=None)


@pytest.mark.parametrize("fig_fn", [fig_bucket_balances, fig_funding_source, fig_strategy_events])
def test_chart_builds_for_annual(annual_results, fig_fn):
    fig = fig_fn(annual_results)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


@pytest.mark.parametrize("fig_fn", [fig_bucket_balances, fig_funding_source, fig_strategy_events])
def test_chart_builds_for_monthly(monthly_results, fig_fn):
    fig = fig_fn(monthly_results)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_terminal_comparison_needs_comparison_key(annual_results):
    params = golden_scenarios._base_params(withdrawal_strategy=_wcfg(), random_seed=42)
    comp_params = dataclasses.replace(params, withdrawal_strategy=None)
    comp = run_simulation(comp_params, guardrails=None)
    results = dict(annual_results)
    results["comparison"] = build_comparison(results, comp)

    fig = fig_terminal_comparison(results)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1  # single box trace; comparator rendered as hline/hrect annotations
