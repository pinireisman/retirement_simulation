"""PRD_FUNDED_RATIO_GUARDRAILS.md Stage 1a: real_discount_rate param plumbing.

Architect-owned acceptance tests (do not edit) for:
  - SimulationParams gaining a real_discount_rate field (default 0.01), used
    only by the funded-ratio guardrail's PV lookahead.
  - SimulationParams.from_scenario reading portfolio["real_discount_rate"],
    falling back to 0.01 when absent (mirrors fat_tails_df's scalar-with-
    default persistence, NOT mu/sigma's None-fallback).
  - xlsx round-trip persistence of the new real_discount_rate scalar column.
"""
import tempfile
from pathlib import Path

from engine.params import SimulationParams, scenario_from_xlsx, scenario_to_xlsx

BASE_PORTFOLIO = {
    "initial_portfolio": 1_000_000,
    "start_age": 60,
    "end_age": 95,
    "market": "US",
    "fat_tails_enabled": True,
    "fat_tails_df": 5,
    "mode": "annual",
    "n_paths": 1000,
    "random_seed": 42,
}


def _scenario(portfolio_overrides=None):
    portfolio = dict(BASE_PORTFOLIO)
    portfolio.update(portfolio_overrides or {})
    return {
        "name": "",
        "$schema": "scenario.v1",
        "portfolio": portfolio,
        "spending_bands": [],
        "income_bands": [],
        "lumps": [],
        "properties": [],
    }


def test_default_real_discount_rate_is_one_percent():
    params = SimulationParams(
        start_age=60, end_age=95, initial_portfolio=1_000_000,
        real_return_mean=0.05, real_return_sd=0.16, fat_tails_df=5,
    )
    assert params.real_discount_rate == 0.01


def test_from_scenario_uses_default_when_absent():
    s = _scenario()  # no "real_discount_rate" key at all
    params = SimulationParams.from_scenario(s)
    assert params.real_discount_rate == 0.01


def test_from_scenario_uses_custom_value_when_present():
    s = _scenario({"real_discount_rate": 0.025})
    params = SimulationParams.from_scenario(s)
    assert params.real_discount_rate == 0.025


def test_from_scenario_zero_is_not_treated_as_absent():
    """0.0 is a legitimate discount rate -- must not be coerced to the 0.01
    default the way an absent key is."""
    s = _scenario({"real_discount_rate": 0.0})
    params = SimulationParams.from_scenario(s)
    assert params.real_discount_rate == 0.0


def test_xlsx_roundtrip_persists_custom_real_discount_rate():
    s = _scenario({"real_discount_rate": 0.0175})
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "s.xlsx"
        scenario_to_xlsx(s, path)
        result = scenario_from_xlsx(path)
    assert result["portfolio"]["real_discount_rate"] == 0.0175
    # and it survives the trip all the way into params
    assert SimulationParams.from_scenario(result).real_discount_rate == 0.0175


def test_xlsx_roundtrip_absent_reads_back_as_default():
    s = _scenario()  # no real_discount_rate set
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "s.xlsx"
        scenario_to_xlsx(s, path)
        result = scenario_from_xlsx(path)
    assert result["portfolio"]["real_discount_rate"] == 0.01


def test_legacy_xlsx_without_column_still_loads():
    """A file written before this feature existed (no real_discount_rate
    column) must still load, defaulting to 0.01 so from_scenario works."""
    result = scenario_from_xlsx("scenario_data_example.xlsx")
    assert result["portfolio"]["real_discount_rate"] == 0.01
    params = SimulationParams.from_scenario(result)
    assert params.real_discount_rate == 0.01
