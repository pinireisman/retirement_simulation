"""PRD_UNDO_MAXIMIZE_DISTRIBUTION.md Phase C: editable portfolio mu/sigma.

Architect-owned acceptance tests (do not edit) for:
  - SimulationParams.from_scenario falling back to the selected market's
    mu/sigma when portfolio["mu"]/["sigma"] are absent or None, and using
    them directly when present.
  - xlsx round-trip persistence of the new mu/sigma scalar columns, with the
    same None-fallback semantics as fat_tails_df/market/etc.
"""
import tempfile
from pathlib import Path

from engine.markets import MARKETS
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


def test_from_scenario_uses_market_default_when_mu_sigma_absent():
    s = _scenario()  # no "mu"/"sigma" keys at all
    params = SimulationParams.from_scenario(s)
    assert params.real_return_mean == MARKETS["US"]["mu"]
    assert params.real_return_sd == MARKETS["US"]["sigma"]


def test_from_scenario_uses_market_default_when_mu_sigma_none():
    s = _scenario({"mu": None, "sigma": None})
    params = SimulationParams.from_scenario(s)
    assert params.real_return_mean == MARKETS["US"]["mu"]
    assert params.real_return_sd == MARKETS["US"]["sigma"]


def test_from_scenario_uses_custom_mu_sigma_when_present():
    s = _scenario({"mu": 0.09, "sigma": 0.25})
    params = SimulationParams.from_scenario(s)
    assert params.real_return_mean == 0.09
    assert params.real_return_sd == 0.25


def test_custom_mu_sigma_does_not_affect_property_housing_growth():
    """The mu/sigma override is portfolio-only; properties must keep using
    the selected market's own housing_mu/housing_sigma regardless."""
    s = _scenario({"mu": 0.09, "sigma": 0.25})
    s["properties"] = [{"start_age": 60, "initial_value": 500000,
                         "rent_monthly": 1000, "label": "test"}]
    params = SimulationParams.from_scenario(s)
    assert params.properties[0].growth_mean == MARKETS["US"]["housing_mu"]
    assert params.properties[0].growth_sd == MARKETS["US"]["housing_sigma"]


def test_from_scenario_zero_mu_is_not_treated_as_absent():
    """0.0 is a legitimate (if pessimistic) expected return -- must not be
    coerced to the market default the way None/missing is."""
    s = _scenario({"mu": 0.0, "sigma": 0.10})
    params = SimulationParams.from_scenario(s)
    assert params.real_return_mean == 0.0
    assert params.real_return_sd == 0.10


def test_xlsx_roundtrip_persists_custom_mu_sigma():
    s = _scenario({"mu": 0.061, "sigma": 0.171})
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "s.xlsx"
        scenario_to_xlsx(s, path)
        result = scenario_from_xlsx(path)
    assert result["portfolio"]["mu"] == 0.061
    assert result["portfolio"]["sigma"] == 0.171


def test_xlsx_roundtrip_absent_mu_sigma_reads_back_as_none():
    s = _scenario()  # no mu/sigma set
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "s.xlsx"
        scenario_to_xlsx(s, path)
        result = scenario_from_xlsx(path)
    assert result["portfolio"]["mu"] is None
    assert result["portfolio"]["sigma"] is None


def test_legacy_xlsx_without_mu_sigma_columns_still_loads():
    """A file written before this feature existed (no mu/sigma columns at
    all) must still load, with mu/sigma reading back as None so
    from_scenario's market fallback applies."""
    result = scenario_from_xlsx("scenario_data_example.xlsx")
    assert result["portfolio"].get("mu") is None
    assert result["portfolio"].get("sigma") is None
    # and from_scenario must still work end-to-end off that
    params = SimulationParams.from_scenario(result)
    assert params.real_return_mean == MARKETS[result["portfolio"]["market"]]["mu"]
