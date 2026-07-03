"""PRD flow 2.3: Run/Save validate ages + non-empty spending, toast field errors."""
import copy

from engine.params import validate_scenario

VALID = {
    "$schema": "scenario.v1",
    "name": "smoke",
    "portfolio": {
        "initial_portfolio": 3_000_000,
        "start_age": 60,
        "end_age": 95,
        "market": "IL",
        "fat_tails_enabled": True,
        "fat_tails_df": 5,
        "mode": "annual",
        "n_paths": 10000,
        "random_seed": 42,
    },
    "spending_bands": [
        {"id": "sb-1", "age_from": 60, "age_to": 95, "amount_monthly": 25000,
         "label": "Base living", "category": "strict"},
    ],
    "income_bands": [],
    "lumps": [],
    "properties": [],
}


def test_valid_scenario_has_no_errors():
    assert validate_scenario(VALID) == []


def test_empty_spending_is_rejected():
    s = copy.deepcopy(VALID)
    s["spending_bands"] = []
    errors = validate_scenario(s)
    assert any("spending" in e.lower() for e in errors)


def test_start_age_not_less_than_end_age_is_rejected():
    s = copy.deepcopy(VALID)
    s["portfolio"]["start_age"] = 95
    s["portfolio"]["end_age"] = 60
    errors = validate_scenario(s)
    assert any("age" in e.lower() for e in errors)


def test_band_age_from_after_age_to_is_rejected():
    s = copy.deepcopy(VALID)
    s["spending_bands"][0]["age_from"] = 80
    s["spending_bands"][0]["age_to"] = 70
    errors = validate_scenario(s)
    assert any("age_from" in e for e in errors)


def test_monthly_mode_scenario_is_valid():
    s = copy.deepcopy(VALID)
    s["portfolio"]["mode"] = "monthly"
    assert validate_scenario(s) == []
