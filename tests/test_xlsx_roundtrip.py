import pytest
from pathlib import Path
import tempfile

from engine.params import SimulationParams, scenario_to_xlsx, scenario_from_xlsx
from engine.simulation import run_simulation
from cli import read_scenario_data


def test_roundtrip():
    """Test that a scenario dict can be written to xlsx and read back unchanged.

    Portfolio scalars are deliberately non-default (market US not IL, mode
    monthly not annual, random_seed 7 not 42, n_paths 500 not 10000) so this
    actually exercises the PRD 4.4 round-trip guarantee for those columns
    instead of trivially matching scenario_from_xlsx's fallback defaults.
    """
    # Build a test scenario with realistic values
    s = {
        "name": "",  # xlsx schema doesn't carry name; it's always "" on read back
        "$schema": "scenario.v1",
        "portfolio": {
            "initial_portfolio": 3000000,
            "start_age": 60,
            "end_age": 95,
            "market": "US",
            "fat_tails_enabled": True,
            "fat_tails_df": 5,
            "mode": "monthly",
            "n_paths": 500,
            "random_seed": 7
        },
        "spending_bands": [
            {
                "id": "sb-1",
                "age_from": 60,
                "age_to": 95,
                "amount_monthly": 25000,
                "label": "Base living",
                "category": "strict"
            },
            {
                "id": "sb-2",
                "age_from": 60,
                "age_to": 80,
                "amount_monthly": 5000,
                "label": "Travel",
                "category": "lifestyle"
            }
        ],
        "income_bands": [
            {
                "id": "ib-1",
                "age_from": 60,
                "age_to": 67,
                "amount_monthly": 12000,
                "label": "Consulting"
            }
        ],
        "lumps": [
            {
                "id": "lp-1",
                "age": 70,
                "amount": -400000,
                "label": "Gift to kids",
                "category": "gifts"
            }
        ],
        "properties": [
            {
                "id": "pr-1",
                "start_age": 60,
                "initial_value": 2500000,
                "rent_monthly": 6000,
                "label": "TLV apartment"
            }
        ]
    }

    # Write to xlsx and read back
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_scenario.xlsx"
        
        # Write to xlsx
        scenario_to_xlsx(s, test_file)
        
        # Read back from xlsx
        result = scenario_from_xlsx(test_file)
        
        # Remove all id fields from both for comparison
        def remove_ids(obj):
            if isinstance(obj, dict):
                return {k: remove_ids(v) for k, v in obj.items() if k != "id"}
            elif isinstance(obj, list):
                return [remove_ids(item) for item in obj]
            else:
                return obj
        
        s_cleaned = remove_ids(s)
        result_cleaned = remove_ids(result)
        
        # Compare the cleaned results
        assert s_cleaned == result_cleaned


def test_legacy_file_loads():
    """Test that the legacy example file loads without error."""
    # This should not raise any exceptions
    result = scenario_from_xlsx("scenario_data_example.xlsx")
    
    # Should have non-empty spending_bands
    assert len(result["spending_bands"]) > 0
    
    # Should have correct portfolio values (matches scenario_data_example.xlsx row 0)
    assert result["portfolio"]["start_age"] == 50
    assert result["portfolio"]["end_age"] == 95
    assert result["portfolio"]["initial_portfolio"] == 8000000
    
    # Should have the right schema
    assert result["$schema"] == "scenario.v1"

    # Untagged legacy rows default to "strict"; travel-derived rows default to "lifestyle"
    travel_bands = [b for b in result["spending_bands"] if "travel" in b["label"].lower()]
    regular_bands = [b for b in result["spending_bands"] if b not in travel_bands]
    assert travel_bands, "expected the legacy file's travel row to migrate into spending_bands"
    assert all(b["category"] == "lifestyle" for b in travel_bands)
    assert all(b["category"] == "strict" for b in regular_bands)


def test_legacy_loader_interop():
    """PRD 4.4: a file written by scenario_to_xlsx must still load through the
    old CLI path (cli.read_scenario_data + SimulationParams.from_legacy_scenario_data),
    which selects columns by name and ignores the new scalar columns entirely.
    Both loading paths must agree on the simulation result for a scenario whose
    values equal from_legacy_scenario_data's own defaults (market IL, annual,
    fat_tails_df 5, n_paths 10000, random_seed 42) -- the one combination where
    "ignoring the new columns" and "reading them" produce identical params.
    """
    s = {
        "name": "",
        "$schema": "scenario.v1",
        "portfolio": {
            "initial_portfolio": 3000000,
            "start_age": 60,
            "end_age": 95,
            "market": "IL",
            "fat_tails_enabled": True,
            "fat_tails_df": 5,
            "mode": "annual",
            "n_paths": 10000,
            "random_seed": 42
        },
        "spending_bands": [
            {"id": "sb-1", "age_from": 60, "age_to": 95, "amount_monthly": 25000,
             "label": "Base living", "category": "strict"},
        ],
        "income_bands": [
            {"id": "ib-1", "age_from": 60, "age_to": 67, "amount_monthly": 12000,
             "label": "Consulting"},
        ],
        "lumps": [
            {"id": "lp-1", "age": 70, "amount": -400000, "label": "Gift to kids",
             "category": "gifts"},
        ],
        "properties": [
            {"id": "pr-1", "start_age": 60, "initial_value": 2500000,
             "rent_monthly": 6000, "label": "TLV apartment"},
        ],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "interop.xlsx"
        scenario_to_xlsx(s, path)

        # New path: scenario_from_xlsx -> SimulationParams.from_scenario
        params_new = SimulationParams.from_scenario(scenario_from_xlsx(path))

        # Legacy path: cli.read_scenario_data -> SimulationParams.from_legacy_scenario_data
        params_legacy = SimulationParams.from_legacy_scenario_data(read_scenario_data(path))

        result_new = run_simulation(params_new)
        result_legacy = run_simulation(params_legacy)

        assert result_new["summary"]["ruin_probability"] == result_legacy["summary"]["ruin_probability"]