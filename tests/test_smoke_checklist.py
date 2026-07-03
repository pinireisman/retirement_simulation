"""Automates the cheap top-5 items of PRD §9.2's manual smoke checklist.

No browser/Selenium: exercises the same functions the UI callbacks call
(execute_run, scenario_to_xlsx/scenario_from_xlsx) directly, which is the
"automated where cheap" instruction in PRD §8 Phase 6 without adding a new
test-automation dependency.
"""
import copy

from cli import read_scenario_data
from engine.params import SimulationParams, scenario_to_xlsx, scenario_from_xlsx
from engine.simulation import run_simulation
from webapp.callbacks import execute_run
from webapp.layout import DEFAULT_SCENARIO

SCENARIO = copy.deepcopy(DEFAULT_SCENARIO)
SCENARIO["name"] = "smoke"
SCENARIO["portfolio"]["initial_portfolio"] = 3_000_000
SCENARIO["spending_bands"] = [
    {"id": "sb-1", "age_from": 60, "age_to": 95, "amount_monthly": 25000,
     "label": "Base living", "category": "strict"},
]


def test_save_then_load_roundtrips(tmp_path):
    # item 2: Save As -> file written
    path = tmp_path / "smoke.xlsx"
    scenario_to_xlsx(SCENARIO, path)
    assert path.exists()

    # item 3: Load -> tabs repopulate from the saved file
    reloaded = scenario_from_xlsx(path)
    assert reloaded["portfolio"]["initial_portfolio"] == SCENARIO["portfolio"]["initial_portfolio"]
    assert len(reloaded["spending_bands"]) == 1
    assert reloaded["spending_bands"][0]["amount_monthly"] == 25000


def test_run_simulation_returns_figure_and_ruin_badge():
    # item 4: Run Simulation -> figure + ruin probability
    run_id, figure, summary, badges, guardrail_stats = execute_run(
        SCENARIO, {"guardrails": []}, [], include_historic=False,
    )
    assert run_id
    assert figure is not None
    assert 0.0 <= summary["ruin_probability"] <= 1.0
    assert badges == []  # no playground events, no guardrails -> baseline run
    assert guardrail_stats is None


def test_saved_scenario_matches_cli_ruin_probability(tmp_path):
    # item 5: a scenario saved from the web app runs unchanged through the CLI
    # and produces the same ruin probability (same seed) -- PRD §1.4.
    path = tmp_path / "smoke.xlsx"
    scenario_to_xlsx(SCENARIO, path)

    _, _, web_summary, _, _ = execute_run(SCENARIO, {"guardrails": []}, [], include_historic=False)

    cli_params = SimulationParams.from_legacy_scenario_data(read_scenario_data(path))
    cli_result = run_simulation(cli_params)

    assert round(web_summary["ruin_probability"], 3) == round(cli_result["summary"]["ruin_probability"], 3)
