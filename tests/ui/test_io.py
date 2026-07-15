"""Scenario I/O journeys: upload, save (download), load, localStorage
persistence, toast positioning."""
from __future__ import annotations

import pytest
from playwright.sync_api import expect

from engine.params import scenario_from_xlsx
from tests.ui import journeys
from tests.ui.conftest import EXAMPLE_XLSX, REPO

pytestmark = pytest.mark.ui


@pytest.fixture
def no_new_scenario_files():
    """Guard: saving now downloads to the browser (PRD §3.1) — scenarios/
    must gain no new files across the test."""
    before = set((REPO / "scenarios").glob("*.xlsx"))
    yield
    after = set((REPO / "scenarios").glob("*.xlsx"))
    assert after == before, f"scenarios/ gained files from a save: {after - before}"


@pytest.mark.ux("UX-IO-01")
def test_upload_populates_plan(page, app_url, example_scenario):
    """Upload populates the whole plan."""
    page.goto(app_url)
    journeys.upload_scenario(page, str(EXAMPLE_XLSX))
    journeys.goto_view(page, "plan")

    assert journeys.row_count(page, "spending") == len(example_scenario["spending_bands"])
    assert journeys.row_count(page, "income") == len(example_scenario["income_bands"])
    assert journeys.row_count(page, "lumps") == len(example_scenario["lumps"])
    assert journeys.row_count(page, "properties") == len(example_scenario["properties"])

    # App bar reflects the loaded scenario's name.
    expect(page.locator("#header-scenario-name")).to_be_visible()


@pytest.mark.ux("UX-IO-02")
def test_save_creates_scenario(loaded_page, tmp_path, no_new_scenario_files):
    """Save downloads an .xlsx the engine can round-trip; nothing is written
    server-side and the download never appears in the Load dropdown."""
    download = journeys.save_scenario(loaded_page, "uxtest-roundtrip")

    assert download.suggested_filename == "uxtest-roundtrip.xlsx"
    dest = tmp_path / download.suggested_filename
    download.save_as(dest)
    scenario = scenario_from_xlsx(dest)
    assert scenario["portfolio"]

    assert not journeys.scenario_in_load_list(loaded_page, "uxtest-roundtrip")


# UX-IO-03 (overwrite flow) retired: PRD §3.1 removed server-side save and the
# chk-overwrite checkbox — saving is now a stateless browser download with no
# name-collision concept to test.


@pytest.mark.ux("UX-IO-04")
def test_reload_and_load(page, app_url):
    """Dropdown load still works locally; localStorage persists edits across
    a reload and falls back to the default scenario if the store is corrupt
    (PRD §3.3, §6.4)."""
    page.goto(app_url)

    # --- load from dropdown: scenarios/ still feeds it in local dev ---
    # (name must be unambiguous against sibling files, e.g. "scenario_example"
    # also prefix-matches "scenario_example_orig" in this repo's scenarios/)
    dropdown_scenario = scenario_from_xlsx(REPO / "scenarios" / "two_bucket_example.xlsx")
    journeys.load_scenario(page, "two_bucket_example")
    journeys.goto_view(page, "plan")
    assert journeys.row_count(page, "spending") == len(dropdown_scenario["spending_bands"])
    assert journeys.row_count(page, "income") == len(dropdown_scenario["income_bands"])
    assert journeys.row_count(page, "lumps") == len(dropdown_scenario["lumps"])
    assert journeys.row_count(page, "properties") == len(dropdown_scenario["properties"])

    # --- edits persist via localStorage across a reload, no Load needed ---
    journeys.open_tab(page, "Portfolio")
    page.locator("#inp-initial-portfolio").fill("999999")
    page.keyboard.press("Tab")
    expect(page.locator("#inp-initial-portfolio")).to_have_value("999999")
    # The edit round-trips through a server callback before it lands in
    # localStorage; wait for that write so reload() below doesn't race it.
    page.wait_for_function(
        "() => { try { return JSON.parse(localStorage.getItem('store-scenario'))"
        ".portfolio.initial_portfolio === 999999; } catch(e) { return false; } }"
    )
    # Reloading mid-flight of an unrelated pending callback makes dash-renderer
    # log a "callback failed" console error (cancelled request) — drain the
    # network before navigating so that's not this test's problem.
    page.wait_for_load_state("networkidle")

    page.reload()
    journeys.goto_view(page, "plan")
    journeys.open_tab(page, "Portfolio")
    expect(page.locator("#inp-initial-portfolio")).to_have_value("999999")

    # --- corrupt store-scenario -> falls back to DEFAULT_SCENARIO, no blank UI ---
    page.evaluate("() => localStorage.setItem('store-scenario', 'not json {{{')")
    page.wait_for_load_state("networkidle")
    page.reload()
    journeys.goto_view(page, "plan")
    journeys.open_tab(page, "Portfolio")
    expect(page.locator("#inp-initial-portfolio")).to_have_value("0")
    expect(page.locator("#header-scenario-name")).to_have_text("untitled")


@pytest.mark.ux("UX-TOAST-01")
def test_toast_position_and_dismiss(loaded_page, no_new_scenario_files):
    """Toast appears bottom-right (clear of app bar/form) and auto-dismisses."""
    journeys.save_scenario(loaded_page, "uxtest-toast")

    toast = loaded_page.locator("#toast")
    expect(toast).to_be_visible()
    bbox = toast.bounding_box()
    vw = loaded_page.evaluate("() => window.innerWidth")
    vh = loaded_page.evaluate("() => window.innerHeight")
    # Bottom-right: within 120px of both the right and bottom viewport edges.
    assert bbox["x"] + bbox["width"] >= vw - 120
    assert bbox["y"] + bbox["height"] >= vh - 120

    # Auto-dismisses (dbc.Toast duration=4000ms) within small tolerance.
    expect(toast).to_be_hidden(timeout=6_000)
