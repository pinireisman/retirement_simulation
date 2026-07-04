"""Scenario I/O journeys: upload, save, load, overwrite, toast positioning."""
from __future__ import annotations

from pathlib import Path

import pytest
from playwright.sync_api import expect

from tests.ui import journeys
from tests.ui.conftest import EXAMPLE_XLSX, REPO

pytestmark = pytest.mark.ui


@pytest.fixture
def clean_uxtest():
    """Remove test-created scenario files before and after, so save/overwrite
    tests start from a known state and leave nothing behind."""
    def _wipe():
        for p in (REPO / "scenarios").glob("uxtest*.xlsx"):
            p.unlink()
    _wipe()
    yield
    _wipe()


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

    # The app bar reflects the loaded scenario's name.
    expect(page.locator("#header-scenario-name")).to_be_visible()


@pytest.mark.ux("UX-IO-02")
def test_save_creates_scenario(loaded_page, clean_uxtest):
    """Save creates a loadable scenario that appears in the Load dropdown."""
    text = journeys.save_scenario(loaded_page, "uxtest-roundtrip")
    assert "saved" in text.lower()
    assert journeys.scenario_in_load_list(loaded_page, "uxtest-roundtrip")


@pytest.mark.ux("UX-IO-03")
def test_overwrite_flow(loaded_page, clean_uxtest):
    """Saving an existing name is rejected until the overwrite box is checked.

    The app shows the overwrite checkbox based on the *loaded* scenario's name
    (the modal prefills it), so we save then load the scenario — that makes the
    app-bar name match the file on disk and the conflict path reachable.
    """
    assert "saved" in journeys.save_scenario(loaded_page, "uxtest-roundtrip").lower()
    journeys.load_scenario(loaded_page, "uxtest-roundtrip")
    # Same name without overwrite → rejected with feedback about the conflict.
    rejected = journeys.save_scenario(loaded_page, "uxtest-roundtrip", overwrite=False).lower()
    assert "exists" in rejected or "overwrite" in rejected
    # With the overwrite box checked it succeeds.
    assert "saved" in journeys.save_scenario(loaded_page, "uxtest-roundtrip", overwrite=True).lower()


@pytest.mark.ux("UX-IO-04")
def test_reload_and_load(loaded_page, clean_uxtest, example_scenario):
    """Hard refresh + Load restores the plan."""
    assert "saved" in journeys.save_scenario(loaded_page, "uxtest-roundtrip", overwrite=True).lower()

    loaded_page.reload()
    journeys.load_scenario(loaded_page, "uxtest-roundtrip")
    journeys.goto_view(loaded_page, "plan")

    assert journeys.row_count(loaded_page, "spending") == len(example_scenario["spending_bands"])
    assert journeys.row_count(loaded_page, "income") == len(example_scenario["income_bands"])
    assert journeys.row_count(loaded_page, "lumps") == len(example_scenario["lumps"])
    assert journeys.row_count(loaded_page, "properties") == len(example_scenario["properties"])


@pytest.mark.ux("UX-TOAST-01")
def test_toast_position_and_dismiss(loaded_page, clean_uxtest):
    """Toast appears bottom-right (clear of the app bar/form) and auto-dismisses."""
    journeys.save_scenario(loaded_page, "uxtest-toast", overwrite=True)

    toast = loaded_page.locator("#toast")
    expect(toast).to_be_visible()
    bbox = toast.bounding_box()
    vw = loaded_page.evaluate("() => window.innerWidth")
    vh = loaded_page.evaluate("() => window.innerHeight")
    # Bottom-right: within 120px of both the right and bottom viewport edges.
    assert bbox["x"] + bbox["width"] >= vw - 120
    assert bbox["y"] + bbox["height"] >= vh - 120

    # Auto-dismisses (dbc.Toast duration=4000ms) within a small tolerance.
    expect(toast).to_be_hidden(timeout=6_000)
