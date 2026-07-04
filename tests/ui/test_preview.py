"""Cash-flow preview journeys: live update and zoom persistence.

EXEMPLAR FILE — chart-state assertions go through journeys helpers
(read_preview_zoom / zoom_preview / trace_count), never raw Plotly access.
"""
from __future__ import annotations

import pytest

from tests.ui import journeys

pytestmark = pytest.mark.ui


@pytest.mark.ux("UX-PREV-01")
def test_preview_renders_scenario_series(loaded_page, example_scenario):
    """After upload the preview shows one trace per band/lump + net line."""
    journeys.goto_view(loaded_page, "plan")
    expected_min = (len(example_scenario["spending_bands"])
                    + len(example_scenario["income_bands"])
                    + len(example_scenario["lumps"]))
    assert journeys.trace_count(loaded_page, journeys.PREVIEW) >= expected_min


@pytest.mark.ux("UX-PREV-02")
def test_preview_updates_on_edit(loaded_page):
    """Preview updates live on table edit."""
    journeys.goto_view(loaded_page, "plan")
    # Capture trace count before edit
    initial_count = journeys.trace_count(loaded_page, journeys.PREVIEW)
    # Add a lumpy
    journeys.open_tab(loaded_page, "Lumps")
    journeys.add_row(loaded_page, "lumps")
    journeys.set_table_cell(loaded_page, "lumps", row=0, column="Age", value=60)
    journeys.set_table_cell(loaded_page, "lumps", row=0, column="Amount", value=100000)
    # Wait for preview to re-render
    loaded_page.wait_for_timeout(700)
    # Trace count should increase by 1 (new lump adds a marker)
    assert journeys.trace_count(loaded_page, journeys.PREVIEW) > initial_count


@pytest.mark.ux("UX-PREV-03")
def test_zoom_survives_table_edit(loaded_page):
    """Regression (2026-07-04): every Income cell edit reset the preview zoom
    because the rebuilt figure had no uirevision."""
    journeys.goto_view(loaded_page, "plan")
    journeys.zoom_preview(loaded_page, 60, 70)
    journeys.open_tab(loaded_page, "Income")
    journeys.set_table_cell(loaded_page, "income", row=0, column="Amount Monthly", value=9999)
    # wait for the preview to re-render with the edit, then check the zoom
    loaded_page.wait_for_timeout(700)
    assert journeys.read_preview_zoom(loaded_page) == [60, 70]


@pytest.mark.ux("UX-PREV-04")
def test_dashboard_chart_zoom_survives_rerun(loaded_page):
    """Dashboard chart zoom survives re-run."""
    journeys.set_fast_run(loaded_page)
    journeys.run_simulation(loaded_page)
    # Zoom the results chart
    journeys.zoom_preview(loaded_page, 60, 70, graph="#graph-results")
    # Edit and re-run
    journeys.goto_view(loaded_page, "plan")
    journeys.open_tab(loaded_page, "Spending")
    journeys.set_table_cell(loaded_page, "spending", row=0, column="Amount Monthly", value=30000)
    journeys.run_simulation(loaded_page)
    # Verify zoom is preserved
    assert journeys.read_preview_zoom(loaded_page, graph="#graph-results") == [60, 70]
