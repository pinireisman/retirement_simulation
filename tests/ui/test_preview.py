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
