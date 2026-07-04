"""Plan-view editing journeys: tables, dropdowns, validation.

EXEMPLAR FILE — new UI tests follow this pattern:
- @pytest.mark.ux("<id>") binds the test to its UX_TEST_PLAN.md case.
- All DOM interaction goes through journeys.py helpers; assert user-visible
  outcomes via journeys return values or expect() on component ids.
"""
from __future__ import annotations

import pytest
from playwright.sync_api import expect

from tests.ui import journeys

pytestmark = pytest.mark.ui


@pytest.mark.ux("UX-DD-01")
def test_spending_category_dropdown_selects(loaded_page):
    """Regression (2026-07-04): the category dropdown menu was clipped
    invisible by the table container's overflow — 'not really a drop down'."""
    journeys.goto_view(loaded_page, "plan")
    journeys.open_tab(loaded_page, "Spending")
    journeys.select_table_category(loaded_page, "spending", row=0, value="lifestyle")
    # and back, so the example scenario is left semantically intact
    journeys.select_table_category(loaded_page, "spending", row=0, value="strict")


@pytest.mark.ux("UX-DD-02")
def test_lumps_category_dropdown_selects(loaded_page):
    journeys.goto_view(loaded_page, "plan")
    journeys.open_tab(loaded_page, "Lumps")
    journeys.select_table_category(loaded_page, "lumps", row=0, value="gifts")


@pytest.mark.ux("UX-FORM-04")
def test_table_cell_edit_persists(loaded_page, example_scenario):
    """Editing an amount updates the cell and survives a tab round-trip."""
    journeys.goto_view(loaded_page, "plan")
    journeys.open_tab(loaded_page, "Income")
    journeys.set_table_cell(loaded_page, "income", row=0, column="Amount Monthly", value=12345)
    journeys.open_tab(loaded_page, "Spending")
    journeys.open_tab(loaded_page, "Income")
    expect(journeys.table_cell(loaded_page, "income", row=0, column="Amount Monthly")
           ).to_contain_text("12345")


@pytest.mark.ux("UX-FORM-05")
def test_add_band_appends_row(loaded_page, example_scenario):
    journeys.goto_view(loaded_page, "plan")
    journeys.open_tab(loaded_page, "Spending")
    before = journeys.row_count(loaded_page, "spending")
    assert before == len(example_scenario["spending_bands"])
    journeys.add_row(loaded_page, "spending")
    journeys.expect_row_count(loaded_page, "spending", before + 1)
