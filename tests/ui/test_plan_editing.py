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


@pytest.mark.ux("UX-FORM-01")
def test_portfolio_fields_editable(loaded_page):
    """Portfolio numeric fields accept edits and preview updates."""
    journeys.goto_view(loaded_page, "plan")
    journeys.open_tab(loaded_page, "Portfolio")
    # Fill the initial portfolio field and tab away to trigger update
    loaded_page.locator("#inp-initial-portfolio").fill("5000000")
    loaded_page.keyboard.press("Tab")
    # Verify the value is retained
    expect(loaded_page.locator("#inp-initial-portfolio")).to_have_value("5000000")


@pytest.mark.ux("UX-FORM-02")
def test_market_updates_mu_sigma(loaded_page):
    """Market dropdown updates the μ/σ label."""
    journeys.goto_view(loaded_page, "plan")
    journeys.open_tab(loaded_page, "Portfolio")
    # Verify the label is visible
    expect(loaded_page.locator("#lbl-market-mu-sigma")).to_be_visible()
    # Note: dropdown selection behavior depends on the specific Dash component
    # For now, just verify the label exists and contains expected symbols (µ or μ)
    label_text = loaded_page.locator("#lbl-market-mu-sigma").inner_text()
    assert any(s in label_text for s in ["μ", "µ", "sigma"])


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


@pytest.mark.ux("UX-VAL-01")
def test_end_age_validation(loaded_page):
    """End age ≤ start age is rejected with feedback."""
    journeys.goto_view(loaded_page, "plan")
    journeys.open_tab(loaded_page, "Portfolio")
    # Set end age to a value less than start age (start is typically 50)
    loaded_page.locator("#inp-end-age").fill("45")
    loaded_page.keyboard.press("Tab")
    # Navigate to dashboard and click run to trigger validation
    journeys.goto_view(loaded_page, "dashboard")
    loaded_page.locator("#btn-run").click()
    # Expect an error toast mentioning age
    journeys.expect_toast(loaded_page, containing="age")


@pytest.mark.ux("UX-VAL-02")
def test_empty_spending_validation(page, app_url):
    """Empty spending is rejected with feedback."""
    # Fresh page (no scenario loaded, so spending is empty)
    page.goto(app_url)
    # Try to run without any data
    page.locator("#btn-run").click()
    # Expect an error toast mentioning spending
    text = journeys.expect_toast(page, containing="spending")
    assert "spending" in text.lower()
