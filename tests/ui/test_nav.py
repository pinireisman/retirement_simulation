"""Navigation and empty-state tests — UX-NAV-01 through UX-EMPTY-02.

All DOM interaction goes through journeys.py helpers; assert user-visible
outcomes via journeys return values or expect() on component ids.
"""
from __future__ import annotations

import pytest
from playwright.sync_api import expect

from tests.ui import journeys

pytestmark = pytest.mark.ui


@pytest.mark.ux("UX-NAV-01")
def test_opens_on_dashboard(page, app_url):
    """App opens on the Dashboard with em-dash placeholder and prompt."""
    page.goto(app_url)
    # Expected: dashboard view visible, Plan view not
    expect(page.locator("#div-view-dashboard")).to_be_visible()
    expect(page.locator("#div-view-plan")).not_to_be_visible()
    # Hero shows em-dash placeholder numeral and "Run a simulation…" prompt
    expect(page.locator("#hero-numeral")).to_contain_text("—")


@pytest.mark.ux("UX-NAV-02")
def test_view_toggle(page, app_url):
    """View toggle switches Dashboard ↔ Plan."""
    page.goto(app_url)
    # Toggle to plan, then back to dashboard
    journeys.goto_view(page, "plan")
    journeys.goto_view(page, "dashboard")
    # Expected: exactly one view visible after each toggle
    expect(page.locator("#div-view-dashboard")).to_be_visible()
    expect(page.locator("#div-view-plan")).not_to_be_visible()
    # Toggle back to plan
    journeys.goto_view(page, "plan")
    expect(page.locator("#div-view-dashboard")).not_to_be_visible()
    expect(page.locator("#div-view-plan")).to_be_visible()


@pytest.mark.ux("UX-NAV-03")
def test_all_tabs_open(loaded_page):
    """All five Plan tabs open and render content."""
    journeys.goto_view(loaded_page, "plan")
    # Open each tab; open_tab asserts visibility of its content
    journeys.open_tab(loaded_page, "Portfolio")
    journeys.open_tab(loaded_page, "Spending")
    journeys.open_tab(loaded_page, "Income")
    journeys.open_tab(loaded_page, "Lumps")
    journeys.open_tab(loaded_page, "Properties")


@pytest.mark.ux("UX-NAV-04")
def test_edits_survive_view_toggle(loaded_page):
    """Edits survive a view round-trip."""
    journeys.goto_view(loaded_page, "plan")
    journeys.open_tab(loaded_page, "Income")
    # Edit a cell value
    journeys.set_table_cell(loaded_page, "income", row=0, column="Amount Monthly", value=4242)
    # Round-trip through dashboard
    journeys.goto_view(loaded_page, "dashboard")
    journeys.goto_view(loaded_page, "plan")
    journeys.open_tab(loaded_page, "Income")
    # Expect the cell still shows the edited value
    expect(journeys.table_cell(loaded_page, "income", row=0, column="Amount Monthly")
           ).to_contain_text("4242")


@pytest.mark.ux("UX-EMPTY-01")
def test_empty_state_no_ghost_charts(page, app_url):
    """No ghost charts before the first run; placeholder visible."""
    page.goto(app_url)
    # Expected: chart cards not visible; placeholder text is visible
    expect(page.locator("#div-chart-cards")).to_have_class("d-none")
    expect(page.locator("#div-chart-placeholder")).to_be_visible()
    expect(page.locator("#div-chart-placeholder")).to_contain_text("Charts appear here once you run a simulation")


@pytest.mark.ux("UX-EMPTY-02")
def test_empty_stat_tiles(page, app_url):
    """Stat tiles show placeholders pre-run."""
    page.goto(app_url)
    # Expected: stat tiles render with "—" values
    stat_value_tiles = page.locator("#div-summary h3")
    expect(stat_value_tiles).to_have_count(4)
    # All tiles should show "—"
    for i in range(4):
        expect(stat_value_tiles.nth(i)).to_contain_text("—")
    # Check that "Spending guardrail" label is present
    expect(page.locator("#div-summary")).to_contain_text("SPENDING GUARDRAIL")
