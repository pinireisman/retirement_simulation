"""Run-flow and result surface journeys.

Test simulation run, result display, historic toggle, and re-run behaviors.
"""
from __future__ import annotations

import pytest
from playwright.sync_api import expect

from tests.ui import journeys

pytestmark = pytest.mark.ui


@pytest.mark.ux("UX-RUN-01")
def test_run_populates_results(loaded_page):
    """Run produces the full result surface."""
    journeys.set_fast_run(loaded_page)
    journeys.run_simulation(loaded_page)

    # Hero shows a percentage and a verdict sentence
    expect(loaded_page.locator("#hero-numeral")).to_contain_text("%")

    # All four stat tiles show non-"—" values
    stat_values = loaded_page.locator("#div-summary h3")
    expect(stat_values).to_have_count(4)
    for i in range(4):
        expect(stat_values.nth(i)).not_to_contain_text("—")

    # Chart cards visible (placeholder gone)
    expect(loaded_page.locator("#div-chart-cards")).not_to_have_class("d-none")

    # Cash-flow/portfolio/draw graphs have trace_count > 0
    assert journeys.trace_count(loaded_page, "#graph-results") > 0
    assert journeys.trace_count(loaded_page, "#graph-portfolio") > 0
    assert journeys.trace_count(loaded_page, "#graph-draw") > 0


@pytest.mark.ux("UX-RUN-02")
def test_historic_cards(loaded_page):
    """Historic toggle adds historic cards."""
    journeys.set_fast_run(loaded_page)
    # Navigate to dashboard where the historic toggle is visible
    journeys.goto_view(loaded_page, "dashboard")
    # Enable historic scenarios
    journeys.enable_historic(loaded_page)

    journeys.run_simulation(loaded_page)

    # Historic cards container should be visible with at least one card
    expect(loaded_page.locator("#div-historic-cards")).to_be_visible()

    # At least one chart card in historic section should have traces
    historic_graphs = loaded_page.locator("#div-historic-cards [id*='graph-historic']")
    assert historic_graphs.count() >= 1, "Expected at least one historic graph"
    # Verify at least one has traces
    found_traces = False
    for i in range(historic_graphs.count()):
        graph_id = historic_graphs.nth(i).get_attribute("id")
        if graph_id and journeys.trace_count(loaded_page, f"#{graph_id}") > 0:
            found_traces = True
            break
    assert found_traces, "At least one historic graph should have traces"


@pytest.mark.ux("UX-RUN-03")
def test_rerun_refreshes(loaded_page, example_scenario):
    """Re-run refreshes and re-computes results."""
    journeys.set_fast_run(loaded_page)
    journeys.run_simulation(loaded_page)

    # Verify first result shows a percentage
    expect(loaded_page.locator("#hero-numeral")).to_contain_text("%")

    # Edit spending to trigger a change
    journeys.goto_view(loaded_page, "plan")
    journeys.open_tab(loaded_page, "Spending")
    journeys.set_table_cell(loaded_page, "spending", row=0, column="Amount Monthly", value=500000)

    # Re-run: should re-compute and display results again
    journeys.run_simulation(loaded_page)

    # Verify result is displayed after re-run
    expect(loaded_page.locator("#hero-numeral")).to_contain_text("%")


@pytest.mark.ux("UX-RUN-04")
def test_run_speed_bound(loaded_page):
    """Run completes within interactive bounds (< 30s at 500 paths)."""
    journeys.set_fast_run(loaded_page, n_paths=500)
    # run_simulation has timeout=60_000 by default; use 30s for this test
    journeys.run_simulation(loaded_page, timeout=30_000)
    # If we get here without timeout, the test passes


@pytest.mark.ux("UX-RUN-05")
def test_cash_flow_plot_fits_card(loaded_page):
    """The cash-flow panel grows with its legend; the rendered plot must stay
    inside its own card instead of bleeding over the Annual-draw row below."""
    journeys.set_fast_run(loaded_page)
    journeys.run_simulation(loaded_page)
    loaded_page.wait_for_selector("#graph-results .main-svg", timeout=30_000)
    loaded_page.wait_for_timeout(1000)  # height-sync runs a frame after render

    def box(sel):
        return loaded_page.locator(sel).first.bounding_box()

    cf_card = box("#graph-results >> xpath=ancestor::*[contains(@class,'chart-card')]")
    cf_plot = box("#graph-results .main-svg")
    draw_card = box("#graph-draw >> xpath=ancestor::*[contains(@class,'chart-card')]")

    assert cf_plot["height"] > 100, f"cash-flow plot collapsed: {cf_plot}"
    plot_bottom = cf_plot["y"] + cf_plot["height"]
    card_bottom = cf_card["y"] + cf_card["height"]
    assert plot_bottom <= card_bottom + 2, (
        f"cash-flow plot overflows its card: plot bottom {plot_bottom}, "
        f"card bottom {card_bottom}")
    assert not (cf_card["x"] < draw_card["x"] + draw_card["width"]
                and draw_card["x"] < cf_card["x"] + cf_card["width"]
                and cf_card["y"] < draw_card["y"] + draw_card["height"]
                and draw_card["y"] < card_bottom), (
        f"cards overlap: cash-flow {cf_card}, annual-draw {draw_card}")
