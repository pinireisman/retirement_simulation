"""Playground mode journeys — enabling, event entry, result reporting."""
from __future__ import annotations

import pytest
from playwright.sync_api import expect

from tests.ui import journeys

pytestmark = pytest.mark.ui


@pytest.mark.ux("UX-PG-01")
def test_banner(loaded_page):
    """Enabling playground shows the banner."""
    journeys.enable_playground(loaded_page)
    # The helper asserts the banner is visible
    expect(loaded_page.locator("#banner-playground")).to_be_visible()


@pytest.mark.ux("UX-PG-02")
def test_add_event_chip_and_marker(loaded_page):
    """Chart click opens the event modal and adds a chip."""
    # Enable playground first
    journeys.enable_playground(loaded_page)

    # Capture initial trace count
    initial_count = journeys.trace_count(loaded_page, journeys.PREVIEW)

    # Add an event
    journeys.add_playground_event(loaded_page, -200_000, "test gift")

    # Check that preview gained a marker trace
    final_count = journeys.trace_count(loaded_page, journeys.PREVIEW)
    assert final_count > initial_count, "Preview should gain a marker trace after adding event"

    # Check that a chip appears in the chips container
    chips_container = loaded_page.locator("#div-playground-chips")
    expect(chips_container).to_contain_text("test gift")


@pytest.mark.ux("UX-PG-03")
def test_run_with_events_badge(loaded_page):
    """Run with playground events reports them."""
    # Enable playground and add an event
    journeys.enable_playground(loaded_page)
    journeys.add_playground_event(loaded_page, -200_000, "test gift")

    # Set fast run and run with playground events
    journeys.set_fast_run(loaded_page, n_paths=500)
    journeys.run_simulation(loaded_page, playground=True)

    # Check that result badges are visible (event was included)
    badges_container = loaded_page.locator("#div-result-badges")
    expect(badges_container).to_be_visible()
    # Verify we're on the dashboard with results
    expect(loaded_page.locator("#div-view-dashboard")).to_be_visible()


@pytest.mark.ux("UX-PG-04")
def test_clear_all(loaded_page):
    """Clear-all removes chips and markers."""
    journeys.enable_playground(loaded_page)
    before = journeys.trace_count(loaded_page, journeys.PREVIEW)

    journeys.add_playground_event(loaded_page, -200_000, "test gift")
    loaded_page.wait_for_timeout(500)
    with_event = journeys.trace_count(loaded_page, journeys.PREVIEW)
    assert with_event > before, "Adding an event should add a marker trace"

    journeys.clear_playground_events(loaded_page)
    loaded_page.wait_for_timeout(700)
    # Preview returns to its pre-event trace count.
    assert journeys.trace_count(loaded_page, journeys.PREVIEW) == before
