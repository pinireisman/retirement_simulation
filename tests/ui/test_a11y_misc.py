"""Accessibility and responsiveness tests.

These tests cover keyboard focus, reduced motion, and viewport responsiveness
which are accessibility-related UI behaviors.
"""
from __future__ import annotations

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.ui


@pytest.mark.ux("UX-A11Y-01")
def test_focus_visible(page, app_url):
    """Keyboard focus is visible — press Tab up to 8 times and check outlineStyle."""
    page.goto(app_url)

    # Press Tab up to 8 times and check each active element's outlineStyle
    for i in range(8):
        page.keyboard.press("Tab")
        # Get the active element and its computed outlineStyle
        outline_style = page.evaluate("""
            () => {
                const activeElement = document.activeElement;
                if (!activeElement) return null;
                const computedStyle = window.getComputedStyle(activeElement);
                return computedStyle.outlineStyle;
            }
        """)

        # Check if any focus stop shows a visible outline (solid)
        if outline_style == "solid":
            # At least one focus stop should have a solid outline
            return

    # If we get here without finding a solid outline, fail the test
    pytest.fail("No focus stop found with visible outline (solid)")


@pytest.mark.ux("UX-A11Y-02")
def test_reduced_motion(page, app_url):
    """Reduced motion is honored — no visible ripple animation.

    The ripple CSS sets `animation: none; display: none` under
    prefers-reduced-motion, so a ripple span may still be appended to the DOM
    but must never visibly render.
    """
    page.emulate_media(reduced_motion="reduce")
    page.goto(app_url)

    # Click a ripple-bearing button.
    page.locator("#btn-run").click()

    # No ripple should be visibly rendered.
    expect(page.locator(".ripple:visible")).to_have_count(0)


@pytest.mark.ux("UX-RESP-01")
def test_no_hscroll_1280(page, app_url):
    """No horizontal scroll at 1280×800 viewport."""
    # Set viewport size before navigating
    page.set_viewport_size({"width": 1280, "height": 800})
    page.goto(app_url)

    # After loading and potentially running, check scroll behavior
    page.locator("#btn-run").click()

    # Check that horizontal scroll is not present
    scroll_width = page.evaluate("() => document.documentElement.scrollWidth")
    inner_width = page.evaluate("() => window.innerWidth")

    assert scroll_width <= inner_width, (
        f"Horizontal scroll detected: scrollWidth ({scroll_width}) > innerWidth ({inner_width})"
    )


@pytest.mark.ux("UX-RESP-02")
def test_no_hscroll_tablet(page, app_url):
    """No horizontal scroll at 834×1112 (tablet portrait) viewport."""
    # Set viewport size before navigating
    page.set_viewport_size({"width": 834, "height": 1112})
    page.goto(app_url)

    # After loading and potentially running, check scroll behavior
    page.locator("#btn-run").click()

    # Check that horizontal scroll is not present
    scroll_width = page.evaluate("() => document.documentElement.scrollWidth")
    inner_width = page.evaluate("() => window.innerWidth")

    assert scroll_width <= inner_width, (
        f"Horizontal scroll detected: scrollWidth ({scroll_width}) > innerWidth ({inner_width})"
    )
