"""Cursor/caret behavior for editable dash_table cells.

Mouse-click-to-reposition inside an already-editing cell is a known
dash_table limitation (its internal redux-driven re-render resets the
selection on a later commit than any DOM-event-time hook can beat — verified
empirically, not fixed by this suite). Arrow-key navigation within an
editing cell is the supported workaround and is what's tested here.
"""
from __future__ import annotations

import pytest

from tests.ui import journeys

pytestmark = pytest.mark.ui


def test_caret_visible_while_unfocused_class_present(loaded_page):
    """The CSS fix: dash_table's own 'unfocused' class sets caret-color:
    transparent, but it applies that class even to inputs that currently
    have real DOM focus. Our override must win regardless."""
    journeys.goto_view(loaded_page, "plan")
    journeys.open_tab(loaded_page, "Income")

    cell = journeys.table_cell(loaded_page, "income", row=0, column="Amount Monthly")
    cell.dblclick()
    input_locator = cell.locator("input")

    input_locator.evaluate("el => el.classList.add('unfocused')")
    caret_color = input_locator.evaluate("el => getComputedStyle(el).caretColor")
    assert caret_color != "transparent", f"Expected caret to be visible, got: {caret_color}"


def test_arrow_key_repositions_cursor_and_backspace_works(loaded_page):
    """Keyboard navigation within an editing cell works reliably: arrow
    keys collapse the selection to a specific offset, and backspace at that
    offset removes exactly one character."""
    journeys.goto_view(loaded_page, "plan")
    journeys.open_tab(loaded_page, "Income")

    cell = journeys.table_cell(loaded_page, "income", row=0, column="Amount Monthly")
    cell.dblclick()
    input_locator = cell.locator("input")

    original_value = input_locator.input_value()
    assert original_value == "15000"

    # Double-click selects all; ArrowLeft collapses selection to position 0.
    loaded_page.keyboard.press("ArrowLeft")
    selection_range = input_locator.evaluate("el => [el.selectionStart, el.selectionEnd]")
    assert selection_range == [0, 0], f"Expected cursor collapsed at 0, got: {selection_range}"

    # Backspace at position 0 must be a no-op.
    loaded_page.keyboard.press("Backspace")
    assert input_locator.input_value() == original_value

    # ArrowRight once moves the cursor to position 1; backspace there removes
    # exactly the first character.
    loaded_page.keyboard.press("ArrowRight")
    selection_range = input_locator.evaluate("el => [el.selectionStart, el.selectionEnd]")
    assert selection_range == [1, 1], f"Expected cursor collapsed at 1, got: {selection_range}"

    loaded_page.keyboard.press("Backspace")
    new_value = input_locator.input_value()
    assert new_value == original_value[1:], "Expected only the first character removed"
