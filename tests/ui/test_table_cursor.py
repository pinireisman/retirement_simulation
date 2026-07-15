"""Cursor/caret behavior for editable dash_table cells.

assets/click-to-edit.js makes cells behave like normal text boxes: a single
click enters real edit mode with the caret at the clicked character, clicks
and drags inside an editing cell keep their native caret/selection (dash's
global handlers are isolated away), Ctrl+C/V act on the in-cell selection,
and clicking away commits instead of discarding.
"""
from __future__ import annotations

import time

import pytest
from playwright.sync_api import expect

from tests.ui import journeys


def _click_text_end(page, cell):
    """Click just inside the right edge of a cell — over the end of its
    right-aligned text, so the promoted caret lands at the end."""
    box = cell.bounding_box()
    page.mouse.click(box["x"] + box["width"] - 8, box["y"] + box["height"] / 2)


def _wait_edit_ready(page):
    """Wait until click-to-edit promotion finished: the cell input is in real
    edit mode with a collapsed caret (promotion retries + caret placement are
    async, and take longer when dash is busy with a prior commit)."""
    page.wait_for_function(
        "() => { const el = document.activeElement;"
        " return !!el && el.tagName === 'INPUT'"
        " && el.classList.contains('dash-cell-value')"
        " && el.classList.contains('focused')"
        " && !el.classList.contains('unfocused')"
        " && el.selectionStart === el.selectionEnd; }")

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


def test_single_click_enters_real_edit_mode(loaded_page):
    """Regression (assets/click-to-edit.js): a single click used to leave the
    cell in dash_table's half-edit state — value invisibly select-all'd, so
    Backspace wiped the whole cell and arrow keys jumped to neighboring cells
    instead of moving the caret. A click over the end of the text must land
    in real edit mode with the caret at the end."""
    journeys.goto_view(loaded_page, "plan")
    journeys.open_tab(loaded_page, "Income")

    cell = journeys.table_cell(loaded_page, "income", row=0, column="Amount Monthly")
    _click_text_end(loaded_page, cell)
    input_locator = cell.locator("input")
    _wait_edit_ready(loaded_page)

    value = input_locator.input_value()
    assert value == "15000"
    sel = input_locator.evaluate("el => [el.selectionStart, el.selectionEnd]")
    assert sel == [len(value), len(value)], f"Expected caret at end, got: {sel}"

    # Backspace removes exactly one character; arrows move within the text.
    loaded_page.keyboard.press("Backspace")
    assert input_locator.input_value() == value[:-1]
    loaded_page.keyboard.press("ArrowLeft")
    loaded_page.keyboard.type("9")
    assert input_locator.input_value() == "15090", \
        "Expected ArrowLeft to move the caret within the text, not leave the cell"
    loaded_page.keyboard.press("Escape")


def test_click_positions_caret_at_clicked_char(loaded_page):
    """Regression (assets/click-to-edit.js): promoting a click used to put the
    caret at the end regardless of where the user clicked. The caret must land
    at the clicked character (captured via caretRangeFromPoint at mousedown,
    before dash swaps the display text for an input)."""
    journeys.goto_view(loaded_page, "plan")
    journeys.open_tab(loaded_page, "Income")
    journeys.set_table_cell(loaded_page, "income", row=0, column="Label",
                            value="abcdefghij")
    loaded_page.wait_for_timeout(600)

    cell = journeys.table_cell(loaded_page, "income", row=0, column="Label")
    box = cell.bounding_box()
    # Text is right-aligned; 30px left of the right edge is mid-text.
    loaded_page.mouse.click(box["x"] + box["width"] - 30,
                            box["y"] + box["height"] / 2)
    _wait_edit_ready(loaded_page)

    sel = cell.locator("input").evaluate("el => [el.selectionStart, el.selectionEnd]")
    assert sel[0] == sel[1], f"Expected collapsed caret, got selection {sel}"
    assert 0 < sel[0] < 10, f"Expected caret mid-text, got position {sel[0]}"
    loaded_page.keyboard.press("Escape")


def test_drag_select_copy_paste(loaded_page):
    """Regression (assets/click-to-edit.js): dash resets any mouse-made text
    selection to select-all and hijacks Ctrl+C/V for cell-range copy even
    while editing. Drag-selecting part of the text must survive, Ctrl+C must
    copy exactly the selection, and Ctrl+V must insert at the caret."""
    loaded_page.context.grant_permissions(["clipboard-read", "clipboard-write"])
    journeys.goto_view(loaded_page, "plan")
    journeys.open_tab(loaded_page, "Income")
    journeys.set_table_cell(loaded_page, "income", row=0, column="Label",
                            value="abcdefghij")
    loaded_page.wait_for_timeout(600)

    cell = journeys.table_cell(loaded_page, "income", row=0, column="Label")
    cell.click()
    _wait_edit_ready(loaded_page)
    inp = cell.locator("input")
    box = inp.bounding_box()
    y = box["y"] + box["height"] / 2
    right = box["x"] + box["width"]

    # Drag over part of the (right-aligned) text.
    loaded_page.mouse.move(right - 60, y)
    loaded_page.mouse.down()
    loaded_page.mouse.move(right - 10, y, steps=5)
    loaded_page.mouse.up()
    loaded_page.wait_for_timeout(400)
    start, end, value = inp.evaluate(
        "el => [el.selectionStart, el.selectionEnd, el.value]")
    assert 0 < end - start < len(value), \
        f"Expected a partial selection to survive, got [{start}, {end}]"

    loaded_page.keyboard.press("ControlOrMeta+c")
    loaded_page.wait_for_timeout(200)
    clip = loaded_page.evaluate("() => navigator.clipboard.readText()")
    assert clip == value[start:end], \
        f"Ctrl+C copied {clip!r}, expected selection {value[start:end]!r}"

    # Paste inserts at the caret instead of replacing the cell.
    loaded_page.evaluate("() => navigator.clipboard.writeText('XY')")
    inp.evaluate("el => el.setSelectionRange(4, 4)")
    loaded_page.keyboard.press("ControlOrMeta+v")
    assert inp.input_value() == "abcdXYefghij"
    loaded_page.keyboard.press("Escape")


def test_click_away_commits_edit(loaded_page):
    """Regression (assets/click-to-edit.js): dash_table discards a real-edit-
    mode value on click-away (only Enter/Tab commit natively). Clicking away
    must commit the typed value, spreadsheet-style — both onto another cell
    and outside the table entirely."""
    journeys.goto_view(loaded_page, "plan")
    journeys.open_tab(loaded_page, "Income")

    cell = journeys.table_cell(loaded_page, "income", row=0, column="Amount Monthly")

    # Edit, then click another cell in the same table.
    _click_text_end(loaded_page, cell)
    _wait_edit_ready(loaded_page)
    loaded_page.keyboard.type("9")
    journeys.table_cell(loaded_page, "income", row=0, column="Label").click()
    expect(cell).to_have_text("150009")

    # Edit again, then click outside the table entirely.
    _click_text_end(loaded_page, cell)
    _wait_edit_ready(loaded_page)
    loaded_page.keyboard.press("Backspace")
    loaded_page.locator("#header-scenario-name").click()
    expect(cell).to_have_text("15000")


def test_edit_survives_echo_of_previous_commit(loaded_page):
    """Canary: committing a cell round-trips through store-scenario back to
    hydrate_tabs. That echo must never disturb an edit already in progress
    in another cell (hydrate_tabs skips tables whose content is unchanged;
    dash_table itself must also tolerate prop updates mid-edit)."""
    journeys.goto_view(loaded_page, "plan")
    journeys.open_tab(loaded_page, "Income")

    # Simulate a busy server (preview sims etc.): delay every callback
    # response so the commit's echo lands well after the next edit starts.
    def slow(route):
        time.sleep(0.5)
        route.continue_()
    loaded_page.route("**/_dash-update-component", slow)

    # Commit a label edit (its store echo is now in flight), then
    # immediately start editing another cell — the user fixing a typo.
    journeys.set_table_cell(loaded_page, "income", row=0, column="Label",
                            value="salary edited")
    cell = journeys.table_cell(loaded_page, "income", row=0, column="Amount Monthly")
    cell.dblclick()
    input_locator = cell.locator("input")
    original_value = input_locator.input_value()

    # Wait past the delayed collect_edits -> hydrate_tabs chain landing.
    loaded_page.wait_for_timeout(3000)

    # The edit survived: arrows still reposition and backspace removes
    # exactly one character.
    loaded_page.keyboard.press("ArrowLeft")
    loaded_page.keyboard.press("ArrowRight")
    loaded_page.keyboard.press("Backspace")
    assert input_locator.input_value() == original_value[1:], (
        "in-progress cell edit was reset by the store echo of the previous commit")
