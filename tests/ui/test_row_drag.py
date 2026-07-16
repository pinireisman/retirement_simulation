"""Row reordering via the ⠿ handle column (assets/row-drag.js +
move_table_row callback).

Press the handle, drag over another row, release: a ghost row follows the
cursor, and on drop the table's `data` prop is reordered, which persists
through collect_edits like any other table edit. Clicking the handle without
dragging must not reorder or put a cell into edit mode.
"""
from __future__ import annotations

import pytest

from tests.ui import journeys

pytestmark = pytest.mark.ui


def _labels(page, table):
    return page.eval_on_selector_all(
        f"#tbl-{table} td.dash-cell.column-{journeys._column_index(page, f'tbl-{table}', 'Label')}",
        "els => els.map(e => e.textContent.trim())")


def _handle(page, table, row):
    return page.locator(f'#tbl-{table} td.dash-cell[data-dash-column="drag-handle"]').nth(row)


def _drag_row(page, table, from_row, to_row, check_ghost=False):
    src_box = _handle(page, table, from_row).bounding_box()
    dst_box = _handle(page, table, to_row).bounding_box()
    page.mouse.move(src_box["x"] + src_box["width"] / 2,
                    src_box["y"] + src_box["height"] / 2)
    page.mouse.down()
    page.mouse.move(dst_box["x"] + dst_box["width"] / 2,
                    dst_box["y"] + dst_box["height"] / 2, steps=8)
    if check_ghost:
        assert page.locator("table.row-drag-ghost").is_visible(), \
            "Expected a ghost row to follow the cursor while dragging"
    page.mouse.up()


def test_handle_drag_reorders_rows(loaded_page):
    journeys.goto_view(loaded_page, "plan")
    journeys.open_tab(loaded_page, "Income")

    before = _labels(loaded_page, "income")
    assert len(before) >= 3

    _drag_row(loaded_page, "income", from_row=0, to_row=2, check_ghost=True)
    loaded_page.wait_for_timeout(1000)  # store round-trip + re-render

    assert loaded_page.locator("table.row-drag-ghost").count() == 0, \
        "Ghost row must be removed on drop"
    after = _labels(loaded_page, "income")
    expected = before[1:3] + [before[0]] + before[3:]
    assert after == expected, f"Expected {expected}, got {after}"

    # and back up: drag row 2 to row 0
    _drag_row(loaded_page, "income", from_row=2, to_row=0)
    loaded_page.wait_for_timeout(1000)
    assert _labels(loaded_page, "income") == before


def test_handle_click_without_drag_is_inert(loaded_page):
    journeys.goto_view(loaded_page, "plan")
    journeys.open_tab(loaded_page, "Income")

    before = _labels(loaded_page, "income")
    _handle(loaded_page, "income", 0).click()
    loaded_page.wait_for_timeout(600)

    assert _labels(loaded_page, "income") == before
    assert loaded_page.locator("#tbl-income input.dash-cell-value.focused").count() == 0, \
        "Clicking the drag handle must not enter cell edit mode"
