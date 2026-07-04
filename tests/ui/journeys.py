"""User-journey helpers — the ONLY place UI tests may touch DOM structure.

Test files express user intent (open a tab, edit a cell, run a simulation)
through these helpers and assert user-visible outcomes. Two kinds of selector
are allowed here:

- Dash component ids (#btn-run, #tbl-spending, ...): stable by construction —
  every callback in webapp/callbacks.py references them, so renaming one is a
  functional change, not a cosmetic one.
- Library-internal markup (dash_table cell classes, react-select menus,
  Plotly's figure object): implementation-coupled. Each use below carries a
  "COUPLING:" comment naming the library detail it depends on, so when a Dash
  upgrade breaks a helper, the fix is local to this file.
"""
from __future__ import annotations

import re

from playwright.sync_api import Page, expect

PREVIEW = "#graph-preview"


# ---------- navigation ----------

def goto_view(page: Page, view: str) -> None:
    """Click the Dashboard/Plan pill the way a user does."""
    page.locator(f".view-toggle label:has-text('{view.capitalize()}')").click()
    target = "#div-view-plan" if view.lower() == "plan" else "#div-view-dashboard"
    expect(page.locator(target)).to_be_visible()


def open_tab(page: Page, tab_name: str) -> None:
    """Open a Plan-view tab by its visible label (Portfolio/Spending/...)."""
    page.locator(f"#div-view-plan a.nav-link:has-text('{tab_name}')").click()
    expect(page.locator(f"#tbl-{tab_name.lower()}" if tab_name.lower() != "portfolio"
                        else "#inp-initial-portfolio")).to_be_visible()


# ---------- scenario I/O ----------

def upload_scenario(page: Page, xlsx_path: str) -> None:
    """Upload an .xlsx scenario and wait for the user-visible confirmation."""
    page.locator("#upload-scenario input[type=file]").set_input_files(xlsx_path)
    expect_toast(page, "loaded")


def save_scenario(page: Page, name: str, overwrite: bool = False) -> None:
    page.locator("#btn-save").click()
    expect(page.locator("#modal-save")).to_be_visible()
    page.locator("#input-save-name").fill(name)
    if overwrite:
        page.locator("#chk-overwrite").check()
    page.locator("#btn-save-confirm").click()


def load_scenario(page: Page, name: str) -> None:
    """Pick a scenario in the app-bar dropdown and click Load."""
    page.locator("#dd-load-scenario").click()
    page.get_by_role("option", name=re.compile(name)).click()
    page.locator("#btn-load").click()
    expect_toast(page, "loaded")


# ---------- tables ----------

def _column_index(page: Page, table_id: str, column_name: str) -> int:
    # COUPLING: dash_table renders header cells as th.dash-header.column-<i>
    # with the display name inside span.column-header-name.
    headers = page.locator(f"#{table_id} th.dash-header")
    for i in range(headers.count()):
        th = headers.nth(i)
        if column_name.lower() in th.inner_text().strip().lower():
            m = re.search(r"column-(\d+)", th.get_attribute("class") or "")
            if m:
                return int(m.group(1))
    raise AssertionError(f"column {column_name!r} not found in #{table_id}")


def table_cell(page: Page, table: str, row: int, column: str):
    """Locator for a data cell, addressed by visible column name."""
    idx = _column_index(page, f"tbl-{table}", column)
    # COUPLING: dash_table data cells are td.dash-cell.column-<i>, one per row.
    return page.locator(f"#tbl-{table} td.dash-cell.column-{idx}").nth(row)


def set_table_cell(page: Page, table: str, row: int, column: str, value) -> None:
    """Edit a cell the way a user does: double-click, retype, Enter."""
    cell = table_cell(page, table, row, column)
    cell.dblclick()
    # COUPLING: the focused dash_table cell swaps in an <input>.
    cell.locator("input").fill(str(value))
    page.keyboard.press("Enter")


def select_table_category(page: Page, table: str, row: int, value: str,
                          column: str = "Category") -> None:
    """Open a dropdown cell and pick an option (regression: this menu used to
    be clipped invisible by the table container's overflow)."""
    table_cell(page, table, row, column).click()
    # COUPLING: dash_table dropdown cells embed react-select; the open menu is
    # .Select-menu-outer with role="option" entries.
    menu = page.locator(".Select-menu-outer")
    expect(menu).to_be_visible()
    menu.get_by_role("option", name=value, exact=True).click()
    expect(table_cell(page, table, row, column)).to_contain_text(value)


def row_count(page: Page, table: str) -> int:
    # COUPLING: dash_table renders the header row inside tbody too.
    return page.locator(f"#tbl-{table} tbody tr").count() - 1


def expect_row_count(page: Page, table: str, n: int) -> None:
    """Waiting assertion — table edits go through a server callback round-trip,
    so a plain row_count() right after a click races the re-render."""
    expect(page.locator(f"#tbl-{table} tbody tr")).to_have_count(n + 1)


def add_row(page: Page, table: str) -> None:
    page.locator(f"#btn-add-{table}").click()


# ---------- charts ----------

def read_preview_zoom(page: Page, graph: str = PREVIEW):
    """Current x-axis range of a chart, or None if autoranged."""
    # COUPLING: Plotly stores live axis state on the graph div's _fullLayout.
    return page.evaluate(
        """(sel) => {
            const gd = document.querySelector(sel + ' .js-plotly-plot');
            if (!gd || !gd._fullLayout) return null;
            const ax = gd._fullLayout.xaxis;
            return ax.autorange ? null : ax.range.slice();
        }""", graph)


def zoom_preview(page: Page, x0: float, x1: float, graph: str = PREVIEW) -> None:
    """Zoom a chart to an x-range. Uses Plotly's relayout — the same event a
    user's drag-zoom emits — because pixel-drag coordinates are viewport- and
    layout-dependent (flaky)."""
    # COUPLING: window.Plotly + the graph div; relayout mirrors drag-zoom.
    page.evaluate(
        """([sel, x0, x1]) => {
            const gd = document.querySelector(sel + ' .js-plotly-plot');
            return window.Plotly.relayout(gd, {'xaxis.range': [x0, x1]});
        }""", [graph, x0, x1])
    assert read_preview_zoom(page, graph) == [x0, x1]


def trace_count(page: Page, graph: str) -> int:
    # COUPLING: Plotly graph div .data array.
    return page.evaluate(
        "(sel) => { const gd = document.querySelector(sel + ' .js-plotly-plot');"
        " return gd && gd.data ? gd.data.length : 0; }", graph)


# ---------- runs ----------

def run_simulation(page: Page, playground: bool = False, timeout: int = 60_000) -> None:
    """Click Run and wait for the user-visible result (hero shows a percent)."""
    goto_view(page, "dashboard")
    page.locator("#btn-run-playground" if playground else "#btn-run").click()
    expect(page.locator("#hero-numeral")).to_contain_text("%", timeout=timeout)


def set_fast_run(page: Page, n_paths: int = 500) -> None:
    """Lower the path count so run-flow tests stay fast."""
    goto_view(page, "plan")
    open_tab(page, "Portfolio")
    page.locator("details summary").click()  # the Advanced section
    page.locator("#inp-n-paths").fill(str(n_paths))
    page.keyboard.press("Tab")


# ---------- feedback surfaces ----------

def expect_toast(page: Page, containing: str = "", timeout: int = 10_000) -> str:
    """Wait for the toast and return its text."""
    toast = page.locator("#toast")
    expect(toast).to_be_visible(timeout=timeout)
    if containing:
        expect(toast).to_contain_text(containing, ignore_case=True, timeout=timeout)
    return toast.inner_text()


# ---------- playground ----------

def enable_playground(page: Page) -> None:
    page.locator("#switch-playground label, label[for*='switch-playground']").first.click()
    expect(page.locator("#banner-playground")).to_be_visible()


def add_playground_event(page: Page, amount: int, label: str = "test event") -> None:
    """Click a bar on the preview chart, fill the modal, confirm."""
    plot = page.locator(f"{PREVIEW} .js-plotly-plot")
    box = plot.bounding_box()
    # COUPLING: Plotly click events need a hit on a rendered bar; bars span the
    # full age axis just above/below the zero line, so mid-plot clicks land.
    for frac_y in (0.45, 0.5, 0.55, 0.4):
        page.mouse.click(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * frac_y)
        if page.locator("#modal-playground").is_visible():
            break
    expect(page.locator("#modal-playground")).to_be_visible()
    page.locator("#input-pg-amount").fill(str(amount))
    page.locator("#input-pg-label").fill(label)
    page.locator("#btn-pg-confirm").click()
    expect(page.locator("#modal-playground")).not_to_be_visible()
