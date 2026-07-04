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


def save_scenario(page: Page, name: str, overwrite: bool = False) -> str:
    """Save the current scenario via the Save modal; return the feedback text.

    The overwrite checkbox only appears once a file with this name exists, so
    it is checked only when visible. On a name conflict without overwrite the
    app leaves the modal open with a danger toast — the returned text lets the
    caller assert on that ("exists"/"overwrite") the way a user would read it.
    """
    modal = page.locator("#modal-save")
    # A prior conflict can leave the modal already open; only then skip re-clicking
    # Save (btn-save sits behind the modal backdrop). When the modal is closed,
    # clear any lingering toast first so the text we read back is this save's.
    if not modal.is_visible():
        _dismiss_toast(page)
        page.locator("#btn-save").click()
        expect(modal).to_be_visible()
    page.locator("#input-save-name").fill(name)
    if overwrite and page.locator("#div-overwrite-checkbox").is_visible():
        box = page.locator("#chk-overwrite")
        box.check()
        # Let the checkbox's value prop sync to Dash's store before the confirm
        # callback reads it as State (otherwise it reads the old False).
        expect(box).to_be_checked()
        page.wait_for_timeout(200)
    page.locator("#btn-save-confirm").click()
    # A successful save closes the modal; a conflict/error leaves it open with a
    # danger toast. Wait for whichever settles so we don't read a stale toast.
    toast = page.locator("#toast")
    for _ in range(50):
        page.wait_for_timeout(100)
        if not modal.is_visible():
            break  # saved
        txt = toast.inner_text().lower() if toast.is_visible() else ""
        if any(w in txt for w in ("exist", "overwrite", "cannot", "error")):
            break  # rejected, modal stays open
    return expect_toast(page)


def scenario_in_load_list(page: Page, name: str) -> bool:
    """Open the app-bar Load dropdown and report whether `name` is listed."""
    page.locator("#dd-load-scenario").click()
    # COUPLING: the app-bar dcc.Dropdown renders its open options as elements
    # with role=option (dash-dropdown markup, distinct from the dash_table
    # in-cell dropdowns which use react-select's .Select-menu-outer).
    try:
        expect(page.get_by_role("option").first).to_be_visible(timeout=5_000)
        found = page.get_by_role("option", name=re.compile(re.escape(name))).count() > 0
    finally:
        page.keyboard.press("Escape")
    return found


def load_scenario(page: Page, name: str) -> None:
    """Pick a scenario in the app-bar dropdown and click Load."""
    _dismiss_toast(page)
    page.locator("#dd-load-scenario").click()
    # Options are refreshed on an interval after a reload; wait for them.
    option = page.get_by_role("option", name=re.compile(re.escape(name)))
    expect(option).to_be_visible(timeout=10_000)
    option.click()
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


def _dismiss_toast(page: Page) -> None:
    """Close the notice toast if one is showing (so the next read is fresh).

    Best-effort: a toast mid-fade can be unclickable, so failures are ignored
    rather than allowed to stall the caller.
    """
    toast = page.locator("#toast")
    if not toast.is_visible():
        return
    try:
        # COUPLING: dbc.Toast(dismissable=True) renders a header close button.
        page.locator("#toast button.btn-close").click(timeout=2_000)
        expect(toast).to_be_hidden(timeout=2_000)
    except Exception:
        pass


# ---------- playground ----------

def enable_historic(page: Page) -> None:
    """Toggle the historic scenarios switch."""
    # dbc.Switch renders as a label with a checkbox input
    page.locator("label[for='switch-historic'], #switch-historic").first.click()


def enable_playground(page: Page) -> None:
    # The playground switch lives in the Plan view, below the tabs.
    goto_view(page, "plan")
    # COUPLING: dbc.Switch renders a visually-hidden checkbox + a clickable
    # <label for=...>; click the label (the input itself is opacity:0).
    page.locator("label[for='switch-playground']").click()
    expect(page.locator("#banner-playground")).to_be_visible()


def add_playground_event(page: Page, amount: int, label: str = "test event",
                         age: int = 60) -> None:
    """Open the add-event modal at `age`, fill it, and confirm.

    The app opens the modal from the preview graph's ``clickData`` (the user
    clicks a bar to pick an age). Pixel-clicking a Plotly filled-area trace is
    unreliable, so we emit the same ``plotly_click`` event Dash listens for —
    the exact signal a real bar click produces — with the target age.
    """
    # COUPLING: dcc.Graph updates its clickData prop from Plotly's plotly_click
    # event; emitting it on the graph div drives the same callback a click does.
    page.evaluate(
        """([sel, age]) => {
            const gd = document.querySelector(sel + ' .js-plotly-plot');
            gd.emit('plotly_click', {points: [{x: age, y: 0, curveNumber: 0,
                                               pointNumber: 0}]});
        }""",
        [PREVIEW, age],
    )
    expect(page.locator("#modal-playground")).to_be_visible()
    page.locator("#input-pg-amount").fill(str(amount))
    page.locator("#input-pg-label").fill(label)
    page.locator("#btn-pg-confirm").click()
    expect(page.locator("#modal-playground")).not_to_be_visible()


def clear_playground_events(page: Page) -> None:
    """Click Clear all and wait for the chips container to empty."""
    page.locator("#btn-pg-clear").click()
    expect(page.locator("#div-playground-chips")).to_be_empty()
