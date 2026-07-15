# UX Test Plan — Retirement Simulator Web App

**Scope**: functional UI/UX behavior of the Dash app (`webapp/`), to the standard of a
modern, polished single-user app. Tests are **coupled to functionality and user
journeys, not to DOM structure**: every step below is expressed in the vocabulary of
`tests/ui/journeys.py` helpers, and tests assert user-visible outcomes. When the DOM
changes, only `journeys.py` is updated — never the test files or this plan.

**How tests bind to this plan**: each automated test carries
`@pytest.mark.ux("<ID>")`. The run's `summary.json` reports outcomes per ID, and the
analyst's REPORT.md must account for every ID listed here.

**Severity classes**
- **Blocker** — a primary journey is unusable (can't run, can't save/load, crash)
- **Major** — a feature misbehaves or data displays wrongly, workaround exists
- **Minor** — behavior is off but rare/cosmetic-functional (focus, timing)
- **Polish** — visual/UX refinement judgment

**Conventions for all automated cases**
- Preconditions "loaded" = `loaded_page` fixture (example scenario uploaded).
- Preconditions "fresh" = `page` + `page.goto(app_url)`.
- Every test implicitly asserts **zero non-allowlisted console errors** (autouse
  fixture in `tests/ui/conftest.py`) — this is case UX-CON-01, enforced suite-wide.
- Expected values derive from the `example_scenario` fixture, never magic numbers.

---

## 1. Navigation & views — `tests/ui/test_nav.py`

**UX-NAV-01 — App opens on the Dashboard.**
Pre: fresh. Steps: load the page. Expected: dashboard view visible, Plan view not;
hero shows the em-dash placeholder numeral and the "Run a simulation…" prompt.
Sev: Major. Automated-by: `test_nav.py::test_opens_on_dashboard`.

**UX-NAV-02 — View toggle switches Dashboard ↔ Plan.**
Pre: fresh. Steps: `goto_view("plan")`; `goto_view("dashboard")`. Expected: exactly
one view visible after each toggle. Sev: Blocker.
Automated-by: `test_nav.py::test_view_toggle`.

**UX-NAV-03 — All five Plan tabs open.**
Pre: loaded. Steps: `goto_view("plan")`; `open_tab` each of Portfolio, Spending,
Income, Lumps, Properties. Expected: each tab's content renders (open_tab asserts
it); no console errors. Sev: Blocker. Automated-by: `test_nav.py::test_all_tabs_open`.

**UX-NAV-04 — Edits survive a view round-trip.**
Pre: loaded. Steps: `goto_view("plan")`; `open_tab("Income")`;
`set_table_cell("income", 0, "Amount Monthly", 4242)`; `goto_view("dashboard")`;
`goto_view("plan")`; `open_tab("Income")`. Expected: cell still shows 4242.
Sev: Major. Automated-by: `test_nav.py::test_edits_survive_view_toggle`.

**UX-EMPTY-01 — No ghost charts before the first run.**
Pre: fresh. Steps: none. Expected: chart cards **not visible**; the placeholder text
"Charts appear here once you run a simulation." **is** visible. Sev: Minor.
Automated-by: `test_nav.py::test_empty_state_no_ghost_charts`.

**UX-EMPTY-02 — Stat tiles show placeholders pre-run.**
Pre: fresh. Steps: none. Expected: the four stat tiles render with "—" values;
tile labels include "Spending guardrail" (not a bare "Guardrail"). Sev: Polish.
Automated-by: `test_nav.py::test_empty_stat_tiles`.

## 2. Plan editing — `tests/ui/test_plan_editing.py` *(exemplar file, partly done)*

**UX-FORM-01 — Portfolio numeric fields accept edits.**
Pre: loaded. Steps: `open_tab("Portfolio")`; fill `#inp-initial-portfolio` with a new
value; Tab away. Expected: value retained; preview re-renders without error.
Sev: Major. Automated-by: `test_plan_editing.py::test_portfolio_fields_editable`.

**UX-FORM-02 — Market dropdown updates the μ/σ label.**
Pre: loaded. Steps: `open_tab("Portfolio")`; pick a different market in `#dd-market`.
Expected: `#lbl-market-mu-sigma` text changes to that market's μ/σ. Sev: Major.
Automated-by: `test_plan_editing.py::test_market_updates_mu_sigma`.

**UX-FORM-04 — Table cell edit persists across tab round-trip.**
Pre: loaded. Steps: (see exemplar). Expected: edited value visible after leaving and
re-entering the tab. Sev: Major.
Automated-by: `test_plan_editing.py::test_table_cell_edit_persists` ✅ implemented.

**UX-FORM-05 — Add band appends a row.**
Pre: loaded. Steps: (see exemplar; uses `expect_row_count`). Expected: row count =
example scenario's band count + 1. Sev: Major.
Automated-by: `test_plan_editing.py::test_add_band_appends_row` ✅ implemented.

**UX-DD-01 — Spending category dropdown opens and selects.** *(regression 2026-07-04:
menu was clipped invisible by table-container overflow)*
Pre: loaded. Steps: (see exemplar; `select_table_category`). Expected: menu visible,
option selected, cell text updates. Sev: Major.
Automated-by: `test_plan_editing.py::test_spending_category_dropdown_selects` ✅.

**UX-DD-02 — Lumps category dropdown opens and selects.**
Same as UX-DD-01 on the Lumps tab. Sev: Major.
Automated-by: `test_plan_editing.py::test_lumps_category_dropdown_selects` ✅.

**UX-VAL-01 — End age ≤ start age is rejected with feedback.**
Pre: loaded. Steps: `open_tab("Portfolio")`; set `#inp-end-age` to a value ≤ start
age; `run_simulation` **without** waiting for a % (click `#btn-run` directly);
`expect_toast(containing="age")`. Expected: error toast; hero numeral unchanged; no
crash. Sev: Major. Automated-by: `test_plan_editing.py::test_end_age_validation`.

**UX-VAL-02 — Empty spending is rejected with feedback.**
Pre: fresh (no scenario — spending empty by default). Steps: click `#btn-run`;
`expect_toast(containing="spending")`. Expected: error toast, no crash. Sev: Major.
Automated-by: `test_plan_editing.py::test_empty_spending_validation`.

## 3. Live preview — `tests/ui/test_preview.py` *(exemplar file, partly done)*

**UX-PREV-01 — Preview renders the scenario's series.**
Automated-by: `test_preview.py::test_preview_renders_scenario_series` ✅ implemented.

**UX-PREV-02 — Preview updates live on a table edit.**
Pre: loaded. Steps: `goto_view("plan")`; capture `trace_count`; delete… simpler:
`open_tab("Lumps")`; `add_row("lumps")` then `set_table_cell` age/amount on the new
row; wait; Expected: preview `trace_count` grows by 1. Sev: Major.
Automated-by: `test_preview.py::test_preview_updates_on_edit`.

**UX-PREV-03 — Zoom survives a table edit.** *(regression 2026-07-04: figure rebuilt
without uirevision reset the axes on every edit)*
Automated-by: `test_preview.py::test_zoom_survives_table_edit` ✅ implemented.

**UX-PREV-04 — Dashboard chart zoom survives a re-run.**
Pre: loaded. Steps: `set_fast_run()`; `run_simulation()`; `zoom_preview(60, 70,
graph="#graph-results")`; `run_simulation()` again. Expected:
`read_preview_zoom(graph="#graph-results") == [60, 70]`. Sev: Minor.
Automated-by: `test_preview.py::test_dashboard_zoom_survives_rerun`.

## 4. Run flows — `tests/ui/test_run.py`

**UX-RUN-01 — Run produces the full result surface.**
Pre: loaded. Steps: `set_fast_run()`; `run_simulation()`. Expected: hero shows a
percentage and a verdict sentence; all four stat tiles show non-"—" values; chart
cards visible (placeholder gone); cash-flow/portfolio/draw graphs have
`trace_count > 0`. Sev: Blocker. Automated-by: `test_run.py::test_run_populates_results`.

**UX-RUN-02 — Historic toggle adds historic cards.**
Pre: loaded. Steps: `set_fast_run()`; enable `#switch-historic` (click its label);
`run_simulation()`. Expected: `#div-historic-cards` contains at least one chart card
with traces. Sev: Major. Automated-by: `test_run.py::test_historic_cards`.

**UX-RUN-03 — Re-run refreshes results.**
Pre: loaded. Steps: `set_fast_run()`; `run_simulation()`; capture hero text; edit a
spending amount sharply (e.g. ×10); `run_simulation()`. Expected: hero/verdict
reflect a changed outcome (text differs). Sev: Major.
Automated-by: `test_run.py::test_rerun_refreshes`.

**UX-RUN-04 — Run completes within interactive bounds.**
Pre: loaded. Steps: `set_fast_run(500)`; `run_simulation(timeout=30_000)`.
Expected: completes < 30 s at 500 paths. Sev: Minor.
Automated-by: `test_run.py::test_run_speed_bound`.

## 5. Scenario I/O — `tests/ui/test_io.py`

**UX-IO-01 — Upload populates the whole plan.**
Pre: fresh. Steps: `upload_scenario(EXAMPLE_XLSX)`; `goto_view("plan")`; per table in
(spending, income, lumps, properties): `row_count` equals the corresponding
example_scenario list length. Also `#header-scenario-name` shows the file's scenario
name. Sev: Blocker. Automated-by: `test_io.py::test_upload_populates_plan`.

**UX-IO-02 — Save downloads a loadable .xlsx.**
Pre: loaded. Steps: `save_scenario("uxtest-roundtrip")` (modal prompts for filename,
confirm button labeled "Download .xlsx"). Expected: browser download named
`uxtest-roundtrip.xlsx`; the engine round-trips it (`scenario_from_xlsx` yields a
non-empty `portfolio`); nothing is written server-side and the name never appears in
the Load dropdown (`scenario_in_load_list` is false); `scenarios/` gains no files
(`no_new_scenario_files` fixture). Sev: Blocker.
Automated-by: `test_io.py::test_save_creates_scenario`.

**UX-IO-03 — retired.** Overwrite flow no longer applies: PRD §3.1 removed
server-side save and the `chk-overwrite` checkbox, so saving is a stateless browser
download with no name-collision concept to test. ID kept for history;
was `test_io.py::test_overwrite_flow`.

**UX-IO-04 — Dropdown load, localStorage persistence, and corrupt-store fallback
(PRD §3.3, §6.4).**
Pre: fresh. Steps: `load_scenario("two_bucket_example")` from the Load dropdown
(still fed by `scenarios/` in local dev); `goto_view("plan")` and check row counts
match. Then `open_tab("Portfolio")`, edit `#inp-initial-portfolio`, Tab away, wait for
the value to land in `localStorage["store-scenario"]`, and `page.reload()`: the edit
survives with no Load needed. Then corrupt `localStorage["store-scenario"]` with
invalid JSON and reload: app falls back to the default scenario (portfolio field
reads "0", `#header-scenario-name` reads "untitled") instead of rendering blank.
Sev: Blocker. Automated-by: `test_io.py::test_reload_and_load`.

**UX-TOAST-01 — Toast appears bottom-right and auto-dismisses.**
Pre: loaded. Steps: trigger any toast (e.g. `save_scenario("uxtest-toast")`); read
`#toast` `bounding_box()`. Expected: box bottom within 120 px
of viewport bottom and right within 120 px of viewport right (i.e. never over the
app bar or form fields); toast gone within 4 s + 1 s tolerance. Sev: Minor.
Automated-by: `test_io.py::test_toast_position_and_dismiss`.

## 6. Playground — `tests/ui/test_playground.py`

**UX-PG-01 — Enabling playground shows the banner.**
Pre: loaded, Plan view. Steps: `enable_playground()`. Expected: banner visible
(helper asserts). Sev: Major. Automated-by: `test_playground.py::test_banner`.

**UX-PG-02 — Chart click opens the event modal and adds a chip.**
Pre: loaded, playground on. Steps: `add_playground_event(-200_000, "test gift")`.
Expected: modal flow completes (helper asserts); a chip listing "test gift" appears
in `#div-playground-chips`; preview gains a marker trace (`trace_count` +1).
Sev: Major. Automated-by: `test_playground.py::test_add_event_chip_and_marker`.

**UX-PG-03 — Run with playground events reports them.**
Pre: after UX-PG-02. Steps: `set_fast_run()`; `run_simulation(playground=True)`.
Expected: result badges (`#div-result-badges`) mention the events (count ≥ 1).
Sev: Major. Automated-by: `test_playground.py::test_run_with_events_badge`.

**UX-PG-04 — Clear-all removes chips and markers.**
Pre: after UX-PG-02. Steps: click `#btn-pg-clear`. Expected: chips container empty;
preview `trace_count` back to its pre-event value. Sev: Minor.
Automated-by: `test_playground.py::test_clear_all`.

## 7. Accessibility & responsiveness — `tests/ui/test_a11y_misc.py`

**UX-A11Y-01 — Keyboard focus is visible.**
Pre: fresh. Steps: press Tab up to 8 times; after each, read the active element's
computed `outlineStyle`. Expected: at least one focus stop on a button/input shows a
visible outline (`solid`). Sev: Major. Automated-by: `test_a11y_misc.py::test_focus_visible`.

**UX-A11Y-02 — Reduced motion is honored.**
Pre: fresh with `reduced_motion="reduce"` context. Steps: load; click `#btn-run` (a
ripple-bearing button). Expected: no `.ripple` element appears; page renders normally.
Sev: Minor. Automated-by: `test_a11y_misc.py::test_reduced_motion`.

**UX-RESP-01 — No horizontal scroll at 1280×800.**
Pre: fresh, viewport 1280×800, then loaded + after a run. Expected:
`document.documentElement.scrollWidth <= window.innerWidth`. Sev: Major.
Automated-by: `test_a11y_misc.py::test_no_hscroll_1280`.

**UX-RESP-02 — No horizontal scroll at 834×1112 (tablet portrait).**
Same check at 834×1112. Sev: Minor. Automated-by: `test_a11y_misc.py::test_no_hscroll_tablet`.

## 8. Suite-wide invariant

**UX-CON-01 — Zero console errors on every journey.**
Enforced by the autouse `fail_on_console_errors` fixture on every test above; any
non-allowlisted console error or pageerror fails that test with a `console/*.log`
artifact. Sev: Blocker when it fires.

---

## MANUAL cases (human judgment, not automated)

| ID | Check | Cadence |
|----|-------|---------|
| UX-MAN-01 | Visual polish: spacing, alignment, typography scale reads ProjectionLab-quality, not generic | before release |
| UX-MAN-02 | Chart hover tooltips: content is informative, formatted ₪, no overlap | before release |
| UX-MAN-03 | Color/contrast judgment on real display incl. tone washes | after palette changes |
| UX-MAN-04 | Animation feel (ripple, wash transitions) — quiet, not distracting | after motion changes |
| UX-MAN-05 | Real-device touch pass (iPad): tap targets, table editing | occasionally |

## Implementation status

| File | Cases | Status |
|------|-------|--------|
| `test_plan_editing.py` | UX-DD-01/02, UX-FORM-04/05 ✅; UX-FORM-01/02, UX-VAL-01/02 to implement | partial (exemplar) |
| `test_preview.py` | UX-PREV-01/03 ✅; UX-PREV-02/04 to implement | partial (exemplar) |
| `test_nav.py` | UX-NAV-01..04, UX-EMPTY-01/02 | to implement |
| `test_run.py` | UX-RUN-01..04 | to implement |
| `test_io.py` | UX-IO-01, 02, 04, UX-TOAST-01 (UX-IO-03 retired) | to implement |
| `test_playground.py` | UX-PG-01..04 | to implement |
| `test_a11y_misc.py` | UX-A11Y-01/02, UX-RESP-01/02 | to implement |

New helpers may be added to `journeys.py` when a case needs DOM knowledge no
existing helper covers — never put selectors in test files.
