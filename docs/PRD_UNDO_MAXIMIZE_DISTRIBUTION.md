# Feature PRD — Undo, chart maximize, editable µ/σ + distribution preview

Status: **Ready for implementation** (2026-07-04). Builds on top of the completed R1–R6 redesign
(`docs/REDESIGN_PRD.md`) and the original `docs/PRD.md`. All existing state schemas, callback contracts,
and engine behavior not explicitly changed below remain binding.

Orchestration: three independent phases, delegated **one at a time, in order** (A, then B, then C) to the
local LLM via `delegate-local`, each reviewed and its own `pytest tests/ -q` run green before starting the
next. Do not run them in parallel — A and C both touch `webapp/layout.py` and `webapp/callbacks.py`.

---

## Phase A — Undo for table edits

### A.1 Problem

`webapp/layout.py`'s Spending/Income/Lumps/Properties tables (`tbl-spending`, `tbl-income`, `tbl-lumps`,
`tbl-properties`, built via `webapp/components.py::build_data_table`) are `editable=True,
row_deletable=True`. There is no way to undo an accidental row delete or a bad cell edit.

### A.2 Design

Add a small in-memory undo stack (`dcc.Store(id="store-undo-stack")`) of previous full `store-scenario`
snapshots, pushed to **only** when a table-driven edit happens (row edit, row delete, or "Add row" click) —
not on every keystroke in an unrelated field like "Initial Portfolio". An "Undo" button pops the stack and
restores `store-scenario`. Ctrl+Z / Cmd+Z does the same via a plain JS keydown listener.

This reuses the existing store→widgets→store round trip already in `webapp/callbacks.py`
(`hydrate_tabs` / `collect_edits`, see the module docstring at the top of that file) — no new architecture.

### A.3 Changes

**`webapp/layout.py`**
1. Add a new store next to the other `dcc.Store`s (near `store-run-id`, ~line 291):
   ```python
   dcc.Store(id="store-undo-stack", storage_type="memory", data=[]),
   ```
2. Add an "Undo" button just above the `dbc.Tabs(...)` inside the Plan view panel
   (the `html.Div(build_panel(None, [...` block, right before the `dbc.Tabs` line):
   ```python
   dbc.Button("Undo", id="btn-undo", color="outline-secondary", size="sm",
              className="mb-2", disabled=True),
   ```

**`webapp/callbacks.py`**
1. In the existing `collect_edits` callback (the one with `Output("store-scenario", "data",
   allow_duplicate=True)` and the big Input list including `tbl-spending`/`tbl-income`/`tbl-lumps`/
   `tbl-properties`/`btn-add-spending`/etc.):
   - Add `Output("store-undo-stack", "data", allow_duplicate=True)` to the Outputs.
   - Add `State("store-undo-stack", "data")` to the end of the State list (after
     `State("store-scenario", "data")`), and add the matching `undo_stack` parameter to the function
     signature.
   - At the very end of the function, before `return scenario`, compute the new undo stack and return it
     as a second value (`return scenario, new_undo_stack`):
     ```python
     TABLE_TRIGGER_IDS = {
         "tbl-spending", "tbl-income", "tbl-lumps", "tbl-properties",
         "btn-add-spending", "btn-add-income", "btn-add-lumps", "btn-add-properties",
     }
     triggered_ids = {t["prop_id"].split(".")[0] for t in ctx.triggered}
     new_undo_stack = list(undo_stack or [])
     if (triggered_ids & TABLE_TRIGGER_IDS) and scenario != prev_scenario:
         new_undo_stack = (new_undo_stack + [prev_scenario])[-20:]
     ```
     (`scenario != prev_scenario` is required to stop the hydrate-echo round — triggered when
     `undo_last_edit` below writes to `store-scenario` and `hydrate_tabs` re-populates the table widgets —
     from pushing a duplicate entry back onto the stack. Compare the dicts directly, do not use
     `json.dumps`.)
2. Add two small new callbacks (anywhere after `collect_edits`, e.g. right after it):
   ```python
   @app.callback(
       Output("store-scenario", "data", allow_duplicate=True),
       Output("store-undo-stack", "data", allow_duplicate=True),
       Input("btn-undo", "n_clicks"),
       State("store-undo-stack", "data"),
       prevent_initial_call=True,
   )
   def undo_last_edit(n_clicks, undo_stack):
       undo_stack = list(undo_stack or [])
       if not undo_stack:
           return no_update, no_update
       prev = undo_stack.pop()
       return prev, undo_stack

   @app.callback(
       Output("btn-undo", "disabled"),
       Input("store-undo-stack", "data"),
   )
   def toggle_undo_disabled(undo_stack):
       return not bool(undo_stack)
   ```

**`webapp/assets/undo.js`** (new file, mirrors the existing `webapp/assets/ripple.js` pattern — plain
vanilla JS, auto-loaded by Dash from `webapp/assets/`, no build step):
```js
document.addEventListener("keydown", function (e) {
  if ((e.ctrlKey || e.metaKey) && (e.key === "z" || e.key === "Z")) {
    var btn = document.getElementById("btn-undo");
    if (btn && !btn.disabled) {
      e.preventDefault();
      btn.click();
    }
  }
});
```

### A.4 Acceptance

- `pytest tests/ -q` stays green (22/22 or more).
- No new engine-level unit test is required for this phase (it's Dash-callback-only, no pure-Python logic
  to isolate) — but do NOT write ad-hoc root-level self-check scripts. Verify manually by running the app
  (`python3 webapp/app.py` or however `run_ui_tests.sh`/README says to start it), adding a spending band,
  deleting it, and confirming "Undo" restores it, both via the button and via Ctrl+Z.
- Undo must **not** fire when editing a plain input like "Initial Portfolio" or typing a Label in a table
  cell that isn't yet committed — only on a committed table `data` change (row edit/delete) or an "Add row"
  click.

---

## Phase B — Maximize button on every chart pane

### B.1 Problem

`webapp/components.py::build_chart_card` is the one function that builds every chart pane in the app
(Cash flow, Portfolio & property, Annual draw, and each historic-scenario card built dynamically in
`webapp/callbacks.py`'s `run_simulation_cb` via `build_chart_card(name, f"graph-historic-{i}", figure=fig)`).
The playground preview chart (`graph-preview` in `webapp/layout.py`) is a raw `dcc.Graph`, not built via
`build_chart_card`. None of them can be expanded — useful since the results grid is a cramped 2-column
layout.

### B.2 Design

**Pure CSS + vanilla JS, no new Dash callback, no new component ids.** A maximize button toggles a
`maximized` CSS class on the chart's containing `.chart-card` element; the class makes that card
`position: fixed` and fill most of the viewport. A single delegated `click` listener (event delegation on
`document`) handles every maximize button that exists now or is created later (covers the dynamically
created historic cards automatically — no per-card wiring needed). This is deliberately **not** a
Dash `Output`/pattern-matching callback: it needs zero round trip to the server and works uniformly for
every chart including ones that don't exist yet at page load.

Because the Plotly figures set an explicit pixel `height` in their layout (see
`engine/figures.py::_cash_flow_panel_height` and similar), simply resizing the CSS container is not enough
— Plotly.js keeps the figure's own explicit height unless told to recompute. Two things fix this together:
`config={"responsive": True}` on every `dcc.Graph` (already the officially supported "make Plotly follow
its container's size" flag) plus dispatching a `window` `resize` event after the CSS class toggles, which
is what `responsive: True` listens for internally.

### B.3 Changes

**`webapp/components.py`** — `build_chart_card`, replace the whole function body with:
```python
def build_chart_card(title, graph_id, figure=None):
    """Build a chart card with title, a maximize button, and the graph."""
    if figure is None:
        figure = go.Figure(layout={"template": PLOTLY_TEMPLATE})

    header = html.Div([
        html.H5(title, className="mb-0"),
        dbc.Button("⤢", className="btn-maximize p-0", color="link", size="sm",
                   title="Maximize / restore"),
    ], className="d-flex justify-content-between align-items-center mb-2")

    return dbc.Card(
        dbc.CardBody([header, dcc.Graph(id=graph_id, figure=figure, config={"responsive": True})]),
        className="md-panel chart-card",
    )
```
Note the returned card now has className `"md-panel chart-card"` (both classes) instead of just
`"md-panel"` — this is intentional, `chart-card` is the new hook the CSS/JS below key off of. Do not add
`chart-card` to `build_panel` (used for the Plan-view tabs container, which must NOT be maximizable).

**`webapp/layout.py`** — the playground preview chart is not built via `build_chart_card`; wrap it by hand
so it gets the same maximize behavior. Find this line (Plan view, playground section):
```python
dcc.Graph(id="graph-preview", figure=_EMPTY_FIGURE),
```
Replace it with:
```python
html.Div([
    html.Div(
        dbc.Button("⤢", className="btn-maximize p-0", color="link", size="sm",
                   title="Maximize / restore"),
        className="d-flex justify-content-end",
    ),
    dcc.Graph(id="graph-preview", figure=_EMPTY_FIGURE, config={"responsive": True}),
], className="chart-card"),
```
Do not change the `id="graph-preview"` or anything else about that graph — callbacks reference it by id
and are unaffected by the extra wrapping `div`.

**`webapp/assets/maximize.js`** (new file, same plain-JS convention as `webapp/assets/ripple.js` /
`webapp/assets/undo.js`):
```js
document.addEventListener("click", function (e) {
  var btn = e.target.closest(".btn-maximize");
  if (!btn) return;
  var card = btn.closest(".chart-card");
  if (!card) return;
  card.classList.toggle("maximized");
  setTimeout(function () { window.dispatchEvent(new Event("resize")); }, 50);
});

document.addEventListener("keydown", function (e) {
  if (e.key === "Escape") {
    var opened = document.querySelectorAll(".chart-card.maximized");
    if (!opened.length) return;
    opened.forEach(function (card) { card.classList.remove("maximized"); });
    setTimeout(function () { window.dispatchEvent(new Event("resize")); }, 50);
  }
});
```

**`webapp/assets/style.css`** — append:
```css
.chart-card { position: relative; }
.chart-card.maximized {
  position: fixed;
  inset: 16px;
  z-index: 2000;
  background: var(--md-surface-1, var(--surface));
  box-shadow: 0 8px 40px rgba(0, 0, 0, .5);
  display: flex;
  flex-direction: column;
}
.chart-card.maximized .dash-graph,
.chart-card.maximized .js-plotly-plot,
.chart-card.maximized .plot-container {
  flex: 1 1 auto;
  height: 100% !important;
  width: 100% !important;
}
.btn-maximize {
  cursor: pointer;
  font-size: 1.1rem;
  line-height: 1;
  text-decoration: none;
}
```
If `--md-surface-1` is not defined in this file's `:root` (check first — the redesign may have renamed it
to `--surface`), use whichever surface/background token the rest of `style.css` actually uses for cards; do
not invent a new one.

### B.4 Acceptance

- `pytest tests/ -q` stays green.
- No new engine-level test needed (pure CSS/JS, no Python logic branches).
- Manually verify: run the app, run a simulation so all 4 dashboard chart cards + at least one historic
  card are visible, click each maximize button and confirm the chart fills most of the viewport and is
  still interactive (hover tooltips work); click again (or press Escape) to restore; repeat for the
  playground preview chart in the Plan view.

---

## Phase C — Editable portfolio µ/σ, compact fat-tails slider, live distribution preview

### C.1 Problem

`webapp/layout.py`'s Portfolio tab only lets the user pick a `market` preset (`dd-market`, options from
`engine.markets.MARKETS`); the resolved µ/σ are shown read-only (`lbl-market-mu-sigma`,
wired in `webapp/callbacks.py::market_info`). There's no way to type a custom expected return / volatility.
The fat-tails degrees-of-freedom slider (`slider-df`, `dcc.Slider(min=3, max=10, step=1)`) sits in a
`width=6` column — too wide for a single small control. There's no visual feedback for what the chosen
µ/σ/fat-tails combination actually implies about the return distribution.

### C.2 Design

1. Add two new **editable** portfolio fields, `mu` and `sigma` (decimal fractions, e.g. `0.042` = 4.2% —
   same units `engine/markets.py::MARKETS` already uses; do NOT convert to percent-scaled numbers anywhere,
   that would require conversions in 3+ places for no benefit). They flow through the existing
   store↔widget round trip exactly like `inp-initial-portfolio` etc. already do.
2. The `market` dropdown keeps controlling **housing** µ/σ (used for properties, unchanged) and is now also
   just a *preset picker* for the portfolio µ/σ inputs via a new explicit "Use market default" button —
   it does **not** auto-overwrite `inp-mu`/`inp-sigma` on every dropdown change. This is deliberate: since
   `dd-market`'s value is also set programmatically by `hydrate_tabs` on every scenario load, an
   Input-triggered auto-fill callback would silently clobber a saved custom µ/σ every time a scenario is
   loaded. A separate button, gated on a real `n_clicks`, sidesteps that race entirely (same reasoning as
   why `btn-add-spending` etc. are checked via `ctx.triggered_id`, not via reacting to every table change).
3. `engine/params.py::SimulationParams.from_scenario` uses `portfolio.get("mu")` /
   `portfolio.get("sigma")` when present, falling back to the selected market's own `mu`/`sigma` when
   absent or `None` (backward compatible with scenarios saved before this feature, including anything
   already sitting in a browser's session storage).
4. `mu`/`sigma` are persisted through the xlsx round trip the same way `fat_tails_df` already is (see
   `engine/params.py` — the `scenario_to_xlsx`-side dict construction and `_scenario_from_df`'s read side).
5. Fat-tails slider: put it and its checkbox in a narrower column (`width=6` → `width=3`) — purely a
   layout tweak, one line.
6. New small distribution-preview chart (`graph-return-distribution`) directly below the mu/sigma/fat-tails
   controls in the Portfolio tab, updating live as any of those four inputs change. It draws a histogram of
   a large synthetic sample using the **exact same sampling function the simulation engine uses**
   (`engine.simulation.sample_real_returns`) when fat tails are enabled, or a plain Normal sample when they
   are not — reusing existing code rather than re-deriving a PDF formula, and guaranteeing the preview
   never drifts out of sync with what a real run would actually do. Uses a fixed RNG seed (`0`) so the
   histogram shape doesn't jitter on every keystroke, independent of the scenario's own `random_seed`.

### C.3 Changes

**`engine/params.py`**
1. In `SimulationParams.from_scenario` (~line 76-109), where `market = MARKETS[portfolio["market"]]` is
   resolved and then used for `real_return_mean=market["mu"], real_return_sd=market["sigma"]`, change to:
   ```python
   mu = portfolio.get("mu")
   sigma = portfolio.get("sigma")
   real_return_mean = mu if mu is not None else market["mu"]
   real_return_sd = sigma if sigma is not None else market["sigma"]
   ```
   and use `real_return_mean`/`real_return_sd` in place of `market["mu"]`/`market["sigma"]` in the
   `SimulationParams(...)` construction that follows. Leave every other use of `market[...]` (housing
   mu/sigma) untouched.
2. Near where `portfolio["fat_tails_enabled"]`/`portfolio["fat_tails_df"]` are written into the xlsx-bound
   scalar-column dict (~line 318-321, the `{"market": ..., "fat_tails_enabled": ..., "fat_tails_df": ...}`
   block keyed `if i == 0 else None`), add two more entries following the exact same pattern:
   ```python
   "mu": portfolio.get("mu") if i == 0 else None,
   "sigma": portfolio.get("sigma") if i == 0 else None,
   ```
3. In `_scenario_from_df` (~line 354-368), where `fat_tails_df` is read back with a legacy-file fallback
   (`int(df.iloc[0]["fat_tails_df"]) if "fat_tails_df" in df.columns and pd.notna(...) else 5`), add,
   following the exact same pattern, but falling back to `None` (not a market lookup — `from_scenario`
   already handles the `None` → market-default fallback at read time, don't duplicate that logic here):
   ```python
   "mu": float(df.iloc[0]["mu"]) if "mu" in df.columns and pd.notna(df.iloc[0]["mu"]) else None,
   "sigma": float(df.iloc[0]["sigma"]) if "sigma" in df.columns and pd.notna(df.iloc[0]["sigma"]) else None,
   ```

**`engine/figures.py`**
1. Add `PRIMARY` to the existing `from engine.theme import (...)` block at the top of the file.
2. Add a new function (near the other `fig_*` builders):
   ```python
   def fig_return_distribution(mu, sigma, fat_tails_enabled, fat_tails_df):
       """Histogram preview of the assumed per-period real-return distribution.
       Reuses the simulation engine's own sampler so the preview can't drift
       out of sync with what a real run actually does."""
       rng = np.random.default_rng(0)
       n = 20_000
       if fat_tails_enabled and fat_tails_df:
           from engine.simulation import sample_real_returns
           sample = sample_real_returns(n, mean=mu, std=sigma, df=fat_tails_df,
                                         clipping_thr=(-0.99, 0.99), rng=rng)
       else:
           sample = rng.normal(mu, sigma, n)
       fig = go.Figure(data=[go.Histogram(x=sample, histnorm="probability density",
                                           marker_color=PRIMARY, nbinsx=60)])
       fig.update_layout(
           template=PLOTLY_TEMPLATE, height=220, showlegend=False,
           margin=dict(l=30, r=10, t=10, b=30),
           xaxis_tickformat=".0%",
       )
       return fig
   ```

**`webapp/layout.py`**
1. In the Portfolio tab, replace this block:
   ```python
   dbc.Row([
       dbc.Col([
           html.Label("Market"),
           dcc.Dropdown(id="dd-market", options=list(MARKETS.keys()), value="IL"),
           html.Span(id="lbl-market-mu-sigma", className="small"),
       ], width=12),
   ]),
   ```
   with:
   ```python
   dbc.Row([
       dbc.Col([
           html.Label("Market"),
           dcc.Dropdown(id="dd-market", options=list(MARKETS.keys()), value="IL"),
           html.Span(id="lbl-market-mu-sigma", className="small"),
       ], width=6),
       dbc.Col([
           html.Label("Expected return µ (decimal, e.g. 0.042 = 4.2%)"),
           dbc.Input(id="inp-mu", type="number", step=0.001, value=MARKETS["IL"]["mu"]),
           html.Label("Volatility σ (decimal, e.g. 0.13 = 13%)", className="mt-2"),
           dbc.Input(id="inp-sigma", type="number", step=0.001, value=MARKETS["IL"]["sigma"]),
           dbc.Button("Use market default", id="btn-apply-market-preset", color="outline-secondary",
                      size="sm", className="mt-2"),
       ], width=6),
   ]),
   ```
2. Change the fat-tails row from `width=6` to `width=3` (one-line change):
   ```python
   dbc.Row([
       dbc.Col([
           dbc.Checkbox(id="chk-fat-tails", label="Fat tails (Student-t)", value=True),
           dcc.Slider(id="slider-df", min=3, max=10, step=1, value=5),
       ], width=3),
   ]),
   ```
3. Directly below that row, add the distribution preview graph:
   ```python
   dbc.Row([
       dbc.Col([
           html.Label("Return distribution preview", className="small text-muted"),
           dcc.Graph(id="graph-return-distribution", config={"displayModeBar": False}),
       ], width=6),
   ], className="mb-2"),
   ```
4. Add `"mu": MARKETS["IL"]["mu"], "sigma": MARKETS["IL"]["sigma"],` to the `"portfolio"` dict in
   `DEFAULT_SCENARIO` (near the existing `"fat_tails_enabled"`/`"fat_tails_df"` entries).

**`webapp/callbacks.py`**
1. `hydrate_tabs`: add `Output("inp-mu", "value")` and `Output("inp-sigma", "value")` to the Outputs list,
   and in the function body add (using the same `.get` defensive pattern as elsewhere in this function,
   falling back to the selected market's own values for scenarios saved before this feature):
   ```python
   from engine.markets import MARKETS
   market_preset = MARKETS[portfolio["market"]]
   mu_val = portfolio.get("mu")
   sigma_val = portfolio.get("sigma")
   ```
   returning `mu_val if mu_val is not None else market_preset["mu"]` and the `sigma` equivalent in the
   matching return-tuple positions. **Update the `(no_update,) * 14` early-return line to
   `(no_update,) * 16`** — you added 2 new Outputs to a callback that previously had 14; count the actual
   Output list length yourself rather than trusting this number blindly.
2. `collect_edits`: add `Input("inp-mu", "value")` and `Input("inp-sigma", "value")` to the Input list, add
   matching `mu_in`/`sigma_in` parameters to the function signature (in the same position as the Input
   list), and inside the `"portfolio"` dict being built, add:
   ```python
   "mu": _num(mu_in, float, None),
   "sigma": _num(sigma_in, float, None),
   ```
   (Passing `None` as the `_num` default means "leave unset, let `from_scenario`'s market fallback apply"
   — do not default to a market lookup here, `_num`'s signature is `_num(value, cast=float, default=0)`, so
   pass `None` explicitly as the third argument.)
3. Add a new callback for the preset button:
   ```python
   @app.callback(
       Output("inp-mu", "value", allow_duplicate=True),
       Output("inp-sigma", "value", allow_duplicate=True),
       Input("btn-apply-market-preset", "n_clicks"),
       State("dd-market", "value"),
       prevent_initial_call=True,
   )
   def apply_market_preset(n_clicks, market):
       from engine.markets import MARKETS
       m = MARKETS[market]
       return m["mu"], m["sigma"]
   ```
4. Add a new callback for the live distribution preview:
   ```python
   @app.callback(
       Output("graph-return-distribution", "figure"),
       Input("inp-mu", "value"),
       Input("inp-sigma", "value"),
       Input("chk-fat-tails", "value"),
       Input("slider-df", "value"),
   )
   def update_return_distribution(mu, sigma, fat_tails, df):
       from engine.figures import fig_return_distribution
       mu = _num(mu, float, 0.04)
       sigma = _num(sigma, float, 0.13)
       if sigma <= 0:
           sigma = 0.001
       return fig_return_distribution(mu, sigma, bool(fat_tails), _num(df, int, 5))
   ```
   (The `sigma <= 0` guard exists because a user could type `0` or a negative number in the volatility box
   mid-edit — `np.random.normal`/`sample_real_returns` would raise or produce a degenerate histogram.)

### C.4 Acceptance

Two new pure-Python tests already exist (architect-owned, do not edit them, iterate your implementation
against them via `run_tests`):
- `tests/test_mu_sigma.py`
- `tests/test_return_distribution.py`

Run `run_tests("tests/test_mu_sigma.py")` and `run_tests("tests/test_return_distribution.py")` plus the
full `run_tests("tests/")` before finishing — the full suite must stay green (22 pre-existing tests + the 2
new files).

Manually verify once tests pass: run the app, change the Market dropdown (confirm it does NOT change the
µ/σ inputs), type a custom µ/σ, confirm the distribution preview redraws, click "Use market default" and
confirm it fills in the selected market's values, toggle fat tails on/off and drag the df slider and
confirm the histogram visibly changes shape (fatter tails at low df).

---

## Delegation notes for whoever runs this (per `delegate-local` / the project's `delegation-workflow`)

- Point the local model at this file (`docs/PRD_UNDO_MAXIMIZE_DISTRIBUTION.md`) and the specific phase
  section — do not paste the phase content into the brief, that defeats the token-saving point of using
  the agentic harness.
- Run phases **A, then B, then C**, strictly in order, each in its own `delegate_local.py` invocation, each
  reviewed (`pytest tests/ -q` independently re-run by the reviewer, not just trusted from the junior's own
  `run_tests` calls) before starting the next phase.
- Phase C should be launched with `--readonly tests/test_mu_sigma.py --readonly tests/test_return_distribution.py`
  once those two files exist (see below) — it is the one phase with real numeric logic (fallback
  precedence, xlsx persistence) worth pinning with fixed acceptance tests the junior can't edit around.
- Phases A and B have no pure-Python logic worth a fixed test file (Dash-callback wiring and CSS/JS
  respectively) — brief them to avoid writing ad-hoc root-level self-check scripts (known failure mode,
  see `delegation-workflow` memory) and to finish once `pytest tests/ -q` is green, since that's the only
  regression signal available for those two phases.
