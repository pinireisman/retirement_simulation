# Redesign PRD — ProjectionLab-style light theme + dashboard/plan IA

Status: **Ready for implementation** (design spec locked 2026-07-03, Fable design pass; approved by user).
Supersedes the *visual* portions of `PRD.md` §6.1–6.4. All state schemas (§4), callback contracts (§5),
and engine behavior (§7) remain binding and must survive unchanged.

Orchestration: **Sonnet orchestrates phases R1–R5**, delegating the mechanical items in each phase's
delegation table to the local LLM via `delegate-local` (exact before/after specs, review every diff).
**R6 visual QA is a Fable/Opus gate.** Escalate mid-phase only if a spec here turns out ambiguous.

---

## 1. Why

The app works end-to-end but reads as dated and has real readability failures:

- White-on-white Plotly hover text (`engine/figures.py` ~292/~466: `hoverlabel=dict(bgcolor="white")`,
  no font color, on `plotly_dark`'s light default font).
- Hand-rolled Material-dark theme fights Bootswatch DARKLY + Radix dropdown internals with `!important`.
- Category colors defined twice (CSS `:root` in `style.css` + `CATEGORY_COLORS` in `engine/figures.py`) — drift risk.
- Same DataTable `style_header`/`style_cell` dict pasted 4× (`layout.py:122-131, 176-185, 203-212, 257-266`).
- Results are one fixed-height 1800px 3-row subplot monolith (up to ~3400px with historic scenarios) —
  scrolling inside a single chart.
- Engineer-speak copy ("ruin probability", "Guardrail G1", "▶" glyphs) — part of the readability complaint.

Reference: **ProjectionLab** (projectionlab.com) — Vuetify/Material, Roboto, 4px radius + pill buttons,
"chance of success" framing, praised as "powerful AND pretty". Note: its pastel gradient blobs live on the
*marketing site*; the *app* is a calm Material surface. We take the pastel language but make it semantic (§3).

## 2. Locked decisions

1. Light theme only (no dark-mode requirement).
2. "Larger restructure": top app bar + Dashboard | Plan view toggle. **No `dash.Pages`/URL routing** —
   show/hide via callback (the proven playground-banner/guardrails-collapse pattern). The four existing
   `dcc.Store`s (`store-scenario`, `store-playground`, `store-guardrails`, `store-run-id`) are untouched.
3. No new chart types — but the subplot monolith **is split into separate chart cards** (same traces/data).
4. `dbc.themes.DARKLY` → `dbc.themes.BOOTSTRAP` in `webapp/app.py` (light base ends the `!important` fights).

## 3. Design spec

Subject: a personal Monte Carlo retirement simulator (₪ amounts, one power user). The dashboard's single
job: answer **"what's my chance of success?"** at a glance.

### 3.1 Color tokens

Single Python source of truth: `engine/theme.py`. CSS mirrors hexes in `webapp/assets/style.css` `:root`
with cross-referencing comments (no codegen for ~10 colors — keep the two files' hexes identical by hand).

| Token | Value | Use |
|---|---|---|
| `--canvas` | `#F6F8FC` | Page background. Flat and quiet — no decorative gradients behind data. |
| `--surface` | `#FFFFFF` | Cards. Border `rgba(31,41,51,.08)`; shadow `0 1px 3px rgba(31,41,51,.10), 0 1px 2px rgba(31,41,51,.06)`. |
| `--ink` | `#1F2933` | Primary text. |
| `--ink-2` | `rgba(31,41,51,.64)` | Secondary text, labels. |
| `--primary` | `#3949AB` | Buttons, links, portfolio median line. Hover `#303F9F`. White text on it passes AA. |
| `--success` | `#2E7D32` | Status triad — all three AA on white. |
| `--warning` | `#B26A00` | |
| `--danger` | `#C62828` | |

Tone washes (Signature, §3.4) — used ONLY as the dashboard backdrop tint and stat-tile tints, never behind
chart plot areas: mint `#D7EEEC`, powder `#CFE8FF`, blush `#FCDEDE`, lavender `#E7DEFF`.

Categories (moved to `engine/theme.py`, re-exported from `figures.py` for compat):
strict `#B71C1C` (unchanged) · lifestyle `#D81B60` (darkened from `#F06292` for AA text on light) ·
gifts `#8E44AD` (unchanged) · playground fill `#FF9800`, playground text/border on light `#E65100`.

Chart series mapping: portfolio median = `--primary` · percentile band fill = powder blue low-alpha ·
property = purple family (`#8E44AD` base) · income ramp = `_shade()` off `#2E7D32` · rent ramp = `_shade()`
off `#1565C0` · draw bars = `--danger` / neutral `rgba(31,41,51,.15)` · net cash-flow line = `--ink-2`.

**Mandate:** run the dataviz skill's `validate_palette.js` on the final categorical set before R2 sign-off,
and again in R6 on as-shipped hexes.

### 3.2 Type

Roboto stays (already loaded; it is what ProjectionLab uses). Add **Roboto Mono** to the Google Fonts link.

- Body: Roboto 400, 14–15px.
- Labels/overlines: Roboto 500, 12px, `letter-spacing: .04em`.
- Hero numeral: Roboto 300, ~64px, tight tracking, tone-colored.
- ₪ amounts in tables and chart hover labels: Roboto Mono (tabular figures are the app's real content).

### 3.3 Shape & motion

Radius 4px (controls), 8px (cards), 24px pill (Dashboard/Plan toggle, Run button). Keep `ripple.js`;
recolor ripple to `rgba(57,73,171,.25)`. Respect `prefers-reduced-motion` (disable ripple + wash
transition). Responsive floor: views are full-width stacks; tables scroll horizontally on narrow screens;
visible keyboard focus on all interactive elements.

### 3.4 Signature: the verdict atmosphere

Top of the Dashboard: the **"Chance of success" hero** — one huge thin numeral (Roboto 300), tone-colored,
with a plain-language verdict line, e.g. *"Your plan holds in 9,900 of 10,000 simulated futures."*
Paired with a **semantic tone wash**: a soft gradient tint at the top of the canvas reflecting the latest
result — lavender-grey before any run, mint on success, powder on borderline, blush when the plan is in
danger. Implemented as one CSS class swap (`.wash-neutral/.wash-success/.wash-borderline/.wash-danger`) on
the dashboard wrapper, driven by the tone the engine already computes (`get_ruin_explanation()`), ~500ms
ease transition. The pastel palette becomes information: the page's atmosphere is the plan's health.
Everything else stays quiet — white cards on a calm canvas.

Tone thresholds reuse the existing ruin-icon logic in `figures.py` (green/orange/red bands) mapped to
success/borderline/danger.

### 3.5 Empty state (the new front door)

Dashboard is the landing view, so pre-first-run it must be designed, not blank: neutral wash, muted hero
placeholder ("—"), one line — *"Run a simulation to see how your plan holds up."* — and a single primary
**Run simulation** button as the focal CTA. No disabled ghost charts.

### 3.6 Copy rules

Sweep all user-facing text: buttons say what they do ("Run simulation", "Save scenario", "Add band" — drop
the ▶ glyphs); consistent verbs end-to-end (Run → "Simulation complete" toast); jargon translated at the
surface ("ruin probability" → "chance of success"; "Guardrail G1" → "Spending guardrail", parameters in
plain terms; "Playground" stays but its banner explains what it does); errors state what went wrong and how
to fix it. `validate_scenario` messages surface verbatim (already plain). English UI, ₪ amounts.

### 3.7 IA wireframe

```
┌────────────────────────────────────────────────────────────────────┐
│ APP BAR (white, subtle shadow)                                     │
│ Retirement Simulator   scenario-name •     [Dashboard | Plan]      │
│                                        Save  Load ▾  Upload .xlsx  │
├────────────────────────────────────────────────────────────────────┤
│ DASHBOARD view (landing, full width, tone-washed top band)         │
│ ┌────────────────────────────────────────────────────────────────┐ │
│ │  CHANCE OF SUCCESS          [Run simulation] [Run w/ events]   │ │
│ │      99.0%                  [x] Include historic scenarios     │ │
│ │  Your plan holds in 9,900 of 10,000 simulated futures.        │ │
│ └────────────────────────────────────────────────────────────────┘ │
│ [Median portfolio] [Median property] [Median estate] [Guardrail…]  │  ← stat tiles
│ ┌──────────────────────────┐ ┌──────────────────────────┐          │
│ │ Cash flow                │ │ Portfolio & property     │          │  ← chart cards
│ └──────────────────────────┘ └──────────────────────────┘          │
│ ┌──────────────────────────┐ ┌──────────────────────────┐          │
│ │ Annual draw              │ │ Historic: 1929 crash …   │(if on)   │
│ └──────────────────────────┘ └──────────────────────────┘          │
├────────────────────────────────────────────────────────────────────┤
│ PLAN view (toggled, full width)                                    │
│ [Portfolio][Spending][Income][Lumps][Properties]  ← existing tabs  │
│ Playground panel · Guardrails panel  ← existing, restyled          │
└────────────────────────────────────────────────────────────────────┘
```

New store: `dcc.Store(id="store-active-view", data="dashboard")` + one callback toggling the two view
containers' `style` (copy the `_banner_style()` display-dict pattern in `callbacks.py`).

## 4. Phases

Every phase ends with `pytest tests/ -q` green (baseline ruin 0.9355 + guardrail invariants prove the
engine untouched) and a fresh-profile headless-Chrome screenshot (project memory: Chrome disk cache serves
stale screenshots on reused profiles).

### R1 — Theme foundation
Files: new `engine/theme.py`; rewrite `webapp/assets/style.css`; `webapp/app.py` (BOOTSTRAP + Roboto Mono).
- `engine/theme.py`: `PALETTE`, `CATEGORY_COLORS` (moved from `figures.py`, re-export kept), `PLAYGROUND_COLOR`, status triad, tone-wash map.
- `style.css`: light tokens per §3.1, wash classes per §3.4, strip now-unneeded `!important` dark overrides.
- Delegate local: token rename pass + override deletions from an exact mapping table. Sonnet: wash CSS, review.
- Verify: `pytest tests/test_webapp_smoke.py -q`; screenshot of empty shell.

### R2 — Plotly template + panel split
Files: `engine/figures.py`, `engine/theme.py`; template call sites in `webapp/callbacks.py:95`, `webapp/layout.py:7`.
- `PLOTLY_TEMPLATE = go.layout.Template(...)` in `theme.py`: Roboto, white paper/plot bg, light gridlines,
  **`layout.hoverlabel` set once** (white bg + ink font) — delete both per-annotation `bgcolor="white"`
  overrides (root cause, not patch).
- Split `plot_cash_flow`/`plot_with_historic` into per-panel builders `fig_cash_flow(...)`,
  `fig_portfolio(...)`, `fig_draw(...)`, `fig_historic(...)` each returning one `go.Figure` (~420px,
  autosize), same traces/data as today's rows. Keep `build_cash_flow_series`, percentile logic, `_shade()`
  untouched. Thin compat wrappers may keep old names for tests/CLI until R5. Check `cli.py` call sites —
  CLI parity must hold.
- Replace every hardcoded color literal (`"blue"`, `"grey"`, `#1f77b4`, `#9467bd`, `#d62728`, `#2ca02c`,
  `#EEEEEE`, green/blue 5-step ramps, green/orange/red icon colors) with `engine.theme` references per §3.1.
- Remove the in-chart ruin ℹ️ annotation (hero replaces it in R4); keep `get_ruin_explanation()`.
- Delegate local: literal→constant find/replace + template swaps from a mapping table. Sonnet: the Template
  object + the panel-split refactor (structurally risky — do directly, verify trace-for-trace against a
  before screenshot); run `validate_palette.js`.
- Verify: `pytest tests/test_engine_baseline.py tests/test_guardrails.py -q`; per-panel screenshots incl. a
  hover tooltip; `python3 cli.py scenario_data_example.xlsx` → ruin 0.010% unchanged.

### R3 — Component library
Files: new `webapp/components.py`; call sites in `webapp/layout.py`.
- `build_panel(title, children)` · `build_data_table(id, columns, ..., category_col=None)` (kills the 4×
  dup dict; category conditional styling optional — only Spending/Lumps use it) · `build_stat_tile(label,
  value, tone=None)` (real card; hero variant at display size) · `build_badge_row(items)` (generalizes
  `_result_badges`/`_playground_chips`) · `build_chart_card(title, graph_id)`.
- Delegate local: the best local-LLM phase — mechanical extraction of byte-identical dicts; Sonnet writes
  signatures + one worked example, reviews for behavioral equivalence.
- Verify: full `pytest tests/ -q`; screenshot diff vs R2 shows ~no change (pure refactor).

### R4 — Nav shell & IA restructure
Files: `webapp/layout.py`; one new callback in `webapp/callbacks.py`.
- App bar + two view containers per §3.7. Builder content moved (not modified) into `div-view-plan`.
  Dashboard assembled: washed hero band, stat-tile row, chart card grid (3 `dcc.Graph`s replacing
  `graph-results`, + historic-cards container), empty state per §3.5. `store-active-view` + toggle callback.
- Copy pass over all static text in `layout.py` per §3.6.
- Delegate local: block moves into wrappers + the show/hide callback, after Sonnet fixes exact container
  boundaries. Sonnet: app-bar/dashboard composition, empty state, copy.
- Verify: smoke tests; live click-through — Dashboard↔Plan toggle, stores persist across toggle; grep every
  id referenced in `callbacks.py` against the new `layout.py` (no orphaned Inputs/Outputs).

### R5 — Callback wiring
Files: `webapp/callbacks.py`.
- `run_simulation_cb` outputs per-panel figures to the 3 graph ids (+ historic container) instead of one
  monolith; `_summary_cards()` → `build_stat_tile()`s incl. hero fed by `get_ruin_explanation()`; wash-class
  output on the dashboard wrapper; badges/chips → `build_badge_row()`; `_EMPTY_DARK_FIGURE` → light empty
  figure from `PLOTLY_TEMPLATE`; preview-figure template swap; toast/error copy per §3.6.
- Tests that asserted the monolith figure shape get updated here, deliberately, not silently.
- Delegate local: call-site swaps against R3 signatures. Sonnet: integration review + first full screenshot
  pass (this is where the whole app is coherent for the first time).
- Verify: full `pytest tests/ -q`; walkthrough: fresh load (empty state) → load scenario → edit band →
  toggle playground → run (wash transition, hero, cards) → historic toggle → save/load.

### R6 — Visual QA & hardening  *(Fable/Opus gate)*
- Screenshot-driven bug list across all views/states; fix pass; re-run `validate_palette.js` on as-shipped
  hexes; keyboard-focus + reduced-motion check; update `PRD.md` §6.1–6.4 to the new IA.
- Judgment question: does it read as ProjectionLab-quality or generic-AI-pastel; is the wash legible but quiet.
- Verify: full `pytest tests/ -q` + full click-through of every view/tab/toggle/modal, including chart hover
  tooltips and toast messages (the original complaint).

## 5. Delegation summary

| Phase | Local LLM (exact spec required) | Sonnet directly | Fable/Opus |
|---|---|---|---|
| R1 | CSS token renames, override deletions | wash CSS, token review | — |
| R2 | color find/replace, template swaps | Template object, panel split, palette validation | — |
| R3 | component extraction + call-site rewrites | signatures, equivalence review | — |
| R4 | block moves, show/hide callback | composition, empty state, copy | — |
| R5 | call-site swaps | integration review, screenshot pass | — |
| R6 | PRD prose mechanics | fix implementation | QA judgment + fix list |
