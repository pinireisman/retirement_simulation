# Feature PRD — Funded Ratio Guardrails (G2)

Status: **Draft, ready for review** (2026-07-09). Builds on the existing guardrail
plug-in system shipped in `engine/guardrails.py` (G1 = `VolatilityDiscretionaryScaling`,
PRD §7.3 in `docs/PRD.md`). Source spec: `docs/funded_ratio_guardrails_approach.md`
(pasted design notes — this document turns them into an implementable, staged plan
against the actual codebase).

---

## 1. How it works, in plain terms

G1 (today's guardrail) reacts to **the market**: "did my portfolio drop a lot last
year? Cut spending." It has no idea whether the plan is actually in trouble — a path
can get a bad year and cut spending even if it's still miles ahead of what it needs.

Funded Ratio Guardrails react to **the plan**, not the market. Each year, for each
simulated path, we ask one question:

> *"How does what I have compare to what I still need?"*

```text
funded_ratio = current portfolio / (present value of everything I still plan to spend)
```

- `funded_ratio` around **1.0** → on track, spend as planned.
- `funded_ratio` well **below 1.0** (e.g. < 0.85) → behind plan → dial back
  discretionary spending.
- `funded_ratio` well **above 1.0** (e.g. > 1.30) → ahead of plan → allow a raise.

The "present value of everything I still need" is computed once per path/year by
discounting all future planned spending (minus future planned income) back to today.
It already accounts for known future one-off events — a wedding gift due in 8 years is
baked into "what I still need" starting now. So a year where that gift actually gets
paid is **not** a shock to the model: the portfolio drops, but the remaining liability
drops by the same amount, so the funded ratio barely moves. This is the key advantage
over a naive "withdrawal-rate this year" rule, which would misread every lumpy expense
year as reckless overspending.

Only **discretionary** spending (lifestyle extras, gifts) moves. Essential spending
(housing, food, health) is never touched, in every version of the rule below.

The cut/raise itself isn't a cliff-edge: when a path crosses a guardrail, its spending
multiplier moves partway toward a target multiplier (25%/year by default) that would
restore the target funded ratio, capped at a max ±10%/year change — so plans get
nudged, not shocked.

This is "cheap": no simulation-inside-a-simulation. The present-value lookahead for
each future year is precomputed once (an O(T²) loop over ~40 years — trivial), then
each of the 10,000 Monte Carlo paths does an O(1) lookup per year.

---

## 2. Why this fits the existing engine almost for free

This is the single most important implementation finding, and it changes the scope
from "new feature" to "new guardrail handler":

**The 3-way spending split the funded-ratio approach wants — protected / flexible /
lumpy-optional — already exists in this codebase.** `engine/params.py`'s `Band` and
`Lump` dataclasses already tag every spending item with `category: strict | lifestyle
| gifts` (`engine/guardrails.py:11`), and `engine/simulation.py` already aggregates
them into separate per-year arrays before the main loop:

```python
spend_by_cat = {cat: ... for cat in CATEGORIES}   # strict / lifestyle / gifts, per band
lump_out_by_cat = {cat: np.zeros(n_periods) for cat in CATEGORIES}  # strict / lifestyle / gifts, per lump
disc_out = spend_by_cat["lifestyle"] + spend_by_cat["gifts"] + lump_out_by_cat["lifestyle"] + lump_out_by_cat["gifts"]
strict_out = spend_by_cat["strict"] + lump_out_by_cat["strict"]
```

Map that straight onto the source doc's buckets:

| Source doc bucket | Existing engine array |
|---|---|
| `essential_recurring` (protected) | `spend_by_cat["strict"]` |
| `discretionary_recurring` (primary dial) | `spend_by_cat["lifestyle"]` |
| `planned_lumpy_fixed` (committed one-offs) | `lump_out_by_cat["strict"]` |
| `planned_lumpy_flexible` (optional gifts/weddings/car upgrades) | `spend_by_cat["gifts"]` + `lump_out_by_cat["gifts"]` |

No new fields on `Band`/`Lump` are needed for the first version. A user who wants a
required home-repair lump *not* to be cuttable just tags it `strict`; an optional
gift/wedding lump tagged `gifts` is already the "flexible lumpy" bucket. This is
exactly the `strict`/`lifestyle`/`gifts` taxonomy the app already asks users to fill
in on the Spending/Lumps tables today.

**Scope decision:** ship the two-bucket version first — one combined
`discretionary = lifestyle + gifts` multiplier, matching `disc_out` exactly as it
exists today (zero changes to how outflow is assembled). The source doc's own
"What I would implement first" section (its final section) recommends starting here,
not with the fancier 3-multiplier hierarchy (separate optional-lumpy multiplier,
reserved-liability bucket, tiered cut order). That richer version is real and
described in the doc, but it requires splitting `disc_out` into two separately-scaled
arrays inside the simulation loop — a bigger, higher-risk change for a benefit that's
unproven until the simple version is tested. Flagged as **Stage 4 / deferred** below;
build only if Stage 1–3 show the simple version isn't nuanced enough.

---

## 3. Data structure & simulation engine changes (Stage 1)

### 3.1 New param

`engine/params.py::SimulationParams` gets one new field — nothing today represents a
liability-discount rate (only `real_return_mean`/`real_return_sd`, which describe the
*stochastic portfolio return*, a different concept):

```python
real_discount_rate: float = 0.01   # used only by funded-ratio guardrails' PV lookahead
```

Persisted through `from_scenario`/xlsx round-trip the same way `fat_tails_df` already
is (`engine/params.py`, the `if i == 0 else None` scalar-column block).

### 3.2 New engine module code — `engine/guardrails.py`

```python
def precompute_pv(cashflows: np.ndarray, discount_rate: float) -> np.ndarray:
    """pv[t] = present value of cashflows[t:], discounted at `discount_rate`."""
    T = len(cashflows)
    pv = np.zeros(T)
    for t in range(T):
        years = np.arange(T - t)
        pv[t] = np.sum(cashflows[t:] / (1 + discount_rate) ** years)
    return pv


class FundedRatioGuardrail:
    """funded_ratio = portfolio / PV(remaining committed + discretionary need).
    Planned gifts/weddings/upgrades (category "gifts") are already inside
    PV_discretionary, so paying one out does not itself look like overspending —
    the liability and the cash both drop together."""

    KEY = "funded_ratio_guardrail"

    def __init__(self, options: dict, context: dict | None = None):
        self.fr_lower = options["fr_lower"]
        self.fr_target = options["fr_target"]
        self.fr_upper = options["fr_upper"]
        self.adjustment_fraction = options.get("adjustment_fraction", 0.25)
        self.max_cut_per_year = options.get("max_cut_per_year", 0.10)
        self.max_raise_per_year = options.get("max_raise_per_year", 0.10)
        self.min_multiplier = options.get("min_multiplier", 0.40)
        self.max_multiplier = options.get("max_multiplier", 1.50)
        self.pv_committed = context["pv_committed"]        # shape (T,)
        self.pv_discretionary = context["pv_discretionary"]  # shape (T,)
        self.mult: np.ndarray | None = None
        self.triggered_down: np.ndarray | None = None
        self.triggered_up: np.ndarray | None = None
        self.adjustments: list = []

    def multiplier(self, year_idx, port_ret, n_paths, bal=None) -> np.ndarray:
        if self.mult is None:
            self.mult = np.ones(n_paths)
            self.triggered_down = np.zeros(n_paths, dtype=bool)
            self.triggered_up = np.zeros(n_paths, dtype=bool)

        pv_committed_t = self.pv_committed[year_idx]
        pv_disc_t = max(self.pv_discretionary[year_idx], 1.0)

        pv_need = np.maximum(pv_committed_t + self.mult * pv_disc_t, 1.0)
        funded_ratio = bal / pv_need

        cut = funded_ratio < self.fr_lower
        raise_ = funded_ratio > self.fr_upper

        target = (bal / self.fr_target - pv_committed_t) / pv_disc_t
        target = np.clip(target, self.min_multiplier, self.max_multiplier)

        proposed = self.mult + self.adjustment_fraction * (target - self.mult)
        proposed = np.where(cut, np.maximum(proposed, self.mult * (1 - self.max_cut_per_year)), proposed)
        proposed = np.where(raise_, np.minimum(proposed, self.mult * (1 + self.max_raise_per_year)), proposed)

        self.mult = np.where(cut | raise_, proposed, self.mult)
        self.triggered_down |= cut
        self.triggered_up |= raise_
        return self.mult
```

`VolatilityDiscretionaryScaling.multiplier()` gains an unused `bal=None` kwarg (one
line) so both handlers share one call signature.

### 3.3 Wiring into `run_simulation` (`engine/simulation.py`)

Two changes, both additive:

1. **Reorder:** `build_handlers(guardrails)` currently runs before `spend_by_cat`
   exists. Move it to just after `disc_out`/`strict_out`/`incomes`/`lump_in` are
   built, and compute the PV context there:

   ```python
   rent_by_period = np.array([
       sum(p.rent_annual / factor for p in params.properties
           if period >= (p.start_age if params.annual else p.start_age * 12))
       for period in periods
   ])
   committed_net = strict_out - incomes - rent_by_period - lump_in
   pv_context = {
       "pv_committed": precompute_pv(committed_net, params.real_discount_rate),
       "pv_discretionary": precompute_pv(disc_out, params.real_discount_rate),
   }
   handlers = build_handlers(guardrails, pv_context)
   ```

   (`rent_by_period` duplicates the per-period rent sum the loop already computes
   inline — deliberately, to avoid touching the loop's proven inline calculation;
   it's a cheap O(T × n_properties) prepass.)

2. **Pass `bal`:** at the existing call site (`mult *= h.multiplier(i, port_ret,
   params.n_paths)`), add `bal=bal`. `bal` at that point in the loop is still the
   *start-of-period* balance (mutated later), matching the funded-ratio formula's
   "portfolio before this year's withdrawal" convention — and matching how G1
   already reads `port_ret[:, year_idx - 1]` as a look-back, not a look-forward.

3. `build_handlers(guardrails, context=None)` passes `context` through to whichever
   handler class needs it (`GUARDRAIL_REGISTRY[g.type](g.options, context)` —
   `VolatilityDiscretionaryScaling.__init__` gains an unused `context=None` param
   for symmetry).

4. Register: `GUARDRAIL_REGISTRY["funded_ratio_guardrail"] = FundedRatioGuardrail`.

No change to `bal_over_time`, `ruin_time`, withdrawal/`wrate` bookkeeping, or any
property-value logic — the guardrail only ever changes the `mult` fed into the
existing `outflow = strict_out[i] + disc_out[i] * mult` line.

### 3.4 Tests — `tests/test_funded_ratio_guardrail.py` (new, mirrors `test_guardrails.py`)

- `precompute_pv` against a hand-computed 3-4 period array (exact expected numbers).
- `FundedRatioGuardrail.multiplier()` unit test with a hand-crafted small `bal`/PV
  scenario: assert the exact multiplier for a path that's clearly underfunded, one
  that's clearly overfunded, and one in the dead zone (no change).
- No-op test: `fr_lower=0, fr_upper=inf` (or equivalent) never triggers →
  `bal_over_time` bit-identical to `guardrails=None`.
- Scenario-level acceptance test on the repo's example scenario: assert it doesn't
  raise, multipliers stay within `[min_multiplier, max_multiplier]` for all paths/years,
  and a year with a large planned "gifts"-category lump does *not* spike the
  fraction of paths cutting that year (the whole point of PV-inclusion) — compare
  `frac_paths_cut` in the gift year vs. neighboring years.
- `build_handlers` still fails fast on an unknown `type` (existing test, should
  keep passing unmodified).

---

## 4. UX changes (Stage 2)

Today's Guardrails panel (`webapp/layout.py:300-317`) is a single collapsible block:
a checkbox + 4 sliders, all G1. Running G1 and a funded-ratio rule simultaneously
would fight each other (two independent multipliers, unclear which "wins" — and
`compute_guardrail_stats` only ever reports `handlers[0]`, `engine/guardrails.py:70-83`).
So: **one active guardrail strategy at a time**, chosen from a dropdown, not two
checkboxes.

```text
Guardrails ▾
┌─────────────────────────────────────┐
│ Strategy: [None ▾]                    │  <- dd-guardrail-strategy
│   options: None / Volatility (G1) / Funded ratio (G2)
│                                        │
│  (G1 block, shown only if strategy=G1) │
│    Drop threshold / Rise threshold / Cut % / Raise %   ← unchanged sliders
│                                        │
│  (G2 block, shown only if strategy=G2) │
│    Lower guardrail (%)      slider, default 85
│    Target funded ratio (%)  slider, default 105
│    Upper guardrail (%)      slider, default 130
│    Discount rate (%)        slider, default 1
└─────────────────────────────────────┘
```

`adjustment_fraction`, `max_cut_per_year`, `max_raise_per_year`, `min/max_multiplier`
stay as engine-side defaults (not sliders) — mirrors G1's own choice not to expose
every internal constant, keeps the panel the same size it is today (4 sliders either
way), and can be promoted to sliders later if real usage shows the defaults are wrong.

### 4.1 `webapp/layout.py`

Replace the checkbox with a strategy dropdown, and gate each slider block behind it:

```python
dbc.Button("Guardrails", id="btn-guardrails-header", color="outline-secondary", className="mb-2"),
dbc.Collapse([
    html.Div("Strategy:", className="mt-2"),
    dcc.Dropdown(id="dd-guardrail-strategy",
                 options=[{"label": "None", "value": "none"},
                          {"label": "Volatility-based (G1)", "value": "volatility_discretionary_scaling"},
                          {"label": "Funded ratio (G2)", "value": "funded_ratio_guardrail"}],
                 value="none", clearable=False),

    html.Div([  # G1 block
        html.Div("Drop threshold (%):", className="mt-2"),
        dcc.Slider(id="slider-g1-drop", min=5, max=50, step=1, value=20, tooltip={"placement": "bottom", "template": "{value}%"}),
        html.Div("Rise threshold (%):", className="mt-2"),
        dcc.Slider(id="slider-g1-rise", min=5, max=50, step=1, value=20, tooltip={"placement": "bottom", "template": "{value}%"}),
        html.Div("Cut percentage (%):", className="mt-2"),
        dcc.Slider(id="slider-g1-cut", min=0, max=50, step=1, value=15, tooltip={"placement": "bottom", "template": "{value}%"}),
        html.Div("Raise percentage (%):", className="mt-2"),
        dcc.Slider(id="slider-g1-raise", min=0, max=50, step=1, value=10, tooltip={"placement": "bottom", "template": "{value}%"}),
    ], id="block-g1"),

    html.Div([  # G2 block
        html.Div("Lower guardrail — funded ratio (%):", className="mt-2"),
        dcc.Slider(id="slider-g2-lower", min=50, max=100, step=1, value=85, tooltip={"placement": "bottom", "template": "{value}%"}),
        html.Div("Target funded ratio (%):", className="mt-2"),
        dcc.Slider(id="slider-g2-target", min=90, max=150, step=1, value=105, tooltip={"placement": "bottom", "template": "{value}%"}),
        html.Div("Upper guardrail — funded ratio (%):", className="mt-2"),
        dcc.Slider(id="slider-g2-upper", min=100, max=200, step=1, value=130, tooltip={"placement": "bottom", "template": "{value}%"}),
        html.Div("Real discount rate (%):", className="mt-2"),
        dcc.Slider(id="slider-g2-discount", min=0, max=4, step=0.25, value=1, tooltip={"placement": "bottom", "template": "{value}%"}),
    ], id="block-g2"),
], id="collapse-guardrails", is_open=False),
```

`store-guardrails`'s default `data=` payload gains a second (disabled) entry for
`funded_ratio_guardrail`, same pattern as the existing G1 default.

A small clientside or Dash callback toggles `block-g1`/`block-g2` `style.display`
based on `dd-guardrail-strategy.value` — same pattern as the existing
`collapse-guardrails` open/close toggle (`webapp/callbacks.py:531-534`), just two
more `Output`s.

### 4.2 `webapp/callbacks.py`

`collect_guardrails` (`webapp/callbacks.py:510-529`) becomes strategy-aware — build
**one** `GuardrailConfig`-shaped dict for whichever strategy is selected, `enabled`
following directly from the dropdown rather than a separate checkbox:

```python
@app.callback(
    Output("store-guardrails", "data"),
    Input("dd-guardrail-strategy", "value"),
    Input("slider-g1-drop", "value"), Input("slider-g1-rise", "value"),
    Input("slider-g1-cut", "value"), Input("slider-g1-raise", "value"),
    Input("slider-g2-lower", "value"), Input("slider-g2-target", "value"),
    Input("slider-g2-upper", "value"), Input("slider-g2-discount", "value"),
)
def collect_guardrails(strategy, drop, rise, cut, raise_pct, fr_lower, fr_target, fr_upper, discount):
    """Callback #15. Exactly one guardrail entry, enabled iff it's the selected strategy."""
    return {"guardrails": [
        {"type": "volatility_discretionary_scaling", "enabled": strategy == "volatility_discretionary_scaling",
         "drop_threshold": drop / 100, "rise_threshold": rise / 100, "cut_pct": cut / 100, "raise_pct": raise_pct / 100},
        {"type": "funded_ratio_guardrail", "enabled": strategy == "funded_ratio_guardrail",
         "fr_lower": fr_lower / 100, "fr_target": fr_target / 100, "fr_upper": fr_upper / 100},
    ]}
```

`real_discount_rate` slider value flows into `store-scenario`'s `portfolio` dict
(like `mu`/`sigma` in the maximize/undo PRD's Phase C), not into the guardrail
config — it's a `SimulationParams` field, not a per-guardrail option, since it also
matters if a future guardrail type needs PV lookahead.

`_GUARDRAIL_DISPLAY_NAMES` (`webapp/callbacks.py:44`) gets one new entry:
`"funded_ratio_guardrail": "Funded ratio guardrail"` — this is the only change
needed for the existing stat tiles (`_stat_tiles`, line 157) to display G2's
trigger/cut/raise stats; no new UI code needed there, `compute_guardrail_stats`
already returns the same shape for any handler.

### 4.3 Deferred UX (not in MVP)

A chart of funded-ratio or multiplier percentiles over time (P10/P50/P90 bands) would
be genuinely useful for judging "is this policy psychologically tolerable" — but it's
a new figure (`engine/figures.py`) plus new per-year percentile bookkeeping in
`run_simulation`'s return dict, which is a real addition, not a plug-in. Build it
after Stage 3 validates the guardrail is worth the visualization investment.

---

## 5. Staged development PRD

Orchestration = the model that writes/refines specs, reorders work, and reviews each
stage's diff against the plan before the next stage starts. Implementation = the
model or worker actually writing code. Three implementer tiers are in play, not two:
a free **local LM Studio model** (via the `delegate-local` skill — "junior worker,"
zero frontier tokens) for mechanical, fully-specified work; **Sonnet 5** as the floor
for anything a local model shouldn't touch; **Opus 4.8** only where the judgment call
itself, not just the code, is the hard part. Haiku is excluded everywhere (no
automode). The local/Sonnet split below follows `delegate-local`'s own stated
boundary: good for "CRUD/layout code, implementing a written spec against existing
code"; bad for "subtle numeric/money math — do those yourself, or the reviewer ends
up rewriting them anyway." That line falls *inside* Stage 1, not around it, so Stage 1
is split into 1a/1b rather than kept monolithic.

| Stage | Scope | Implementer | Reviewer / orchestration | Why |
|---|---|---|---|---|
| **0. Spec** | This document | — (done) | Opus 4.8 | Cross-file architecture call (handler interface change, call-site reorder) — worth the higher tier once, up front. |
| **1a. Param plumbing** | `real_discount_rate` field on `SimulationParams`, xlsx round-trip persistence, `from_scenario` wiring — mechanical scaffolding, no guardrail math. Mirrors the `mu`/`sigma` persistence added in `docs/PRD_UNDO_MAXIMIZE_DISTRIBUTION.md` Phase C, which this same local-delegation flow already built successfully. | **Local model** (`delegate-local`) | **Sonnet 5** reviews | Textbook "implementing a written spec against existing code" — a new scalar field following an established persistence pattern. Zero frontier tokens for typing it; Sonnet review is cheap insurance. Architect (you) should write `tests/test_discount_rate.py` first and launch the junior with `--readonly` on it, mirroring the ownership rule `delegate-local`'s SKILL.md lays out for logic-bearing modules. |
| **1b. Guardrail math** | §3.2-3.3: `precompute_pv`, `FundedRatioGuardrail`, the `run_simulation` reorder + `bal`-passing at the call site, `tests/test_funded_ratio_guardrail.py` | **Sonnet 5** | Opus 4.8 reviews the diff before Stage 2 starts | This *is* the "subtle numeric/money math" `delegate-local` explicitly says to keep off the local model — a sign error or off-by-one in the PV/`bal` timing convention (§3.3.2) would silently corrupt every downstream number, and a 30B local model is exactly where that kind of error slips through unnoticed. Opus reviews because it's easy to miss by just reading green tests. |
| **2. UI wiring** | §4: strategy dropdown, G2 sliders, `collect_guardrails`, display-name entry, block-toggle callback | **Local model** (`delegate-local`) | **Sonnet 5** reviews (`pytest tests/ -q` + a manual run-through) | Same shape as the Undo/Maximize PRD's Phases A and B, both local-delegated: copies an existing, working slider/callback pattern (G1) verbatim with new ids. No new logic, no money math — squarely `delegate-local`'s GOOD list. Brief should point at §4.1/§4.2 above plus the live `webapp/layout.py:300-317` / `webapp/callbacks.py:510-534` blocks rather than re-pasting them. |
| **3. Calibration & acceptance** | Run G2 against the repo's example scenario(s); collect stats (ruin probability, guardrail-trigger fraction, cut/raise frequency, whether the "gifts" lump year spikes cuts); sanity-check defaults (`fr_lower=0.85`, `target=1.05`, `upper=1.30`, `discount=1%`) | **Local model** runs the scenarios and tabulates stats; **Opus 4.8** judges whether the numbers are financially sane and picks final defaults | Opus 4.8 signs off | Split by task type, not by stage: running a scenario and reporting numbers is mechanical (local-delegable, with `run_tests`-style acceptance just being "produced the stats table"); *deciding whether 0.85/1.05/1.30 is actually the right guardrail policy* is a judgment call on the money math's real-world behavior, not on whether code runs — keep that with the model that reviewed Stage 1b. |
| **4. (Deferred, build only if requested)** 3-bucket hierarchy: split `disc_out` into a separately-scaled `lifestyle` vs `gifts`/optional-lumpy multiplier, reserved-liability treatment for large one-offs, tiered cut order (§9-11 of the source doc) | Opus 4.8 | Opus 4.8 | The *design*, not just the code, is genuinely open here — a product call as much as an engineering one. Don't start it speculatively, and don't local-delegate a stage whose spec doesn't exist yet. |

**Sequencing:** 1a → 1b → 2 → 3 strictly in order, each stage's tests green
(`pytest tests/ -q`) before the next starts — same discipline as
`docs/PRD_UNDO_MAXIMIZE_DISTRIBUTION.md`'s delegation notes. 1a and 2 touch
disjoint files from 1b (`engine/params.py`/`webapp/*` vs `engine/guardrails.py`+
`engine/simulation.py`) except for the one shared contract (the `store-guardrails`
payload shape and the `real_discount_rate` field name) — safe to hand each local brief
the finished upstream schema as a fixed interface rather than re-deriving it. Do not
run 1a and 2 concurrently against 1b — 1b changes `build_handlers`'s signature, which
2's `collect_guardrails` brief needs to already exist in its final shape.

**Out of scope for all stages above:** stochastic "random shock" disaster events
(source doc §6) — that's a new stochastic-cashflow feature independent of guardrails,
not needed to make funded-ratio guardrails work, and shouldn't block this PRD.
