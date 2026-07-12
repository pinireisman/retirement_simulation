# Feature PRD — Goal-Based Guardrails (G2.1: confidence-anchored thresholds)

Status: **Draft, ready for review** (2026-07-12). Extends the shipped Funded Ratio
Guardrail (`docs/PRD_FUNDED_RATIO_GUARDRAILS.md`, commit `e32de97` + the
asset-side/playground fixes on top of it). Source method: the "Better: calibrate
funded-ratio thresholds with Monte Carlo" section of
`docs/funded_ratio_guardrails_approach.md`.

---

## 1. Why, in plain terms

The G2 sliders ask the user for funded-ratio percentages (85% / 105% / 130%).
Two problems, both proven live on `scenarios/scenario_data_minus_parents.xlsx`:

1. **Raw thresholds are not portable across plans.** That plan's natural funded
   ratio is ~1.33, so the default "raise above 130%" meant *raise from day one*
   and cost 2% of success. Another plan sitting at FR 0.95 gets the opposite
   policy from the same sliders. The number the user sets doesn't mean what
   they think it means — and means something different for every scenario.
2. **Nobody outside this project has intuition for a funded ratio.** The user's
   actual goals are: (a) protect against sequence-of-returns risk by cutting
   discretionary spending realistically, (b) enjoy the upside when the plan is
   beating expectations, (c) some balance of both. Those are statements about
   *plan confidence*, not about PV ratios.

So: let the user state the goal in confidence language —

> *"Cut discretionary spending if my plan's chance of success falls below 90%.
> Give me a raise if it climbs above 99%."*

— and derive the funded-ratio thresholds from the simulation itself. The method
(from the source doc): run the baseline simulation once, and from its own paths
estimate, for every year, **how much wealth is needed at that age for an X%
chance of success**. Those per-year wealth levels convert directly into per-year
funded-ratio guardrails. No nested Monte Carlo — one extra baseline pass per
run, using paths we already know how to generate.

Bonus: the baseline pass gives us the with/without-policy comparison for free,
which is exactly the report a non-expert needs ("this policy took you from
97.6% → 97.8%; in the worst 5% of futures your lifestyle spending dropped at
most 25% for a few years; median estate unchanged").

**Back-compat:** the existing raw-ratio mode stays, relegated to an "Advanced"
UI section. Same guardrail type/KEY, one new `mode` option — no new dropdown
entry, no store-schema break.

---

## 2. Engine design (Stage 1)

### 2.1 Calibration: wealth-needed-for-confidence table

New function in `engine/guardrails.py`:

```python
def calibrate_wealth_needed(bal_over_time: np.ndarray,   # (T, n_paths), baseline run
                            success: np.ndarray,          # (n_paths,) bool — never ruined
                            confidences: tuple) -> dict:  # e.g. (0.75, 0.85, 0.90, ...)
    """W[p][t] = smallest start-of-year-t liquid balance such that baseline
    paths with at least that balance went on to succeed with frequency >= p.
    np.inf where no observed wealth level reaches p (=> always 'behind' that
    year); 0.0 where even the poorest paths reach p (=> never 'behind')."""
    T, n = bal_over_time.shape
    out = {p: np.zeros(T) for p in confidences}
    for t in range(T):
        order = np.argsort(-bal_over_time[t])          # richest first
        sorted_bal = bal_over_time[t][order]
        frac = np.cumsum(success[order]) / np.arange(1, n + 1)
        for p in confidences:
            ok = np.nonzero(frac >= p)[0]
            out[p][t] = np.inf if ok.size == 0 else sorted_bal[ok[-1]]
    return out
```

Notes:
- `success` comes from the baseline run's existing per-path ruin bookkeeping
  (ruined paths are frozen at 0, so `bal_over_time[-1] > 0` is equivalent if no
  cleaner mask is exposed; use the engine's actual ruined mask, not a re-derivation).
- Deterministic — no RNG. O(T · n log n), trivial next to the simulation itself.
- The estimate assumes baseline (unguarded) future spending, so it is slightly
  conservative once cuts kick in. This is the source doc's own recommended
  approximation ("run calibration jobs once, not inside each path"). Footnote it
  in the UI (§3.4); do not build nested simulation to fix it.
- v1 uses the raw per-year values, no smoothing. If year-to-year noise in the
  thresholds shows up in Stage-4 calibration, add isotonic/rolling smoothing then.

### 2.2 Converting to per-year funded-ratio thresholds

In `run_simulation`, when any enabled guardrail has `mode == "confidence"`:

```python
base = run_simulation(params, guardrails=None)        # same seed => identical
                                                      # return draws as the
                                                      # guarded pass
needed = sorted({c for g in configs for c in confidence_levels(g.options)})
W = calibrate_wealth_needed(base["bal_over_time"], base_success, tuple(needed))
pv_need_base = pv_committed + pv_lifestyle + pv_optional   # multipliers at 1
fr_by_conf = {p: (W[p] + pv_income) / np.maximum(pv_need_base, 1.0) for p in needed}
pv_context["fr_by_confidence"] = fr_by_conf
pv_context["baseline_summary"] = base["summary"]      # free with/without report
```

- `np.inf` propagates correctly: `fr_lower[t] = inf` → that year always reads
  "behind" (cut); `fr_upper[t] = inf` (or `c_raise=None`) → never raise.
- Because path FR and threshold FR share the same denominator, the cut/raise
  comparisons reduce algebraically to `bal < W[p][t]` — the ratio form is kept
  for the target-multiplier formula and continuity with manual mode, but this
  equivalence is why the denominator's shape can't reintroduce the
  minus_parents pathology.
- Guard: recursion depth is exactly one (the inner call passes
  `guardrails=None`). Assert `mode != "confidence"` reaches the inner call
  nowhere.

### 2.3 Handler: thresholds become per-year arrays

`FundedRatioGuardrail.__init__` normalizes every threshold to shape `(T,)`:

```python
T = len(self.pv_committed)
def _per_year(x, default=None):
    x = default if x is None else x
    return np.broadcast_to(np.asarray(x, dtype=float), (T,)).copy()

if options.get("mode") == "confidence":
    tables = context["fr_by_confidence"]
    self.fr_lower  = tables[options["c_cut"]]
    self.fr_target = tables[options["c_target"]]
    self.fr_upper  = tables[options["c_raise"]] if options.get("c_raise") else np.full(T, np.inf)
    self.fr_severe = tables[options["c_severe"]] if options.get("c_severe") else np.zeros(T)
else:                                   # manual mode — today's behavior, exactly
    self.fr_lower  = _per_year(options["fr_lower"])
    self.fr_target = _per_year(options["fr_target"])
    self.fr_upper  = _per_year(options["fr_upper"])
    self.fr_severe = _per_year(options.get("fr_severe", 0.80))
```

`multiplier()` indexes `self.fr_lower[year_idx]` etc. — the only change to the
loop body. All Stage-4 dial mechanics (lifestyle cut+raise, gifts capped at 1.0,
per-year caps, mins/maxes) are untouched.

`compute_guardrail_stats` unchanged. `results` dict gains
`baseline_summary` when a calibration pass ran (else absent).

### 2.4 Tests (`tests/test_confidence_guardrail.py`, new)

- `calibrate_wealth_needed` on a hand-built 4-path × 3-year toy: exact W values,
  including the all-succeed → 0 and unreachable-p → inf edges.
- Confidence-mode determinism: same params + seed → identical results twice.
- Same-seed property: the guarded pass and the internal baseline see identical
  `port_ret` (assert the returned baseline_summary equals a manual
  `run_simulation(params, None)` summary).
- Semantics: on a stressed scenario, `mode=confidence, c_cut=0.90, c_raise=None`
  must give success ≥ the no-guardrail baseline (protect-only can trim, never
  raise — this is the property the whole feature exists for).
- Manual mode bit-compat: existing G2 options dict (no `mode` key) produces
  results identical to before this change (pin against
  `test_funded_ratio_baseline`'s numbers).
- No-op: confidence mode with `c_cut` so low it's never binding and
  `c_raise=None` → bit-identical to `guardrails=None`.

---

## 3. UX design (Stage 2)

### 3.1 The three questions (replace the four G2 sliders)

Shown when strategy = "Funded ratio (G2)". Plain-language controls, no ratios:

```text
What should the guardrails do for you?
  (dd-g2-goal)  ○ Protect the plan   ○ Balanced   ○ Enjoy the upside

In a rough stretch, how much could you realistically trim
lifestyle spending?
  (dd-g2-tolerance)  ○ A little (~10%)  ○ Up to a quarter  ○ Up to half

If the plan were stressed, could you shrink or delay the big
optional gifts/upgrades?
  (chk-g2-flex-lumps)  ☑ yes
```

### 3.2 Preset → engine mapping (initial values; Stage 4 calibrates them)

| Goal | c_cut | c_target | c_raise | reading |
|---|---|---|---|---|
| Protect the plan | 0.90 | 0.97 | None (never raise) | act early, bank the upside |
| Balanced | 0.85 | 0.95 | 0.99 | cut when clearly behind, raise when overwhelmingly ahead |
| Enjoy the upside | 0.75 | 0.90 | 0.95 | tolerate risk, harvest good markets sooner |

| Tolerance | min_multiplier | max_cut_per_year |
|---|---|---|
| A little (~10%) | 0.90 | 0.05 |
| Up to a quarter | 0.75 | 0.10 |
| Up to half | 0.50 | 0.15 |

(Stage-4 sign-off fixes applied: "A little" floor aligned to its ~10% label;
the §3.4 footnote states both bias directions — cuts measured conservatively,
raises optimistically; the report pairs the confidence delta with an "Upside
taken" tile so raise-preset users see what the confidence cost bought.)

`chk-g2-flex-lumps` on → `c_severe = 0.80`; off → `c_severe = None` (gifts dial
never trims). `adjustment_fraction`, raise caps, and the gifts-dial internals
stay engine defaults.

### 3.3 Store payload (`collect_guardrails`)

```python
{"type": "funded_ratio_guardrail", "enabled": strategy == "funded_ratio_guardrail",
 "mode": "confidence",
 "c_cut": 0.85, "c_target": 0.95, "c_raise": 0.99,        # from dd-g2-goal
 "c_severe": 0.80,                                          # or None
 "min_multiplier": 0.75, "max_cut_per_year": 0.10}          # from dd-g2-tolerance
```

`parse_guardrail_configs` already forwards unknown option keys — no change.

### 3.4 Advanced section (back-compat)

The four existing raw sliders (lower/target/upper/discount) move into a
collapsed "Advanced: set thresholds manually" block. A mode toggle
(`chk-g2-manual`) switches the payload to today's `fr_*` scalars (no `mode`
key). Default off. Add one line of small print under the presets:
*"Thresholds are calibrated from your own plan's simulation. Estimates assume
baseline spending, so protection is measured slightly conservatively."*

### 3.5 Outcome report (Stage 3)

- Stat tile rewrite (uses `baseline_summary`, free from the calibration pass):
  `Plan confidence: 97.8% with policy (97.6% without)`.
- New tile from multiplier history: `Worst 5% of futures: lifestyle cut up to
  25% for a time; median future: no change`. Requires the handler to append
  its effective multiplier each year (`self.adjustments` already exists;
  add `self.mult_history`, (T, n_paths) — ~3.6MB at 10k paths, fine).
- New figure (`engine/figures.py`): spending-multiplier percentile bands over
  time (P10/P50/P90) — the deferred §4.3 chart from the previous PRD, now
  justified because it is the feedback loop that lets a non-expert iterate on
  goals. Wire into the results tab next to the existing charts.

---

## 4. Staged plan

Assignments follow the previous PRD's tiering. **Caveat from that build:** local
delegation (`delegate-local` + qwen3-coder-30b) produced zero output in both
attempts (refused-read loop → forced-finish); if it fails again on first try,
the architect implements directly from the brief rather than re-briefing.

| Stage | Scope | Implementer | Reviewer | Why |
|---|---|---|---|---|
| **0. Spec** | This document | — (done) | User review | Product semantics (preset wording, confidence numbers) are the user's call as much as engineering. |
| **1. Engine** | §2: `calibrate_wealth_needed`, per-year threshold arrays in the handler, baseline-pass orchestration in `run_simulation`, `baseline_summary` passthrough, `tests/test_confidence_guardrail.py` | **Architect (session model)** | **Opus subagent** | Money math + a subtle estimator (prefix-success quantiles, inf/0 edges, same-seed property). Exactly the class of change the previous build kept off the local model. |
| **2. UI** | §3.1–3.4: three plain-language controls, preset mapping, store payload, Advanced collapse with manual mode | **Local model** (one attempt, then architect) | **Sonnet subagent** | Copies the existing slider/callback pattern; mapping tables are fully specified above. |
| **3. Report** | §3.5: tiles + multiplier-history + percentile-band figure | Split: figure/tiles mechanical (local, one attempt), wording final call with architect | **Sonnet subagent** | New figure is boilerplate Plotly; the sentence templates are product voice. |
| **4. Calibrate presets** | Run all three presets × repo scenarios (`scenario_data_example.xlsx`, `scenarios/scenario_data_minus_parents.xlsx` ± the three disaster events); verify: Protect ≥ baseline success everywhere, Upside raises early without cratering success, Balanced in between; adjust §3.2 numbers; pin a baseline test per preset in the `test_funded_ratio_baseline.py` style | **Architect** judges; runs are mechanical | **Opus subagent** signs off | Whether 0.90/0.85/0.75 *feel* right is the product judgment this feature exists to encode. |

**Sequencing:** 1 → 2 → 3 → 4 strictly; `pytest tests/ -q` green before each
advance (the pre-existing `test_xlsx_roundtrip::test_roundtrip` int/float red is
excluded — it predates this work). Stage 2 depends on Stage 1's option schema;
Stage 3 depends on `baseline_summary` and `mult_history` from Stage 1.

**Out of scope:** nested-simulation calibration (rejected by the source doc);
threshold smoothing (only if Stage 4 shows noise); per-goal custom confidence
sliders (the Advanced manual mode covers power users); random disaster event
*generation* (playground events already model this by hand).

---

## 5. Risks & mitigations

- **2× simulation cost per run** when confidence mode is on. Current runs are
  sub-second at 10k annual paths; acceptable. If monthly × 10k ever hurts,
  cache the calibration table keyed on the scenario snapshot hash (the
  webapp already computes one for the unsaved-changes dot).
- **Calibration noise at the tails** (few paths near W_99 late in horizon).
  Stage-4 explicitly eyeballs the per-year threshold arrays on real scenarios
  before presets ship; smoothing is the ready remedy.
- **User trust**: a user who sets "protect" and sees success *drop* would be
  right to distrust the feature. The Stage-1 test suite makes
  protect-never-raises ≥ baseline a hard invariant, not a hope.
