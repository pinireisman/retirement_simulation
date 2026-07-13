# Product Requirements Document: Two-Bucket Retirement Withdrawal Strategy

**Repository:** `pinireisman/retirement_simulation`
**Target branch:** `webapp-port`
**Document status:** v1.0 — implementation-ready. All previously open decisions are resolved (§19). Design validated against the actual code on `webapp-port`; every file/line reference below was verified.
**Feature name in UI:** **Growth + Spending Reserve**
**Internal feature name:** `two_bucket_withdrawal_strategy`

---

## 1. Executive summary

Add an optional retirement withdrawal strategy that divides the existing liquid portfolio into two pools:

1. **Growth bucket** — the long-term portfolio, using the application's existing configurable real-return model, including the Student-t distribution.
2. **Spending reserve** — a lower-volatility pool intended to cover a configurable number of years of expected portfolio-funded spending gaps, with its own return assumptions.

Each simulated period, the strategy decides:

- which bucket funds a cash-flow deficit;
- where a cash-flow surplus is deposited;
- when money is transferred from growth to reserve (refill);
- what happens when the preferred bucket cannot cover the required amount (spillover).

The purpose is to reduce forced sales from the growth portfolio after adverse returns. The feature is an **optional withdrawal/allocation strategy**, not a spending guardrail. Existing spending guardrails continue to determine how much is spent; the new strategy determines how the resulting cash-flow deficit is funded.

The application must also make it possible to test whether the strategy actually helps. A spending reserve can improve behavioral comfort and reduce sales in down markets, but it also introduces cash drag and may not improve portfolio survival or terminal wealth. Results must therefore compare the new strategy against the existing single-portfolio strategy under common random numbers (§11).

---

## 2. Current application assessment

The `webapp-port` branch already has the right foundations. Verified facts about the code this feature builds on:

- **Engine:** `engine/simulation.py` (~380 lines). `run_simulation(params, guardrails=None)` is vectorized across paths with a Python loop over periods. Per-period order: guardrail multiplier (from start-of-period balance and/or previous period's return) → cash flows applied to balance → ruin check (`bal <= 0`, ruined paths zeroed and frozen) → returns applied to survivors → post-return balance recorded into `bal_over_time` (shape `(n_periods, n_paths)`; note the transpose relative to `port_ret`, which is `(n_paths, n_periods)`).
- **RNG:** one shared `np.random.default_rng(params.random_seed)` (simulation.py:53). Consumption order: the full `port_ret` matrix first, then one `rng.normal` per property. Playground/unplanned events consume **no** RNG draws — they are deterministic lumps injected into cash-flow arrays.
- **Cash flows:** prebuilt per-period arrays before the loop — `strict_out`, `lifestyle_out`, `gifts_out`, `disc_out`, `incomes`, `lump_in`/`lump_out_by_cat`, `rent_by_period`, with `playground_out`/`playground_in` tracked separately and already stripped from the funded-ratio PV lookahead.
- **Guardrails:** `engine/guardrails.py`. `VolatilityDiscretionaryScaling` reads only `port_ret[:, i-1]`. `FundedRatioGuardrail` reads the start-of-period balance plus precomputed PV arrays; confidence mode calibrates via an inner same-seed `run_simulation(params, guardrails=None)` pass plus isotonic regression.
- **Result:** a plain `dict` (not a class), keys assembled at simulation.py:288-309. Chart percentiles are computed downstream in `engine/figures.py`.
- **Scenario:** a plain JSON-shaped dict with `"$schema": "scenario.v1"` and **flat** portfolio fields (`mu`, `sigma`, `fat_tails_enabled`, `fat_tails_df`). No code branches on the schema string. There is **no JSON file persistence** — the scenario lives in a Dash `dcc.Store`; disk persistence is XLSX-only (`engine/params.py:230-364` write, `367-502` read, with defensive missing-field reads).
- **Historic mode:** `run_historic_scenario` (simulation.py:316) is a single deterministic path with a scalar balance, always annual.
- **Tests:** regression, smoke, return-model, guardrail, and persistence tests exist, including bit-exact `bal_over_time` assertions (`tests/test_engine_baseline.py:33`, `tests/test_funded_ratio_baseline.py:102`). A Playwright UI pipeline exists (`tests/ui/`, `scripts/run_ui_tests.sh`).

The principal limitation is structural: the simulation currently has one liquid balance and one liquid-portfolio return stream. The feature must therefore introduce a strategy abstraction over two liquid balances without duplicating the simulation engine.

### Architectural conclusion

Do **not** create a second, separate Monte Carlo implementation for bucket scenarios.

Refactor the liquid-portfolio portion of the existing simulation loop into a strategy layer with two implementations:

- `SinglePortfolioStrategy` — preserves current behavior exactly (bit-for-bit, verified by golden fixtures captured **before** the refactor).
- `TwoBucketStrategy` — manages growth and reserve balances.

All non-portfolio features — cash-flow schedules, guardrails, properties, playground shocks, funded-ratio calculations, results, persistence — continue to use the shared simulation pipeline.

---

## 3. Goals

### 3.1 Primary goals

1. Allow the user to enable or disable the two-bucket withdrawal strategy per scenario.
2. Size the spending reserve using a configurable number of years of **future positive portfolio-funded cash-flow gaps**, not gross household spending.
3. Simulate separate real returns for the growth bucket and the spending reserve.
4. Fund deficits from growth in favorable markets and from the reserve in adverse markets.
5. Refill the reserve using explicit, configurable, non-look-ahead rules.
6. Preserve all existing income, spending, gifts, lump-sum, property, guardrail, and playground-event behavior.
7. Expose the strategy's effects through clear charts and diagnostics.
8. Support fair side-by-side comparison with the existing single-portfolio strategy using common random numbers.
9. Maintain backward compatibility with existing scenarios and XLSX files.
10. Preserve vectorized performance; no Python loop over Monte Carlo paths.

### 3.2 Deferred to follow-up releases (schema reserved, no implementation — see §21)

- Per-category draw policies for gifts, planned lumps, and unplanned events.
- Recovery-aware high-water-mark market-state rule.
- Reverse rebalancing (excess reserve buys growth in downturns).
- Same-assets static-rebalancing benchmark comparator.
- Correlated growth/reserve returns.
- Pro-rata draw sources, fixed-amount refill, refill caps.

### 3.3 Non-goals for any release

- Tax-lot selection or account-specific withdrawal ordering.
- Modeling Israeli, U.S., or other tax law in detail.
- Treating dividends as a separate source of free return.
- Product recommendations (money-market funds, bonds, ETFs).
- A bond ladder with explicit maturity cash flows.
- Dynamic allocation inside the growth bucket.
- Market valuation forecasts.
- Perfect optimization of reserve size.
- A three-bucket strategy.
- Intraperiod market timing.

---

## 4. Product terminology

Avoid calling the feature merely "buckets" in code or UI — the guardrail machinery already uses "two-bucket scaling" for strict-vs-discretionary **spending** (simulation.py:108-111), and spending has categories (strict, lifestyle, gifts). Use these terms consistently:

| User-facing term | Internal term | Meaning |
|---|---|---|
| Single portfolio | `single_portfolio` | Current behavior: one liquid balance |
| Growth + Spending Reserve | `two_bucket` | New withdrawal strategy |
| Growth bucket | `growth` | Long-term, higher-volatility liquid portfolio |
| Spending reserve | `reserve` | Lower-volatility near-term funding pool |
| Portfolio-funded gap | `planned_gap` | Planned spending outflows not covered by planned inflows |
| Reserve coverage | `coverage_years` | Reserve balance divided by forecast portfolio-funded gaps |
| Refill | `refill` | Internal transfer from growth to reserve |
| Forced growth sale | `forced_sale` | Growth withdrawal in an adverse market state because the reserve was insufficient |

Neither bucket may be labeled "safe." Use "lower volatility" or "spending reserve."

---

## 5. Modeling principles

### 5.1 No look-ahead

The withdrawal source decision for period `t` must not use the return that will be generated in period `t`. The decision uses only information available at the beginning of the period: the previous completed period's growth return.

At the end of period `t`, the newly realized return **may** be used for the refill decision — the transferred money becomes part of ending balances available in period `t+1` and earns no return in period `t`.

### 5.2 Total return, not dividends

The growth return assumption represents total real return — price appreciation plus distributions, net of whatever fees and tax drag the user chooses to reflect. Dividends must not be modeled as additional income; doing so would double-count return. The same principle applies to the reserve return.

### 5.3 Transfers are not gains, losses, income, or spending

Transfers between growth and reserve:

- must sum to zero at the combined-liquid-portfolio level;
- must not affect household cash flow;
- must not affect funded-ratio liabilities;
- must not count as withdrawals delivered to the household;
- may be reported as transactions in diagnostics.

### 5.4 Portfolio ruin uses combined liquid wealth

A reserve balance of zero is **not** portfolio ruin. Ruin occurs only when combined liquid balance cannot fund the required external outflow:

```text
growth_balance + reserve_balance <= 0
```

The two-bucket accounting is arranged so that this condition fires under exactly the same arithmetic as the current `bal <= 0` check (§8.4 step E). Existing handling of properties and other illiquid assets is unchanged.

### 5.5 Common random numbers and RNG streams

Strategy comparisons must use identical growth return shocks, property return shocks, playground events, and all other stochastic components. Adding reserve-return sampling must not shift the RNG sequence used by any existing component.

**Corrected design (replaces the draft's `SeedSequence.spawn(4)` proposal, which would have changed the growth and property sequences and violated the bit-for-bit requirement):**

- The existing generator `np.random.default_rng(params.random_seed)` and its consumption order (`port_ret` block, then per-property normals) are **never touched**. Legacy streams stay on the legacy generator forever.
- New stochastic components get their own independently derived generators via an **append-only stream registry** in `engine/withdrawal_strategies.py`:

```python
# Streams are derived as np.random.default_rng(np.random.SeedSequence([seed, STREAM_ID])).
# NEVER renumber. Append only.
STREAM_RESERVE_RETURNS = 1
# future: STREAM_GROWTH_CORRELATED = 2, ...
```

Consequences, all automatic:

- Two-bucket mode's growth stream is identical to single mode's (same first draw from the same generator) — CRN across strategies for free.
- Adding or removing properties never shifts reserve returns, and the reserve stream can never shift growth/property draws.
- The `constant` reserve distribution consumes no draws at all.
- If `random_seed` is `None`, the reserve stream uses an unseeded `default_rng()` — CRN is impossible without a seed anyway.

---

## 6. Functional requirements

### 6.1 Strategy enablement

Add a scenario-level selector:

```text
Withdrawal strategy
(•) Single portfolio
( ) Growth + Spending Reserve
```

Requirements:

- Existing and imported scenarios default to `single_portfolio`.
- When `single_portfolio` is selected, outputs are bit-for-bit identical to the current engine (verified by golden fixtures, §14.1).
- Bucket-specific controls are hidden or disabled in single-portfolio mode, but entered values are preserved when toggling back and forth.
- Enabling the strategy immediately shows the estimated initial split (live preview, §12.1).

### 6.2 Initial reserve sizing

The reserve target equals the next `N` years of **positive projected portfolio-funded gaps**, not `N × gross annual spending`.

Per period, using the engine's existing prebuilt planned-cash-flow arrays (playground events always excluded — this falls out of the existing `playground_out`/`playground_in` stripping, the same exclusion the funded-ratio PV already applies at simulation.py:132-135):

```text
planned_inflow[t]  = incomes[t] + rent_by_period[t] + (lump_in[t] - playground_in[t])
planned_outflow[t] = per coverage_scope, see below
planned_gap[t]     = max(planned_outflow[t] - planned_inflow[t], 0)
reserve_target[t]  = sum(planned_gap over the next target_years worth of periods)
```

**Coverage scopes** (which planned outflows the reserve is sized to cover):

- `recurring_gap_only` (default) — strict + lifestyle spending bands only.
- `recurring_plus_scheduled_gifts` — adds the gifts band and gift-category lumps.
- `all_planned_outflows` — adds all scheduled negative lumps (strict-category lumps net of playground events, lifestyle, gifts).

Unplanned playground events are never included in the reserve target before they occur, under any scope.

In monthly mode the same computation runs over the next `target_years × 12` monthly periods (arrays are already per-period and `/12`-scaled). Fractional reserve years are allowed; the fractional tail period is prorated (§8.6).

**Initial allocation** at simulation start:

```text
initial_reserve = min(initial_total_liquid_portfolio, reserve_target[0])
initial_growth  = initial_total_liquid_portfolio - initial_reserve
```

If the target exceeds the initial liquid portfolio: cap the reserve at the available amount, start growth at zero, show a high-severity UI warning, and do not reject the scenario. If no planned future gap exists, the reserve starts at zero even when target years are positive (with a UI warning).

### 6.3 Reserve return model

The reserve has its own real-return model:

```yaml
reserve_return_model:
  distribution: constant | normal | student_t
  mean_real: float
  std_real: float
  student_t_df: float | null
```

Rules:

- `constant` requires `std_real = 0` and consumes no RNG draws.
- `normal` ignores `student_t_df`.
- `student_t` requires `df > 2` and reuses the engine's existing `sample_real_returns` (simulation.py:16-24), so it gets the same variance rescale and ±0.75 clip as growth returns.
- Parameters are annualized real arithmetic mean and standard deviation, consistent with the growth-return conventions. Monthly conversion uses the engine's existing conventions: geometric mean conversion `exp(log(1+mean)/12) - 1`, `sd/sqrt(12)`.
- The UI may provide descriptive presets, but all numeric assumptions remain visible and editable. No preset may be promised as risk-free or tax-optimal.

Reserve returns are independent of growth returns in this release. The schema reserves `correlation_with_growth` (must be `0.0`; any other value is a validation error) for a later release.

### 6.4 Market-state rule

```text
growth_market_is_favorable[t] = realized_growth_return[t-1] >= growth_return_threshold
```

Default threshold: `0%` real (annual). In the first period no prior return exists; the configured **first-period source** applies:

```text
First-period deficit source:
(•) Spending reserve first   (default)
( ) Growth first
```

(`pro_rata` is deferred; the schema value is rejected by validation in this release.)

The recovery-aware high-water-mark rule is deferred (§21). The schema field `market_state_rule` exists but only `previous_period_return` is accepted.

### 6.5 Deficit funding

The period's external net cash flow is calculated exactly as today — **after** guardrail adjustments and all scheduled/injected cash flows:

```text
funding_gap = max(-actual_external_net_cashflow, 0)
surplus     = max(actual_external_net_cashflow, 0)
```

Default deficit rule:

- Favorable growth state: growth first, reserve second.
- Adverse growth state: reserve first, growth second.
- If the preferred bucket is insufficient, automatically spill into the other bucket (spillover is always on in this release).
- Neither bucket balance may go negative.
- If combined liquid assets are insufficient, the existing ruin/shortfall behavior applies unchanged.

Vectorized implementation (per path, all NumPy masks — see §8.4 step E for the exact code): compute both draw orders, select by the favorable mask, and let any residual shortfall ride on the growth balance so that `combined = old_combined - gap` holds exactly and the loop's existing ruin check fires under identical arithmetic to single mode.

**Forced growth sale** is recorded when all of: market state adverse, funding gap > 0, reserve could not cover the full gap, growth withdrawal > 0.

### 6.6 Surplus allocation

When the external net cash flow is positive: deposit into the reserve up to the current target, then deposit the remainder into growth (`reserve_then_growth`). This is the only surplus rule in this release; `growth_only` and `pro_rata` are deferred. Positive lump sums follow the surplus rule.

### 6.7 Reserve target over time

The reserve target is rolling and forward-looking; near the end of the plan it naturally declines as fewer future periods remain (the rolling window tapers to zero).

The target is computed from the deterministic baseline planned cash-flow schedule, **precomputed once before the loop** as a vectorized rolling forward sum (§8.6) — an O(1) array lookup per period, identical across paths:

- the target is based on the baseline plan, before stochastic playground events;
- the target does not forecast future guardrail changes;
- actual current-period withdrawals do reflect the current path's guardrail multiplier;
- the target is identical across paths.

This avoids nested simulations and keeps the strategy explainable. A path-dependent target (applying the current spending multiplier forward) is a possible future option, not in this release.

### 6.8 Reserve refill policy

Refill is evaluated **after returns** at the configured cadence (annual only in this release). Using the just-realized return here is legal under §5.1 because the period's draw decision was already made from the prior period's return, and refilled money earns no current-period return.

Default policy:

```yaml
cadence: annual
target_coverage_years: 4.0        # reserve.target_years
trigger_coverage_years: 3.0       # reserve.refill_trigger_years
eligibility_rule: growth_return_at_or_above_threshold
growth_return_threshold_real: 0.0
amount_rule: to_target
```

Sequence at each refill opportunity (end of period `t`; in monthly mode only at sim-year boundaries, `(t+1) % 12 == 0`, including the first sim year whose return is fully realized by period 11):

1. Returns have been applied to both buckets.
2. Look up the **forward** trigger and target amounts for period `t+1` (precomputed arrays, index shift — at the plan horizon the forward window is empty and no refill occurs).
3. Eligible if: reserve balance (after returns) < trigger amount, AND the just-completed growth return meets the eligibility rule, AND the path is alive.
   - Annual mode: just-completed return = `port_ret[:, t]`. Monthly mode: compounded return of the just-completed sim year.
   - `eligibility_rule` values: `growth_return_at_or_above_threshold` (default), `always`, `never`.
4. Transfer from growth to reserve: `min(target_amount - reserve_balance, growth_balance)`, further capped by the amount rule.

**Amount rules** in this release:

- `to_target` (default) — restore the full target.
- `gains_only` — cap the transfer at the period's positive growth gain. **Precise definition:** the exposed balance is the growth balance *after* this period's cash flows and ruin zeroing, *immediately before* the return multiply. Period gain = `exposed_balance × growth_return[t]`. In monthly mode the cap is the sum of the 12 monthly gains of the just-completed sim year (a running accumulator, reset at each refill opportunity). Never infer gains from ending-minus-starting balance — cash flows would distort it.
- `none` — never refill; useful as a diagnostic.

`fixed_amount` and refill caps are deferred (§21).

### 6.9 Reverse rebalancing — deferred

Deferred to a follow-up (§21). The schema field `allow_reverse_rebalance` exists, defaults to `false`, and any other value is a validation error in this release.

### 6.10 Treatment of scheduled gifts and lump sums — deferred

In this release, **all** cash flows aggregate into the single external net cash flow and follow the strategy rule — there are no per-category draw sources. The `cashflow_policy` schema block exists for forward compatibility with all fields defaulting to `strategy_rule` (describing actual behavior; the draft's recommended `growth_first` defaults move to the follow-up that implements them). Non-default values are validation errors.

Guardrail scaling of gifts continues to occur before funding, exactly as today (the strategy sees the post-guardrail net flow).

### 6.11 Monthly-mode behavior

- Cash flows and investment returns continue monthly, exactly as today.
- The draw-source market state is **fixed for each simulation year**, using the previous completed sim year's compounded growth return (`prod(1 + monthly returns) - 1`) against the annual threshold. No monthly threshold conversion.
- The first sim year (all 12 periods) uses the configured first-period source.
- Reserve refill is evaluated annually, at sim-year boundaries.
- Monthly reserve withdrawals are allowed throughout the year.
- The source does not switch on monthly return noise.

The favorable-state matrix is precomputed vectorized before the loop (one cumprod pass; ~4 MB bool matrix at 10k paths × 432 months). A `rolling 12-month | monthly` decision cadence is a possible future option.

### 6.12 Guardrail interaction

The two mechanisms have distinct responsibilities: guardrails calculate path-specific allowed spending; the withdrawal strategy decides which bucket funds the result.

**Verified: no changes to `engine/guardrails.py` are needed.**

- The funded-ratio guardrail receives `bal = strategy.combined_balance()` (start-of-period combined liquid wealth) — same convention as today, one changed argument at the call site.
- The volatility/return-based guardrail already reads `port_ret[:, i-1]`, which **is** the growth-bucket stream in two-bucket mode. No blended return is ever used.
- The reserve target is not a liability and is not added to the spending PV (satisfied by doing nothing — the PV context is built from cash-flow arrays only).
- Confidence-mode guardrail calibration runs its inner baseline via `run_simulation(params, guardrails=None)`; because the strategy config rides on `params`, the baseline automatically uses the **selected strategy** with guardrails off. It cannot silently revert to single-portfolio behavior. The calibration regresses success on start-of-period *combined* wealth, consistent with the runtime guardrail also reading combined wealth.
- Existing guarantee/floor logic is unchanged. The strategy never changes spending amounts directly.

### 6.13 Historic simulation

`run_historic_scenario` (simulation.py:316) is a deterministic single path, always annual. Two-bucket support **ships in this release** by reusing `TwoBucketStrategy` with `n_paths=1`:

- historic growth factors drive the growth bucket (`port_ret = factors - 1`);
- the reserve earns its configured `mean_real` deterministically (`reserve_ret = full(mean_real)`);
- the same no-look-ahead draw and refill rules apply (rule parity for free);
- the UI must state visibly that reserve returns are configured, not historical.

The single-mode historic path is untouched.

---

## 7. Data model

### 7.1 Scenario schema

**The `$schema` string stays `"scenario.v1"`** (change from draft: no v2 bump — nothing in the codebase branches on the version string, so a bump buys nothing and forces churn). The existing **flat** portfolio fields (`mu`, `sigma`, `fat_tails_enabled`, `fat_tails_df`) are **not** renamed or nested (change from draft: the `growth_return_model` migration is dropped — it would ripple through hydration, edit-collection, XLSX, and tests for zero behavioral gain). The flat fields *are* the growth bucket's return model.

New optional top-level block; when absent, the scenario is `single_portfolio` — no migration needed:

```json
{
  "$schema": "scenario.v1",
  "name": "...",
  "portfolio": { "initial_value": 9000000, "mu": 0.05, "sigma": 0.16,
                 "fat_tails_enabled": true, "fat_tails_df": 5 },
  "withdrawal_strategy": {
    "type": "two_bucket",
    "reserve": {
      "target_years": 4.0,
      "refill_trigger_years": 3.0,
      "coverage_scope": "recurring_gap_only",
      "return_model": {
        "distribution": "normal",
        "mean_real": 0.01,
        "std_real": 0.03,
        "student_t_df": null
      }
    },
    "draw_policy": {
      "market_state_rule": "previous_period_return",
      "growth_return_threshold_real": 0.0,
      "first_period_source": "reserve"
    },
    "refill_policy": {
      "cadence": "annual",
      "eligibility_rule": "growth_return_at_or_above_threshold",
      "growth_return_threshold_real": 0.0,
      "amount_rule": "to_target"
    },
    "cashflow_policy": {
      "surplus_allocation": "reserve_then_growth",
      "planned_gift_draw": "strategy_rule",
      "planned_lump_draw": "strategy_rule",
      "unplanned_event_draw": "strategy_rule"
    },
    "correlation_with_growth": 0.0
  },
  "spending_bands": [], "income_bands": [], "lumps": [], "properties": []
}
```

**Integration hazard (must-fix, owned by the UI phase, regression-tested by the engine phases):** `collect_edits` (webapp/callbacks.py:385-453) rebuilds the scenario dict from scratch on every widget edit, copying forward only `$schema` and `name`. The `withdrawal_strategy` block must be threaded through (from widgets or from `prev_scenario`) or it will be silently dropped on the first edit. A regression test asserting the block survives a `collect_edits` round trip is required.

### 7.2 Python configuration objects

Frozen dataclasses in the new `engine/withdrawal_strategies.py`. Deferred fields default to values describing **actual MVP behavior**, so a saved scenario never claims behavior the engine doesn't implement:

```python
@dataclass(frozen=True)
class ReturnModelConfig:
    distribution: str = "normal"          # constant | normal | student_t
    mean_real: float = 0.0
    std_real: float = 0.0
    student_t_df: float | None = None

@dataclass(frozen=True)
class ReserveConfig:
    target_years: float = 4.0
    refill_trigger_years: float = 3.0
    coverage_scope: str = "recurring_gap_only"
    return_model: ReturnModelConfig = ReturnModelConfig()

@dataclass(frozen=True)
class DrawPolicyConfig:
    market_state_rule: str = "previous_period_return"   # "recovery_aware" deferred
    growth_return_threshold_real: float = 0.0
    first_period_source: str = "reserve"                # | "growth"; "pro_rata" deferred

@dataclass(frozen=True)
class RefillPolicyConfig:
    cadence: str = "annual"
    eligibility_rule: str = "growth_return_at_or_above_threshold"  # | always | never
    growth_return_threshold_real: float = 0.0
    amount_rule: str = "to_target"                      # | gains_only | none; "fixed_amount" deferred

@dataclass(frozen=True)
class CashflowPolicyConfig:                             # ALL deferred — schema round-trip only
    surplus_allocation: str = "reserve_then_growth"
    planned_gift_draw: str = "strategy_rule"
    planned_lump_draw: str = "strategy_rule"
    unplanned_event_draw: str = "strategy_rule"

@dataclass(frozen=True)
class WithdrawalStrategyConfig:
    type: str = "single_portfolio"                      # | two_bucket
    reserve: ReserveConfig = ReserveConfig()
    draw_policy: DrawPolicyConfig = DrawPolicyConfig()
    refill_policy: RefillPolicyConfig = RefillPolicyConfig()
    cashflow_policy: CashflowPolicyConfig = CashflowPolicyConfig()
    correlation_with_growth: float = 0.0                # reserved; must be 0.0
```

`SimulationParams` (engine/params.py:50) gains exactly **one** new optional field:

```python
withdrawal_strategy: Optional[WithdrawalStrategyConfig] = None   # None ≡ single portfolio
```

so every existing constructor call site keeps working. `from_scenario` (params.py:74-128) parses `scenario.get("withdrawal_strategy")`; absent → `None`.

### 7.3 Validation

Extend `validate_scenario` (engine/params.py:191), gated on the block being present:

- `type` in `{single_portfolio, two_bucket}`; the rest applies only to `two_bucket`.
- `target_years >= 0`; `0 <= refill_trigger_years <= target_years`.
- `coverage_scope`, `first_period_source`, `eligibility_rule`, `amount_rule`, `distribution` restricted to the **implemented** enums.
- `constant` ⇒ `std_real == 0`; `student_t` ⇒ `student_t_df > 2`; `std_real >= 0`.
- **Deferred-feature guard:** `correlation_with_growth == 0.0`, `market_state_rule == "previous_period_return"`, `cadence == "annual"`, all `cashflow_policy` fields equal to their defaults. Any non-default deferred value produces an explicit "not implemented in this version" **error** — silent acceptance would misrepresent results.

---

## 8. Engine design

### 8.1 Module layout

New module `engine/withdrawal_strategies.py` containing: the configuration dataclasses (§7.2), the RNG stream registry (§5.5), the two strategy classes, the reserve-target pure functions (§8.6), and the reserve return sampler. No UI concepts in this module.

### 8.2 Strategy interface (corrected from draft)

The draft's five-method Protocol (`initialize / apply_external_cashflow / apply_returns / rebalance_after_returns / combined_balance`) with a separate `LiquidStrategyState` does not fit the real loop: ruin bookkeeping is loop-owned (it feeds ruin distributions, ruin tracks, and confidence calibration), the strategy needs the full pre-sampled return matrices (the volatility guardrail and market-state rule both index `port_ret` directly), and a separate state object would have exactly one consumer. Corrected interface — the strategy instance **is** the state, constructed once per `run_simulation` call:

```python
class SinglePortfolioStrategy:
    def __init__(self, params, port_ret, cashflow_ctx=None):
        self.bal = np.full(params.n_paths, params.initial_portfolio)
        self.port_ret = port_ret

    def combined_balance(self) -> np.ndarray: ...   # live array; callers must not mutate
    def apply_cashflow(self, i, net) -> None: ...   # exactly current line 237
    def mark_ruined(self, ruined) -> None: ...      # exactly current line 253
    def apply_returns(self, i, alive) -> None: ...  # exactly current line 254
    def end_of_period(self, i, alive) -> None: ...  # single: no-op; two-bucket: refill
    def record(self, i) -> None: ...                # single: no-op; two-bucket: histories/aggregates
    def extra_results(self) -> dict: ...            # single: {}; two-bucket: new result keys
```

`TwoBucketStrategy` has the same signature and additionally receives `reserve_ret` and a small frozen `CashflowContext` (the prebuilt planned-cash-flow arrays needed for the gap schedule). Duck typing is sufficient; a `Protocol` may be added for type checking.

### 8.3 Call-site mapping in `run_simulation`

This table makes the Phase-3 refactor mechanical and reviewable op-for-op. For single mode, every floating-point operation is the same op on the same values in the same order, so `bal_over_time` stays bit-for-bit.

| Current line | Current op | Becomes |
|---|---|---|
| 185 | `bal = np.full(...)` | `strategy = make_strategy(params, port_ret, reserve_ret, cashflow_ctx)` — constructed after line 199 so `port_ret` exists (moving the init below return sampling is value-neutral; no RNG involved) |
| 212 | `start_bal = bal.copy()` | `start_bal = strategy.combined_balance().copy()` |
| 231 | `h.multiplier(..., bal=bal)` | `bal=strategy.combined_balance()` |
| 237 | `bal = bal + cash_delta_bal` | `strategy.apply_cashflow(i, cash_delta_bal)` |
| 250-252 | ruin masks from `bal` | masks from `strategy.combined_balance()`; loop keeps `ruined`/`ruin_time` |
| 253 | `bal[ruined] = 0.0` | `strategy.mark_ruined(ruined)` |
| 254 | `bal[~ruined] *= (1+port_ret[~ruined,i])` | `strategy.apply_returns(i, ~ruined)` |
| new, between 254 and 255 | — | `strategy.end_of_period(i, ~ruined)`; `strategy.record(i)` |
| 255 | `bal_over_time[i,:] = bal.copy()` | `= strategy.combined_balance().copy()` |
| 266, 293 | `estate = bal + prop_tot`; `"final_portfolio": bal` | use `strategy.combined_balance()` |
| 288-309 | result dict | `result.update(strategy.extra_results())` at the end |

The withdrawal-rate stats block (lines 242-248) stays in the loop untouched — it reads only `cash_delta` and `start_bal`.

### 8.4 Two-bucket period order

`TwoBucketStrategy` state: `growth`, `reserve` (`(n_paths,)` float64), `port_ret`, `reserve_ret` (`(n_paths, n_periods)`), precomputed `gap` / `target_by_period` / `trigger_by_period` (`(n_periods,)`), precomputed `favorable` (`(n_paths, n_periods)` bool), config, history/aggregate buffers. Ordered steps, anchored to the loop:

- **A. Start-of-period snapshot** (loop): `start_bal = combined_balance().copy()` — withdrawal-stat definition unchanged.
- **B. Guardrail multiplier** (loop): funded ratio gets start-of-period combined balance; volatility guardrail reads `port_ret[:, i-1]` (the growth stream).
- **C. Net external cash flow** (loop, unchanged): `net = incomes[i] + rent_i + lump_in[i] - (strict_out[i] + disc_out[i] * mult)`.
- **D. Market state** (precomputed in `__init__`): annual — `favorable[:, i] = port_ret[:, i-1] >= threshold` for `i >= 1`; period 0 (annual) / sim year 0 (monthly) from `first_period_source` (`reserve` ⇒ all-False). Monthly — fixed per sim year from the previous sim year's compounded return (§6.11).
- **E. Deficit/surplus split** (inside `apply_cashflow`, fully vectorized):

  ```python
  gap = np.maximum(-net, 0.0); surplus = np.maximum(net, 0.0)
  pref_growth = self.favorable[:, i]
  r1 = np.minimum(gap, self.reserve); g1 = np.minimum(gap - r1, self.growth)   # reserve-first
  g2 = np.minimum(gap, self.growth); r2 = np.minimum(gap - g2, self.reserve)   # growth-first
  from_res = np.where(pref_growth, r2, r1)
  from_gro = np.where(pref_growth, g2, g1)
  resid = gap - from_res - from_gro            # > 0 only when combined is insufficient
  self.growth  = self.growth - from_gro - resid   # residual rides on growth → combined ≤ 0
  self.reserve = self.reserve - from_res
  to_res = np.minimum(surplus, np.maximum(self.target_by_period[i] - self.reserve, 0.0))
  self.reserve = self.reserve + to_res
  self.growth  = self.growth + (surplus - to_res)
  forced = (~pref_growth) & (from_gro > 0)     # forced-sale diagnostic
  ```

  Putting the residual on growth makes `combined = old_combined - gap` **exactly**, so the loop's ruin condition fires under precisely the same arithmetic as single mode, and the reserve never goes negative.
- **F. Ruin** (loop): masks from combined balance; `mark_ruined` zeroes **both** buckets; ruined paths stay frozen (all subsequent ops are min/max/add against 0 with `alive` masking on returns). Reserve depletion alone is not ruin — falls out naturally.
- **G. Returns to survivors** (`apply_returns`):

  ```python
  self._growth_pre_ret = self.growth.copy()               # "exposed" balance for gains_only
  self.growth[alive]  *= (1 + self.port_ret[alive, i])
  self.reserve[alive] *= (1 + self.reserve_ret[alive, i])
  self._year_gain += self._growth_pre_ret * self.port_ret[:, i]   # monthly accumulator; annual: overwrite
  ```

- **H. Refill** (`end_of_period`, after returns; annual mode every period, monthly mode only when `(i+1) % 12 == 0`): per §6.8 — forward `t+1` trigger/target lookups, eligibility mask, transfer capped by need / growth balance / amount rule; reset `_year_gain`. Refilled money earns no current-period return (transfer happens post-return; §5.1 satisfied). Transfers sum to zero on combined wealth by construction.
- **I. Record** (`record(i)`): `growth_over_time[i,:] = growth` (post-refill, i.e., the split carried into `i+1`); aggregate accumulators (refill/forced-sale/depletion probabilities per period, per-path counters). Combined recording stays the loop's `bal_over_time`.
- **J. Carry-over:** none needed — "previous return" is an index into the precomputed matrices.

Document this order in code comments and pin it with the deterministic tests (§14.2).

### 8.5 Return sampling

```python
def sample_reserve_returns(cfg: ReturnModelConfig, n_paths, n_periods, seed, annual) -> np.ndarray
```

Applies the engine's existing monthly conversion, returns a constant matrix without touching RNG for `constant`, otherwise draws from the registry stream (§5.5), reusing `sample_real_returns` for Student-t so the variance rescale and clip match growth conventions.

### 8.6 Planned gap schedule helpers

Two pure functions (unit-testable without the engine):

```python
def planned_gap_schedule(*, spend_strict, spend_lifestyle, spend_gifts,
                         lump_out_by_cat, playground_out, incomes,
                         rent_by_period, lump_in, playground_in,
                         coverage_scope) -> np.ndarray   # (n_periods,) positive planned gap
```

Fed directly by the arrays already built at simulation.py:70-135 (already per-period and monthly-scaled). Playground exclusion falls out of the existing `playground_out`/`playground_in` stripping. Gift-category lumps can never be playground events (playground lumps are always `strict`, params.py:99-100), so no stripping is needed there.

```python
def rolling_gap_target(gap: np.ndarray, years: float, periods_per_year: int) -> np.ndarray:
    """target[t] = sum(gap[t : t+w]) + frac * gap[t+w]; zero past the horizon."""
    w_f = years * periods_per_year
    w, frac = int(w_f), w_f - int(w_f)
    c = np.concatenate([[0.0], np.cumsum(gap)])
    t = np.arange(len(gap))
    full = c[np.minimum(t + w, len(gap))] - c[t]
    tail = frac * np.where(t + w < len(gap), gap[np.minimum(t + w, len(gap) - 1)], 0.0)
    return full + tail
```

Vectorized cumsum; O(T) once per run. Three arrays are precomputed before the loop (`gap`, target, trigger); the per-period cost inside the loop is an O(1) lookup.

### 8.7 Accounting invariant

Per period, on every path (enforced in tests; available as a debug assertion):

```text
combined[t] == combined[t-1] + net_external_cashflow[t]
             + growth_gain[t] + reserve_gain[t]
```

Internal transfers cancel. Do not hide accounting discrepancies by clamping before the diagnostic is calculated.

### 8.8 Memory and performance

Required stored histories beyond today's: `growth_over_time` only (reserve is derived, §9). Event diagnostics are aggregated per period plus per-path lifetime counters — **no** full boolean event matrices.

Monthly / 10k paths / 432 periods: one float64 matrix ≈ 34.6 MB. Today's run holds ≈ 104-138 MB of matrices; two-bucket adds `reserve_ret` (34.6 MB) + `growth_over_time` (34.6 MB) + `favorable` bool (4.3 MB) ≈ **+74 MB in-run (~+55%)**. `reserve_ret` and `favorable` are engine-internal and freed after the run, so the `RESULTS_CACHE` increment is only `growth_over_time` (+34.6 MB per cached run).

Requirements:

- no Python loop over Monte Carlo paths (period loop is fine, as today);
- annual 10,000 × 40 simulations remain interactive;
- benchmark single vs two-bucket runtime and memory; record in the Phase-5 PR;
- target: two-bucket runtime ≤ 2× the single-portfolio baseline for the same scenario shape (expected ~2× per-period array ops, so within budget).

---

## 9. Result model

The engine result is a plain dict. New keys are contributed by `strategy.extra_results()` — in single mode **zero** new keys are added except `withdrawal_strategy_type: "single_portfolio"`, preserving the existing contract:

| Key | Shape | Stored or derived |
|---|---|---|
| `withdrawal_strategy_type` | scalar | stored in both modes |
| `growth_over_time` | `(n_periods, n_paths)` — same transposed layout as `bal_over_time` | **stored** (the one new matrix) |
| reserve over time | — | **derived**: `bal_over_time - growth_over_time` (≈1-ulp fp noise; clamp at 0 for display only) |
| `reserve_target_by_period`, `reserve_trigger_by_period`, `planned_gap_by_period` | `(n_periods,)` | stored |
| `refill_probability_by_period`, `forced_sale_probability_by_period`, `reserve_depletion_probability_by_period` | `(n_periods,)` | stored (aggregated in-loop) |
| `refill_count_per_path`, `forced_sale_count_per_path`, `reserve_depleted_ever_per_path`, `refill_total_per_path`, `deficit_from_reserve_per_path`, `deficit_from_growth_per_path` | `(n_paths,)` | stored |
| `final_growth`, `final_reserve` | `(n_paths,)` | stored |
| `comparison` | small dict | attached by `execute_run` (§11), not by the engine |

---

## 10. Metrics and diagnostics

### Reserve behavior

- median reserve balance by age;
- median reserve coverage (years) by age;
- probability the reserve is depleted at least once; depletion probability by age;
- median number of refills; average total amount transferred into the reserve;
- median share of lifetime deficits funded from the reserve;
- reserve share of total liquid assets (cash-drag indicator).

### Sequence-risk diagnostics

- probability of at least one forced growth sale in an adverse state; forced-sale probability by age;
- number of forced growth sales; amount sold from growth during adverse states;
- longest consecutive run of periods funded primarily from the reserve.

### Comparison deltas (when comparison is enabled)

- change in ruin probability;
- change in 10th-percentile and median terminal wealth;
- change in lifetime spending delivered;
- change in forced down-market growth sales;
- change in guardrail cuts;
- change in average reserve allocation.

Do not label any delta universally "better." Present tradeoffs (§20).

---

## 11. Comparison mode

Add a results toggle, default **on** when a two-bucket scenario runs:

```text
Compare with current single-portfolio strategy
```

**Where it lives:** `execute_run` (webapp/callbacks.py:206-248), **not** inside `run_simulation` — the engine already self-recurses for confidence calibration, and `execute_run` already orchestrates auxiliary runs and owns the cache.

```python
results = run_simulation(params, guardrails=guardrails)
if two_bucket and compare_enabled:
    comp_params = dataclasses.replace(params, withdrawal_strategy=None)
    comp = run_simulation(comp_params, guardrails=guardrails)   # same seed → same shocks
    results["comparison"] = build_comparison(results, comp)     # compact extract; comp freed
```

The comparator uses identical initial total liquid wealth, the same growth-return shock matrix (automatic under §5.5), the same cash flows, events, properties, and guardrail settings, and the current single-balance logic. It must be labeled clearly: all assets in the comparator receive the growth return model.

`build_comparison` stores only compact data: the comparator's summary, ruin probability, terminal-wealth percentiles, guardrail stats, and per-period p10/p50/p90 of its `bal_over_time` (3 × n_periods) for the overlay chart. Only the main result enters `RESULTS_CACHE`.

**Run-count accounting:** worst case (two-bucket + comparison + confidence guardrail) = 4 engine runs — main + its calibration baseline + comparator + its calibration baseline; 2 runs without a confidence guardrail. The two calibrations cannot soundly share (wealth-needed thresholds depend on strategy dynamics). Escape hatch if 4 ever hurts: pass a precomputed `fr_by_confidence` into `run_simulation` — future work.

The same-assets static-rebalancing benchmark (isolating asset-mix effect from withdrawal-rule effect) is the **highest-priority follow-up** (§21): without it, users may attribute to the withdrawal policy an outcome actually caused by the different asset mix.

---

## 12. UX requirements

### 12.1 Placement and builder controls

Follow the established guardrail template (commit b29cbb2): **strategy configuration in the Plan view** (new section in the Portfolio tab, layout.py:167-237 pattern), **run-level toggles on the Dashboard** (comparison toggle next to the Run controls, like `switch-guardrails-enabled`, layout.py:93-96).

Strategy selector:

```text
Withdrawal strategy
[ Single portfolio ]  [ Growth + Spending Reserve ]
```

When Growth + Spending Reserve is selected, show four cards:

**Card 1 — Reserve size:** target reserve years; refill trigger years; coverage scope; first-period draw source. With a live preview:

```text
Estimated spending reserve: ₪1,280,000
Covers projected portfolio-funded gaps from age 51 through age 55.
Initial growth bucket: ₪7,720,000
Reserve share of liquid portfolio: 14.2%
```

The preview recomputes from the current scenario schedules whenever spending, income, gifts, lumps, properties, start age, or reserve settings change. **Implementation:** follow the `update_return_distribution` live-preview pattern (callbacks.py:673+) — a dedicated callback computing a cheap deterministic estimate, with the estimator (`planned_gap_schedule` + `rolling_gap_target`) living in `engine/` so it cannot drift from the real run.

**Card 2 — Reserve investment assumptions:** return distribution; expected real return; real volatility; Student-t df when applicable. Helper text: "These assumptions apply only to the spending reserve. Enter the total real return net of the fees and tax drag you intend to model."

**Card 3 — Draw policy:** real-return threshold; first-period source. (Market-state rule is fixed to previous-period return in this release; favorable/adverse sources are fixed to growth/reserve; spillover always on — show these as static explanatory text, not controls.)

**Card 4 — Refill policy:** eligibility rule; refill threshold; amount rule. Plain-language preview:

```text
At year-end, if reserve coverage is below 3.0 years and the growth
bucket had a nonnegative real return, transfer enough from Growth to
restore 4.0 years of projected portfolio-funded gaps.
```

### 12.2 Validation and warnings

Inline validation errors (block the run, from §7.3): trigger years > target years; negative target years; Student-t df ≤ 2; negative volatility; deferred-feature values.

Warnings (explain consequences, never block valid scenarios):

- reserve target consumes a large share of liquid assets (configurable threshold);
- reserve target exceeds available liquid assets (starts capped, growth at zero);
- no projected funding gap exists, so the reserve starts at zero;
- configured reserve volatility is close to or above growth volatility;
- planned gifts/lumps are excluded from the reserve target under the selected scope;
- historic mode uses configured (not historical) reserve returns;
- the market-state threshold is real, not nominal;
- the first period has no prior return and uses the selected fallback source.

### 12.3 Results page

**KPI row** (extend `_stat_tiles`, callbacks.py:159-203, using `build_stat_tile`): success/ruin probability; median terminal portfolio; reserve depletion probability; forced down-market growth-sale probability; median number of refills. When comparison is enabled, show deltas under each metric — same pattern as the existing "X% with policy · Y% without" tiles.

**Charts** (each via `build_chart_card`; conditionally-shown strategy cards use the `div-historic-cards` injection-slot pattern, callbacks.py:874-885):

1. **Combined portfolio over time** — retain the current chart.
2. **Bucket balances over time** — clone the `fig_guardrail_multiplier` band template (figures.py:320-354): median growth and median reserve lines with percentile bands, plus the deterministic reserve-target line.
3. **Funding source by age** — share of deficits funded from reserve vs growth.
4. **Strategy events by age** — probability of refill, reserve depletion, forced growth sale (the three per-period probability arrays).
5. **Terminal outcome comparison** — single vs two-bucket distributions (comparison mode).

Never stack percentile bands to imply balances add across different percentile paths: plot median growth and median reserve separately with a note, or offer a path-percentile selector based on combined wealth showing that same path's bucket balances.

**Explanation panel** — a generated neutral summary:

```text
In this scenario, the reserve changed the probability of selling growth
assets in a down market from X% to Y%, while median terminal wealth
changed from A to B. Portfolio-ruin probability changed from C% to D%.
```

Never claim causality beyond the compared simulation assumptions.

### 12.4 Accessibility, theming, usability

- All controls have labels and helper text (`html.Small` + `text-muted` convention); tooltips for "real return," "portfolio-funded gap," "reserve trigger," "forced growth sale."
- Do not rely on color alone to distinguish buckets or market states.
- **Every new trace color must be registered in `engine/theme.py` `_DARK_STATIC`/`_DARK_SHADE_BASES` (theme.py:106-129)** or it will not invert in dark mode.
- Currency formatting follows the existing `f"₪{x:,.0f}"` convention.
- Preserve entered values when toggling temporarily back to single mode.
- Do not hide advanced values from scenario export.

---

## 13. Persistence compatibility

**Corrected from draft: there is no JSON file persistence.** The scenario dict lives in a Dash store; the "JSON" requirements reduce to (a) the dict round-tripping through `collect_edits` (§7.1 hazard) and (b) `from_scenario` defaulting an absent block to single portfolio.

### XLSX

Maintain the current workbook format, adding optional **row-0 scalar columns** (append after `random_seed` in `scenario_to_xlsx`, params.py:324-336; defensive `pd.notna` reads in `_scenario_from_df`, params.py:372-385 — exactly the existing `mu`/`sigma` pattern):

```text
withdrawal_strategy_type, reserve_target_years, reserve_trigger_years,
reserve_coverage_scope, reserve_return_distribution, reserve_return_mean,
reserve_return_std, reserve_return_df, draw_market_threshold,
draw_first_period_source, refill_eligibility, refill_threshold,
refill_amount_rule
```

- Workbooks without these columns load as single portfolio (all existing files remain importable).
- Deferred/`cashflow_policy` fields are not exported to XLSX (the scenario dict preserves them).
- Do not repurpose existing columns.
- Round-trip tests for both strategy types (§14.1).

---

## 14. Testing plan

### 14.1 Regression and golden fixtures

**Golden-fixture capture procedure (must land before any engine edit — Phase 1):**

1. New `scripts/capture_golden_fixtures.py` runs `run_simulation` on 4 pinned param sets: (a) annual, no guardrails (reuse `make_params` from tests/test_engine_baseline.py); (b) annual + volatility guardrail; (c) annual + funded-ratio **confidence** mode (exercises the inner-baseline path); (d) monthly, `n_paths=500`. Saves `np.savez_compressed` of `bal_over_time`, `prop_over_time`, `final_portfolio`, `ruined`, withdrawal-rate stats, plus `summary` JSON, into `tests/fixtures/golden/`.
2. New `tests/test_golden_single_mode.py` asserts `np.testing.assert_array_equal` (bit-exact) against the fixtures.
3. Fixtures are regenerated **only** on a documented numpy-version bump, never during the feature phases. Pin numpy in the capture PR.

Plus, unmodified: existing engine baselines (`test_engine_baseline.py`, `test_guardrails.py`, `test_funded_ratio_baseline.py`, `test_funded_ratio_guardrail.py`, `test_confidence_guardrail.py`), scenario-compat extensions (`test_xlsx_roundtrip.py`, `test_validation.py`: absent block → single; new columns round-trip; old workbook loads), smoke (`test_webapp_smoke.py`, `test_smoke_checklist.py`), and a new `collect_edits` passthrough regression test.

### 14.2 Deterministic strategy tests — new `tests/test_two_bucket_strategy.py`

Two layers: (i) **unit** tests constructing `TwoBucketStrategy` directly with hand-built 2-4-path `port_ret`/`reserve_ret` matrices (mixed up/down sequences give exact assertions); (ii) **integration** through `run_simulation` with `real_return_sd=0, fat_tails_df=None` (fully deterministic) and constant reserve returns, hand-computed balances.

1. Positive previous growth return causes growth-first draw.
2. Negative previous growth return causes reserve-first draw.
3. First period uses the configured fallback source.
4. Preferred-bucket insufficiency spills into the other bucket.
5. Reserve depletion alone does not mark ruin.
6. Combined insufficiency marks ruin with existing semantics (both buckets zeroed, frozen).
7. Surplus fills reserve to target, then growth.
8. Refill occurs only below the trigger.
9. Refill does not occur when eligibility fails.
10. `to_target` reaches the target when growth has enough funds.
11. `gains_only` never transfers more than the calculated positive gain.
12. Internal transfers preserve combined wealth (accounting invariant, §8.7).
13. Returns apply only to the balance exposed in each bucket.
14. Refill-transferred money earns no current-period return.
15. The same-period return is never used for the same-period draw decision.

(High-water-mark and reverse-rebalance tests are deferred with their features.)

### 14.3 Reserve-target tests — new `tests/test_reserve_target.py`

Pure-function tests: scope inclusion/exclusion per §6.2 (playground always excluded; gifts per scope); rolling-window taper at end of plan; fractional years; monthly ≡ annual for equivalent flat schedules; zero-gap plans produce zero targets.

### 14.4 Guardrail integration — extend `tests/test_two_bucket_strategy.py`

1. Funded ratio uses growth + reserve (invariant: with identical growth/reserve returns and zero volatility, two-bucket multipliers equal single-mode multipliers).
2. Volatility guardrail fires on a growth crash while the reserve stays constant.
3. Guardrail-adjusted spending changes the actual funded amount.
4. Gift multipliers apply before funding.
5. Confidence calibration reflects the selected strategy (baseline_summary differs when the reserve mean differs).

### 14.5 RNG tests — new `tests/test_two_bucket_rng.py`

1. **Killer test:** `target_years=0` two-bucket run has `growth_over_time` bit-equal to the single run's `bal_over_time` (the §8.4-E arithmetic makes this exact, proving growth-stream identity).
2. `prop_over_time` bit-equal single vs two-bucket (property stream unshifted).
3. `sample_reserve_returns` reproducible from seed alone; unaffected by property count.
4. Stream-registry constants pinned.
5. Same scenario + seed reproduces every result.

### 14.6 Statistical sanity — new `tests/test_two_bucket_sanity.py`

1. Zero reserve years ≡ single mode.
2. With identical growth/reserve returns, transfer decisions neither create nor destroy wealth (combined path equal under `to_target` vs `none`).
3. Zero volatility and no gaps ⇒ deterministic compounding.
4. Reserve return sample moments meet the existing tolerance patterns (`test_return_distribution.py`).
5. Larger reserve years ⇒ higher average reserve allocation (no monotonic claim about success).

### 14.7 UI and persistence tests (Playwright, via `tests/ui/journeys.py` helpers only)

1. Selecting dual mode reveals the cards; single mode hides them but preserves values.
2. Live initial-split preview updates when schedules change.
3. Invalid trigger/target combinations block the run with a clear error.
4. XLSX round trip preserves all strategy fields; old files load as single portfolio.
5. Comparison mode shows deltas; charts handle a zero reserve gracefully.
6. Save → reload → run works end to end.

---

## 15. Acceptance criteria

1. A user can enable the strategy, configure it, save/reload it, and run it in the web app.
2. Single-portfolio mode passes all existing tests **and the golden fixtures bit-exact**.
3. The dual strategy works with current schedules, guardrails, properties, and playground shocks.
4. No withdrawal decision uses a future return.
5. The reserve is sized from forecast portfolio-funded gaps.
6. The simulation records separate bucket balances and the combined balance.
7. Refills and spillovers obey the documented rules.
8. Combined accounting reconciles on every tested path and period.
9. Results expose reserve depletion, refills, and forced down-market growth sales.
10. Comparison mode shares all applicable stochastic paths.
11. Old scenarios and XLSX files load as single portfolio.
12. No Python loop over Monte Carlo paths.
13. Runtime stays within 2× the current baseline for the same simulation shape (benchmarked).
    Recorded via `scripts/benchmark_two_bucket.py` (10k paths, M-series Mac):
    annual 0.02s → 0.03s (1.33×); monthly 0.20s → 0.32s (1.57×). Both well under 2×.
14. Documentation explains return timing, reserve sizing, cash-flow scope, and exclusions.
15. The app never presents the strategy as a guaranteed improvement.

---

## 16. Implementation plan and LLM delegation

Workflow: **Fable** (session model) writes each phase spec (compact, file-pointer style per the `delegate-local` skill) and gates every phase transition on green tests. **Sonnet** subagents implement the numerically risky engine phases and review local-model output. The **local LM Studio model** implements well-specified, low-risk phases. Phases 3 and 5 get Fable review regardless of implementer.

| Phase | Content | Files | Implementer | Reviewer | Gate | Risk |
|---|---|---|---|---|---|---|
| **1. Golden fixtures** | Capture script + fixture tests; zero engine changes | `scripts/capture_golden_fixtures.py`, `tests/test_golden_single_mode.py`, `tests/fixtures/golden/` | Local model | Sonnet | full suite + goldens green | low |
| **2. Config + persistence** | Dataclasses, stream-registry constants, `from_scenario` parsing, validation rules, XLSX columns, `collect_edits` passthrough test | `engine/withdrawal_strategies.py` (config half), `engine/params.py`, tests | Local model | Sonnet | round-trip + validation tests; goldens untouched | low |
| **3. Strategy refactor (single only)** | `SinglePortfolioStrategy` + loop rewire per §8.3 table; no two-bucket code | `engine/simulation.py`, `engine/withdrawal_strategies.py` | **Sonnet** | **Fable, op-for-op vs the §8.3 table** | **golden fixtures bit-exact** + full suite | **HIGH** |
| **4. Reserve math + sampler** | `planned_gap_schedule`, `rolling_gap_target`, `sample_reserve_returns` — pure functions | `engine/withdrawal_strategies.py`, `tests/test_reserve_target.py` | **Sonnet** | Sonnet, Fable spot-check | §14.3 + sampler moments | low-med |
| **5. TwoBucketStrategy (annual)** | Init split, §8.4 steps D-I, aggregates, result keys | `engine/withdrawal_strategies.py`, `engine/simulation.py` | **Sonnet** | **Fable** | §14.2 deterministic set, §14.5 RNG set, §14.6 equivalences; benchmark recorded | **HIGH** |
| **6. Monthly + guardrails + historic** | Per-year favorable precompute, year-end refill + gains accumulator, §14.4 tests, historic via `n_paths=1` reuse | `engine/simulation.py`, `engine/withdrawal_strategies.py` | **Sonnet** | Fable | monthly deterministic + §14.4 + historic tests | med (calendar off-by-ones) |
| **7. Comparison harness** | `build_comparison`, deltas, cache discipline | `webapp/callbacks.py`, tests | Local model | Sonnet | harness unit test + smoke | low |
| **8. Web UX** | Selector, four cards, live preview, warnings, charts, KPIs, dark-mode registration | `webapp/layout.py`, `webapp/callbacks.py`, `engine/figures.py`, `engine/theme.py`, `tests/ui/` | Local model (frontend-design skill for cards) | Sonnet + UI test pipeline (`scripts/run_ui_tests.sh --analyze`) | §14.7 + smoke | low-med |
| **9. Docs + benchmark** | Tooltips, sample two-bucket scenario, runtime/memory benchmark, historic-mode note | docs, sample files | Local model | Sonnet | full suite | low |

Each phase is a separately reviewable PR-sized unit with no behavior change outside its scope. Phases 1-2 can run in parallel; 3 → 4 → 5 → 6 are sequential; 7-9 depend on 5.

---

## 17. Pull-request checklist

Each phase PR includes: files changed; exact period-order semantics touched (if any); test evidence (which gates ran); for Phase 3, the op-for-op mapping confirmation; for Phase 5, the runtime/memory benchmark vs single mode; any deviation from this PRD called out explicitly.

---

## 18. Recommended defaults

A coherent first-release default set (not personalized investment advice):

```yaml
withdrawal_strategy:
  type: two_bucket
  reserve:
    target_years: 4.0
    refill_trigger_years: 3.0
    coverage_scope: recurring_gap_only
    return_model:
      distribution: normal
      # No universal numeric mean/std preset is silently imposed.
      # Values are visible and editable in the UI.
  draw_policy:
    market_state_rule: previous_period_return
    growth_return_threshold_real: 0.0
    first_period_source: reserve
  refill_policy:
    cadence: annual
    eligibility_rule: growth_return_at_or_above_threshold
    growth_return_threshold_real: 0.0
    amount_rule: to_target
  cashflow_policy:
    surplus_allocation: reserve_then_growth
    planned_gift_draw: strategy_rule
    planned_lump_draw: strategy_rule
    unplanned_event_draw: strategy_rule
  correlation_with_growth: 0.0
```

---

## 19. Resolved decisions

All of the draft's 28 open decisions are resolved. Owner-decided (2026-07-12): **MVP is core-only** (per-category draw policies, recovery-aware rule, reverse rebalancing, and the static benchmark are deferred — §21); **full strategy abstraction** with golden-fixture verification (not an in-loop branch); **historic mode ships** with constant configured reserve returns; **delegation** per §16.

The remaining draft recommendations are adopted as written: gap-based (not gross-spending) sizing; 4/3 target/trigger defaults; gifts and lumps excluded from the default coverage scope; reserve-first first period; full top-up default with gains-only available; small positive years may trigger refill under the simple rule; income surpluses refill the reserve first; constant/normal/Student-t reserve distributions; independent reserve returns; all returns net of assumed fees/tax drag with no dividend model; comparison auto-runs with a disable toggle; "Spending reserve" label; bit-for-bit single mode is a hard requirement; all existing XLSX files remain importable; the export may add optional columns (without a schema bump — §7.1).

---

## 20. Research and product caution

The spending-reserve approach is implemented as a hypothesis the simulator tests, not a guaranteed improvement. Potential advantages: fewer forced growth-asset sales after adverse returns; clearer short-term liquidity; behavioral comfort; an explicit operational refill policy. Potential disadvantages: lower expected return from holding a large reserve; rule-dependent results; refilling after a small recovery while markets remain depressed; failure to buy growth assets at low prices; a false sense that reserve depletion equals plan failure; apparent improvement actually caused by the changed asset mix rather than the withdrawal rule.

The comparison design, forced-sale diagnostics, and the (follow-up) same-assets benchmark are therefore essential parts of the feature, not reporting enhancements.

---

## 21. Follow-ups (deferred, in priority order)

1. **Same-assets static-rebalancing benchmark** (§11) — highest priority: isolates the asset-mix effect from the withdrawal-rule effect.
2. **Per-category draw policies** for gifts / planned lumps / unplanned events (`growth_first` etc.) — requires per-category funding splits inside the vectorized loop; revisit the draft's `growth_first` recommended defaults then.
3. **Recovery-aware high-water-mark market-state rule** — unitized return index (never the cash-flow-affected balance), `allowed_drawdown` parameter.
4. **Reverse rebalancing** — excess reserve above target buys growth in adverse states; advanced, off by default, labeled experimental.
5. **Correlated reserve returns** (`correlation_with_growth ≠ 0`) — uses `STREAM_GROWTH_CORRELATED = 2`.
6. **Pro-rata sources, fixed-amount refill, refill caps; rolling/monthly decision cadence; path-dependent reserve target; precomputed calibration reuse** for the 4-run worst case.

---

## Appendix: changelog vs Draft 0.9

- **RNG design replaced** (§5.5): the draft's `SeedSequence.spawn(4)` re-streaming would have changed existing growth/property sequences, violating the draft's own bit-for-bit requirement. Now: legacy generator untouched; append-only registry for new streams only.
- **`scenario.v2` bump and `growth_return_model` migration dropped** (§7.1): nothing branches on the schema string; flat `mu`/`sigma` fields stay.
- **JSON persistence section rewritten** (§13): no JSON file persistence exists; the real risks are the `collect_edits` rebuild (new must-fix hazard) and absent-block defaulting.
- **Strategy interface corrected** (§8.2): 7 duck-typed methods, instance-is-state, full return matrices at construction, ruin loop-owned; `LiquidStrategyState` and `MarketContext`/`CashflowContext` protocol parameters dropped. Exact call-site mapping table added (§8.3).
- **Scope trimmed to core** (owner decision): §6.9/§6.10 deferred with schema-only presence and validation guards; deferred defaults describe actual behavior.
- **Guardrail section verified against code** (§6.12): zero guardrail changes needed; calibration inherits the strategy automatically.
- **gains_only, refill timing, and monthly cadence made precise** (§6.8, §6.11, §8.4): exposed-balance definition, sim-year boundaries, first-year handling, forward `t+1` windows as O(1) lookups.
- **Deficit-residual accounting rule added** (§8.4-E) so ruin arithmetic exactly matches single mode.
- **Result model corrected** (§9): plain dict; store `growth_over_time` only, derive reserve; aggregates instead of boolean matrices; memory quantified.
- **Comparison harness located and costed** (§11): `execute_run`-level, `dataclasses.replace`, 2-4 run accounting.
- **Golden-fixture procedure specified** (§14.1) and the `target_years=0` bit-equality RNG killer test added (§14.5).
- **Historic mode resolved** (§6.13): ships via `n_paths=1` strategy reuse with constant reserve returns.
- **UX anchored to concrete existing patterns** (§12): guardrail placement split, live-preview and band-chart templates, dark-mode color registration, `journeys.py` selector rule.
- **Implementation plan re-phased with LLM delegation** (§16): 9 gated phases, implementer/reviewer/risk per phase.
- **Prose rewritten in full sentences** throughout (the draft had garbled/compressed wording).
