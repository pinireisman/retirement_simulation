"""Config layer for the two-bucket withdrawal strategy (PRD §7).

Frozen dataclasses, scenario-dict parsing, validation rules, and the
append-only RNG stream registry constant.  No engine/simulation behavior —
pure configuration plumbing only."""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np


# ------------------------------------------------------------------- RNG streams
# Streams derived as np.random.default_rng(np.random.SeedSequence([seed, STREAM_ID])).
# NEVER renumber. Append only.  (PRD §5.5)
STREAM_RESERVE_RETURNS = 1


# ------------------------------------------------------------------- frozen dataclasses (PRD §7.2)

@dataclasses.dataclass(frozen=True)
class ReturnModelConfig:
    distribution: str = "normal"  # constant | normal | student_t
    mean_real: float = 0.0
    std_real: float = 0.0
    student_t_df: Optional[float] = None


@dataclasses.dataclass(frozen=True)
class ReserveConfig:
    target_years: float = 4.0
    refill_trigger_years: float = 3.0
    coverage_scope: str = "recurring_gap_only"
    return_model: ReturnModelConfig = dataclasses.field(default_factory=ReturnModelConfig)


@dataclasses.dataclass(frozen=True)
class DrawPolicyConfig:
    market_state_rule: str = "previous_period_return"  # "recovery_aware" deferred
    growth_return_threshold_real: float = 0.0
    first_period_source: str = "reserve"  # | "growth"; "pro_rata" deferred


@dataclasses.dataclass(frozen=True)
class RefillPolicyConfig:
    cadence: str = "annual"
    eligibility_rule: str = "growth_return_at_or_above_threshold"  # | always | never
    growth_return_threshold_real: float = 0.0
    amount_rule: str = "to_target"  # | gains_only | none; "fixed_amount" deferred


@dataclasses.dataclass(frozen=True)
class CashflowPolicyConfig:
    # ALL deferred — schema round-trip only
    surplus_allocation: str = "reserve_then_growth"
    planned_gift_draw: str = "strategy_rule"
    planned_lump_draw: str = "strategy_rule"
    unplanned_event_draw: str = "strategy_rule"


@dataclasses.dataclass(frozen=True)
class WithdrawalStrategyConfig:
    type: str = "single_portfolio"  # | two_bucket
    reserve: ReserveConfig = dataclasses.field(default_factory=ReserveConfig)
    draw_policy: DrawPolicyConfig = dataclasses.field(default_factory=DrawPolicyConfig)
    refill_policy: RefillPolicyConfig = dataclasses.field(default_factory=RefillPolicyConfig)
    cashflow_policy: CashflowPolicyConfig = dataclasses.field(default_factory=CashflowPolicyConfig)
    correlation_with_growth: float = 0.0  # reserved; must be 0.0


# ------------------------------------------------------------------- parsing

def parse_withdrawal_strategy(block: dict | None) -> Optional[WithdrawalStrategyConfig]:
    """Build a WithdrawalStrategyConfig from a scenario dict block.

    * ``None`` or missing → ``None`` (single-portfolio default).
    * Otherwise construct nested dataclasses with ``.get(...)`` defaults,
      storing unknown string values as-is (validation rejects them later).
    """
    if block is None:
        return None

    # --- reserve.return_model ---
    rm_block = block.get("reserve", {}).get("return_model", {}) if isinstance(block.get("reserve"), dict) else {}
    return_model = ReturnModelConfig(
        distribution=rm_block.get("distribution", "normal"),
        mean_real=float(rm_block.get("mean_real", 0.0)),
        std_real=float(rm_block.get("std_real", 0.0)),
        student_t_df=rm_block.get("student_t_df"),  # may be None
    )

    # --- reserve ---
    res_block = block.get("reserve", {}) if isinstance(block.get("reserve"), dict) else {}
    reserve = ReserveConfig(
        target_years=float(res_block.get("target_years", 4.0)),
        refill_trigger_years=float(res_block.get("refill_trigger_years", 3.0)),
        coverage_scope=res_block.get("coverage_scope", "recurring_gap_only"),
        return_model=return_model,
    )

    # --- draw_policy ---
    dp_block = block.get("draw_policy", {}) if isinstance(block.get("draw_policy"), dict) else {}
    draw_policy = DrawPolicyConfig(
        market_state_rule=dp_block.get("market_state_rule", "previous_period_return"),
        growth_return_threshold_real=float(dp_block.get("growth_return_threshold_real", 0.0)),
        first_period_source=dp_block.get("first_period_source", "reserve"),
    )

    # --- refill_policy ---
    rp_block = block.get("refill_policy", {}) if isinstance(block.get("refill_policy"), dict) else {}
    refill_policy = RefillPolicyConfig(
        cadence=rp_block.get("cadence", "annual"),
        eligibility_rule=rp_block.get("eligibility_rule", "growth_return_at_or_above_threshold"),
        growth_return_threshold_real=float(rp_block.get("growth_return_threshold_real", 0.0)),
        amount_rule=rp_block.get("amount_rule", "to_target"),
    )

    # --- cashflow_policy ---
    cp_block = block.get("cashflow_policy", {}) if isinstance(block.get("cashflow_policy"), dict) else {}
    cashflow_policy = CashflowPolicyConfig(
        surplus_allocation=cp_block.get("surplus_allocation", "reserve_then_growth"),
        planned_gift_draw=cp_block.get("planned_gift_draw", "strategy_rule"),
        planned_lump_draw=cp_block.get("planned_lump_draw", "strategy_rule"),
        unplanned_event_draw=cp_block.get("unplanned_event_draw", "strategy_rule"),
    )

    return WithdrawalStrategyConfig(
        type=block.get("type", "single_portfolio"),
        reserve=reserve,
        draw_policy=draw_policy,
        refill_policy=refill_policy,
        cashflow_policy=cashflow_policy,
        correlation_with_growth=float(block.get("correlation_with_growth", 0.0)),
    )


# ------------------------------------------------------------------- validation (PRD §7.3)

_VALID_TYPES = {"single_portfolio", "two_bucket"}
_VALID_COVERAGE_SCOPES = {"recurring_gap_only", "recurring_plus_scheduled_gifts", "all_planned_outflows"}
_VALID_DISTRIBUTIONS = {"constant", "normal", "student_t"}
_VALID_FIRST_PERIOD_SOURCES = {"reserve", "growth"}
_VALID_ELIGIBILITY_RULES = {"growth_return_at_or_above_threshold", "always", "never"}
_VALID_AMOUNT_RULES = {"to_target", "gains_only", "none"}


def validate_withdrawal_strategy_block(block: dict | None) -> list[str]:
    """Return human-readable error strings for *block*.

    Only validates further when ``type == "two_bucket"``.  Every error
    message contains the offending field name so callers can grep for it.
    """
    errors: list[str] = []

    if block is None:
        return errors

    wtype = block.get("type", "single_portfolio")

    # --- type enum (always validated) ---
    if wtype not in _VALID_TYPES:
        errors.append(f"type '{wtype}' is not a valid withdrawal strategy type")

    # --- single_portfolio: skip all two-bucket checks ---
    if wtype != "two_bucket":
        return errors

    # Helper to safely navigate nested dicts
    reserve = block.get("reserve", {}) or {}
    return_model = reserve.get("return_model", {}) or {}
    draw_policy = block.get("draw_policy", {}) or {}
    refill_policy = block.get("refill_policy", {}) or {}
    cashflow_policy = block.get("cashflow_policy", {}) or {}

    # --- reserve.target_years >= 0 ---
    target_years = reserve.get("target_years", 4.0)
    if not isinstance(target_years, (int, float)) or target_years < 0:
        errors.append(f"target_years must be >= 0, got {target_years}")

    # --- 0 <= refill_trigger_years <= target_years ---
    trigger_years = reserve.get("refill_trigger_years", 3.0)
    if not isinstance(trigger_years, (int, float)):
        errors.append(f"refill_trigger_years must be a number, got {trigger_years}")
    elif trigger_years < 0:
        errors.append(f"refill_trigger_years must be >= 0, got {trigger_years}")
    elif isinstance(target_years, (int, float)) and target_years >= 0 and trigger_years > target_years:
        errors.append(f"refill_trigger_years ({trigger_years}) must be <= target_years ({target_years})")

    # --- coverage_scope enum ---
    coverage = reserve.get("coverage_scope", "recurring_gap_only")
    if coverage not in _VALID_COVERAGE_SCOPES:
        errors.append(f"coverage_scope '{coverage}' is not a valid value")

    # --- return_model.distribution enum ---
    distribution = return_model.get("distribution", "normal")
    if distribution not in _VALID_DISTRIBUTIONS:
        errors.append(f"distribution '{distribution}' is not a valid return model distribution")

    # --- std_real >= 0 ---
    std_real = return_model.get("std_real", 0.0)
    if not isinstance(std_real, (int, float)) or std_real < 0:
        errors.append(f"std_real must be >= 0, got {std_real}")

    # --- constant requires std_real == 0 ---
    if distribution == "constant" and isinstance(std_real, (int, float)) and std_real != 0:
        errors.append(f"constant distribution requires std_real == 0, got {std_real}")

    # --- student_t requires student_t_df > 2 ---
    if distribution == "student_t":
        df_val = return_model.get("student_t_df")
        if df_val is None:
            errors.append(f"student_t distribution requires student_t_df > 2, got None")
        elif not isinstance(df_val, (int, float)) or df_val <= 2:
            errors.append(f"student_t distribution requires student_t_df > 2, got {df_val}")

    # --- first_period_source enum ---
    fps = draw_policy.get("first_period_source", "reserve")
    if fps not in _VALID_FIRST_PERIOD_SOURCES:
        errors.append(f"first_period_source '{fps}' is not a valid value")

    # --- eligibility_rule enum ---
    elig = refill_policy.get("eligibility_rule", "growth_return_at_or_above_threshold")
    if elig not in _VALID_ELIGIBILITY_RULES:
        errors.append(f"eligibility_rule '{elig}' is not a valid value")

    # --- amount_rule enum ---
    ar = refill_policy.get("amount_rule", "to_target")
    if ar not in _VALID_AMOUNT_RULES:
        errors.append(f"amount_rule '{ar}' is not a valid value")

    # --- deferred-feature guards ---
    corr = block.get("correlation_with_growth", 0.0)
    if isinstance(corr, (int, float)) and corr != 0:
        errors.append(f"correlation_with_growth != 0 is not implemented")

    mkr = draw_policy.get("market_state_rule", "previous_period_return")
    if mkr != "previous_period_return":
        errors.append(f"market_state_rule '{mkr}' is not implemented")

    cadence = refill_policy.get("cadence", "annual")
    if cadence != "annual":
        errors.append(f"cadence '{cadence}' is not implemented")

    # cashflow_policy deferred fields
    cp_defaults = CashflowPolicyConfig()
    for field_name in ("surplus_allocation", "planned_gift_draw", "planned_lump_draw", "unplanned_event_draw"):
        cp_val = cashflow_policy.get(field_name, getattr(cp_defaults, field_name))
        if cp_val != getattr(cp_defaults, field_name):
            errors.append(f"{field_name} '{cp_val}' is not implemented")

    return errors


# ------------------------------------------------------------------- strategy interface (PRD §8.2)

class SinglePortfolioStrategy:
    """Preserves current single-balance behavior bit-for-bit (PRD §8.2, §8.3)."""

    def __init__(self, params, port_ret):
        self.bal = np.full(params.n_paths, params.initial_portfolio)
        self.port_ret = port_ret

    def combined_balance(self) -> np.ndarray:
        return self.bal

    def apply_cashflow(self, i, net) -> None:
        self.bal = self.bal + net  # exactly current line 237

    def mark_ruined(self, ruined) -> None:
        self.bal[ruined] = 0.0  # exactly current line 253

    def apply_returns(self, i, alive) -> None:
        self.bal[alive] *= (1 + self.port_ret[alive, i])  # exactly current line 254

    def end_of_period(self, i, alive) -> None:
        pass  # no-op in single mode

    def record(self, i) -> None:
        pass  # no-op in single mode

    def extra_results(self) -> dict:
        return {"withdrawal_strategy_type": "single_portfolio"}


# ------------------------------------------------------------------- reserve-target pure functions (PRD §8.6)

def planned_gap_schedule(*, spend_strict, spend_lifestyle, spend_gifts, lump_out_by_cat,
                          playground_out, incomes, rent_by_period, lump_in, playground_in,
                          coverage_scope) -> np.ndarray:
    """Positive planned portfolio-funded gap per period (PRD §8.6).

    All arrays are the engine's already-built, already-period-scaled
    cash-flow arrays (simulation.py:70-135). Playground events are excluded
    by construction: they never enter ``spend_*``/`lump_out_by_cat`` (except
    as part of the "strict" lump total, which is only pulled in at
    ``all_planned_outflows`` scope and explicitly netted against
    ``playground_out`` there) and are stripped from inflow via
    ``lump_in - playground_in``.
    """
    planned_inflow = incomes + rent_by_period + (lump_in - playground_in)

    planned_outflow = spend_strict + spend_lifestyle
    if coverage_scope in ("recurring_plus_scheduled_gifts", "all_planned_outflows"):
        planned_outflow = planned_outflow + spend_gifts + lump_out_by_cat["gifts"]
    if coverage_scope == "all_planned_outflows":
        planned_outflow = (planned_outflow + lump_out_by_cat["lifestyle"]
                            + (lump_out_by_cat["strict"] - playground_out))

    return np.maximum(planned_outflow - planned_inflow, 0.0)


def rolling_gap_target(gap: np.ndarray, years: float, periods_per_year: int) -> np.ndarray:
    """target[t] = sum(gap[t : t+w]) + frac * gap[t+w]; tapers to zero past the plan horizon.

    Verbatim vectorized cumsum implementation from PRD §8.6.
    """
    w_f = years * periods_per_year
    w, frac = int(w_f), w_f - int(w_f)
    c = np.concatenate([[0.0], np.cumsum(gap)])
    t = np.arange(len(gap))
    full = c[np.minimum(t + w, len(gap))] - c[t]
    tail = frac * np.where(t + w < len(gap), gap[np.minimum(t + w, len(gap) - 1)], 0.0)
    return full + tail


# ------------------------------------------------------------------- reserve return sampler (PRD §5.5, §8.5)

def sample_reserve_returns(cfg: ReturnModelConfig, n_paths: int, n_periods: int, seed, annual: bool) -> np.ndarray:
    """Sample the spending-reserve return matrix.

    ``constant`` returns a full matrix and consumes NO RNG draws. Otherwise
    draws from a generator independently derived via the append-only stream
    registry (STREAM_RESERVE_RETURNS) so the shared growth/property RNG
    sequence used elsewhere in ``run_simulation`` is never shifted.
    ``student_t`` reuses ``engine.simulation.sample_real_returns`` so the
    variance rescale and clip match growth-return conventions exactly
    (imported inside the function body: engine.simulation imports
    ``make_strategy`` from this module, so a top-level import would cycle).
    """
    mean_real = cfg.mean_real
    std_real = cfg.std_real
    if not annual:
        mean_real = np.exp(np.log(1 + mean_real) / 12) - 1
        std_real = std_real / np.sqrt(12)

    if cfg.distribution == "constant":
        return np.full((n_paths, n_periods), mean_real)

    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(np.random.SeedSequence([seed, STREAM_RESERVE_RETURNS]))

    if cfg.distribution == "student_t":
        from engine.simulation import sample_real_returns  # deferred: avoid module cycle
        return sample_real_returns((n_paths, n_periods), mean_real, std_real, cfg.student_t_df, rng=rng)

    return rng.normal(mean_real, std_real, size=(n_paths, n_periods))


# ------------------------------------------------------------------- TwoBucketStrategy, annual mode (PRD §8.4)

class TwoBucketStrategy:
    """Growth + Spending Reserve withdrawal strategy (PRD §8.4; annual and
    monthly modes).

    Same 7-method duck-typed interface as SinglePortfolioStrategy.
    """

    def __init__(self, params, port_ret, reserve_ret, planned_gap):
        cfg = params.withdrawal_strategy
        self.reserve_cfg = cfg.reserve
        self.draw_policy = cfg.draw_policy
        self.refill_policy = cfg.refill_policy

        self.port_ret = port_ret
        self.reserve_ret = reserve_ret
        self.planned_gap = planned_gap

        n_paths, n_periods = port_ret.shape
        self.n_paths = n_paths
        self.n_periods = n_periods
        # PRD §6.11: refill cadence and the market-state matrix are fixed at
        # 12 periods/year in monthly mode (cadence="annual" is the only
        # implemented value, validated in engine/params.py).
        self.periods_per_year = 1 if params.annual else 12

        # Precomputed once before the loop (PRD §6.7, §8.6): O(1) array
        # lookup per period. periods_per_year=1 for annual mode reproduces
        # the original hardcoded call exactly.
        self.target_by_period = rolling_gap_target(planned_gap, self.reserve_cfg.target_years,
                                                     self.periods_per_year)
        self.trigger_by_period = rolling_gap_target(planned_gap, self.reserve_cfg.refill_trigger_years,
                                                      self.periods_per_year)

        # Init split (PRD §6.2).
        reserve0 = min(params.initial_portfolio, self.target_by_period[0])
        growth0 = params.initial_portfolio - reserve0
        self.growth = np.full(n_paths, growth0, dtype=float)
        self.reserve = np.full(n_paths, reserve0, dtype=float)

        # Favorable/market-state matrix (PRD §6.4, §8.4 step D, §6.11).
        threshold = self.draw_policy.growth_return_threshold_real
        first_source_growth = (self.draw_policy.first_period_source == "growth")
        favorable = np.zeros((n_paths, n_periods), dtype=bool)
        if self.periods_per_year == 1:
            # Annual: favorable[:, i] = port_ret[:, i-1] >= threshold for
            # i >= 1; period 0 falls back to the configured
            # first_period_source. (Unchanged from the annual-only version.)
            if n_periods > 1:
                favorable[:, 1:] = port_ret[:, :-1] >= threshold
            favorable[:, 0] = first_source_growth
        else:
            # Monthly (PRD §6.11): fixed per sim-year from the PREVIOUS sim
            # year's compounded trailing-12-month growth return; the first
            # sim year (all 12 periods) uses the configured fallback source.
            # ponytail: loop is over sim-years (tens, not Monte Carlo paths --
            # each year's compounded return is vectorized across paths), so
            # this stays well inside "no loop over paths"; a single fused
            # cumprod pass is possible but not simpler to read or verify.
            n_years = -(-n_periods // self.periods_per_year)  # ceil div
            for y in range(n_years):
                start = y * self.periods_per_year
                end = min(start + self.periods_per_year, n_periods)
                if y == 0:
                    favorable[:, start:end] = first_source_growth
                else:
                    prev_start = (y - 1) * self.periods_per_year
                    prev_end = y * self.periods_per_year
                    compounded = np.prod(1 + port_ret[:, prev_start:prev_end], axis=1) - 1
                    favorable[:, start:end] = (compounded >= threshold)[:, None]
        self.favorable = favorable

        # Result-model histories/aggregates (PRD §9). No full boolean event
        # matrices -- per-period aggregates plus per-path lifetime counters only.
        self.growth_over_time = np.zeros((n_periods, n_paths))
        self.refill_probability_by_period = np.zeros(n_periods)
        self.forced_sale_probability_by_period = np.zeros(n_periods)
        self.reserve_depletion_probability_by_period = np.zeros(n_periods)
        self.refill_count_per_path = np.zeros(n_paths, dtype=int)
        self.forced_sale_count_per_path = np.zeros(n_paths, dtype=int)
        self.reserve_depleted_ever_per_path = np.zeros(n_paths, dtype=bool)
        self.refill_total_per_path = np.zeros(n_paths)
        self.deficit_from_reserve_per_path = np.zeros(n_paths)
        self.deficit_from_growth_per_path = np.zeros(n_paths)
        # PRD §12.3 "funding source by age" chart: alive-gated per-period sums
        # of the same from_res/from_gro values already computed each period in
        # apply_cashflow -- purely additive bookkeeping, no effect on any other
        # stored/derived value.
        self.deficit_from_reserve_by_period = np.zeros(n_periods)
        self.deficit_from_growth_by_period = np.zeros(n_periods)

        # Per-period scratch state, refreshed each period.
        self._period_gain = np.zeros(n_paths)          # gains_only exposed-balance gain
        self._year_gain = np.zeros(n_paths)             # monthly-only: running sim-year accumulator
        self._forced_this_period = np.zeros(n_paths, dtype=bool)
        self._refilled_this_period = np.zeros(n_paths, dtype=bool)
        self._from_res_this_period = np.zeros(n_paths)
        self._from_gro_this_period = np.zeros(n_paths)
        self._alive = np.ones(n_paths, dtype=bool)

    def combined_balance(self) -> np.ndarray:
        return self.growth + self.reserve

    def apply_cashflow(self, i, net) -> None:
        # PRD §8.4 step E -- exact vectorized deficit/surplus code.
        gap = np.maximum(-net, 0.0)
        surplus = np.maximum(net, 0.0)
        pref_growth = self.favorable[:, i]
        r1 = np.minimum(gap, self.reserve); g1 = np.minimum(gap - r1, self.growth)  # reserve-first
        g2 = np.minimum(gap, self.growth); r2 = np.minimum(gap - g2, self.reserve)  # growth-first
        from_res = np.where(pref_growth, r2, r1)
        from_gro = np.where(pref_growth, g2, g1)
        resid = gap - from_res - from_gro  # 0 unless combined insufficient
        self.growth = self.growth - from_gro - resid  # residual rides on growth -> combined = old - gap exactly
        self.reserve = self.reserve - from_res
        to_res = np.minimum(surplus, np.maximum(self.target_by_period[i] - self.reserve, 0.0))
        self.reserve = self.reserve + to_res
        self.growth = self.growth + (surplus - to_res)
        forced = (~pref_growth) & (from_gro > 0)  # forced-sale diagnostic

        self.deficit_from_reserve_per_path += from_res
        self.deficit_from_growth_per_path += from_gro
        self.forced_sale_count_per_path += forced.astype(int)
        self._forced_this_period = forced
        self._from_res_this_period = from_res
        self._from_gro_this_period = from_gro

    def mark_ruined(self, ruined) -> None:
        self.growth[ruined] = 0.0
        self.reserve[ruined] = 0.0

    def apply_returns(self, i, alive) -> None:
        exposed = self.growth.copy()  # "exposed" balance for gains_only (PRD §6.8)
        self.growth[alive] *= (1 + self.port_ret[alive, i])
        self.reserve[alive] *= (1 + self.reserve_ret[alive, i])
        if self.periods_per_year == 1:
            self._period_gain = exposed * self.port_ret[:, i]  # annual mode: overwrite each period
        else:
            # Monthly (PRD §6.8): gains_only caps the refill at the SUM of
            # the just-completed sim year's 12 monthly gains, each computed
            # against that month's own exposed (pre-return) growth balance
            # -- never inferred from ending-minus-starting balance. Reset at
            # the start of each sim year; end_of_period reads the total
            # after the year's last month is added.
            if i % self.periods_per_year == 0:
                self._year_gain = np.zeros(self.n_paths)
            self._year_gain = self._year_gain + exposed * self.port_ret[:, i]
            self._period_gain = self._year_gain
        self._alive = alive

    def end_of_period(self, i, alive) -> None:
        # PRD §6.8/§8.4 step H. Forward t+1 window; at the final period there
        # is no forward window, so no refill can occur.
        if i + 1 >= self.n_periods:
            self._refilled_this_period = np.zeros(self.n_paths, dtype=bool)
            return

        if self.periods_per_year > 1 and (i + 1) % self.periods_per_year != 0:
            # Monthly mode (PRD §6.11): refill is evaluated only at sim-year
            # boundaries; intermediate months are a no-op (refilled stays
            # False for record()'s per-period probability tracking).
            self._refilled_this_period = np.zeros(self.n_paths, dtype=bool)
            return

        trigger_next = self.trigger_by_period[i + 1]
        target_next = self.target_by_period[i + 1]
        if self.periods_per_year == 1:
            just_completed_return = self.port_ret[:, i]
        else:
            # Monthly (PRD §8.4 step H): compounded return of the
            # just-completed sim year, not the last month's return alone.
            year_start = i - self.periods_per_year + 1
            just_completed_return = np.prod(1 + self.port_ret[:, year_start:i + 1], axis=1) - 1

        rule = self.refill_policy.eligibility_rule
        if rule == "always":
            elig_return = np.ones(self.n_paths, dtype=bool)
        elif rule == "never":
            elig_return = np.zeros(self.n_paths, dtype=bool)
        else:  # growth_return_at_or_above_threshold
            elig_return = just_completed_return >= self.refill_policy.growth_return_threshold_real

        eligible = (self.reserve < trigger_next) & elig_return & alive
        need = np.maximum(target_next - self.reserve, 0.0)

        amount_rule = self.refill_policy.amount_rule
        if amount_rule == "gains_only":
            amount = np.minimum(need, np.maximum(self._period_gain, 0.0))
        elif amount_rule == "none":
            amount = np.zeros(self.n_paths)
        else:  # to_target
            amount = need

        transfer = np.where(eligible, np.minimum(amount, self.growth), 0.0)
        self.growth = self.growth - transfer
        self.reserve = self.reserve + transfer

        refilled = transfer > 0
        self.refill_count_per_path += refilled.astype(int)
        self.refill_total_per_path += transfer
        self._refilled_this_period = refilled

    def record(self, i) -> None:
        self.growth_over_time[i, :] = self.growth
        alive = self._alive
        n_alive = int(alive.sum())
        if n_alive > 0:
            depleted = (self.reserve <= 1e-9) & alive
            self.forced_sale_probability_by_period[i] = float(np.sum(self._forced_this_period & alive)) / n_alive
            self.refill_probability_by_period[i] = float(np.sum(self._refilled_this_period & alive)) / n_alive
            self.reserve_depletion_probability_by_period[i] = float(np.sum(depleted)) / n_alive
            self.reserve_depleted_ever_per_path |= depleted
        self.deficit_from_reserve_by_period[i] = float(np.sum(self._from_res_this_period[alive]))
        self.deficit_from_growth_by_period[i] = float(np.sum(self._from_gro_this_period[alive]))

    def extra_results(self) -> dict:
        return {
            "withdrawal_strategy_type": "two_bucket",
            "growth_over_time": self.growth_over_time,
            "reserve_target_by_period": self.target_by_period,
            "reserve_trigger_by_period": self.trigger_by_period,
            "planned_gap_by_period": self.planned_gap,
            "refill_probability_by_period": self.refill_probability_by_period,
            "forced_sale_probability_by_period": self.forced_sale_probability_by_period,
            "reserve_depletion_probability_by_period": self.reserve_depletion_probability_by_period,
            "deficit_from_reserve_by_period": self.deficit_from_reserve_by_period,
            "deficit_from_growth_by_period": self.deficit_from_growth_by_period,
            "refill_count_per_path": self.refill_count_per_path,
            "forced_sale_count_per_path": self.forced_sale_count_per_path,
            "reserve_depleted_ever_per_path": self.reserve_depleted_ever_per_path,
            "refill_total_per_path": self.refill_total_per_path,
            "deficit_from_reserve_per_path": self.deficit_from_reserve_per_path,
            "deficit_from_growth_per_path": self.deficit_from_growth_per_path,
            "final_growth": self.growth.copy(),
            "final_reserve": self.reserve.copy(),
        }


def make_strategy(params, port_ret, reserve_ret=None, planned_gap=None):
    """Construct the withdrawal strategy for a run (PRD §8.2)."""
    if params.withdrawal_strategy is None or params.withdrawal_strategy.type == "single_portfolio":
        return SinglePortfolioStrategy(params, port_ret)
    if params.withdrawal_strategy.type == "two_bucket":
        return TwoBucketStrategy(params, port_ret, reserve_ret, planned_gap)
    raise NotImplementedError(f"withdrawal strategy type {params.withdrawal_strategy.type!r} not implemented")
