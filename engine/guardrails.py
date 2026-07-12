"""Guardrail engine: configurable spending-adjustment rules layered onto
run_simulation (PRD §7). guardrails=None/[] must leave run_simulation's
output bit-identical to having no guardrail engine at all."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

CATEGORIES = ("strict", "lifestyle", "gifts")


@dataclass
class GuardrailConfig:
    type: str
    options: dict


class VolatilityDiscretionaryScaling:
    """PRD §7.3: scales lifestyle+gifts outflows based on each path's own
    realized portfolio return in the previous period. Strict spending and
    all inflows are never touched."""

    KEY = "volatility_discretionary_scaling"

    def __init__(self, options: dict, context: dict | None = None):
        self.drop = options["drop_threshold"]
        self.rise = options["rise_threshold"]
        self.cut = options["cut_pct"]
        self.raise_ = options["raise_pct"]
        self.triggered_down: Optional[np.ndarray] = None
        self.triggered_up: Optional[np.ndarray] = None
        self.adjustments: list = []

    def multiplier(self, year_idx: int, port_ret: np.ndarray, n_paths: int, bal=None) -> np.ndarray:
        """Per-path multiplier for this year's discretionary outflows.
        `bal` is unused here (this rule reacts to returns, not the balance);
        it exists so all handlers share one call signature."""
        if self.triggered_down is None:
            self.triggered_down = np.zeros(n_paths, dtype=bool)
            self.triggered_up = np.zeros(n_paths, dtype=bool)
        mult = np.ones(n_paths)
        if year_idx == 0:
            return mult
        prev = port_ret[:, year_idx - 1]
        down = prev <= -self.drop
        up = prev >= self.rise
        mult[down] = 1.0 - self.cut
        mult[up] = 1.0 + self.raise_
        self.triggered_down |= down
        self.triggered_up |= up
        return mult


def _pav_nondecreasing(y: np.ndarray) -> np.ndarray:
    """Isotonic (non-decreasing) least-squares fit via pool-adjacent-violators.
    Returns the fitted values, same length as y. O(n)."""
    # pools of (mean, count), merged whenever a new value breaks monotonicity
    means: list = []
    counts: list = []
    for v in y:
        m, c = float(v), 1
        while means and means[-1] > m:
            pm, pc = means.pop(), counts.pop()
            m = (m * c + pm * pc) / (c + pc)
            c += pc
        means.append(m)
        counts.append(c)
    return np.repeat(means, counts)


def calibrate_wealth_needed(start_bal: np.ndarray, success: np.ndarray,
                            confidences: tuple) -> dict:
    """W[p][t] = smallest start-of-period-t liquid balance at which baseline
    paths' CONDITIONAL success probability (success regressed on wealth,
    isotonic fit) reaches p (PRD_GOAL_BASED_GUARDRAILS.md §2.1). np.inf where
    no observed wealth level reaches p (that year always reads 'behind').

    The conditional fit matters: a cumulative "everyone richer than W succeeds
    >= p of the time" construction degenerates whenever the plan's OVERALL
    success rate exceeds p — the healthy mass drags the running average up, W
    collapses to ~0, and cut triggers only fire on already-doomed paths (seen
    live: protect mode recovering nothing on a 95%-success plan with a huge
    trimmable-gifts ceiling). Isotonic regression answers the local question
    "how do paths AT this wealth level fare", so thresholds land at the
    genuinely at-risk tail regardless of overall plan health. Deterministic;
    estimates assume baseline (unguarded) spending, per the source doc's
    one-shot calibration method."""
    T, n = start_bal.shape
    out = {p: np.zeros(T) for p in confidences}
    for t in range(T):
        order = np.argsort(start_bal[t])               # poorest first
        sorted_bal = start_bal[t][order]
        fit = _pav_nondecreasing(success[order].astype(float))
        for p in confidences:
            idx = np.searchsorted(fit, p, side="left")  # fit is non-decreasing
            out[p][t] = np.inf if idx == n else sorted_bal[idx]
    return out


def precompute_pv(cashflows: np.ndarray, discount_rate: float) -> np.ndarray:
    """pv[t] = present value of cashflows[t:], discounted at `discount_rate`."""
    T = len(cashflows)
    pv = np.zeros(T)
    for t in range(T):
        years = np.arange(T - t)
        pv[t] = np.sum(cashflows[t:] / (1 + discount_rate) ** years)
    return pv


class FundedRatioGuardrail:
    """funded_ratio = (portfolio + PV(planned income)) / PV(remaining spending).

    Income sits on the ASSET side (actuarial convention), not netted into the
    liability. Netting (portfolio / PV(spending - income)) breaks down for
    income-rich plans: the denominator goes negative/near-zero, the ratio
    explodes to millions, the raise leg fires for every path regardless of
    fr_upper, and the cut leg can never fire — the guardrail degenerates into
    an unconditional spending raise. Asset-side keeps both sides positive and
    the ratio near 1.x, where the thresholds actually discriminate. The two
    forms agree at funded_ratio == 1 (assets exactly cover needs), so the
    0.85 / 1.05 / 1.30 defaults keep their meaning.

    Three-bucket model (PRD Stage 4 / source doc §9-11). Two independently-scaled
    discretionary buckets, both already inside the PV lookahead so paying a
    planned lump is not itself read as overspending:

    - `lifestyle` (recurring discretionary): the PRIMARY dial — cut when behind
      plan, raised when ahead, within [min_multiplier, max_multiplier].
    - `gifts` (optional lumpy — weddings/gifts/upgrades): a SECONDARY dial capped
      at 1.0 (never given above plan) and only trimmed under *severe* stress
      (funded ratio < fr_severe), recovering back toward 1.0 when the plan is
      healthy again. This is what stops a rich early year from inflating a large
      planned gift and then ruining the plan when markets turn.

    Essential/committed spending is never scaled. The handler returns a single
    *effective* multiplier on the combined discretionary array so the simulation
    loop's `disc_out[i] * mult` line stays untouched."""

    KEY = "funded_ratio_guardrail"

    def __init__(self, options: dict, context: dict | None = None):
        T = len(context["pv_committed"])

        def _per_year(x):
            return np.broadcast_to(np.asarray(x, dtype=float), (T,)).copy()

        if options.get("mode") == "confidence":
            # Goal-based mode (PRD_GOAL_BASED_GUARDRAILS.md): thresholds are
            # per-year funded-ratio arrays derived in run_simulation from a
            # same-seed baseline pass ("wealth needed at age t for p% success").
            tables = context["fr_by_confidence"]
            self.fr_lower = tables[options["c_cut"]]
            self.fr_target = tables[options["c_target"]]
            self.fr_upper = (tables[options["c_raise"]]
                             if options.get("c_raise") is not None else np.full(T, np.inf))
            self.fr_severe = (tables[options["c_severe"]]
                              if options.get("c_severe") is not None else np.zeros(T))
        else:
            # Manual mode — scalar thresholds, broadcast so multiplier() has one
            # code path. Optional-lumpy (gifts) tier default stays engine-side.
            self.fr_lower = _per_year(options["fr_lower"])
            self.fr_target = _per_year(options["fr_target"])
            self.fr_upper = _per_year(options["fr_upper"])
            self.fr_severe = _per_year(options.get("fr_severe", 0.80))
        self.adjustment_fraction = options.get("adjustment_fraction", 0.25)
        self.max_cut_per_year = options.get("max_cut_per_year", 0.10)
        # Raise more cautiously than we cut (source doc's lower bound, 5%/yr):
        # keeps the lifestyle-raise leg from over-spending a temporarily
        # overfunded stressed plan into higher ruin.
        self.max_raise_per_year = options.get("max_raise_per_year", 0.05)
        self.min_multiplier = options.get("min_multiplier", 0.40)
        self.max_multiplier = options.get("max_multiplier", 1.50)
        self.optional_cut_per_year = options.get("optional_cut_per_year", 0.10)
        self.optional_recover_per_year = options.get("optional_recover_per_year", 0.10)
        self.min_optional_multiplier = options.get("min_optional_multiplier", 0.0)
        self.pv_committed = context["pv_committed"]      # shape (T,), committed OUTFLOWS only
        self.pv_income = context["pv_income"]            # shape (T,), planned inflows (asset side)
        self.pv_lifestyle = context["pv_lifestyle"]      # shape (T,)
        self.pv_optional = context["pv_optional"]        # shape (T,)
        self.lifestyle_out = context["lifestyle_out"]    # shape (T,)
        self.gifts_out = context["gifts_out"]            # shape (T,)
        self.lifestyle_mult: Optional[np.ndarray] = None
        self.optional_mult: Optional[np.ndarray] = None
        self.triggered_down: Optional[np.ndarray] = None
        self.triggered_up: Optional[np.ndarray] = None
        self.adjustments: list = []
        self.mult_history: list = []   # per-year effective multipliers (report UI)

    def multiplier(self, year_idx: int, port_ret: np.ndarray, n_paths: int, bal=None) -> np.ndarray:
        if self.lifestyle_mult is None:
            self.lifestyle_mult = np.ones(n_paths)
            self.optional_mult = np.ones(n_paths)
            self.triggered_down = np.zeros(n_paths, dtype=bool)
            self.triggered_up = np.zeros(n_paths, dtype=bool)

        pv_committed_t = self.pv_committed[year_idx]
        pv_life_t = max(self.pv_lifestyle[year_idx], 1.0)
        pv_opt_t = self.pv_optional[year_idx]

        # Assets = liquid portfolio + PV of remaining planned income; needs =
        # PV of remaining spending, scaled by the current dials. Both positive.
        assets = bal + self.pv_income[year_idx]
        pv_need = np.maximum(
            pv_committed_t + self.lifestyle_mult * pv_life_t + self.optional_mult * pv_opt_t,
            1.0,
        )
        funded_ratio = assets / pv_need

        cut = funded_ratio < self.fr_lower[year_idx]
        raise_ = funded_ratio > self.fr_upper[year_idx]
        severe = funded_ratio < self.fr_severe[year_idx]

        # Lifestyle dial: solve for the multiplier that hits fr_target given the
        # committed liabilities AND the current optional-lumpy reservation
        # (source doc §9), then move partway there, capped per year.
        target = (assets / self.fr_target[year_idx]
                  - pv_committed_t - self.optional_mult * pv_opt_t) / pv_life_t
        target = np.clip(target, self.min_multiplier, self.max_multiplier)
        proposed = self.lifestyle_mult + self.adjustment_fraction * (target - self.lifestyle_mult)
        proposed = np.where(cut, np.maximum(proposed, self.lifestyle_mult * (1 - self.max_cut_per_year)), proposed)
        proposed = np.where(raise_, np.minimum(proposed, self.lifestyle_mult * (1 + self.max_raise_per_year)), proposed)
        self.lifestyle_mult = np.where(cut | raise_, proposed, self.lifestyle_mult)

        # Optional-lumpy dial: trim under severe stress, recover toward 1.0 when
        # ahead of plan, never exceed 1.0 (source doc §8/§11).
        opt = self.optional_mult
        opt = np.where(severe, opt * (1 - self.optional_cut_per_year), opt)
        opt = np.where(raise_, opt + self.optional_recover_per_year, opt)
        self.optional_mult = np.clip(opt, self.min_optional_multiplier, 1.0)

        self.triggered_down |= cut
        self.triggered_up |= raise_

        # Collapse the two dials into one effective multiplier on the combined
        # discretionary outflow, so the loop's disc_out[i]*mult line is unchanged.
        # life_t/gift_t are scalars for this year; a zero-discretionary year has
        # no outflow to scale, so the multiplier is irrelevant -> return 1.0.
        life_t = self.lifestyle_out[year_idx]
        gift_t = self.gifts_out[year_idx]
        disc_t = life_t + gift_t
        if disc_t <= 1e-9:
            effective = np.ones(n_paths)
        else:
            effective = (life_t * self.lifestyle_mult + gift_t * self.optional_mult) / disc_t
        self.mult_history.append(effective)
        return effective


GUARDRAIL_REGISTRY: dict = {
    VolatilityDiscretionaryScaling.KEY: VolatilityDiscretionaryScaling,
    FundedRatioGuardrail.KEY: FundedRatioGuardrail,
}


def build_handlers(guardrails: Optional[list], context: Optional[dict] = None) -> list:
    """Instantiate handlers for a list of GuardrailConfig. Unknown type ->
    ValueError before the simulation loop starts (PRD §7.5). `context` carries
    precomputed data (e.g. funded-ratio PV lookahead) to handlers that need it."""
    handlers = []
    for g in (guardrails or []):
        if g.type not in GUARDRAIL_REGISTRY:
            raise ValueError(f"Unknown guardrail type: {g.type!r}")
        handlers.append(GUARDRAIL_REGISTRY[g.type](g.options, context))
    return handlers


def compute_guardrail_stats(handlers: list) -> Optional[dict]:
    """PRD §7.4. Only one guardrail is supported today, so stats reflect
    the first handler; extend when a second guardrail type ships."""
    if not handlers:
        return None
    h = handlers[0]
    adjust = np.stack(h.adjustments, axis=0)
    stats = {
        "type": h.KEY,
        "frac_paths_triggered": float((h.triggered_down | h.triggered_up).mean()),
        "frac_paths_cut": float(h.triggered_down.mean()),
        "frac_paths_raised": float(h.triggered_up.mean()),
        "median_adjustment": float(np.median(adjust[adjust != 0])) if np.any(adjust != 0) else 0.0,
    }
    if getattr(h, "mult_history", None):
        # Deepest cut / highest raise each path ever experienced -> report-UI
        # wording ("worst 5% of futures: cut up to X%" / "median peak +Y%").
        # The raise side must be shown alongside the confidence delta so a user
        # who chose upside/balanced sees what the lost confidence bought.
        hist = np.stack(h.mult_history, axis=0)          # (T, n_paths)
        deepest = np.min(hist, axis=0)
        peak = np.max(hist, axis=0)
        stats["worst5_min_mult"] = float(np.percentile(deepest, 5))
        stats["median_min_mult"] = float(np.percentile(deepest, 50))
        stats["median_max_mult"] = float(np.percentile(peak, 50))
    return stats


def parse_guardrail_configs(guardrail_cfg: Optional[dict]) -> list:
    """Convert the §4.3 dcc.Store payload into enabled GuardrailConfig objects."""
    if not guardrail_cfg:
        return []
    configs = []
    for g in guardrail_cfg.get("guardrails", []):
        if not g.get("enabled"):
            continue
        options = {k: v for k, v in g.items() if k not in ("type", "enabled")}
        configs.append(GuardrailConfig(type=g["type"], options=options))
    return configs
