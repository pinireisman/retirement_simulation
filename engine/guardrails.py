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

    def __init__(self, options: dict):
        self.drop = options["drop_threshold"]
        self.rise = options["rise_threshold"]
        self.cut = options["cut_pct"]
        self.raise_ = options["raise_pct"]
        self.triggered_down: Optional[np.ndarray] = None
        self.triggered_up: Optional[np.ndarray] = None
        self.adjustments: list = []

    def multiplier(self, year_idx: int, port_ret: np.ndarray, n_paths: int) -> np.ndarray:
        """Per-path multiplier for this year's discretionary outflows."""
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


GUARDRAIL_REGISTRY: dict = {
    VolatilityDiscretionaryScaling.KEY: VolatilityDiscretionaryScaling,
}


def build_handlers(guardrails: Optional[list]) -> list:
    """Instantiate handlers for a list of GuardrailConfig. Unknown type ->
    ValueError before the simulation loop starts (PRD §7.5)."""
    handlers = []
    for g in (guardrails or []):
        if g.type not in GUARDRAIL_REGISTRY:
            raise ValueError(f"Unknown guardrail type: {g.type!r}")
        handlers.append(GUARDRAIL_REGISTRY[g.type](g.options))
    return handlers


def compute_guardrail_stats(handlers: list) -> Optional[dict]:
    """PRD §7.4. Only one guardrail is supported today, so stats reflect
    the first handler; extend when a second guardrail type ships."""
    if not handlers:
        return None
    h = handlers[0]
    adjust = np.stack(h.adjustments, axis=0)
    return {
        "type": h.KEY,
        "frac_paths_triggered": float((h.triggered_down | h.triggered_up).mean()),
        "frac_paths_cut": float(h.triggered_down.mean()),
        "frac_paths_raised": float(h.triggered_up.mean()),
        "median_adjustment": float(np.median(adjust[adjust != 0])) if np.any(adjust != 0) else 0.0,
    }


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
