"""Golden-scenario builders for single-portfolio regression fixtures.

This module is NOT a test file (no `test_` prefix) — it defines the four
pinned scenarios whose outputs are captured as golden fixtures and verified
bit-exactly in `tests/test_golden_single_mode.py`.

Each builder returns a ``(params, guardrails)`` tuple ready for
``engine.simulation.run_simulation(params, guardrails=guardrails)``.
"""

from engine.guardrails import GuardrailConfig
from engine.params import Band, Lump, Property, SimulationParams


def _base_params(**over):
    """Rich plan with spending bands in all three categories, an income band,
    one positive lump, one negative lump, and a property with rent."""
    base = dict(
        start_age=60, end_age=95, initial_portfolio=3_000_000,
        real_return_mean=0.042, real_return_sd=0.13, fat_tails_df=5,
        annual=True, n_paths=1000, random_seed=42,
        spending_bands=[
            Band(60, 95, 300_000, "base", "strict"),
            Band(60, 80, 60_000, "travel", "lifestyle"),
            Band(75, 85, 20_000, "grandkids", "gifts"),
        ],
        income_bands=[Band(60, 67, 144_000, "consulting")],
        lumps=[
            Lump(65, 200_000, "bonus", "strict"),
            Lump(70, -400_000, "gift", "gifts"),
        ],
        properties=[Property(60, 2_500_000, 72_000, 0.018, 0.08, "apt")],
    )
    base.update(over)
    return SimulationParams(**base)


def annual_plain():
    """Annual mode, no guardrails, fat_tails_df=5 (Student-t paths)."""
    return _base_params(), None


def annual_volatility():
    """Annual mode with a volatility-discretionary-scaling guardrail."""
    params = _base_params()
    guardrails = [GuardrailConfig(
        type="volatility_discretionary_scaling",
        options=dict(drop_threshold=0.20, rise_threshold=0.20,
                     cut_pct=0.15, raise_pct=0.10),
    )]
    return params, guardrails


def annual_confidence():
    """Annual mode with a funded-ratio guardrail in confidence mode.

    This exercises the engine's internal baseline calibration pass
    (same-seed inner run_simulation with guardrails=None).
    """
    params = _base_params()
    guardrails = [GuardrailConfig(
        type="funded_ratio_guardrail",
        options=dict(mode="confidence", c_cut=0.85, c_target=0.95,
                     c_raise=0.99, c_severe=0.80),
    )]
    return params, guardrails


def monthly_plain():
    """Monthly mode, no guardrails, fewer paths for speed."""
    return _base_params(annual=False, n_paths=500), None


SCENARIOS = {
    "annual_plain": annual_plain,
    "annual_volatility": annual_volatility,
    "annual_confidence": annual_confidence,
    "monthly_plain": monthly_plain,
}
