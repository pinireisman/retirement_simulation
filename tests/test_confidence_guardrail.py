"""PRD_GOAL_BASED_GUARDRAILS.md Stage 1: confidence-anchored guardrails (G2.1).

Covers calibrate_wealth_needed (exact toy values + edges), confidence-mode
determinism, the same-seed baseline property, protect-mode's hard invariant
(success >= no-guardrail baseline), manual-mode compatibility, and the
confidence-mode no-op.
"""
import numpy as np
import pytest

from engine.guardrails import GuardrailConfig, calibrate_wealth_needed
from engine.params import Band, Lump, Property, SimulationParams
from engine.simulation import run_simulation
from cli import read_scenario_data


def make_params(**over):
    base = dict(
        start_age=60, end_age=95, initial_portfolio=3_000_000,
        real_return_mean=0.042, real_return_sd=0.13, fat_tails_df=5,
        annual=True, n_paths=2_000, random_seed=42,
        spending_bands=[Band(60, 95, 120_000, "base", "strict"),
                        Band(60, 90, 60_000, "travel", "lifestyle")],
        income_bands=[Band(60, 67, 144_000, "consulting")],
        lumps=[Lump(75, -400_000, "wedding gift", "gifts")],
        properties=[Property(60, 1_500_000, 60_000, 0.018, 0.08, "apt")],
    )
    base.update(over)
    return SimulationParams(**base)


###############################################################################
# calibrate_wealth_needed — exact toy values
###############################################################################

def test_calibrate_exact_on_hand_built_paths():
    """4 paths, 2 years. Balances at year 0: [400, 300, 200, 100]; success =
    [T, T, F, T] (the 200-path fails). Richest-first prefix success fractions:
    400->1/1, 300->2/2, 200->2/3, 100->3/4.
      p=1.00: last prefix with frac>=1 is the top-2 -> W = 300.
      p=0.75: full prefix qualifies (3/4) -> W = 100 (poorest).
      p=0.90: only top-2 (1.0) qualify -> W = 300.
    """
    start_bal = np.array([[400.0, 300.0, 200.0, 100.0],
                          [40.0, 30.0, 20.0, 10.0]])
    success = np.array([True, True, False, True])
    W = calibrate_wealth_needed(start_bal, success, (1.00, 0.90, 0.75))
    assert W[1.00][0] == 300.0
    assert W[0.90][0] == 300.0
    assert W[0.75][0] == 100.0
    # year 1 has the same ordering, scaled balances
    assert W[1.00][1] == 30.0 and W[0.75][1] == 10.0


def test_calibrate_edges_all_succeed_and_unreachable():
    start_bal = np.array([[100.0, 50.0]])
    W_all = calibrate_wealth_needed(start_bal, np.array([True, True]), (0.95,))
    assert W_all[0.95][0] == 50.0          # even the poorest qualifies
    W_none = calibrate_wealth_needed(start_bal, np.array([False, False]), (0.95,))
    assert W_none[0.95][0] == np.inf       # no wealth level reaches 95%


###############################################################################
# Confidence mode, end to end
###############################################################################

CONF_PROTECT = dict(mode="confidence", c_cut=0.90, c_target=0.97, c_raise=None)
CONF_BALANCED = dict(mode="confidence", c_cut=0.85, c_target=0.95, c_raise=0.99,
                     c_severe=0.80)


def _g2(options):
    return [GuardrailConfig(type="funded_ratio_guardrail", options=options)]


def test_confidence_mode_is_deterministic():
    a = run_simulation(make_params(), guardrails=_g2(CONF_BALANCED))
    b = run_simulation(make_params(), guardrails=_g2(CONF_BALANCED))
    assert a["summary"] == b["summary"]
    np.testing.assert_array_equal(a["bal_over_time"], b["bal_over_time"])


def test_baseline_summary_matches_a_manual_baseline_run():
    """The internal calibration pass reuses the same seed, so its summary must
    equal an explicit unguarded run of the same params."""
    guarded = run_simulation(make_params(), guardrails=_g2(CONF_BALANCED))
    manual = run_simulation(make_params())
    assert guarded["baseline_summary"] == manual["summary"]
    # and manual mode / no guardrails do not produce one
    assert run_simulation(make_params())["baseline_summary"] is None


def test_protect_mode_never_reduces_success():
    """The invariant this feature exists for: cut-only policy (c_raise=None)
    can trim discretionary spending but never raise it, so ruin must not
    increase vs baseline. Checked on a stressed scenario where cuts do fire."""
    sd = read_scenario_data("scenario_data_example.xlsx")
    params = SimulationParams.from_legacy_scenario_data(sd)
    params.initial_portfolio = 1_500_000    # stressed: baseline ruin ~14%
    base = run_simulation(params)
    prot = run_simulation(params, guardrails=_g2(CONF_PROTECT))
    assert prot["summary"]["ruin_probability"] <= base["summary"]["ruin_probability"]
    assert prot["guardrail_stats"]["frac_paths_raised"] == 0.0
    # guard against a silently-inert feature: cuts must actually fire here
    assert prot["guardrail_stats"]["frac_paths_cut"] > 0.0


def test_confidence_noop_when_thresholds_never_bind():
    """c_cut at the observable floor (poorest baseline path) and c_raise=None:
    no dial ever moves -> bit-identical to guardrails=None."""
    tiny = dict(mode="confidence", c_cut=0.0001, c_target=0.0001, c_raise=None)
    baseline = run_simulation(make_params())
    guarded = run_simulation(make_params(), guardrails=_g2(tiny))
    assert baseline["summary"] == guarded["summary"]
    np.testing.assert_array_equal(baseline["bal_over_time"], guarded["bal_over_time"])


def test_manual_mode_unchanged_by_confidence_feature():
    """Options without a mode key follow the manual path: no baseline pass, and
    the pinned funded-ratio baseline (test_funded_ratio_baseline) still holds —
    here just assert no calibration side effects leak in."""
    r = run_simulation(make_params(), guardrails=_g2(
        dict(fr_lower=0.85, fr_target=1.05, fr_upper=1.30)))
    assert r["baseline_summary"] is None
    assert r["guardrail_stats"]["type"] == "funded_ratio_guardrail"
