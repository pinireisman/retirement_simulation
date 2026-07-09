"""Stage 3 acceptance: pin FundedRatioGuardrail (G2) behavior on the real
example scenario at the chosen defaults, so future changes to the guardrail
math can't silently drift the numbers. Mirrors test_engine_baseline.py.

Three-bucket model (Stage 4): committed (strict) is protected; `lifestyle` is
the primary cut+raise dial; `gifts` is an optional-lumpy dial capped at 1.0
(never given above plan) and trimmed only under severe stress. Chosen defaults
(judged sensible in Stage 3/4 calibration on the realistic 8M example):
fr_lower=0.85, fr_target=1.05, fr_upper=1.30, fr_severe=0.80,
real_discount_rate=0.01. On that scenario G2 is near-neutral — a heavily-
overfunded plan spends a touch more (estate median ~62.3M -> ~61.6M) with a
negligible ruin change (0.0001 -> 0.0003).

Stage 4 fixed the two-bucket blowup where the raise leg inflated large planned
gifts and drove ruin up on cash-starved variants (e.g. 1.5M initial: 0.14 ->
0.76 under the old two-bucket, now 0.14 -> 0.19). Capping the gifts dial at 1.0
is what removed it; the lifestyle raise leg is also throttled to 5%/yr. The
residual modest increase on stressed plans is the intended lifestyle-raise
behavior, not a defect.
"""
import numpy as np

from engine.guardrails import GuardrailConfig
from engine.params import SimulationParams
from engine.simulation import run_simulation
from cli import read_scenario_data

G2_DEFAULTS = dict(fr_lower=0.85, fr_target=1.05, fr_upper=1.30)


def _example_params():
    sd = read_scenario_data("scenario_data_example.xlsx")
    return SimulationParams.from_legacy_scenario_data(sd)  # 8M, real_discount_rate=0.01


def test_g2_defaults_pinned_on_example_scenario():
    params = _example_params()
    out = run_simulation(params, guardrails=[
        GuardrailConfig(type="funded_ratio_guardrail", options=G2_DEFAULTS),
    ])
    ruin = out["summary"]["ruin_probability"]
    estate = out["summary"]["estate_median"]
    gs = out["guardrail_stats"]

    # Pinned from the verified Stage-4 build (seed fixed). Loose bounds cover
    # numpy RNG-stream drift across platforms/versions.
    assert abs(ruin - 0.0003) < 0.003
    assert abs(estate - 61_638_251) < 0.01 * 61_638_251   # within 1%
    assert gs["type"] == "funded_ratio_guardrail"
    assert gs["frac_paths_raised"] > 0.9                   # overfunded plan -> raises lifestyle
    assert gs["frac_paths_cut"] < 0.05                     # rarely behind here


def test_g2_is_seeded_deterministic():
    params = _example_params()
    a = run_simulation(params, guardrails=[
        GuardrailConfig(type="funded_ratio_guardrail", options=G2_DEFAULTS)])
    b = run_simulation(params, guardrails=[
        GuardrailConfig(type="funded_ratio_guardrail", options=G2_DEFAULTS)])
    assert a["summary"]["ruin_probability"] == b["summary"]["ruin_probability"]
    np.testing.assert_array_equal(a["bal_over_time"], b["bal_over_time"])
