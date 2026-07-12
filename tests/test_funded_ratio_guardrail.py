"""FundedRatioGuardrail (G2) — three-bucket model (PRD Stage 4 / source §9-11).

Covers precompute_pv, the effective multiplier the handler returns for the
combined discretionary outflow, the two independent dials (lifestyle: cut+raise;
optional-lumpy/gifts: trimmed only under severe stress, never raised above 1.0),
the no-op equivalence to guardrails=None, and the central claim: a planned gift
is baked into the PV lookahead so paying it does not itself trigger a cut.
"""
import numpy as np
import pytest

from engine.guardrails import (
    FundedRatioGuardrail,
    GuardrailConfig,
    build_handlers,
    precompute_pv,
)
from engine.params import Band, Lump, Property, SimulationParams
from engine.simulation import run_simulation
from cli import read_scenario_data

G2_DEFAULTS = dict(fr_lower=0.85, fr_target=1.05, fr_upper=1.30)


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


def _handler(pv_committed, pv_lifestyle, pv_optional, lifestyle_out, gifts_out,
             pv_income=None, **opts):
    """pv_income defaults to zeros: assets == bal, so hand-computed expected
    values below stay exact."""
    if pv_income is None:
        pv_income = np.zeros(len(np.atleast_1d(pv_committed)))
    ctx = {k: np.asarray(v, dtype=float) for k, v in dict(
        pv_committed=pv_committed, pv_income=pv_income,
        pv_lifestyle=pv_lifestyle, pv_optional=pv_optional,
        lifestyle_out=lifestyle_out, gifts_out=gifts_out).items()}
    return FundedRatioGuardrail({**G2_DEFAULTS, **opts}, ctx)


###############################################################################
# precompute_pv
###############################################################################

def test_precompute_pv_matches_hand_computed():
    # 10% discount rate gives clean integers: pv = [300, 220, 121]
    cf = np.array([100.0, 110.0, 121.0])
    np.testing.assert_allclose(precompute_pv(cf, 0.10), [300.0, 220.0, 121.0])


def test_precompute_pv_zero_rate_is_reverse_cumsum():
    cf = np.array([100.0, 100.0, 100.0, 100.0])
    np.testing.assert_allclose(precompute_pv(cf, 0.0), [400.0, 300.0, 200.0, 100.0])


###############################################################################
# Lifestyle dial — exact hand-crafted cases (pure lifestyle, no gifts, so the
# effective multiplier equals the lifestyle multiplier directly)
###############################################################################

def test_lifestyle_multiplier_cut_raise_deadzone_exact():
    """3 paths at year 0, pv_committed=0, pv_lifestyle=1000, no gifts:
      path A bal=700  -> funded 0.70 < 0.85 -> cut
      path B bal=1500 -> funded 1.50 > 1.30 -> raise (capped +5%/yr default)
      path C bal=1000 -> funded 1.00 in band -> no change
    With gifts_out=0 the returned effective multiplier == lifestyle multiplier.
    """
    h = _handler([0.0], [1000.0], [0.0], [100.0], [0.0])
    bal = np.array([700.0, 1500.0, 1000.0])
    eff = h.multiplier(0, port_ret=None, n_paths=3, bal=bal)
    # A: partial move toward target 0.66667; B: raise capped at +5% -> 1.05
    np.testing.assert_allclose(eff, [1 - 0.25 * (1 - 700 / 1.05 / 1000), 1.05, 1.0])
    assert h.triggered_down.tolist() == [True, False, False]
    assert h.triggered_up.tolist() == [False, True, False]


def test_dials_stay_within_bounds_over_many_years():
    """Drive the handler over noisy balances; lifestyle mult stays in
    [min,max], optional (gifts) mult stays in [0,1], effective in [0,max]."""
    T, n = 40, 500
    rng = np.random.default_rng(0)
    pv = np.linspace(1_000_000, 10_000, T)
    h = _handler(0.3 * pv, 0.5 * pv, 0.2 * pv,
                 np.full(T, 50_000.0), np.full(T, 20_000.0),
                 min_multiplier=0.40, max_multiplier=1.50)
    for t in range(T):
        bal = rng.uniform(0, 4_000_000, size=n)
        eff = h.multiplier(t, port_ret=None, n_paths=n, bal=bal)
        assert h.lifestyle_mult.min() >= 0.40 - 1e-9
        assert h.lifestyle_mult.max() <= 1.50 + 1e-9
        assert h.optional_mult.min() >= -1e-9
        assert h.optional_mult.max() <= 1.0 + 1e-9
        assert eff.min() >= -1e-9 and eff.max() <= 1.50 + 1e-9


###############################################################################
# Optional-lumpy (gifts) dial — the Stage-4 fix
###############################################################################

def test_optional_lumpy_is_never_raised_above_plan():
    """A wildly overfunded path raises lifestyle but must NOT inflate the
    planned gift: optional_mult stays pinned at 1.0, so the effective multiplier
    on a gift-dominated year barely moves above 1.0 (only the small lifestyle
    slice is raised). This is what the old two-bucket rule got wrong."""
    # gift-dominated year: lifestyle 100, gift 1000
    h = _handler([0.0], [100.0], [1000.0], [100.0], [1000.0])
    bal = np.array([1_000_000.0])  # funded ratio enormous -> raise
    eff = h.multiplier(0, port_ret=None, n_paths=1, bal=bal)
    assert h.optional_mult[0] == 1.0                  # gift never scaled up
    assert h.lifestyle_mult[0] > 1.0                  # lifestyle did raise
    # effective = (100*lifestyle_mult + 1000*1.0)/1100 -> only slightly > 1
    assert eff[0] < 1.02


def test_optional_lumpy_is_trimmed_only_under_severe_stress():
    h = _handler([0.0, 0.0], [1000.0, 1000.0], [1000.0, 1000.0],
                 [50.0, 50.0], [50.0, 50.0], fr_severe=0.80)
    # funded 0.90: not below fr_lower(0.85) or fr_severe(0.80) -> gift untouched
    h.multiplier(0, None, 1, bal=np.array([0.90 * 2000]))
    assert h.optional_mult[0] == 1.0
    # funded 0.70: below fr_severe -> gift trimmed 10%
    h.multiplier(1, None, 1, bal=np.array([0.70 * 2000]))
    assert h.optional_mult[0] == pytest.approx(0.90)


###############################################################################
# No-op equivalence
###############################################################################

def test_noop_when_thresholds_disabled_is_bit_identical():
    """fr_lower=0, fr_upper=inf, fr_severe=0 -> no dial ever moves -> identical
    to guardrails=None."""
    baseline = run_simulation(make_params())
    guarded = run_simulation(make_params(), guardrails=[
        GuardrailConfig(type="funded_ratio_guardrail",
                        options={"fr_lower": 0.0, "fr_target": 1.05,
                                 "fr_upper": float("inf"), "fr_severe": 0.0}),
    ])
    assert baseline["summary"] == guarded["summary"]
    np.testing.assert_array_equal(baseline["bal_over_time"], guarded["bal_over_time"])


###############################################################################
# The core claim: a planned gift is in the PV, so paying it is not a "shock"
###############################################################################

def test_planned_gift_payment_does_not_itself_trigger_a_cut():
    """A perfectly-funded path that pays a large planned gift stays at funded
    ratio ~1.0 across the gift year and is NEVER cut — the gift sits inside
    pv_optional, so balance and remaining need drop together. An underfunded
    path is still cut, proving the rule isn't inert."""
    lifestyle_out = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    gifts_out = np.array([0.0, 0.0, 400.0, 0.0, 0.0])   # 400 gift in year 2
    pv_life = precompute_pv(lifestyle_out, 0.0)          # [500,400,300,200,100]
    pv_opt = precompute_pv(gifts_out, 0.0)               # [400,400,400,0,0]
    pv_need = pv_life + pv_opt                            # [900,800,700,200,100]
    h = _handler(np.zeros(5), pv_life, pv_opt, lifestyle_out, gifts_out)

    for t in range(len(lifestyle_out)):
        bal = np.array([pv_need[t], 0.5 * pv_need[t]])   # funded 1.0 and 0.5
        h.multiplier(t, None, 2, bal=bal)

    assert h.triggered_down[0] == False, "funded gift-payer must never be cut"
    assert h.triggered_down[1] == True, "underfunded path must be cut"


###############################################################################
# Scenario-level: runs clean on the real example, stats have the right shape
###############################################################################

def test_g2_runs_on_example_scenario_and_reports_stats():
    scenario_data = read_scenario_data("scenario_data_example.xlsx")
    params = SimulationParams.from_legacy_scenario_data(scenario_data)
    res = run_simulation(params, guardrails=[
        GuardrailConfig(type="funded_ratio_guardrail", options=G2_DEFAULTS),
    ])
    stats = res["guardrail_stats"]
    assert stats["type"] == "funded_ratio_guardrail"
    assert 0.0 <= stats["frac_paths_triggered"] <= 1.0
    assert 0.0 <= res["summary"]["ruin_probability"] <= 1.0


def test_income_rich_plan_keeps_a_finite_meaningful_funded_ratio():
    """Asset-side regression (scenario_data_minus_parents bug, 2026-07-09):
    when planned income exceeds committed spending, liability-netting made the
    denominator negative -> floored to 1 -> funded ratio in the millions -> the
    raise leg fired regardless of fr_upper and the cut leg was unreachable.
    Asset-side: FR = (bal + pv_income)/pv_need stays near 1.x and thresholds
    discriminate. Here: bal=100, pv_income=5000, needs=1000+1000 -> FR=2.55.
    Netting would give FR = 100/max(1000-5000+1000, 1) = 100 -> raise even at
    fr_upper=3. Asset-side must NOT raise at 3.0 and MUST raise at 2.0."""
    for fr_upper, should_raise in [(3.0, False), (2.0, True)]:
        h = _handler([1000.0], [1000.0], [0.0], [100.0], [0.0],
                     pv_income=[5000.0], fr_upper=fr_upper)
        h.multiplier(0, None, 1, bal=np.array([100.0]))
        assert bool(h.triggered_up[0]) == should_raise, f"fr_upper={fr_upper}"
        assert not h.triggered_down[0]


def test_playground_events_are_unplanned_shocks_not_reserved_liabilities():
    """A playground (what-if) event must hit the portfolio like any lump, but
    NOT be reserved in the funded-ratio PV. So: identical portfolio path to a
    planned strict lump when there is no guardrail (same cashflow, flag ignored),
    but a DIFFERENT run under G2 — the planned lump is discounted into pv_committed
    (funded ratio barely moves when paid) while the playground shock is not
    (funded ratio drops at the shock, driving future cuts)."""
    age, amt = 72, -1_000_000
    planned = make_params(lumps=[Lump(age, amt, "planned repair", "strict")])
    playgr = make_params(lumps=[Lump(age, amt, "unplanned repair", "strict", playground=True)])

    # Same cashflow: without guardrails the playground flag is invisible.
    np.testing.assert_array_equal(
        run_simulation(planned)["bal_over_time"],
        run_simulation(playgr)["bal_over_time"])

    cfg = [GuardrailConfig(type="funded_ratio_guardrail", options=G2_DEFAULTS)]
    r_planned = run_simulation(planned, guardrails=cfg)
    r_playgr = run_simulation(playgr, guardrails=cfg)
    # With G2 they must diverge — proving the playground shock is kept out of the
    # PV lookahead (revert the exclusion and these become identical).
    assert not np.array_equal(r_planned["bal_over_time"], r_playgr["bal_over_time"])


def test_g2_monthly_mode_discounts_per_month_and_stays_sane():
    """Monthly mode: the real discount rate must be converted to a monthly
    equivalent (else far-future liabilities are over-discounted to ~0 and the
    funded ratio is garbage). Guard against crashes/NaNs and degenerate output."""
    res = run_simulation(make_params(annual=False, n_paths=500), guardrails=[
        GuardrailConfig(type="funded_ratio_guardrail", options=G2_DEFAULTS)])
    ruin = res["summary"]["ruin_probability"]
    assert 0.0 <= ruin <= 1.0
    assert np.isfinite(res["summary"]["estate_median"])
    gs = res["guardrail_stats"]
    assert 0.0 <= gs["frac_paths_triggered"] <= 1.0


def test_unknown_guardrail_type_still_raises_with_context_arg():
    with pytest.raises(ValueError):
        build_handlers([GuardrailConfig(type="nope", options={})], context={})
