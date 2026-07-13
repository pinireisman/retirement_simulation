"""Deterministic strategy tests for TwoBucketStrategy (PRD §8.4, §14.2).

Two layers:
  (i)  unit tests constructing TwoBucketStrategy directly with hand-built
       2-4-path port_ret/reserve_ret matrices (mixed up/down sequences give
       exact assertions);
  (ii) integration tests through run_simulation with real_return_sd=0,
       fat_tails_df=None (fully deterministic) and constant reserve
       returns, with hand-computed balances.
"""

import numpy as np
import pytest

from engine.params import Band, SimulationParams
from engine.simulation import run_simulation
from engine.withdrawal_strategies import (
    DrawPolicyConfig,
    RefillPolicyConfig,
    ReserveConfig,
    ReturnModelConfig,
    TwoBucketStrategy,
    WithdrawalStrategyConfig,
)


# ---------------------------------------------------------------- unit-test scaffolding

class _FakeParams:
    """Minimal params stand-in -- TwoBucketStrategy.__init__ only reads
    .annual, .initial_portfolio, .withdrawal_strategy."""

    def __init__(self, initial_portfolio, withdrawal_strategy, annual=True):
        self.annual = annual
        self.initial_portfolio = initial_portfolio
        self.withdrawal_strategy = withdrawal_strategy


def _make_strategy(port_ret, reserve_ret, planned_gap, *, initial_portfolio=0.0,
                    target_years=4.0, trigger_years=3.0, first_period_source="reserve",
                    threshold=0.0, eligibility_rule="growth_return_at_or_above_threshold",
                    amount_rule="to_target", refill_threshold=0.0):
    wcfg = WithdrawalStrategyConfig(
        type="two_bucket",
        reserve=ReserveConfig(target_years=target_years, refill_trigger_years=trigger_years,
                               coverage_scope="recurring_gap_only",
                               return_model=ReturnModelConfig()),
        draw_policy=DrawPolicyConfig(market_state_rule="previous_period_return",
                                      growth_return_threshold_real=threshold,
                                      first_period_source=first_period_source),
        refill_policy=RefillPolicyConfig(cadence="annual", eligibility_rule=eligibility_rule,
                                          growth_return_threshold_real=refill_threshold,
                                          amount_rule=amount_rule),
    )
    return TwoBucketStrategy(_FakeParams(initial_portfolio, wcfg), port_ret, reserve_ret, planned_gap)


# ---------------------------------------------------------------- (i) unit tests, §14.2 items 1-15

def test_1_positive_previous_return_draws_growth_first():
    port_ret = np.array([[0.0, 0.05]])
    strat = _make_strategy(port_ret, np.zeros((1, 2)), np.zeros(2))
    strat.growth = np.array([100.0]); strat.reserve = np.array([100.0])
    strat.apply_cashflow(1, np.array([-30.0]))  # favorable[:,1] = port_ret[:,0]=0.0 >= 0 -> True
    assert strat.growth[0] == pytest.approx(70.0)
    assert strat.reserve[0] == pytest.approx(100.0)


def test_2_negative_previous_return_draws_reserve_first():
    port_ret = np.array([[-0.1, 0.05]])
    strat = _make_strategy(port_ret, np.zeros((1, 2)), np.zeros(2))
    strat.growth = np.array([100.0]); strat.reserve = np.array([100.0])
    strat.apply_cashflow(1, np.array([-30.0]))  # favorable[:,1] = -0.1 >= 0 -> False
    assert strat.reserve[0] == pytest.approx(70.0)
    assert strat.growth[0] == pytest.approx(100.0)


def test_3_first_period_uses_configured_fallback_source():
    port_ret = np.zeros((1, 2))
    strat_reserve_first = _make_strategy(port_ret, np.zeros((1, 2)), np.zeros(2),
                                          first_period_source="reserve")
    strat_reserve_first.growth = np.array([100.0]); strat_reserve_first.reserve = np.array([100.0])
    strat_reserve_first.apply_cashflow(0, np.array([-30.0]))
    assert strat_reserve_first.reserve[0] == pytest.approx(70.0)
    assert strat_reserve_first.growth[0] == pytest.approx(100.0)

    strat_growth_first = _make_strategy(port_ret, np.zeros((1, 2)), np.zeros(2),
                                         first_period_source="growth")
    strat_growth_first.growth = np.array([100.0]); strat_growth_first.reserve = np.array([100.0])
    strat_growth_first.apply_cashflow(0, np.array([-30.0]))
    assert strat_growth_first.growth[0] == pytest.approx(70.0)
    assert strat_growth_first.reserve[0] == pytest.approx(100.0)


def test_4_preferred_bucket_insufficiency_spills_into_other_bucket():
    port_ret = np.array([[0.0, 0.05]])  # favorable -> growth preferred at period 1
    strat = _make_strategy(port_ret, np.zeros((1, 2)), np.zeros(2))
    strat.growth = np.array([20.0]); strat.reserve = np.array([100.0])
    strat.apply_cashflow(1, np.array([-30.0]))
    assert strat.growth[0] == pytest.approx(0.0)
    assert strat.reserve[0] == pytest.approx(90.0)  # 10 spilled over from growth's shortfall


def test_5_reserve_depletion_alone_does_not_mark_ruin():
    port_ret = np.array([[-0.1, 0.0]])  # reserve preferred
    strat = _make_strategy(port_ret, np.zeros((1, 2)), np.zeros(2))
    strat.growth = np.array([1000.0]); strat.reserve = np.array([10.0])
    strat.apply_cashflow(1, np.array([-30.0]))
    assert strat.reserve[0] == pytest.approx(0.0)
    assert strat.combined_balance()[0] == pytest.approx(980.0)  # 1010 - 30, positive => not ruin


def test_6_combined_insufficiency_marks_ruin_with_existing_semantics():
    port_ret = np.zeros((1, 2))
    strat = _make_strategy(port_ret, np.zeros((1, 2)), np.zeros(2))
    strat.growth = np.array([10.0]); strat.reserve = np.array([5.0])
    strat.apply_cashflow(1, np.array([-30.0]))
    combined = strat.combined_balance()
    assert combined[0] == pytest.approx(-15.0)  # old_combined(15) + net(-30) exactly
    ruined = combined <= 0
    strat.mark_ruined(ruined)
    assert strat.growth[0] == 0.0
    assert strat.reserve[0] == 0.0


def test_7_surplus_fills_reserve_to_target_then_growth():
    strat = _make_strategy(np.zeros((1, 2)), np.zeros((1, 2)), np.zeros(2))
    strat.target_by_period = np.array([50.0, 50.0])
    strat.growth = np.array([0.0]); strat.reserve = np.array([20.0])
    strat.apply_cashflow(0, np.array([100.0]))  # surplus 100
    assert strat.reserve[0] == pytest.approx(50.0)  # 30 to fill reserve to target
    assert strat.growth[0] == pytest.approx(70.0)   # remainder 70 to growth


def test_8_refill_only_below_trigger():
    port_ret = np.array([[0.05, 0.0]])
    strat = _make_strategy(port_ret, np.zeros((1, 2)), np.zeros(2), eligibility_rule="always")
    strat.trigger_by_period = np.array([0.0, 40.0])
    strat.target_by_period = np.array([0.0, 100.0])
    strat.growth = np.array([200.0]); strat.reserve = np.array([50.0])  # 50 >= trigger(40): not eligible
    strat.apply_returns(0, np.array([True]))
    strat.end_of_period(0, np.array([True]))
    assert strat.reserve[0] == pytest.approx(50.0)
    assert strat.growth[0] == pytest.approx(210.0)  # only the return applied, no refill


def test_9_refill_does_not_occur_when_eligibility_fails():
    port_ret = np.array([[-0.05, 0.0]])  # just-completed return below threshold
    strat = _make_strategy(port_ret, np.zeros((1, 2)), np.zeros(2),
                            eligibility_rule="growth_return_at_or_above_threshold", refill_threshold=0.0)
    strat.trigger_by_period = np.array([0.0, 40.0])
    strat.target_by_period = np.array([0.0, 100.0])
    strat.growth = np.array([200.0]); strat.reserve = np.array([10.0])  # below trigger
    strat.apply_returns(0, np.array([True]))
    strat.end_of_period(0, np.array([True]))
    assert strat.reserve[0] == pytest.approx(10.0)  # unchanged: eligibility failed


def test_10_to_target_reaches_target_when_growth_has_enough_funds():
    port_ret = np.array([[0.05, 0.0]])
    strat = _make_strategy(port_ret, np.zeros((1, 2)), np.zeros(2), eligibility_rule="always")
    strat.trigger_by_period = np.array([0.0, 40.0])
    strat.target_by_period = np.array([0.0, 100.0])
    strat.growth = np.array([200.0]); strat.reserve = np.array([10.0])
    strat.apply_returns(0, np.array([True]))  # growth -> 210
    strat.end_of_period(0, np.array([True]))
    assert strat.reserve[0] == pytest.approx(100.0)  # need = 100-10 = 90, growth has plenty
    assert strat.growth[0] == pytest.approx(120.0)   # 210 - 90


def test_11_gains_only_never_transfers_more_than_the_positive_gain():
    port_ret = np.array([[0.05, 0.0]])
    strat = _make_strategy(port_ret, np.zeros((1, 2)), np.zeros(2),
                            eligibility_rule="always", amount_rule="gains_only")
    strat.trigger_by_period = np.array([0.0, 1000.0])  # always below trigger
    strat.target_by_period = np.array([0.0, 1000.0])   # need is huge, not the binding constraint
    strat.growth = np.array([200.0]); strat.reserve = np.array([0.0])
    strat.apply_returns(0, np.array([True]))  # exposed=200, gain=200*0.05=10; growth -> 210
    strat.end_of_period(0, np.array([True]))
    assert strat.reserve[0] == pytest.approx(10.0)   # capped at the gain, not the (huge) need
    assert strat.growth[0] == pytest.approx(200.0)   # 210 - 10


def test_12_internal_transfers_preserve_combined_wealth():
    port_ret = np.array([[0.05, 0.0]])
    strat = _make_strategy(port_ret, np.zeros((1, 2)), np.zeros(2), eligibility_rule="always")
    strat.trigger_by_period = np.array([0.0, 40.0])
    strat.target_by_period = np.array([0.0, 100.0])
    strat.growth = np.array([200.0]); strat.reserve = np.array([10.0])
    strat.apply_returns(0, np.array([True]))
    combined_before_refill = strat.combined_balance().copy()
    strat.end_of_period(0, np.array([True]))
    np.testing.assert_allclose(strat.combined_balance(), combined_before_refill)


def test_13_returns_apply_only_to_the_balance_exposed_in_each_bucket():
    port_ret = np.array([[0.10, 0.0]])
    reserve_ret = np.array([[0.02, 0.0]])
    strat = _make_strategy(port_ret, reserve_ret, np.zeros(2))
    strat.growth = np.array([100.0]); strat.reserve = np.array([100.0])
    strat.apply_returns(0, np.array([True]))
    assert strat.growth[0] == pytest.approx(110.0)
    assert strat.reserve[0] == pytest.approx(102.0)


def test_14_refilled_money_earns_no_current_period_return():
    port_ret = np.array([[0.05, 0.20]])  # huge next-period return must not leak into this period
    strat = _make_strategy(port_ret, np.zeros((1, 2)), np.zeros(2), eligibility_rule="always")
    strat.trigger_by_period = np.array([0.0, 40.0])
    strat.target_by_period = np.array([0.0, 100.0])
    strat.growth = np.array([200.0]); strat.reserve = np.array([10.0])
    strat.apply_returns(0, np.array([True]))   # growth -> 210
    strat.end_of_period(0, np.array([True]))   # transfer 90 -> reserve = 100 (NOT 100*1.20)
    assert strat.reserve[0] == pytest.approx(100.0)


def test_15_same_period_return_never_used_for_same_period_draw_decision():
    # period 1's favorable state depends only on port_ret[:,0]; the period-1
    # return itself must be irrelevant to the period-1 draw decision.
    port_ret_a = np.array([[0.0, 0.99]])
    port_ret_b = np.array([[0.0, -0.99]])
    strat_a = _make_strategy(port_ret_a, np.zeros((1, 2)), np.zeros(2))
    strat_b = _make_strategy(port_ret_b, np.zeros((1, 2)), np.zeros(2))
    for s in (strat_a, strat_b):
        s.growth = np.array([100.0]); s.reserve = np.array([100.0])
    strat_a.apply_cashflow(1, np.array([-30.0]))
    strat_b.apply_cashflow(1, np.array([-30.0]))
    np.testing.assert_allclose(strat_a.growth, strat_b.growth)
    np.testing.assert_allclose(strat_a.reserve, strat_b.reserve)


# ---------------------------------------------------------------- §8.7 accounting invariant

def test_accounting_invariant_holds_every_period():
    """combined[t] == combined[t-1] + net + growth_gain + reserve_gain on
    every period, checked in two composable steps (cashflow, then returns);
    the refill step must leave combined wealth untouched."""
    n_periods = 3
    port_ret = np.array([[0.05, -0.03, 0.08],
                          [0.02, 0.01, -0.02]])
    reserve_ret = np.array([[0.01, 0.01, 0.01],
                             [0.02, 0.00, 0.03]])
    planned_gap = np.zeros(n_periods)
    strat = _make_strategy(port_ret, reserve_ret, planned_gap)
    strat.growth = np.array([500.0, 300.0])
    strat.reserve = np.array([200.0, 400.0])
    nets = [np.array([-50.0, 30.0]), np.array([20.0, -80.0]), np.array([-10.0, -5.0])]
    alive = np.array([True, True])

    for i in range(n_periods):
        combined_before = strat.combined_balance().copy()
        strat.apply_cashflow(i, nets[i])
        combined_after_cf = strat.combined_balance().copy()
        np.testing.assert_allclose(combined_after_cf, combined_before + nets[i])

        exposed_growth = strat.growth.copy()
        exposed_reserve = strat.reserve.copy()
        strat.apply_returns(i, alive)
        growth_gain = exposed_growth * port_ret[:, i]
        reserve_gain = exposed_reserve * reserve_ret[:, i]
        combined_after_ret = strat.combined_balance().copy()
        np.testing.assert_allclose(combined_after_ret, combined_after_cf + growth_gain + reserve_gain)

        strat.end_of_period(i, alive)
        np.testing.assert_allclose(strat.combined_balance(), combined_after_ret)  # refill cancels out
        strat.record(i)


# ---------------------------------------------------------------- (ii) integration, hand-computed

def _wcfg(target_years, trigger_years, coverage_scope="recurring_gap_only",
          distribution="constant", mean_real=0.0, std_real=0.0):
    return WithdrawalStrategyConfig(
        type="two_bucket",
        reserve=ReserveConfig(target_years=target_years, refill_trigger_years=trigger_years,
                               coverage_scope=coverage_scope,
                               return_model=ReturnModelConfig(distribution=distribution,
                                                               mean_real=mean_real, std_real=std_real)),
    )


def test_integration_two_period_hand_computed_balances():
    """start_age=60, end_age=61 (2 annual periods). Flat strict spend 300/yr,
    no income. Growth return 10% deterministic; reserve constant 2%.
    target_years=1.0 -> target[t] = gap[t] = 300 each period (1-period window).

    Hand trace:
      init: reserve0=min(1000,300)=300, growth0=700
      period 0: gap=300, reserve-first (default) -> from_res=300, from_gro=0
                growth=700, reserve=0; surplus=0
                returns: growth=700*1.10=770, reserve=0*1.02=0
                refill (need=300-0=300, elig: 0.10>=0 True): transfer=min(300,770)=300
                -> growth=470, reserve=300  (bal=770 recorded post-refill... wait growth_over_time
                   records POST-refill growth=470; combined recorded by loop = 470+300=770)
      period 1: favorable = port_ret[:,0]=0.10>=0 -> True (growth-first)
                gap=300, growth=470,reserve=300 -> from_gro=300, from_res=0
                growth=170, reserve=300
                returns: growth=170*1.10=187, reserve=300*1.02=306
                no forward window (last period) -> no refill
    Final: growth=187, reserve=306, combined=493.
    """
    wcfg = _wcfg(target_years=1.0, trigger_years=1.0, mean_real=0.02)
    params = SimulationParams(
        start_age=60, end_age=61, initial_portfolio=1000.0,
        real_return_mean=0.10, real_return_sd=0.0, fat_tails_df=None,
        annual=True, n_paths=1, random_seed=0,
        spending_bands=[Band(60, 61, 300.0, "base", "strict")],
        withdrawal_strategy=wcfg,
    )
    result = run_simulation(params, guardrails=None)

    np.testing.assert_allclose(result["growth_over_time"][:, 0], [470.0, 187.0])
    np.testing.assert_allclose(result["bal_over_time"][:, 0], [770.0, 493.0])
    assert result["final_growth"][0] == pytest.approx(187.0)
    assert result["final_reserve"][0] == pytest.approx(306.0)
    assert result["withdrawal_strategy_type"] == "two_bucket"
