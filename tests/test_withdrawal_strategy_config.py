"""Acceptance tests for Phase 2 of the two-bucket strategy PRD (docs/
two_bucket_retirement_strategy_PRD.md §7): config dataclasses, scenario
parsing, validation, and XLSX persistence. Architect-owned — implementation
must satisfy these tests, not the other way around."""

import copy
import dataclasses

import pytest

from engine.params import SimulationParams, validate_scenario, scenario_to_xlsx, scenario_from_xlsx
from engine.withdrawal_strategies import (
    ReturnModelConfig,
    ReserveConfig,
    DrawPolicyConfig,
    RefillPolicyConfig,
    CashflowPolicyConfig,
    WithdrawalStrategyConfig,
    parse_withdrawal_strategy,
    validate_withdrawal_strategy_block,
    STREAM_RESERVE_RETURNS,
)

BASE_SCENARIO = {
    "$schema": "scenario.v1",
    "name": "wd-config-test",
    "portfolio": {
        "initial_portfolio": 5_000_000,
        "start_age": 60,
        "end_age": 90,
        "market": "IL",
        "fat_tails_enabled": False,
        "fat_tails_df": 5,
        "mode": "annual",
        "n_paths": 200,
        "random_seed": 7,
        "mu": 0.05,
        "sigma": 0.15,
        "real_discount_rate": 0.01,
    },
    "spending_bands": [
        {"age_from": 60, "age_to": 90, "amount_monthly": 20000,
         "label": "base", "category": "strict"},
    ],
    "income_bands": [],
    "lumps": [],
    "properties": [],
}

# Deliberately non-default everywhere the XLSX columns can express.
TWO_BUCKET_BLOCK = {
    "type": "two_bucket",
    "reserve": {
        "target_years": 5.0,
        "refill_trigger_years": 2.5,
        "coverage_scope": "all_planned_outflows",
        "return_model": {
            "distribution": "student_t",
            "mean_real": 0.015,
            "std_real": 0.05,
            "student_t_df": 6.0,
        },
    },
    "draw_policy": {
        "market_state_rule": "previous_period_return",
        "growth_return_threshold_real": -0.02,
        "first_period_source": "growth",
    },
    "refill_policy": {
        "cadence": "annual",
        "eligibility_rule": "always",
        "growth_return_threshold_real": 0.01,
        "amount_rule": "gains_only",
    },
}


def scenario_with(block):
    s = copy.deepcopy(BASE_SCENARIO)
    if block is not None:
        s["withdrawal_strategy"] = copy.deepcopy(block)
    return s


# ---------------------------------------------------------------- dataclasses

def test_defaults_describe_single_portfolio():
    cfg = WithdrawalStrategyConfig()
    assert cfg.type == "single_portfolio"
    assert cfg.reserve.target_years == 4.0
    assert cfg.reserve.refill_trigger_years == 3.0
    assert cfg.reserve.coverage_scope == "recurring_gap_only"
    assert cfg.draw_policy.market_state_rule == "previous_period_return"
    assert cfg.draw_policy.first_period_source == "reserve"
    assert cfg.refill_policy.cadence == "annual"
    assert cfg.refill_policy.amount_rule == "to_target"
    assert cfg.correlation_with_growth == 0.0
    # deferred fields must default to actual MVP behavior, not aspirations
    assert cfg.cashflow_policy.planned_gift_draw == "strategy_rule"
    assert cfg.cashflow_policy.planned_lump_draw == "strategy_rule"
    assert cfg.cashflow_policy.unplanned_event_draw == "strategy_rule"
    assert cfg.cashflow_policy.surplus_allocation == "reserve_then_growth"


def test_configs_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        WithdrawalStrategyConfig().type = "two_bucket"
    with pytest.raises(dataclasses.FrozenInstanceError):
        ReserveConfig().target_years = 1.0


def test_stream_registry_pinned():
    # Append-only registry: renumbering breaks reproducibility of saved seeds.
    assert STREAM_RESERVE_RETURNS == 1


# ------------------------------------------------------------------- parsing

def test_parse_absent_is_none():
    assert parse_withdrawal_strategy(None) is None


def test_parse_full_block():
    cfg = parse_withdrawal_strategy(TWO_BUCKET_BLOCK)
    assert cfg.type == "two_bucket"
    assert cfg.reserve.target_years == 5.0
    assert cfg.reserve.refill_trigger_years == 2.5
    assert cfg.reserve.coverage_scope == "all_planned_outflows"
    assert cfg.reserve.return_model == ReturnModelConfig(
        distribution="student_t", mean_real=0.015, std_real=0.05, student_t_df=6.0)
    assert cfg.draw_policy.growth_return_threshold_real == -0.02
    assert cfg.draw_policy.first_period_source == "growth"
    assert cfg.refill_policy.eligibility_rule == "always"
    assert cfg.refill_policy.growth_return_threshold_real == 0.01
    assert cfg.refill_policy.amount_rule == "gains_only"


def test_parse_minimal_block_fills_defaults():
    cfg = parse_withdrawal_strategy({"type": "two_bucket"})
    assert cfg == WithdrawalStrategyConfig(type="two_bucket")


def test_from_scenario_threads_strategy():
    assert SimulationParams.from_scenario(scenario_with(None)).withdrawal_strategy is None
    params = SimulationParams.from_scenario(scenario_with(TWO_BUCKET_BLOCK))
    assert params.withdrawal_strategy == parse_withdrawal_strategy(TWO_BUCKET_BLOCK)


def test_existing_call_sites_unaffected():
    # One new optional field only: constructing without it must keep working.
    p = SimulationParams(start_age=60, end_age=90, initial_portfolio=1.0,
                         real_return_mean=0.05, real_return_sd=0.1, fat_tails_df=None)
    assert p.withdrawal_strategy is None


# ---------------------------------------------------------------- validation

def _errors(block):
    return validate_scenario(scenario_with(block))


def test_valid_two_bucket_block_passes():
    assert _errors(TWO_BUCKET_BLOCK) == []
    assert _errors(None) == []
    assert _errors({"type": "single_portfolio"}) == []


@pytest.mark.parametrize("mutate,token", [
    (lambda b: b.__setitem__("type", "three_bucket"), "type"),
    (lambda b: b["reserve"].__setitem__("target_years", -1.0), "target_years"),
    (lambda b: b["reserve"].__setitem__("refill_trigger_years", 9.0), "refill_trigger_years"),
    (lambda b: b["reserve"].__setitem__("coverage_scope", "everything"), "coverage_scope"),
    (lambda b: b["reserve"]["return_model"].__setitem__("std_real", -0.1), "std_real"),
    (lambda b: b["reserve"]["return_model"].__setitem__("student_t_df", 2.0), "student_t_df"),
    (lambda b: b["reserve"]["return_model"].__setitem__("distribution", "lognormal"), "distribution"),
    (lambda b: b["draw_policy"].__setitem__("first_period_source", "pro_rata"), "first_period_source"),
    (lambda b: b["refill_policy"].__setitem__("eligibility_rule", "sometimes"), "eligibility_rule"),
    (lambda b: b["refill_policy"].__setitem__("amount_rule", "fixed_amount"), "amount_rule"),
])
def test_invalid_values_rejected(mutate, token):
    block = copy.deepcopy(TWO_BUCKET_BLOCK)
    mutate(block)
    errors = _errors(block)
    assert errors, f"expected a validation error for bad {token}"
    assert any(token in e for e in errors), f"no error mentions {token}: {errors}"


def test_constant_requires_zero_std():
    block = copy.deepcopy(TWO_BUCKET_BLOCK)
    block["reserve"]["return_model"] = {"distribution": "constant",
                                        "mean_real": 0.01, "std_real": 0.02}
    errors = _errors(block)
    assert any("std_real" in e for e in errors)


@pytest.mark.parametrize("mutate", [
    lambda b: b.__setitem__("correlation_with_growth", 0.3),
    lambda b: b["draw_policy"].__setitem__("market_state_rule", "recovery_aware"),
    lambda b: b["refill_policy"].__setitem__("cadence", "monthly"),
    lambda b: b.__setitem__("cashflow_policy", {"planned_gift_draw": "growth_first"}),
])
def test_deferred_features_are_explicit_errors(mutate):
    # PRD §7.3: silently accepting a deferred option would misrepresent results.
    block = copy.deepcopy(TWO_BUCKET_BLOCK)
    mutate(block)
    errors = _errors(block)
    assert any("not implemented" in e for e in errors), errors


def test_single_portfolio_skips_two_bucket_validation():
    # A single_portfolio block with junk two_bucket sub-config must not error.
    block = {"type": "single_portfolio",
             "reserve": {"target_years": -5, "coverage_scope": "junk"}}
    assert _errors(block) == []


# --------------------------------------------------------------- persistence

def test_xlsx_roundtrip_two_bucket(tmp_path):
    path = tmp_path / "two_bucket.xlsx"
    scenario_to_xlsx(scenario_with(TWO_BUCKET_BLOCK), path)
    loaded = scenario_from_xlsx(path)
    # Compare parsed configs, not raw dicts: XLSX does not carry the deferred
    # cashflow_policy/correlation fields, which parse to defaults on both sides.
    assert (parse_withdrawal_strategy(loaded["withdrawal_strategy"])
            == parse_withdrawal_strategy(TWO_BUCKET_BLOCK))


def test_xlsx_roundtrip_normal_distribution_df_none(tmp_path):
    block = copy.deepcopy(TWO_BUCKET_BLOCK)
    block["reserve"]["return_model"] = {"distribution": "normal",
                                        "mean_real": 0.0, "std_real": 0.03,
                                        "student_t_df": None}
    path = tmp_path / "normal.xlsx"
    scenario_to_xlsx(scenario_with(block), path)
    loaded = scenario_from_xlsx(path)
    rm = parse_withdrawal_strategy(loaded["withdrawal_strategy"]).reserve.return_model
    assert rm == ReturnModelConfig(distribution="normal", mean_real=0.0,
                                   std_real=0.03, student_t_df=None)


def test_xlsx_without_block_loads_as_absent(tmp_path):
    path = tmp_path / "plain.xlsx"
    scenario_to_xlsx(scenario_with(None), path)
    loaded = scenario_from_xlsx(path)
    assert loaded.get("withdrawal_strategy") is None
    assert SimulationParams.from_scenario(loaded).withdrawal_strategy is None


def test_xlsx_explicit_single_portfolio_roundtrip(tmp_path):
    path = tmp_path / "single.xlsx"
    scenario_to_xlsx(scenario_with({"type": "single_portfolio"}), path)
    loaded = scenario_from_xlsx(path)
    ws = loaded.get("withdrawal_strategy")
    assert ws is None or parse_withdrawal_strategy(ws).type == "single_portfolio"
