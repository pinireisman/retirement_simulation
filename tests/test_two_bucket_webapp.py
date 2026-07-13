"""Smoke coverage for Phase 8 web wiring: execute_run must build the four
two-bucket chart figures + comparison-aware KPI stats for a two_bucket
scenario, and must build none of that for a plain single_portfolio scenario
(same shape collect_edits now always emits)."""

import copy

import numpy as np

from webapp.callbacks import RESULTS_CACHE, _stat_tiles, execute_run
from webapp.layout import DEFAULT_SCENARIO

_TWO_BUCKET_BLOCK = {
    "type": "two_bucket",
    "reserve": {"target_years": 4.0, "refill_trigger_years": 3.0,
                "coverage_scope": "recurring_gap_only",
                "return_model": {"distribution": "normal", "mean_real": 0.01,
                                  "std_real": 0.03, "student_t_df": None}},
    "draw_policy": {"growth_return_threshold_real": 0.0, "first_period_source": "reserve"},
    "refill_policy": {"eligibility_rule": "growth_return_at_or_above_threshold",
                       "growth_return_threshold_real": 0.0, "amount_rule": "to_target"},
}


def _scenario(withdrawal_strategy):
    s = copy.deepcopy(DEFAULT_SCENARIO)
    s["portfolio"]["initial_portfolio"] = 5_000_000
    s["portfolio"]["n_paths"] = 300
    s["spending_bands"] = [{"id": "sb-1", "age_from": 60, "age_to": 95,
                             "amount_monthly": 10000, "label": "", "category": "lifestyle"}]
    s["withdrawal_strategy"] = withdrawal_strategy
    return s


def test_execute_run_builds_two_bucket_charts_and_comparison():
    run_id, figures, summary, badges, guardrail_stats, baseline_summary = execute_run(
        _scenario(_TWO_BUCKET_BLOCK), {}, [], False, True)
    names = [n for n, _ in figures["two_bucket"]]
    assert names == ["Growth vs. reserve balance", "Funding source by age",
                      "Strategy events", "Terminal balance vs. single portfolio"]
    assert "comparison" in RESULTS_CACHE[run_id]


def test_execute_run_no_two_bucket_charts_for_single_portfolio():
    block = {**_TWO_BUCKET_BLOCK, "type": "single_portfolio"}
    run_id, figures, summary, badges, guardrail_stats, baseline_summary = execute_run(
        _scenario(block), {}, [], False, True)
    assert figures["two_bucket"] == []
    assert "comparison" not in RESULTS_CACHE[run_id]


def test_stat_tiles_with_two_bucket_stats_adds_three_tiles():
    run_id, figures, summary, badges, guardrail_stats, baseline_summary = execute_run(
        _scenario(_TWO_BUCKET_BLOCK), {}, [], False, True)
    results = RESULTS_CACHE[run_id]
    tb_stats = {
        "reserve_depletion_probability": float(np.max(results["reserve_depletion_probability_by_period"])),
        "forced_sale_probability": float(np.mean(results["forced_sale_count_per_path"] > 0)),
        "median_refills": float(np.median(results["refill_count_per_path"])),
    }
    without = _stat_tiles(summary, guardrail_stats, baseline_summary)
    with_tb = _stat_tiles(summary, guardrail_stats, baseline_summary, tb_stats)
    assert len(with_tb.children) == len(without.children) + 3
