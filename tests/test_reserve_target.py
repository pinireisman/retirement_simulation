"""Pure-function tests for the reserve-target helpers (PRD §8.6, §14.3).

`planned_gap_schedule` and `rolling_gap_target` are unit-testable without
the engine -- hand-built flat arrays with known, hand-computed results.
"""

import numpy as np
import pytest

from engine.withdrawal_strategies import planned_gap_schedule, rolling_gap_target


def _flat(n, val):
    return np.full(n, val, dtype=float)


# ---------------------------------------------------------------- planned_gap_schedule

def test_recurring_gap_only_excludes_gifts_and_all_lumps():
    n = 5
    gap = planned_gap_schedule(
        spend_strict=_flat(n, 100.0), spend_lifestyle=_flat(n, 20.0), spend_gifts=_flat(n, 1000.0),
        lump_out_by_cat={"strict": _flat(n, 999.0), "lifestyle": _flat(n, 999.0), "gifts": _flat(n, 999.0)},
        playground_out=np.zeros(n), incomes=np.zeros(n), rent_by_period=np.zeros(n),
        lump_in=np.zeros(n), playground_in=np.zeros(n), coverage_scope="recurring_gap_only",
    )
    np.testing.assert_allclose(gap, _flat(n, 120.0))


def test_recurring_plus_scheduled_gifts_adds_gift_band_and_gift_lumps_only():
    n = 3
    gap = planned_gap_schedule(
        spend_strict=_flat(n, 100.0), spend_lifestyle=_flat(n, 20.0), spend_gifts=_flat(n, 30.0),
        lump_out_by_cat={"strict": _flat(n, 999.0), "lifestyle": _flat(n, 999.0), "gifts": _flat(n, 40.0)},
        playground_out=np.zeros(n), incomes=np.zeros(n), rent_by_period=np.zeros(n),
        lump_in=np.zeros(n), playground_in=np.zeros(n),
        coverage_scope="recurring_plus_scheduled_gifts",
    )
    # strict/lifestyle lumps are NOT included at this scope
    np.testing.assert_allclose(gap, _flat(n, 100.0 + 20.0 + 30.0 + 40.0))


def test_all_planned_outflows_adds_remaining_lump_categories():
    n = 2
    gap = planned_gap_schedule(
        spend_strict=_flat(n, 100.0), spend_lifestyle=_flat(n, 20.0), spend_gifts=_flat(n, 30.0),
        lump_out_by_cat={"strict": _flat(n, 50.0), "lifestyle": _flat(n, 10.0), "gifts": _flat(n, 40.0)},
        playground_out=np.zeros(n), incomes=np.zeros(n), rent_by_period=np.zeros(n),
        lump_in=np.zeros(n), playground_in=np.zeros(n), coverage_scope="all_planned_outflows",
    )
    np.testing.assert_allclose(gap, _flat(n, 100.0 + 20.0 + 30.0 + 40.0 + 10.0 + 50.0))


def test_playground_out_netted_from_strict_lumps_at_all_planned_outflows_scope():
    n = 2
    gap = planned_gap_schedule(
        spend_strict=_flat(n, 100.0), spend_lifestyle=np.zeros(n), spend_gifts=np.zeros(n),
        lump_out_by_cat={"strict": _flat(n, 50.0), "lifestyle": np.zeros(n), "gifts": np.zeros(n)},
        playground_out=_flat(n, 50.0),  # the entire strict lump is an unplanned playground event
        incomes=np.zeros(n), rent_by_period=np.zeros(n),
        lump_in=np.zeros(n), playground_in=np.zeros(n), coverage_scope="all_planned_outflows",
    )
    np.testing.assert_allclose(gap, _flat(n, 100.0))


def test_playground_out_never_included_at_narrower_scopes():
    """Playground events live only in the "strict" lump total, which
    recurring_gap_only / recurring_plus_scheduled_gifts never pull in at
    all -- so the exclusion holds trivially at those scopes too."""
    n = 2
    for scope in ("recurring_gap_only", "recurring_plus_scheduled_gifts"):
        gap = planned_gap_schedule(
            spend_strict=_flat(n, 100.0), spend_lifestyle=np.zeros(n), spend_gifts=np.zeros(n),
            lump_out_by_cat={"strict": _flat(n, 5000.0), "lifestyle": np.zeros(n), "gifts": np.zeros(n)},
            playground_out=np.zeros(n),  # even with playground_out=0, strict lump must not leak in
            incomes=np.zeros(n), rent_by_period=np.zeros(n),
            lump_in=np.zeros(n), playground_in=np.zeros(n), coverage_scope=scope,
        )
        np.testing.assert_allclose(gap, _flat(n, 100.0))


def test_playground_in_excluded_from_planned_inflow():
    n = 2
    gap = planned_gap_schedule(
        spend_strict=_flat(n, 100.0), spend_lifestyle=np.zeros(n), spend_gifts=np.zeros(n),
        lump_out_by_cat={"strict": np.zeros(n), "lifestyle": np.zeros(n), "gifts": np.zeros(n)},
        playground_out=np.zeros(n), incomes=np.zeros(n), rent_by_period=np.zeros(n),
        lump_in=_flat(n, 40.0), playground_in=_flat(n, 40.0),  # windfall is a playground event
        coverage_scope="recurring_gap_only",
    )
    np.testing.assert_allclose(gap, _flat(n, 100.0))


def test_positive_net_cashflow_gives_zero_gap():
    n = 4
    gap = planned_gap_schedule(
        spend_strict=_flat(n, 10.0), spend_lifestyle=np.zeros(n), spend_gifts=np.zeros(n),
        lump_out_by_cat={"strict": np.zeros(n), "lifestyle": np.zeros(n), "gifts": np.zeros(n)},
        playground_out=np.zeros(n), incomes=_flat(n, 100.0), rent_by_period=np.zeros(n),
        lump_in=np.zeros(n), playground_in=np.zeros(n), coverage_scope="recurring_gap_only",
    )
    np.testing.assert_allclose(gap, np.zeros(n))


def test_rent_counts_as_planned_inflow():
    n = 2
    gap = planned_gap_schedule(
        spend_strict=_flat(n, 100.0), spend_lifestyle=np.zeros(n), spend_gifts=np.zeros(n),
        lump_out_by_cat={"strict": np.zeros(n), "lifestyle": np.zeros(n), "gifts": np.zeros(n)},
        playground_out=np.zeros(n), incomes=np.zeros(n), rent_by_period=_flat(n, 30.0),
        lump_in=np.zeros(n), playground_in=np.zeros(n), coverage_scope="recurring_gap_only",
    )
    np.testing.assert_allclose(gap, _flat(n, 70.0))


# ---------------------------------------------------------------- rolling_gap_target

def test_flat_schedule_sanity_far_from_horizon():
    gap = _flat(20, 10.0)
    target = rolling_gap_target(gap, years=3.0, periods_per_year=1)
    assert target[0] == pytest.approx(30.0)
    assert target[5] == pytest.approx(30.0)


def test_taper_at_end_of_plan():
    n = 10
    gap = _flat(n, 10.0)
    target = rolling_gap_target(gap, years=3.0, periods_per_year=1)
    assert target[n - 1] == pytest.approx(10.0)   # window collapses to itself only
    assert target[n - 2] == pytest.approx(20.0)
    assert target[n - 3] == pytest.approx(30.0)   # last period with the full 3-period window
    assert target[n - 4] == pytest.approx(30.0)


def test_fractional_years_prorate_the_tail_period():
    gap = _flat(10, 12.0)
    target = rolling_gap_target(gap, years=2.5, periods_per_year=1)
    assert target[0] == pytest.approx(2 * 12.0 + 0.5 * 12.0)


def test_zero_target_years_gives_zero_target_regardless_of_gap():
    gap = _flat(5, 100.0)
    target = rolling_gap_target(gap, years=0.0, periods_per_year=1)
    np.testing.assert_allclose(target, np.zeros(5))


def test_zero_gap_plan_gives_zero_target():
    gap = np.zeros(8)
    target = rolling_gap_target(gap, years=4.0, periods_per_year=1)
    np.testing.assert_allclose(target, np.zeros(8))


def test_monthly_equivalent_to_annual_for_flat_schedule():
    annual_gap = _flat(10, 120.0)   # 120/year for 10 years
    monthly_gap = _flat(120, 10.0)  # 10/month for 10 years -- same total rate
    annual_target = rolling_gap_target(annual_gap, years=3.0, periods_per_year=1)
    monthly_target = rolling_gap_target(monthly_gap, years=3.0, periods_per_year=12)
    assert annual_target[0] == pytest.approx(360.0)
    assert monthly_target[0] == pytest.approx(360.0)
