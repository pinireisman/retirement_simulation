import numpy as np
from engine.params import SimulationParams, Band, Lump, Property
from engine.simulation import run_simulation


def make_params(**over):
    base = dict(
        start_age=60, end_age=95, initial_portfolio=3_000_000,
        real_return_mean=0.042, real_return_sd=0.13, fat_tails_df=5,
        annual=True, n_paths=10_000, random_seed=42,
        spending_bands=[Band(60, 95, 300_000, "base", "strict"),
                        Band(60, 80, 60_000, "travel", "lifestyle")],
        income_bands=[Band(60, 67, 144_000, "consulting")],
        lumps=[Lump(70, -400_000, "gift", "gifts")],
        properties=[Property(60, 2_500_000, 72_000, 0.018, 0.08, "apt")],
    )
    base.update(over)
    return SimulationParams(**base)


def test_ruin_probability_known_scenario():
    out = run_simulation(make_params())
    ruin = out["summary"]["ruin_probability"]
    # Baseline captured once from the verified Phase-1 build; pinned because
    # the seed is fixed. The looser bound covers numpy RNG-stream drift
    # across platforms/versions.
    baseline = 0.9355
    assert abs(ruin - baseline) < 0.005
    assert 0.0 <= ruin <= 1.0

    again = run_simulation(make_params())
    assert again["summary"]["ruin_probability"] == ruin          # seeded determinism
    np.testing.assert_array_equal(out["bal_over_time"], again["bal_over_time"])


def test_guardrails_none_is_noop():
    a = run_simulation(make_params())
    b = run_simulation(make_params(), guardrails=[])
    assert a["summary"] == b["summary"]
