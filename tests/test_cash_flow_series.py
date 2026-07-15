"""Duplicate labels in build_cash_flow_series must merge, not overwrite (playground bug)."""

import numpy as np

from engine.figures import build_cash_flow_series
from engine.params import Lump, SimulationParams


def test_duplicate_lump_labels_merge():
    params = SimulationParams(
        start_age=50, end_age=60, initial_portfolio=1_000_000,
        real_return_mean=0.04, real_return_sd=0.12, fat_tails_df=5,
        annual=True, n_paths=10, random_seed=1,
        lumps=[
            Lump(52, -300_000, "disaster", "strict"),
            Lump(55, -1_000_000, "disaster", "strict"),
        ],
    )
    ages, series, _ = build_cash_flow_series(params)
    arr = series["Lump · disaster"]
    assert arr[np.where(ages == 52)[0][0]] == -300_000
    assert arr[np.where(ages == 55)[0][0]] == -1_000_000
