from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from engine.params import SimulationParams


###############################################################################
# Return sampling
###############################################################################

def sample_real_returns(n, mean=0.05, std=0.16, df=5, clipping_thr=(-0.75, 0.75), rng=None):
    """lower df = fatter tails. try 4-7"""
    if rng is None:
        rng = np.random.default_rng()
    # Student-t has variance df/(df-2), so rescale to target std
    raw = rng.standard_t(df=df, size=n)
    scaled = raw / np.sqrt(df / (df - 2))
    normalized = mean + std * scaled
    clipped = np.clip(normalized, clipping_thr[0], clipping_thr[1])
    return clipped


def _summary_percentiles(bal: np.ndarray, prop_tot: np.ndarray, estate: np.ndarray, ruined: np.ndarray) -> dict:
    pct = lambda arr: np.percentile(arr, [25, 50, 75])
    p_port, p_prop, p_est = pct(bal), pct(prop_tot), pct(estate)
    return {
        "ruin_probability": ruined.mean(),
        "portfolio_pct25": p_port[0], "portfolio_median": p_port[1], "portfolio_pct75": p_port[2],
        "property_pct25": p_prop[0], "property_median": p_prop[1], "property_pct75": p_prop[2],
        "estate_pct25": p_est[0], "estate_median": p_est[1], "estate_pct75": p_est[2],
    }


def _ruin_distribution_by_year(ruined: np.ndarray, ruin_time: np.ndarray, factor: int) -> list:
    ruined_paths, = np.nonzero(ruined)
    ruin_years_for_ruined = ruin_time[ruined_paths] // factor
    counts = {}
    for y in np.sort(np.unique(ruin_years_for_ruined)):
        counts[int(y)] = int(np.sum(ruin_years_for_ruined == y))
    return sorted(counts.items(), key=lambda x: x[0])


###############################################################################
# Simulation engine
###############################################################################

def run_simulation(params: SimulationParams, guardrails: Optional[list] = None) -> Dict[str, object]:
    if guardrails:
        raise NotImplementedError("guardrails are not implemented until Phase 5; pass [] or None")

    rng = np.random.default_rng(params.random_seed)
    factor = 1 if params.annual else 12

    if params.annual:
        ages = np.arange(params.start_age, params.end_age + 1)
        periods = np.arange(params.start_age, params.end_age + 1)
    else:
        periods = np.arange(params.start_age * 12, (params.end_age + 1) * 12)
        ages = periods // 12

    n_periods = len(periods)

    spend = np.array([params.spending_fn()(a) / factor for a in ages])
    incomes = np.array([params.income_fn()(a) / factor for a in ages])
    lump_map: dict = {}
    for lump in params.lumps:
        key = lump.age if params.annual else lump.age * 12
        lump_map[key] = lump_map.get(key, 0) + lump.amount

    bal_over_time = np.full((n_periods, params.n_paths), 0.0)
    prop_over_time = np.full((n_periods, params.n_paths), 0.0)
    bal = np.full(params.n_paths, params.initial_portfolio)
    ruined = np.zeros(params.n_paths, dtype=bool)
    ruin_time = np.full(params.n_paths, -1)

    real_return_mean = params.real_return_mean
    real_return_sd = params.real_return_sd
    if not params.annual:
        real_return_mean = np.exp(np.log(1 + params.real_return_mean) / 12) - 1
        real_return_sd = params.real_return_sd / np.sqrt(12)

    if params.fat_tails_df:
        port_ret = sample_real_returns((params.n_paths, n_periods), real_return_mean,
                                       real_return_sd, params.fat_tails_df, rng=rng)
    else:
        port_ret = rng.normal(real_return_mean, real_return_sd, size=(params.n_paths, n_periods))

    prop_vals = [np.zeros(params.n_paths) for _ in params.properties]
    prop_growth_mean = [p.growth_mean if params.annual else (np.exp(np.log(1 + p.growth_mean) / 12) - 1)
                        for p in params.properties]
    prop_growth_sd = [p.growth_sd if params.annual else p.growth_sd / np.sqrt(12) for p in params.properties]
    prop_ret = [rng.normal(prop_growth_mean[j], prop_growth_sd[j], size=(params.n_paths, n_periods))
               for j in range(len(params.properties))]

    mean_wr_list, max_wr_list, mean_withdraw_val_list = [], [], []

    for i, period in enumerate(periods):
        age = ages[i]
        start_bal = bal.copy()
        cash_delta = incomes[i] - spend[i]

        for p_idx, p in enumerate(params.properties):
            start_period = p.start_age if params.annual else p.start_age * 12
            if period >= start_period:
                cash_delta += p.rent_annual / factor
        this_period_lump = lump_map.get(period, 0)
        bal = bal + cash_delta + this_period_lump

        # withdrawal/wrate stats exclude lumps, matching the original engine's
        # cash-flow-only definition of "withdrawal"
        withdrawal = np.where(cash_delta < 0, -cash_delta, 0.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            wrate = np.where(start_bal > 0, withdrawal / start_bal, 0.0)
            withdraw_vals = np.where(start_bal > 0, withdrawal, 0.0)
        mean_wr_list.append(wrate.mean())
        max_wr_list.append(wrate.max())
        mean_withdraw_val_list.append(withdraw_vals.mean())

        just_ruined = (bal <= 0) & (~ruined)
        ruin_time[just_ruined] = period
        ruined |= bal <= 0
        bal[ruined] = 0.0
        bal[~ruined] *= (1 + port_ret[~ruined, i])
        bal_over_time[i, :] = bal.copy()

        for j, p in enumerate(params.properties):
            start_period = p.start_age if params.annual else p.start_age * 12
            if period >= start_period and prop_vals[j].sum() == 0:
                prop_vals[j] = prop_vals[j] + p.initial_value
            active = prop_vals[j] > 0
            prop_vals[j][active] *= (1 + prop_ret[j][active, i])
        prop_over_time[i, :] = np.sum(prop_vals, axis=0)

    prop_tot = sum(prop_vals) if params.properties else np.zeros(params.n_paths)
    estate = bal + prop_tot

    ruin_tracks_log = []
    time_key = "ruin_year" if params.annual else "ruin_month"
    for idx in np.where(ruined)[0]:
        ruin_tracks_log.append({
            "path_index": int(idx),
            time_key: int(ruin_time[idx]),
            "portfolio_sequence": bal_over_time[:, idx].tolist(),
        })

    return {
        "summary": _summary_percentiles(bal, prop_tot, estate, ruined),
        "guardrail_stats": None,
        "final_portfolio": bal,
        "final_property": prop_tot,
        "estate_total": estate,
        "mean_withdraw_rate": np.array(mean_wr_list),
        "max_withdraw_rate": np.array(max_wr_list),
        "mean_withdraw_value": np.array(mean_withdraw_val_list),
        "ages": ages,
        "time_axis": periods,
        "params": params,
        "scenario_name": params.scenario_name,
        "ruined": ruined,
        "ruin_distribution_by_year": _ruin_distribution_by_year(ruined, ruin_time, factor),
        "bal_over_time": bal_over_time,
        "prop_over_time": prop_over_time,
        "estate_final": estate,
        "ruin_tracks_log": ruin_tracks_log,
    }


###############################################################################
# Historic scenario runner
###############################################################################

def run_historic_scenario(params: SimulationParams, real_factors, property_factors=None) -> Dict:
    """Run a single deterministic path using a historical growth-factor sequence.

    real_factors: array-like of annual portfolio growth multipliers
                  (e.g. 0.95 = -5%, 1.15 = +15%).
    After the sequence is exhausted, uses 1 + params.real_return_mean for
    remaining years. Always runs in annual mode regardless of params.annual.
    property_factors: optional array of property growth multipliers;
                      uses p.growth_mean when absent or exhausted.
    """
    ages = np.arange(params.start_age, params.end_age + 1)
    n_historic = len(real_factors)
    mean_factor = 1.0 + params.real_return_mean

    spend_fn = params.spending_fn()
    income_fn = params.income_fn()

    lump_map: dict = {}
    for lump in params.lumps:
        lump_map[lump.age] = lump_map.get(lump.age, 0) + lump.amount

    bal = float(params.initial_portfolio)
    ruined = False
    ruin_age = None
    prop_vals = [0.0] * len(params.properties)
    portfolio_over_time: list = []
    property_over_time: list = []

    for i, age in enumerate(ages):
        cash_delta = income_fn(age) - spend_fn(age)
        for p in params.properties:
            if age >= p.start_age:
                cash_delta += p.rent_annual
        bal += cash_delta + lump_map.get(int(age), 0)

        if not ruined and bal <= 0:
            ruined = True
            ruin_age = int(age)
            bal = 0.0

        if not ruined:
            pf = float(real_factors[i]) if i < n_historic else mean_factor
            bal *= pf

        for j, p in enumerate(params.properties):
            if age >= p.start_age and prop_vals[j] == 0.0:
                prop_vals[j] = p.initial_value
            if prop_vals[j] > 0.0:
                if property_factors is not None and i < len(property_factors):
                    prop_vals[j] *= float(property_factors[i])
                else:
                    prop_vals[j] *= (1.0 + p.growth_mean)

        portfolio_over_time.append(bal)
        property_over_time.append(sum(prop_vals))

    return {
        "ages": ages,
        "portfolio_over_time": portfolio_over_time,
        "property_over_time": property_over_time,
        "ruined": ruined,
        "ruin_age": ruin_age,
        "terminal_portfolio": bal,
        "terminal_property": sum(prop_vals),
        "n_historic_years": n_historic,
    }
