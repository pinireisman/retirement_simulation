# retirement_monte_carlo.py
"""Monte‑Carlo retirement simulator — *real (inflation‑adjusted) shekels*
Run demo
--------
```python
import retirement_monte_carlo as rmc
out = rmc.run_simulation(rmc.SimulationParams())
print(out["summary"])           # ruin probability + percentiles
```
"""
from __future__ import annotations
import argparse
from typing import Dict
import numpy as np
import pandas as pd
from simulation_params import SimulationParams
from visualization import plot_cash_flow, plot_with_historic


###############################################################################
# Simulation engine
###############################################################################
def sample_real_returns(
    n,
    mean=0.05,
    std=0.16,
    df=5,          # lower = fatter tails. try 4–7
    clipping_thr=(-0.75,0.75), # clip min,max step change
    rng=None
):
    if rng is None:
        rng = np.random.default_rng()
    # Student-t has variance df/(df-2), so rescale to target std
    raw = rng.standard_t(df=df, size=n)
    scaled = raw / np.sqrt(df / (df - 2))
    normalized = mean + std * scaled
    clipped = np.clip(normalized, clipping_thr[0], clipping_thr[1])
    return clipped

def run_simulation(params: SimulationParams) -> Dict[str, object]:
    annual = params.annual
    if annual:
        rng = np.random.default_rng(params.random_seed)
        ages = np.arange(params.start_age, params.end_age + 1)
        years = len(ages)

        spend = np.array([params.spending_fn()(a) for a in ages])
        incomes = np.array([params.income_fn()(a) for a in ages])
        lump_map = {}
        for lump in params.lumps:
            if lump.age in lump_map:
                lump_map[lump.age] += lump.amount
            else:
                lump_map[lump.age] = lump.amount

        bal_over_age = np.full((years, params.n_paths), 0.0)
        prop_over_age = np.full((years, params.n_paths), 0.0)
        bal = np.full(params.n_paths, params.initial_portfolio)
        ruined = np.zeros(params.n_paths, dtype=bool)
        if params.fat_tails_df:
            # use standard_t to use fat tails
            print(f"using standard_t distribution with mean: {params.real_return_mean} std: {params.real_return_sd}, df: {params.fat_tails_df}")
            port_ret = sample_real_returns((params.n_paths, years), params.real_return_mean, params.real_return_sd, params.fat_tails_df)
        else:
            port_ret = rng.normal(params.real_return_mean, params.real_return_sd,
                              size=(params.n_paths, years))

        prop_vals = [np.zeros(params.n_paths) for _ in params.properties]
        prop_ret = [rng.normal(p.growth_mean, p.growth_sd, size=(params.n_paths, years))
                    for p in params.properties]

        mean_wr_list = []  # store mean withdrawal rate per year
        max_wr_list = []  # store max withdrawal rate per year
        mean_withdraw_val_list = []

        ruin_year = np.full(params.n_paths, -1)  # -1 means not ruined

        for i, age in enumerate(ages):
            start_bal = bal.copy()
            # cash‑flows before growth
            cash_delta = incomes[i] - spend[i]
            this_year_lump = 0

            for p_idx, p in enumerate(params.properties):
                if age >= p.start_age:
                    cash_delta += p.rent_annual
            if age in lump_map:
                this_year_lump += lump_map[age]
            bal += cash_delta + this_year_lump

            # withdrawal amount is negative cash_delta
            withdrawal = np.where(cash_delta < 0, -cash_delta, 0.0)
            with np.errstate(divide='ignore', invalid='ignore'):
                wrate = np.where(start_bal > 0, withdrawal / start_bal, 0.0)
                withdraw_vals = np.where(start_bal > 0, withdrawal, 0.0)
            mean_wr_list.append(wrate.mean())
            max_wr_list.append(wrate.max())
            mean_withdraw_val_list.append(withdraw_vals.mean())

            # ruin & growth
            just_ruined = (bal <= 0) & (~ruined)
            ruin_year[just_ruined] = age
            ruined |= bal <= 0
            bal[ruined] = 0.0
            bal[~ruined] *= (1 + port_ret[~ruined, i])
            bal_over_age[i, :] = bal.copy()
            # property growth
            for j, p in enumerate(params.properties):
                if age >= p.start_age and prop_vals[j].sum() == 0:  # only add initial value at the first year it becomes active
                    prop_vals[j] += p.initial_value
                active = prop_vals[j] > 0
                prop_vals[j][active] *= (1 + prop_ret[j][active, i])
            prop_over_age[i, :] = np.sum(prop_vals, axis=0)  # shape (years, paths)

        prop_tot = sum(prop_vals)
        estate = bal + prop_tot

        pct = lambda arr: np.percentile(arr, [25, 50, 75])
        p_port, p_prop, p_est = pct(bal), pct(prop_tot), pct(estate)

        summary = {
            "ruin_probability": ruined.mean(),
            "portfolio_pct25": p_port[0], "portfolio_median": p_port[1], "portfolio_pct75": p_port[2],
            "property_pct25": p_prop[0], "property_median": p_prop[1], "property_pct75": p_prop[2],
            "estate_pct25": p_est[0], "estate_median": p_est[1], "estate_pct75": p_est[2],
        }

        # Calculate ruin distribution by year
        ruined_paths, = np.nonzero(ruined)
        ruin_years_for_ruined = ruin_year[ruined_paths]
        ruin_by_year_counts = {}
        for y in np.sort(np.unique(ruin_years_for_ruined)):
            ruin_by_year_counts[int(y)] = int(np.sum(ruin_years_for_ruined == y))

        out_ruin_distribution = [(year, count) for year, count in ruin_by_year_counts.items()]
        out_ruin_distribution.sort(key=lambda x: x[0])

        # Logging ruined tracks
        ruin_tracks_log = []
        for idx in np.where(ruined)[0]:
            ruin_info = {
                "path_index": int(idx),
                "ruin_year": int(ruin_year[idx]),
                "portfolio_sequence": bal_over_age[:, idx].tolist()
            }
            ruin_tracks_log.append(ruin_info)

        return {
            "summary": summary,
            "final_portfolio": bal,
            "final_property": prop_tot,
            "estate_total": estate,
            "mean_withdraw_rate": np.array(mean_wr_list),
            "max_withdraw_rate": np.array(max_wr_list),
            "mean_withdraw_value": np.array(mean_withdraw_val_list),
            "ages": ages,
            "params": params,
            "ruined": ruined,
            "ruin_distribution_by_year": out_ruin_distribution,
            "bal_over_age": bal_over_age,
            "prop_over_age": prop_over_age,
            "estate_final": estate,  # total estate at the end of the simulation
            "ruin_tracks_log": ruin_tracks_log,
        }
    else:  # monthly simulation

        rng = np.random.default_rng(params.random_seed)
        months = np.arange(params.start_age * 12, (params.end_age + 1) * 12)
        n_months = len(months)

        spend = np.array([params.spending_fn()(m // 12) / 12 for m in months])  # Monthly spending
        incomes = np.array([params.income_fn()(m // 12) / 12 for m in months])  # Monthly income
        lump_map = {lump.age * 12: lump.amount for lump in params.lumps}  # Lump sums by month

        bal_over_month = np.full((n_months, params.n_paths), 0.0)
        prop_over_month = np.full((n_months, params.n_paths), 0.0)
        bal = np.full(params.n_paths, params.initial_portfolio)
        ruined = np.zeros(params.n_paths, dtype=bool)
        a = 1 + params.real_return_mean
        real_return_mean_monthly = np.exp( np.log(a) / 12) - 1
        monthly_ret = rng.normal(real_return_mean_monthly, params.real_return_sd / np.sqrt(12),
                                 size=(params.n_paths, n_months))

        prop_vals = [np.zeros(params.n_paths) for _ in params.properties]
        prop_ret = [rng.normal(p.growth_mean / 12, p.growth_sd / np.sqrt(12), size=(params.n_paths, n_months))
                    for p in params.properties]

        ruin_month = np.full(params.n_paths, -1)  # -1 means not ruined
        for i, month in enumerate(months):
            start_bal = bal.copy()
            # Cash-flows before growth
            cash_delta = incomes[i] - spend[i]
            this_month_lump = 0

            for p_idx, p in enumerate(params.properties):
                if month >= p.start_age * 12:
                    cash_delta += p.rent_annual / 12  # Monthly rent
            if month in lump_map:
                this_month_lump += lump_map[month]
            bal += cash_delta + this_month_lump

            # Withdrawal amount is negative cash_delta
            withdrawal = np.where(cash_delta < 0, -cash_delta, 0.0)
            with np.errstate(divide='ignore', invalid='ignore'):
                wrate = np.where(start_bal > 0, withdrawal / start_bal, 0.0)
            just_ruined = (bal <= 0) & (~ruined)
            ruin_month[just_ruined] = month
            ruined |= bal <= 0
            bal[ruined] = 0.0
            bal[~ruined] *= (1 + monthly_ret[~ruined, i])  # Monthly growth
            bal_over_month[i, :] = bal.copy()

            # Property growth
            for j, p in enumerate(params.properties):
                if month == p.start_age * 12:
                    prop_vals[j] += p.initial_value
                active = prop_vals[j] > 0
                prop_vals[j][active] *= (1 + prop_ret[j][active, i])  # Monthly growth
            prop_over_month[i, :] = np.sum(prop_vals, axis=0)

        prop_tot = sum(prop_vals)
        estate = bal + prop_tot

        # Calculate ruin distribution by year (from month)
        ruined_paths, = np.nonzero(ruined)
        ruin_months_for_ruined = ruin_month[ruined_paths]
        ruin_years_for_ruined = ruin_months_for_ruined // 12
        ruin_by_year_counts = {}
        for y in np.sort(np.unique(ruin_years_for_ruined)):
            ruin_by_year_counts[int(y)] = int(np.sum(ruin_years_for_ruined == y))

        out_ruin_distribution = [(year, count) for year, count in ruin_by_year_counts.items()]
        out_ruin_distribution.sort(key=lambda x: x[0])

        # Logging ruined tracks
        ruin_tracks_log = []
        for idx in np.where(ruined)[0]:
            ruin_info = {
                "path_index": int(idx),
                "ruin_month": int(ruin_month[idx]),
                "portfolio_sequence": bal_over_month[:, idx].tolist()
            }
            ruin_tracks_log.append(ruin_info)

        summary = {
            "ruin_probability": ruined.mean(),
            "final_portfolio": bal,
            "final_property": prop_tot,
            "estate_total": estate,
        }

        return {
            "summary": summary,
            "bal_over_month": bal_over_month,
            "prop_over_month": prop_over_month,
            "estate_final": estate,
            "final_portfolio": bal,
            "final_property": prop_tot,
            "estate_total": estate,
            "ruin_tracks_log": ruin_tracks_log,
        }


###############################################################################
# Historic scenario runner
###############################################################################

def _format_historic_name(key: str) -> str:
    parts = key.split("_")
    year = parts[0]
    name = " ".join(word.capitalize() for word in parts[1:])
    return f"{name} ({year})"


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


###############################################################################
# CLI demo
###############################################################################

def _read_numbers_as_df(path):
    from numbers_parser import Document
    doc = Document(path)
    table = doc.sheets[0].tables[0]
    rows = table.rows()
    headers = [cell.value for cell in rows[0]]
    data = [[cell.value for cell in row] for row in rows[1:]]
    return pd.DataFrame(data, columns=headers)


def read_scenario_data(path):
    print(f"Reading scenario data from {path}...")
    if str(path).lower().endswith('.numbers'):
        pdf = _read_numbers_as_df(path)
    else:
        pdf = pd.read_excel(path)
    scenario_data = {}
    scenario_data['initial_portfolio'] = pdf['initial_portfolio'].iloc[0]
    scenario_data['start_age'] = pdf['start_age'].iloc[0]
    scenario_data['end_age'] = pdf['end_age'].iloc[0]
    scenario_data['spending'] = pdf[['spending_age_from',
                                     'spending_age_to',
                                     'spending_amount_monthly',
                                     'spending_comment']].dropna()
    scenario_data['income'] = pdf[['income_age_from',
                                   'income_age_to',
                                   'income_amount_monthly',
                                   'income_comment']].dropna()
    scenario_data['lumps'] = pdf[['lump_age',
                                  'lump_amount',
                                  'lump_comment']].dropna()

    scenario_data['properties'] = pdf[['properties_age_from',
                                       'properties_initial_value',
                                       'properties_rent_monthly',
                                       'properties_comment']].dropna()
    scenario_data['travel'] = pdf[['travel_age_from',
                                   'travel_age_to',
                                   'travel_amount_annual',
                                   'travel_comment']].dropna()
    print(f"done reading. starting simulation")
    return scenario_data


def _cli():
    parser = argparse.ArgumentParser(description="Retirement Monte‑Carlo simulator")
    parser.add_argument("input", type=str, nargs="?", default="./scenario_data_example.xlsx",
                        help="Path to the scenario data .xlsx file (default: ./scenario_data_example.xlsx)")

    parser.add_argument("--plot", action="store_true", help="Show interactive cash‑flow plot")
    parser.add_argument("--historic", action="store_true",
                        help="Run historical return sequences and display portfolio trajectories below the main chart")
    args = parser.parse_args()

    scenario_data = read_scenario_data(args.input)
    params = SimulationParams(scenario_data=scenario_data)

    result = run_simulation(params)
    print("Ruin probability:", f"{result['summary']['ruin_probability']:.3%}")
    # Print ruin tracks log
    ruin_tracks_log = result.get("ruin_tracks_log", [])
    if ruin_tracks_log:
        print("\nTracks that ended in ruin:")
        for track in ruin_tracks_log:
            if 'ruin_year' in track:
                print(f"Path {track['path_index']} ruined in year {track['ruin_year']}. Portfolio sequence: {track['portfolio_sequence']}")
            elif 'ruin_month' in track:
                print(f"Path {track['path_index']} ruined in month {track['ruin_month']}. Portfolio sequence: {track['portfolio_sequence']}")
        # Print distribution of ruined paths per year/month
        ruin_time_key = 'ruin_year' if 'ruin_year' in ruin_tracks_log[0] else 'ruin_month'
        ruin_time_counts = {}
        for track in ruin_tracks_log:
            time = track[ruin_time_key]
            if time not in ruin_time_counts:
                ruin_time_counts[time] = 0
            ruin_time_counts[time] += 1
        print("\nDistribution of ruined paths per {} (sorted):".format('year' if ruin_time_key == 'ruin_year' else 'month'))
        for time in sorted(ruin_time_counts):
            print(f"{ruin_time_key.replace('ruin_', '').capitalize()} {time}: {ruin_time_counts[time]} paths ruined")
    else:
        print("\nNo tracks ended in ruin.")

    result["input_file"] = args.input
    if args.historic:
        from historic_returns import historical_stress_real_factors_70_30
        historic_results = []
        print("\nHistoric scenario results:")
        for key, seq in historical_stress_real_factors_70_30.items():
            r = run_historic_scenario(params, seq["real_factors"], seq.get("property_factors"))
            r["name"] = _format_historic_name(key)
            r["start_year"] = int(seq["years"][0])
            r["end_year"] = int(seq["years"][-1])
            historic_results.append(r)
            status = (f"RUIN at age {r['ruin_age']}" if r["ruined"]
                      else f"survived ₪{r['terminal_portfolio']:,.0f}")
            print(f"  [{status}]  {r['name']}")
        plot_with_historic(result, historic_results)
    elif args.plot:
        plot_cash_flow(result)


if __name__ == "__main__":
    _cli()
