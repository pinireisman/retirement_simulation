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
from visualization import plot_cash_flow


###############################################################################
# Simulation engine
###############################################################################


def run_simulation(params: SimulationParams) -> Dict[str, object]:
    annual = params.annual
    if annual:
        rng = np.random.default_rng(params.random_seed)
        ages = np.arange(params.start_age, params.end_age + 1)
        years = len(ages)

        spend = np.array([params.spending_fn()(a) for a in ages])
        incomes = np.array([params.income_fn()(a) for a in ages])
        lump_map = {lump.age: lump.amount for lump in params.lumps}

        bal_over_age = np.full((years, params.n_paths), 0.0)
        prop_over_age = np.full((years, params.n_paths), 0.0)
        bal = np.full(params.n_paths, params.initial_portfolio)
        ruined = np.zeros(params.n_paths, dtype=bool)
        port_ret = rng.normal(params.real_return_mean, params.real_return_sd,
                              size=(params.n_paths, years))

        prop_vals = [np.zeros(params.n_paths) for _ in params.properties]
        prop_ret = [rng.normal(p.growth_mean, p.growth_sd, size=(params.n_paths, years))
                    for p in params.properties]

        mean_wr_list = []  # store mean withdrawal rate per year
        max_wr_list = []  # store max withdrawal rate per year
        mean_withdraw_val_list = []

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
            ruined |= bal <= 0
            bal[ruined] = 0.0
            bal[~ruined] *= (1 + port_ret[~ruined, i])
            bal_over_age[i, :] = bal.copy()
            # property growth
            for j, p in enumerate(params.properties):
                if age == p.start_age:
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
            "bal_over_age": bal_over_age,
            "prop_over_age": prop_over_age,
            "estate_final": estate,  # total estate at the end of the simulation
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
        # "ages": ages,
        "months": months,
        "params": params,
        "ruined": ruined,
    }


###############################################################################
# CLI demo
###############################################################################

def read_scenario_data(path):
    print(f"Reading scenario data from {path}...")
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
    args = parser.parse_args()

    scenario_data = read_scenario_data(args.input)
    params = SimulationParams(scenario_data=scenario_data)

    result = run_simulation(params)
    print("Ruin probability:", f"{result['summary']['ruin_probability']:.3%}")

    if args.plot:
        result["input_file"] = args.input
        plot_cash_flow(result)


if __name__ == "__main__":
    _cli()
