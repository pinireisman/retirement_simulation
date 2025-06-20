# retirement_monte_carlo.py
"""Monte‑Carlo retirement simulator — *real (inflation‑adjusted) shekels*

This version fixes the unterminated `if __name__ ==` guard and cleans up a
few odds and ends so the module runs flawlessly when executed directly or
imported.  It still models:

* **₪1 M gifts** to each daughter when she turns 25 (father ages 60, 62, 69).
* **₪40 K/year travel** allowance ages 50‑60.
* **Jerusalem apartment** tracked for market value; rent already baked into
  spending bands so `rent_annual = 0` here.
* **Rehovot cottage** inherited at age 65, worth ₪4.5 M and renting for
  ₪144 K/year thereafter.
* **Diur‑mugan deposit** ₪1.5 M at age 80, 40 % refunded at death.
* Probabilities and percentile breakdowns for **liquid portfolio, property
  bucket, and total estate**.

Run demo
--------
```python
import retirement_monte_carlo as rmc
out = rmc.run_simulation(rmc.SimulationParams())
print(out["summary"])           # ruin probability + percentiles
```
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Dict, Optional
import numpy as np
import pandas as pd

MARKET = "IL"  #"US", "UK", "IL"
real_return_mean_by_market = {
    "US": 0.03,  # long‑run real return µ in US
    "UK": 0.053,  # long‑run real return µ in UK
    "IL": 0.06  # long‑run real return µ in IL
}
real_return_sd_by_market = {
    "US": 0.12,
    "UK": 0.20,
    "IL": 0.23
}
growth_mean_by_market = {
    "US": 0.01,
    "UK": 0.024,
    "IL": 0.018
}
growth_sd_by_market = {
    "US": 0.06,
    "UK": 0.07,
    "IL": 0.08
}

###############################################################################
# Utility: build age→amount functions where overlapping bands are summed.
###############################################################################

def aggregate_schedule(bands: List[Tuple[int, int, float]]) -> Callable[[int], float]:
    """Return a schedule function that adds all (start, end, amount) bands."""

    def _fn(age: int) -> float:
        return sum(amount for a0, a1, amount in bands if a0 <= age <= a1)

    return _fn


###############################################################################
# Core data structures
###############################################################################

@dataclass
class Lump:
    age: int
    amount: float  # positive=inflow, negative=outflow


@dataclass
class Property:
    """Real‑estate asset kept outside the liquid portfolio."""

    start_age: int  # first year property exists
    initial_value: float  # market value at start_age (real ₪)
    rent_annual: float  # net rent added to cash‑flow each year ≥ start_age
    growth_mean: float = growth_mean_by_market[MARKET]  # long‑run real appreciation µ
    growth_sd: float = growth_sd_by_market[MARKET]  # annual real volatility σ


@dataclass
class SimulationParams:

    def __init__(self, scenario_data: dict):

        # Horizon --------------------------------------------------------------
        self.start_age: int = scenario_data['start_age']
        self.end_age: int = scenario_data['end_age']

        # Portfolio ------------------------------------------------------------
        self.initial_portfolio: float = scenario_data['initial_portfolio']
        self.real_return_mean: float = real_return_mean_by_market[MARKET]
        self.real_return_sd: float = real_return_sd_by_market[MARKET]

        # Monte‑Carlo ----------------------------------------------------------
        self.n_paths: int = 10_000
        self.random_seed: Optional[int] = 10  # 42

        # Core spending (real) -------------------------------------------------
        self.spending_core: List[Tuple[int, int, float]] = []
        for i, row in scenario_data['spending'].iterrows():
            self.spending_core.append((row.spending_age_from, row.spending_age_to, row.spending_amount_monthly * 12))

        # Extra travel allowance --------------------------------

        self.travel_annual: float = scenario_data['travel'].travel_amount_annual.iloc[0]  # this is beyond 40K that is in the spending_core
        self.travel_annual_start: int = scenario_data['travel'].travel_age_from.iloc[0]
        self.travel_annual_end: int = scenario_data['travel'].travel_age_to.iloc[0]

        # Income bands (real); overlaps add up -------------------------------
        self.income_bands: List[Tuple[int, int, float]] = []
        # age is husband age.
        for i, row in scenario_data['income'].iterrows():
            self.income_bands.append((row.income_age_from, row.income_age_to, row.income_amount_monthly * 12))

        # One‑off lumps --------------------------------------------------------
        self.lumps: List[Lump] = []
        for i, row in scenario_data['lumps'].iterrows():
            self.lumps.append(Lump(age=row.lump_age, amount=row.lump_amount))

        # Property list --------------------------------------------------------
        self.properties: List[Property] = []
        for i, row in scenario_data['properties'].iterrows():
            self.properties.append(
                Property(
                    start_age=row.properties_age_from,
                    initial_value=row.properties_initial_value,
                    rent_annual=row.properties_rent_monthly * 12,
                )
            )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def spending_fn(self) -> Callable[[int], float]:
        bands = list(self.spending_core)  # copy to mutate safely
        if self.travel_annual:
            bands.append((self.travel_annual_start,
                          self.travel_annual_end,
                          self.travel_annual))
        return aggregate_schedule(bands)

    def income_fn(self) -> Callable[[int], float]:
        return aggregate_schedule(self.income_bands)


###############################################################################
# Simulation engine
###############################################################################

def run_simulation(params: SimulationParams) -> Dict[str, object]:
    rng = np.random.default_rng(params.random_seed)

    ages = np.arange(params.start_age, params.end_age + 1)
    years = len(ages)

    spend = np.array([params.spending_fn()(a) for a in ages])
    inc = np.array([params.income_fn()(a) for a in ages])
    lump_map = {l.age: l.amount for l in params.lumps}
    bal_over_age = np.full((years, params.n_paths), 0.0)
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
        cash_delta = inc[i] - spend[i]
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
        bal_over_age[i,:] = bal.copy()
        # property growth
        for j, p in enumerate(params.properties):
            if age == p.start_age:
                prop_vals[j] += p.initial_value
            active = prop_vals[j] > 0
            prop_vals[j][active] *= (1 + prop_ret[j][active, i])

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
        "estate_final": estate,  # total estate at the end of the simulation
    }



###############################################################################
# Basic regression tests
###############################################################################

def _run_basic_tests(scenario_data) -> None:
    """Light sanity tests to spot regressions."""
    base = run_simulation(SimulationParams(scenario_data))
    assert base["summary"]["ruin_probability"] < 0.05, "Base scenario ruin too high"

    high_spend = SimulationParams(scenario_data)
    high_spend.spending_core = [(50, 95, 2_000_000.0)]
    hs = run_simulation(high_spend)
    assert hs["summary"]["ruin_probability"] > 0.9, "Ruin too low for huge spending"


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


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import matplotlib.ticker as ticker
    import matplotlib
    matplotlib.use('QT5Agg')  # Use the Agg backend
    plt.ion()
    scenario_data = read_scenario_data("/Users/pinir/personal/scenario_data.xlsx")
    _run_basic_tests(scenario_data)
    demo = run_simulation(SimulationParams(scenario_data=scenario_data))
    print("*** Retirement simulation summary (real ₪) ***")
    for k, v in demo["summary"].items():
        if k == "ruin_probability":
            print(f"{k:20}: {v:.2%}")
        else:
            print(f"{k:20}: ₪{v:,.0f}")
    #print(f"wr per year: {demo['mean_withdraw_rate']}")
    #print(f"wv per year: {demo['mean_withdraw_value']}")
    # print(f"num ruined {np.sum(demo['ruined'])}")
    # print(demo['ruined'])
    # print(f"bal_over_age {(demo['bal_over_age'][:,demo['ruined']])[:,0]}")
    print(demo["estate_final"])


    # Plotting results
    plt.figure(figsize=(10, 12))
    plt.subplot(4,1,1)
    plt.plot(demo["ages"], demo["mean_withdraw_rate"], label="Withdrawl rate")
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x*100:.2f}%"))
    plt.grid(True)
    plt.subplot(4, 1, 3)
    plt.plot(demo["ages"], demo["mean_withdraw_value"], label="Withdrawl sum")
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K NIS"))
    plt.grid(True)
    plt.subplot(4, 1, 4)
    plt.hist(demo["estate_final"], bins=range(0,100*10**6,10**6), label="estate histogram")
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 1e6:.0f}M"))
    plt.grid(True)
    plt.show()
