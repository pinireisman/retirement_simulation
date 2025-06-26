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
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
try:
    import plotly.graph_objects as go
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    _HAS_PLOTLY = True
except ImportError:  # graceful fallback
    import matplotlib.pyplot as plt  # type: ignore
    _HAS_PLOTLY = False


MARKET = "IL"  #"US", "UK", "IL"
real_return_mean_by_market = {
    "US": 0.03,  # long‑run real return µ in US
    "UK": 0.053,  # long‑run real return µ in UK
    "IL": 0.055  # 0.055 (Gemini) 0.05 (chatgpt) long‑run real return µ in IL
}
real_return_sd_by_market = {
    "US": 0.12,
    "UK": 0.20,
    "IL": 0.23 # 0.155 (Gemini) 0.117 (chatgpt)
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

def aggregate_schedule(bands: List[Band]) -> Callable[[int], float]:
    """Return a schedule function that adds all (start, end, amount) bands."""

    def _fn(age: int) -> float:
        return sum(b.annual for b in bands if b.start <= age <= b.end)

    return _fn


###############################################################################
# Core data structures
###############################################################################

@dataclass
class Band:
    start: int
    end: int
    annual: float
    label: str

@dataclass
class Lump:
    age: int
    amount: float  # positive=inflow, negative=outflow
    label: str = ""  # description of the lump


@dataclass
class Property:
    """Real‑estate asset kept outside the liquid portfolio."""
    start_age: int  # first year property exists
    initial_value: float  # market value at start_age (real ₪)
    rent_annual: float  # net rent added to cash‑flow each year ≥ start_age
    growth_mean: float = growth_mean_by_market[MARKET]  # long‑run real appreciation µ
    growth_sd: float = growth_sd_by_market[MARKET]  # annual real volatility σ
    label: str = ""


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
        self.spending_bands: List[Band] = []
        for i, row in scenario_data['spending'].iterrows():
            self.spending_bands.append(Band(row.spending_age_from, row.spending_age_to,
                                           row.spending_amount_monthly * 12,
                                           row.spending_comment))

        # Extra travel allowance --------------------------------

        self.travel_annual: float = scenario_data['travel'].travel_amount_annual.iloc[0]  # this is beyond 40K that is in the spending_bands
        self.travel_annual_start: int = scenario_data['travel'].travel_age_from.iloc[0]
        self.travel_annual_end: int = scenario_data['travel'].travel_age_to.iloc[0]
        if self.travel_annual:
            self.spending_bands.append(Band(self.travel_annual_start,
                          self.travel_annual_end,
                          self.travel_annual,
                          "extra travel allowance"))

        # Income bands (real); overlaps add up -------------------------------
        self.income_bands: List[Band] = []
        # age is husband age.
        for i, row in scenario_data['income'].iterrows():
            self.income_bands.append(Band(row.income_age_from, row.income_age_to,
                                          row.income_amount_monthly * 12,
                                          row.income_comment))

        # One‑off lumps --------------------------------------------------------
        self.lumps: List[Lump] = []
        for i, row in scenario_data['lumps'].iterrows():
            self.lumps.append(Lump(age=row.lump_age, amount=row.lump_amount, label=row.lump_comment))

        # Property list --------------------------------------------------------
        self.properties: List[Property] = []
        for i, row in scenario_data['properties'].iterrows():
            self.properties.append(
                Property(
                    start_age=row.properties_age_from,
                    initial_value=row.properties_initial_value,
                    rent_annual=row.properties_rent_monthly * 12,
                    label= row.properties_comment
                )
            )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def spending_fn(self) -> Callable[[int], float]:
        return aggregate_schedule(self.spending_bands)

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

###############################################################################
# Cash‑flow series builder (segmented)
###############################################################################

def build_cash_flow_series(params: SimulationParams) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    ages = np.arange(params.start_age, params.end_age + 1)
    n = len(ages)

    series: "OrderedDict[str, np.ndarray]" = OrderedDict()
    colors: "OrderedDict[str, str]" = OrderedDict()

    greens = ["#00CF00", "#00AF00", "#008F00", "#004F00", "#002F00"]
    blues = ["#0000CF", "#0000AF", "#00008F", "#00004F", "#00002F"]
    reds = ["#CF0000", "#BF0000", "#AF0000", "#9F0000", "#8F0000"]
    # Income components
    for i,band in enumerate(params.income_bands):
        arr = np.zeros(n)
        mask = (ages >= band.start) & (ages <= band.end)
        arr[mask] = band.annual
        series[f"Income · {band.label}"] = arr
        colors[f"Income · {band.label}"] = greens[i % len(greens)]  # cycle through green colors

    # Rent components
    for i,prop in enumerate(params.properties):
        arr = np.zeros(n)
        arr[ages >= prop.start_age] = prop.rent_annual
        series[f"Rent · {prop.label}"] = arr
        colors[f"Rent · {prop.label}"] = blues[i % len(blues)]  # cycle through green colors

    # Spending components (negative values)
    for band in params.spending_bands:
        arr = np.zeros(n)
        mask = (ages >= band.start) & (ages <= band.end)
        arr[mask] = -band.annual
        series[f"Spend · {band.label}"] = arr
        colors[f"Spend · {band.label}"] = reds[len(series) % len(reds)]

    # Lump components (+/–)
    for lp in params.lumps:
        arr = np.zeros(n)
        arr[ages == lp.age] = lp.amount
        series[f"Lump · {lp.label}"] = arr
        colors[f"Lump · {lp.label}"] = "grey"

    return ages, series, colors

###############################################################################
# Plot cash‑flows (segmented)
###############################################################################

def plot_cash_flow(results: Dict, interactive: bool = True):
    ages, series, colors = build_cash_flow_series(results['params'])
    if interactive and _HAS_PLOTLY:
        fig = go.Figure()
        # colors = go.Figure().layout.colorway  # default plotly palette
        # if not colors:
        #     import plotly.express as px
        #     colors = px.colors.qualitative.Plotly

        port_paths = results["bal_over_age"]
        prop_paths = results["prop_over_age"]
        total_paths = port_paths + prop_paths
        med_port = np.percentile(port_paths, 50, axis=1)
        p05_port = np.percentile(port_paths, 5, axis=1)
        p95_port = np.percentile(port_paths, 95, axis=1)
        p25 = np.percentile(total_paths, 25, axis=1)
        p75 = np.percentile(total_paths, 75, axis=1)
        med_prop = np.percentile(prop_paths, 50, axis=1)
        total_med = med_port + med_prop

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("Cash‑flow breakdown", "Portfolio & Property"))




        for idx, (name, arr) in enumerate(series.items()):
            fig.add_bar(
                x=ages,
                y=arr,
                name=name,
                marker_color=colors[name],
                hovertemplate="Age %{x}: ₪ %{y:,.0f}<extra>" + name + "</extra>",
            )
        net = np.sum(list(series.values()), axis=0)
        fig.add_scatter(x=ages, y=net, mode="lines", name="Net cash‑flow", line=dict(color="black"))
        ruin = results["summary"]["ruin_probability"]
        title = (f"Segmented annual cash‑flow (real ₪) - ruin probability: {ruin:.3%} "
                 f"success: {1 - ruin:.3%}")
        fig.update_layout(barmode="relative",
                          title=title,
                          xaxis_title="Age", yaxis_title="₪ per year")

        # ----- pane B: balances -----
        fig.add_scatter(
            x=ages, y=med_port, name="Median portfolio",
            line=dict(color="#1f77b4"), row=2, col=1
        )
        fig.add_scatter(
            x=ages, y=med_prop, name="Median property",
            line=dict(color="#9467bd"), row=2, col=1
        )
        fig.add_scatter(
            x=ages, y=total_med, name="Total estate",
            line=dict(color="black", dash="dot"), row=2, col=1
        )
        # 5–95 % ribbon on portfolio
        fig.add_traces(
            [
                go.Scatter(
                    x=np.concatenate([ages, ages[::-1]]),
                    y=np.concatenate([p05_port, p95_port[::-1]]),
                    fill="toself",
                    fillcolor="rgba(31,119,180,0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    name="5–95 % portfolio band",
                )
            ],
            rows=[2], cols=[1],
        )

        # 25–75 % ribbon on portfolio
        fig.add_traces(
            [
                go.Scatter(
                    x=np.concatenate([ages, ages[::-1]]),
                    y=np.concatenate([p25, p75[::-1]]),
                    fill="toself",
                    fillcolor="rgba(0,0,0,0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    name="25–75 % estate band",
                )
            ],
            rows=[2], cols=[1],
        )


        # ----- layout tweaks -----
        fig.update_layout(
            barmode="relative",
            legend=dict(orientation="h", y=-0.15),
            xaxis_title="Age",
            yaxis_title="₪ / year (real)",
            yaxis2_title="₪ balance (real)",
            height=1400,
        )

        fig.show()

    else:
        import matplotlib.pyplot as plt  # type: ignore
        fig, ax = plt.subplots(figsize=(13, 6))
        bottom = np.zeros_like(ages, dtype=float)
        for name, arr in series.items():
            ax.bar(ages, arr, bottom=bottom, label=name)
            bottom += arr
        ax.plot(ages, bottom, color="black", label="Net cash‑flow")
        ax.set_xlabel("Age"); ax.set_ylabel("₪ / year (real)")
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left"); plt.tight_layout(); plt.show()

###############################################################################
# Basic regression tests
###############################################################################

def _run_basic_tests(scenario_data) -> None:
    """Light sanity tests to spot regressions."""
    base = run_simulation(SimulationParams(scenario_data))
    assert base["summary"]["ruin_probability"] < 0.05, "Base scenario ruin too high"

    high_spend = SimulationParams(scenario_data)
    high_spend.spending_bands = [(50, 95, 2_000_000.0)]
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
        plot_cash_flow(result, interactive=True)

if __name__ == "__main__":
    _cli()