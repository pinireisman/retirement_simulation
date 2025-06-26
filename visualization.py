from typing import Tuple, Dict
from collections import OrderedDict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from simulation_params import SimulationParams
import numpy as np
from configuration import MARKET


###############################################################################
# Cash‑flow series builder (segmented)
###############################################################################

def build_cash_flow_series(params: SimulationParams) \
        -> Tuple[np.ndarray, OrderedDict[str, np.ndarray], OrderedDict[str, str]]:

    ages = np.arange(params.start_age, params.end_age + 1)
    n = len(ages)

    series: "OrderedDict[str, np.ndarray]" = OrderedDict()
    colors: "OrderedDict[str, str]" = OrderedDict()

    greens = ["#00CF00", "#00AF00", "#008F00", "#004F00", "#002F00"]
    blues = ["#0000CF", "#0000AF", "#00008F", "#00004F", "#00002F"]
    reds = ["#CF0000", "#BF0000", "#AF0000", "#9F0000", "#8F0000"]
    # Income components
    for i, band in enumerate(params.income_bands):
        arr = np.zeros(n)
        mask = (ages >= band.start) & (ages <= band.end)
        arr[mask] = band.annual
        series[f"Income · {band.label}"] = arr
        colors[f"Income · {band.label}"] = greens[i % len(greens)]  # cycle through green colors

    # Rent components
    for i, prop in enumerate(params.properties):
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
    if interactive:
        go.Figure()
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
                 f"success: {1 - ruin:.3%} "
                 f"market: {MARKET} scenario: {results['input_file']}")
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
