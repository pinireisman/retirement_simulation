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

def build_cash_flow_series(params: SimulationParams, annual=True) \
        -> Tuple[np.ndarray, OrderedDict[str, np.ndarray], OrderedDict[str, str]]:


    factor = 1 if annual else 12  # convert to monthly if not annual
    ages = np.arange(params.start_age * factor, (params.end_age + 1) * factor)  # convert to monthly if needed

    n = len(ages)

    series: "OrderedDict[str, np.ndarray]" = OrderedDict()
    colors: "OrderedDict[str, str]" = OrderedDict()

    greens = ["#00CF00", "#00AF00", "#008F00", "#004F00", "#002F00"]
    blues = ["#0000CF", "#0000AF", "#00008F", "#00004F", "#00002F"]
    reds = ["#CF0000", "#BF0000", "#AF0000", "#9F0000", "#8F0000"]
    # Income components
    for i, band in enumerate(params.income_bands):
        arr = np.zeros(n)
        mask = (ages >= band.start*factor) & (ages <= band.end*factor)
        arr[mask] = band.annual/factor # convert to monthly if needed
        series[f"Income · {band.label}"] = arr
        colors[f"Income · {band.label}"] = greens[i % len(greens)]  # cycle through green colors

    # Rent components
    for i, prop in enumerate(params.properties):
        arr = np.zeros(n)
        arr[ages >= prop.start_age*factor] = prop.rent_annual/factor
        series[f"Rent · {prop.label}"] = arr
        colors[f"Rent · {prop.label}"] = blues[i % len(blues)]  # cycle through green colors

    # Spending components (negative values)
    for band in params.spending_bands:
        arr = np.zeros(n)
        mask = (ages >= band.start*factor) & (ages <= band.end*factor)
        arr[mask] = -band.annual/factor  # convert to monthly if needed
        series[f"Spend · {band.label}"] = arr
        colors[f"Spend · {band.label}"] = reds[len(series) % len(reds)]

    # Lump components (+/–)
    for lp in params.lumps:
        arr = np.zeros(n)
        arr[ages == lp.age*factor] = lp.amount  # lumps are one‑off at a specific age
        series[f"Lump · {lp.label}"] = arr
        colors[f"Lump · {lp.label}"] = "grey"

    return ages, series, colors


###############################################################################
# Plot cash‑flows (segmented)
###############################################################################

def plot_cash_flow(results: Dict):
    annual = "bal_over_age" in results
    ages, series, colors = build_cash_flow_series(results['params'], annual=annual)

    go.Figure()

    factor = 1 if annual else 12.0  # convert to monthly if not annual
    if annual:
        port_paths = results["bal_over_age"]
        prop_paths = results["prop_over_age"]
    else:
        port_paths = results["bal_over_month"]
        prop_paths = results["prop_over_month"]
    total_paths = port_paths + prop_paths

    p05_port = np.percentile(port_paths, 5, axis=1)
    p25_port = np.percentile(port_paths, 25, axis=1)
    med_port = np.percentile(port_paths, 50, axis=1)
    p75_port = np.percentile(port_paths, 75, axis=1)
    p95_port = np.percentile(port_paths, 95, axis=1)

    med_prop = np.percentile(prop_paths, 50, axis=1)
    total_med = med_port + med_prop
    ages_for_plotting = ages / factor  # convert to years for plotting

    ruin = results["summary"]["ruin_probability"]
    title = (f"Segmented annual cash‑flow (real ₪) - ruin probability: {ruin:.3%} "
             f"success: {1 - ruin:.3%} "
             f"market: {MARKET} scenario: {results['input_file']}")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Cash‑flow breakdown", "Portfolio & Property"))
    # plot the bars of cach‑flows
    for idx, (name, arr) in enumerate(series.items()):
        fig.add_bar(
            x=ages_for_plotting,
            y=arr,
            name=name,
            marker_color=colors[name],
            hovertemplate="Age %{x}: ₪ %{y:,.0f}<extra>" + name + "</extra>",
        )
    net = np.sum(list(series.values()), axis=0)
    fig.add_scatter(x=ages_for_plotting, y=net, mode="lines", name="Net cash‑flow", line=dict(color="black"))
    fig.update_layout(barmode="relative",
                      title=title,
                      xaxis_title="Age", yaxis_title="₪ per year")

    # ----- pane B: balances -----
    fig.add_scatter(
        x=ages_for_plotting, y=med_port, name="Median portfolio",
        line=dict(color="#1f77b4"), row=2, col=1
    )
    fig.add_scatter(
        x=ages_for_plotting, y=med_prop, name="Median property",
        line=dict(color="#9467bd"), row=2, col=1
    )
    fig.add_scatter(
        x=ages_for_plotting, y=total_med, name="Total estate",
        line=dict(color="black", dash="dot"), row=2, col=1
    )

    percentiles = [ (p05_port, "5% portfolio"),
                    (p25_port, "25% portfolio"),
                    (p75_port, "75% portfolio"),
                    (p95_port, "95% portfolio") ]
    for port, name in percentiles:
        fig.add_scatter(
            x=ages_for_plotting, y=port, name=name,
            line=dict(color="blue", width=1), row=2, col=1
        )

    # 5–95 % ribbon on portfolio
    fig.add_traces(
        [
            go.Scatter(
                x=np.concatenate([ages_for_plotting, ages_for_plotting[::-1]]),
                y=np.concatenate([p25_port, p75_port[::-1]]),
                fill="toself",
                fillcolor="rgba(0,0,0,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="25–75 % portfolio band",
            )
        ],
        rows=[2], cols=[1],
    )

    # 25–75 % ribbon on portfolio
    fig.add_traces(
        [
            go.Scatter(
                x=np.concatenate([ages_for_plotting, ages_for_plotting[::-1]]),
                y=np.concatenate([p05_port, p95_port[::-1]]),
                fill="toself",
                fillcolor="rgba(31,119,180,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="5–95 % estate band",
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
