from typing import Tuple, Dict
from collections import OrderedDict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from simulation_params import SimulationParams
import numpy as np
from configuration import MARKET


###############################################################################
# Ruin probability explanation
###############################################################################

def get_ruin_explanation(ruin_prob):
    table = [
        (0.00, 0.01, "0–1%", "Too defensive / ultra-safe",
         "Financially excellent, but you may be over-protecting capital unless leaving a large estate is a major goal. Worth asking: 'Are we underspending life?'"),
        (0.01, 0.03, "1–3%", "Super safe",
         "Very strong. Even with fat tails, this is a high-confidence plan. For your long horizon and family/legacy goals, this is a very comfortable target zone."),
        (0.03, 0.05, "3–5%", "Safe",
         "Still solid. Many planners would view this as a good 'green zone,' especially if the plan is reviewed every few years. Guyton-Klinger-style work often used 95% success as a minimum acceptable target, i.e. 5% failure."),
        (0.05, 0.10, "5–10%", "Reasonable only with flexibility",
         "This is where your 'I can reduce spending by 10%' matters. If 10% of spending is genuinely discretionary and cuts happen early when markets disappoint, this can be acceptable. Without guardrails, I'd be cautious."),
        (0.10, 0.15, "10–15%", "Risky but maybe manageable",
         "Needs explicit guardrails: reduce travel, delay gifts, avoid large lump spending in bad markets, maybe extend consulting. I would not call it 'safe' for a rigid plan."),
        (0.15, 0.25, "15–25%", "Risky",
         "Too dependent on favorable markets. A 10% spending cut may not be enough unless applied aggressively and permanently during bad sequences."),
        (0.25, 1.01, ">25%", "Speculative / underfunded",
         "The plan likely needs structural changes: lower baseline spending, more income, fewer/later gifts, later retirement, or a different asset/liability setup."),
    ]
    for low, high, ruin_range, title, explanation in table:
        if low <= ruin_prob < high:
            return ruin_range, title, explanation
    return "N/A", "Unknown", "No explanation available."


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
    for i, band in enumerate(params.spending_bands):
        arr = np.zeros(n)
        mask = (ages >= band.start*factor) & (ages <= band.end*factor)
        arr[mask] = -band.annual/factor  # convert to monthly if needed
        series[f"Spend  {band.label} {i}"] = arr
        colors[f"Spend  {band.label} {i}"] = reds[len(series) % len(reds)]

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
    ruin_range, ruin_title, ruin_explanation = get_ruin_explanation(ruin)

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

    # ----- Ruin probability icon annotation -----
    # Determine icon color based on ruin level
    if ruin < 0.03:
        icon_color = "green"
    elif ruin < 0.10:
        icon_color = "orange"
    else:
        icon_color = "red"

    fig.add_annotation(
        xref="paper", yref="paper",
        x=1.0, y=1.07,
        showarrow=False,
        text=f'<span style="font-size:22px; cursor:pointer;">ℹ️</span>',
        font=dict(size=22, color=icon_color),
        align="right",
        hovertext=f"<b>[{ruin_range}] {ruin_title}</b><br><br>{ruin_explanation}",
        hoverlabel=dict(bgcolor="white", font=dict(size=13)),
    )

    # Also add a visible label next to the icon showing the category title
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.98, y=1.07,
        showarrow=False,
        text=f"<b>[{ruin_range}] {ruin_title}</b>",
        font=dict(size=13, color=icon_color),
        align="right",
        xanchor="right",
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
