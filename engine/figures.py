from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from engine.params import SimulationParams
from engine.theme import (  # re-exported for compat
    CATEGORY_COLORS,
    PLAYGROUND_COLOR,
    PLOTLY_TEMPLATE,
    SERIES_DRAW_ACTIVE,
    SERIES_DRAW_NEUTRAL,
    SERIES_INCOME_BASE,
    SERIES_NET_CASH_FLOW,
    SERIES_PERCENTILE_LINE,
    SERIES_PORTFOLIO_MEDIAN,
    SERIES_POSITIVE_LUMP,
    SERIES_PROPERTY_MEDIAN,
    SERIES_RENT_BASE,
    SERIES_TOTAL_ESTATE,
    BAND_25_75_FILL,
    BAND_5_95_FILL,
    HISTORIC_BOUNDARY_FILL,
    HISTORIC_BOUNDARY_LINE,
    DANGER,
    SUCCESS,
    PRIMARY,
    _shade,
)

PANEL_HEIGHT = 420
_PANEL_LEGEND = dict(orientation="h", y=-0.25)


def _cash_flow_panel_height(n_legend_items: int) -> int:
    """Cash-flow legend lists every income/spend/lump series by name, so a
    scenario-heavy plan can have 15+ entries. Card-width labels are long
    enough that Plotly renders one legend entry per row rather than
    wrapping — a fixed PANEL_HEIGHT then clips it behind Plotly's own
    scroll handle (the "scrolling inside a chart" complaint this redesign
    was meant to kill). Grow the panel with the legend instead."""
    return max(PANEL_HEIGHT, 340 + n_legend_items * 30)


###############################################################################
# Ruin probability explanation
###############################################################################

def get_ruin_explanation(ruin_prob):
    table = [
        (0.00, 0.01, "0-1%", "Too defensive / ultra-safe",
         "Financially excellent, but you may be over-protecting capital unless leaving a large estate is a major goal. Worth asking: 'Are we underspending life?'"),
        (0.01, 0.03, "1-3%", "Super safe",
         "Very strong. Even with fat tails, this is a high-confidence plan. For your long horizon and family/legacy goals, this is a very comfortable target zone."),
        (0.03, 0.05, "3-5%", "Safe",
         "Still solid. Many planners would view this as a good 'green zone,' especially if the plan is reviewed every few years. Guyton-Klinger-style work often used 95% success as a minimum acceptable target, i.e. 5% failure."),
        (0.05, 0.10, "5-10%", "Reasonable only with flexibility",
         "This is where your 'I can reduce spending by 10%' matters. If 10% of spending is genuinely discretionary and cuts happen early when markets disappoint, this can be acceptable. Without guardrails, I'd be cautious."),
        (0.10, 0.15, "10-15%", "Risky but maybe manageable",
         "Needs explicit guardrails: reduce travel, delay gifts, avoid large lump spending in bad markets, maybe extend consulting. I would not call it 'safe' for a rigid plan."),
        (0.15, 0.25, "15-25%", "Risky",
         "Too dependent on favorable markets. A 10% spending cut may not be enough unless applied aggressively and permanently during bad sequences."),
        (0.25, 1.01, ">25%", "Speculative / underfunded",
         "The plan likely needs structural changes: lower baseline spending, more income, fewer/later gifts, later retirement, or a different asset/liability setup."),
    ]
    for low, high, ruin_range, title, explanation in table:
        if low <= ruin_prob < high:
            return ruin_range, title, explanation
    return "N/A", "Unknown", "No explanation available."


###############################################################################
# Cash-flow series builder (segmented)
###############################################################################

def net_cash_flow(series: dict, ages_for_plotting: np.ndarray) -> np.ndarray:
    """Sum of all series, safe for an empty scenario (no spending/income/lumps yet)."""
    if not series:
        return np.zeros(len(ages_for_plotting))
    return np.sum(list(series.values()), axis=0)


def build_cash_flow_series(params: SimulationParams, annual=True) \
        -> Tuple[np.ndarray, "OrderedDict[str, np.ndarray]", "OrderedDict[str, str]"]:

    factor = 1 if annual else 12  # convert to monthly if not annual
    ages = np.arange(params.start_age * factor, (params.end_age + 1) * factor)

    n = len(ages)

    series: "OrderedDict[str, np.ndarray]" = OrderedDict()
    colors: "OrderedDict[str, str]" = OrderedDict()

    # Income components
    for i, band in enumerate(params.income_bands):
        arr = np.zeros(n)
        mask = (ages >= band.start * factor) & (ages <= band.end * factor)
        arr[mask] = band.annual / factor
        series[f"Income · {band.label}"] = arr
        colors[f"Income · {band.label}"] = _shade(SERIES_INCOME_BASE, i)

    # Rent components
    for i, prop in enumerate(params.properties):
        arr = np.zeros(n)
        arr[ages >= prop.start_age * factor] = prop.rent_annual / factor
        series[f"Rent · {prop.label}"] = arr
        colors[f"Rent · {prop.label}"] = _shade(SERIES_RENT_BASE, i)

    # Spending components (negative values) — colored by category (PRD §6.3)
    cat_counts: dict = {}
    for band in params.spending_bands:
        arr = np.zeros(n)
        mask = (ages >= band.start * factor) & (ages <= band.end * factor)
        arr[mask] = -band.annual / factor
        label = f"Spend · {band.label}"
        series[label] = arr
        idx = cat_counts.get(band.category, 0)
        cat_counts[band.category] = idx + 1
        colors[label] = _shade(CATEGORY_COLORS.get(band.category, CATEGORY_COLORS["strict"]), idx)

    # Lump components (+/-) — negative (outflow) lumps colored by category, positive stay grey
    lump_cat_counts: dict = {}
    for lp in params.lumps:
        arr = np.zeros(n)
        arr[ages == lp.age * factor] = lp.amount
        label = f"Lump · {lp.label}"
        series[label] = arr
        if lp.playground:
            colors[label] = PLAYGROUND_COLOR  # match the diamond markers
        elif lp.amount < 0:
            idx = lump_cat_counts.get(lp.category, 0)
            lump_cat_counts[lp.category] = idx + 1
            colors[label] = _shade(CATEGORY_COLORS.get(lp.category, CATEGORY_COLORS["strict"]), idx)
        else:
            colors[label] = SERIES_POSITIVE_LUMP

    return ages, series, colors


###############################################################################
# Shared trace builders — used by both the panel-split figures and the
# monolith compat wrappers below, so the two never drift apart.
###############################################################################

def _cash_flow_bar_traces(ages_for_plotting, series, colors) -> List[go.Bar]:
    return [
        go.Bar(
            x=ages_for_plotting, y=arr, name=name, marker_color=colors[name],
            hovertemplate="Age %{x}: ₪ %{y:,.0f}<extra>" + name + "</extra>",
        )
        for name, arr in series.items()
    ]


def _net_cash_flow_trace(ages_for_plotting, net) -> go.Scatter:
    return go.Scatter(x=ages_for_plotting, y=net, mode="lines", name="Net cash-flow",
                       line=dict(color=SERIES_NET_CASH_FLOW))


def _percentiles(port_paths) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p05 = np.percentile(port_paths, 5, axis=1)
    p25 = np.percentile(port_paths, 25, axis=1)
    med = np.percentile(port_paths, 50, axis=1)
    p75 = np.percentile(port_paths, 75, axis=1)
    p95 = np.percentile(port_paths, 95, axis=1)
    return p05, p25, med, p75, p95


def _portfolio_traces(ages_for_plotting, p05, p25, med_port, p75, p95, med_prop, total_med) -> List[go.Scatter]:
    traces = [
        go.Scatter(x=ages_for_plotting, y=med_port, name="Median portfolio",
                   line=dict(color=SERIES_PORTFOLIO_MEDIAN)),
        go.Scatter(x=ages_for_plotting, y=med_prop, name="Median property",
                   line=dict(color=SERIES_PROPERTY_MEDIAN)),
        go.Scatter(x=ages_for_plotting, y=total_med, name="Total estate",
                   line=dict(color=SERIES_TOTAL_ESTATE, dash="dot")),
    ]
    for port, name in [(p05, "5% portfolio"), (p25, "25% portfolio"),
                       (p75, "75% portfolio"), (p95, "95% portfolio")]:
        traces.append(go.Scatter(x=ages_for_plotting, y=port, name=name,
                                  line=dict(color=SERIES_PERCENTILE_LINE, width=1)))
    traces.append(go.Scatter(
        x=np.concatenate([ages_for_plotting, ages_for_plotting[::-1]]),
        y=np.concatenate([p25, p75[::-1]]),
        fill="toself", fillcolor=BAND_25_75_FILL,
        line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip", name="25-75% portfolio band",
    ))
    traces.append(go.Scatter(
        x=np.concatenate([ages_for_plotting, ages_for_plotting[::-1]]),
        y=np.concatenate([p05, p95[::-1]]),
        fill="toself", fillcolor=BAND_5_95_FILL,
        line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip", name="5-95% portfolio band",
    ))
    return traces


def _draw_trace(ages_for_plotting, series, med_port, annual=True) -> go.Bar:
    """Annual portfolio draw bars, in annual currency terms regardless of
    simulation mode. Hover reveals annual draw, monthly draw, and draw rate
    as % of the median MC portfolio at that age."""
    factor = 1 if annual else 12
    net = net_cash_flow(series, ages_for_plotting)
    draw_per_period = np.maximum(0.0, -net)
    annual_draw = draw_per_period * factor
    monthly_draw = annual_draw / 12.0

    with np.errstate(divide='ignore', invalid='ignore'):
        draw_pct = np.where(med_port > 0, annual_draw / med_port * 100.0, 0.0)

    bar_colors = [SERIES_DRAW_ACTIVE if d > 0 else SERIES_DRAW_NEUTRAL for d in draw_per_period]

    return go.Bar(
        x=ages_for_plotting,
        y=annual_draw,
        name="Portfolio draw",
        marker_color=bar_colors,
        customdata=np.stack([monthly_draw, draw_pct], axis=1),
        hovertemplate=(
            "<b>Age %{x}</b><br>"
            "Annual draw: ₪%{y:,.0f}<br>"
            "Monthly draw: ₪%{customdata[0]:,.0f}<br>"
            "Draw rate: %{customdata[1]:.1f}% of median portfolio<br>"
            "<extra>Portfolio Draw</extra>"
        ),
    )


def _historic_traces_and_shapes(params, hr) -> Tuple[List[go.Scatter], list]:
    color = DANGER if hr["ruined"] else SUCCESS
    ages_h = hr["ages"]
    boundary_age = params.start_age + hr["n_historic_years"]

    shapes = []
    if boundary_age <= params.end_age:
        shapes.append(dict(
            type="rect", x0=boundary_age, x1=params.end_age + 0.5, y0=0, y1=1,
            xref="x", yref="y domain", fillcolor=HISTORIC_BOUNDARY_FILL,
            line=dict(width=0), layer="below",
        ))
        shapes.append(dict(
            type="line", x0=boundary_age, x1=boundary_age, y0=0, y1=1,
            xref="x", yref="y domain",
            line=dict(color=HISTORIC_BOUNDARY_LINE, dash="dash", width=1),
        ))

    years_h = [
        str(hr["start_year"] + i) if i < hr["n_historic_years"] else "—"
        for i in range(len(ages_h))
    ]
    hover_tpl = "<b>Age %{x}</b>  <i>(%{customdata})</i><br>₪%{y:,.0f}<br><extra></extra>"

    traces = [
        go.Scatter(x=ages_h, y=hr["portfolio_over_time"], name=f"{hr['name']} · portfolio",
                   line=dict(color=color, width=2), customdata=years_h, hovertemplate=hover_tpl),
        go.Scatter(x=ages_h, y=hr["property_over_time"], name=f"{hr['name']} · property",
                   line=dict(color=color, width=1, dash="dot"), customdata=years_h, hovertemplate=hover_tpl),
    ]
    return traces, shapes


###############################################################################
# Panel-split figures (R2) — one go.Figure per chart card, ~420px, autosize.
# These feed the dashboard chart-card grid (R4/R5).
###############################################################################

def fig_cash_flow(results: Dict) -> go.Figure:
    params = results["params"]
    annual = params.annual
    ages, series, colors = build_cash_flow_series(params, annual=annual)
    factor = 1 if annual else 12.0
    ages_for_plotting = ages / factor
    net = net_cash_flow(series, ages_for_plotting)

    fig = go.Figure(data=_cash_flow_bar_traces(ages_for_plotting, series, colors)
                     + [_net_cash_flow_trace(ages_for_plotting, net)])
    fig.update_layout(
        template=PLOTLY_TEMPLATE, barmode="relative", height=_cash_flow_panel_height(len(series) + 1),
        legend=_PANEL_LEGEND, xaxis_title="Age", yaxis_title="₪ / year (real)",
    )
    return fig


def fig_portfolio(results: Dict) -> go.Figure:
    params = results["params"]
    factor = 1 if params.annual else 12.0
    port_paths = results["bal_over_time"]
    prop_paths = results["prop_over_time"]
    p05, p25, med_port, p75, p95 = _percentiles(port_paths)
    med_prop = np.percentile(prop_paths, 50, axis=1)
    total_med = med_port + med_prop
    ages_for_plotting = np.arange(params.start_age * factor, (params.end_age + 1) * factor) / factor

    fig = go.Figure(data=_portfolio_traces(ages_for_plotting, p05, p25, med_port, p75, p95, med_prop, total_med))
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=PANEL_HEIGHT, legend=_PANEL_LEGEND,
        xaxis_title="Age", yaxis_title="₪ balance (real)",
    )
    return fig


def fig_draw(results: Dict) -> go.Figure:
    params = results["params"]
    annual = params.annual
    ages, series, _ = build_cash_flow_series(params, annual=annual)
    factor = 1 if annual else 12.0
    ages_for_plotting = ages / factor
    med_port = np.percentile(results["bal_over_time"], 50, axis=1)

    fig = go.Figure(data=[_draw_trace(ages_for_plotting, series, med_port, annual=annual)])
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=PANEL_HEIGHT, legend=_PANEL_LEGEND,
        xaxis_title="Age", yaxis_title="₪ / year (real)",
    )
    return fig


def fig_guardrail_multiplier(results: Dict) -> go.Figure:
    """PRD_GOAL_BASED_GUARDRAILS.md §3.5: spending-multiplier percentile bands
    over time — the feedback loop showing what the guardrail policy actually
    did to discretionary spending (median line, P10-P90 band, 1.0 = planned
    level). Reuses the portfolio chart's band styling so the results row reads
    as one system."""
    pct = results["guardrail_mult_percentiles"]
    params = results["params"]
    factor = 1 if params.annual else 12.0
    n = len(pct["p50"])
    ages = (np.arange(n) + params.start_age * factor) / factor

    fig = go.Figure(data=[
        go.Scatter(
            x=np.concatenate([ages, ages[::-1]]),
            y=np.concatenate([pct["p10"], pct["p90"][::-1]]),
            fill="toself", fillcolor=BAND_25_75_FILL,
            line=dict(color="rgba(255,255,255,0)"),
            name="10-90% of futures", hoverinfo="skip",
        ),
        go.Scatter(x=ages, y=pct["p90"], name="90th percentile",
                   line=dict(color=SERIES_PERCENTILE_LINE, width=1)),
        go.Scatter(x=ages, y=pct["p10"], name="10th percentile",
                   line=dict(color=SERIES_PERCENTILE_LINE, width=1)),
        go.Scatter(x=ages, y=pct["p50"], name="Median spending level",
                   line=dict(color=SERIES_PORTFOLIO_MEDIAN, width=2)),
    ])
    fig.add_hline(y=1.0, line_dash="dot", line_color=SERIES_PERCENTILE_LINE,
                  annotation_text="planned level", annotation_position="bottom right")
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=PANEL_HEIGHT, legend=_PANEL_LEGEND,
        xaxis_title="Age", yaxis_title="Discretionary spending × plan",
        yaxis_tickformat=".0%",
    )
    return fig


def fig_historic(results: Dict, hr: Dict) -> go.Figure:
    """One chart card per historic scenario."""
    params = results["params"]
    traces, shapes = _historic_traces_and_shapes(params, hr)
    fig = go.Figure(data=traces)
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=PANEL_HEIGHT, legend=_PANEL_LEGEND,
        xaxis_title="Age", yaxis_title="₪ balance (real)", shapes=shapes,
    )
    return fig


def fig_return_distribution(mu, sigma, fat_tails_enabled, fat_tails_df):
    """Histogram preview of the assumed per-period real-return distribution.
    Reuses the simulation engine's own sampler so the preview can't drift
    out of sync with what a real run actually does."""
    rng = np.random.default_rng(0)
    n = 20_000
    if fat_tails_enabled and fat_tails_df:
        from engine.simulation import sample_real_returns
        sample = sample_real_returns(n, mean=mu, std=sigma, df=fat_tails_df,
                                      clipping_thr=(-0.99, 0.99), rng=rng)
    else:
        sample = rng.normal(mu, sigma, n)
    fig = go.Figure(data=[go.Histogram(x=sample.tolist(), histnorm="probability density",
                                        marker_color=PRIMARY, nbinsx=60)])
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=220, showlegend=False,
        margin=dict(l=30, r=10, t=10, b=30),
        xaxis_tickformat=".0%",
    )
    return fig


###############################################################################
# Monolith compat wrappers — kept for cli.py and existing tests until R5.
# Same traces/data as the panel-split figures above, assembled as subplots.
###############################################################################

def plot_cash_flow(results: Dict) -> go.Figure:
    params = results["params"]
    annual = params.annual
    ages, series, colors = build_cash_flow_series(params, annual=annual)

    factor = 1 if annual else 12.0
    port_paths = results["bal_over_time"]
    prop_paths = results["prop_over_time"]

    p05_port, p25_port, med_port, p75_port, p95_port = _percentiles(port_paths)
    med_prop = np.percentile(prop_paths, 50, axis=1)
    total_med = med_port + med_prop
    ages_for_plotting = ages / factor

    ruin = results["summary"]["ruin_probability"]
    ruin_range, ruin_title, _ = get_ruin_explanation(ruin)

    mu = params.real_return_mean
    sigma = params.real_return_sd
    title = (f"Segmented annual cash-flow (real ₪) - ruin probability: {ruin:.3%} "
             f"success: {1 - ruin:.3%} [{ruin_range} {ruin_title}] "
             f"(µ={mu:.1%} σ={sigma:.1%}) scenario: {results['scenario_name']}")

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=("Cash-flow breakdown", "Portfolio & Property",
                                        "Annual Portfolio Draw"))
    for trace in _cash_flow_bar_traces(ages_for_plotting, series, colors):
        fig.add_trace(trace, row=1, col=1)
    net = net_cash_flow(series, ages_for_plotting)
    fig.add_trace(_net_cash_flow_trace(ages_for_plotting, net), row=1, col=1)

    for trace in _portfolio_traces(ages_for_plotting, p05_port, p25_port, med_port, p75_port, p95_port, med_prop, total_med):
        fig.add_trace(trace, row=2, col=1)

    fig.add_trace(_draw_trace(ages_for_plotting, series, med_port, annual=annual), row=3, col=1)

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        barmode="relative",
        title=title,
        legend=dict(orientation="h", y=-0.08),
        xaxis_title="Age",
        yaxis_title="₪ / year (real)",
        yaxis2_title="₪ balance (real)",
        yaxis3_title="₪ / year (real)",
        height=1800,
    )

    return fig


def plot_with_historic(results: Dict, historic_results: list) -> go.Figure:
    """Build a single Plotly figure with the standard 2-panel MC view on top
    and one additional panel per historic scenario below."""
    params = results["params"]
    annual = params.annual
    ages, series, colors = build_cash_flow_series(params, annual=annual)

    factor = 1 if annual else 12.0
    ages_for_plotting = ages / factor

    port_paths = results["bal_over_time"]
    prop_paths = results["prop_over_time"]
    p05_port, p25_port, med_port, p75_port, p95_port = _percentiles(port_paths)
    med_prop = np.percentile(prop_paths, 50, axis=1)
    total_med = med_port + med_prop

    ruin = results["summary"]["ruin_probability"]
    ruin_range, ruin_title, _ = get_ruin_explanation(ruin)

    n = len(historic_results)

    hist_titles = []
    for hr in historic_results:
        boundary_age = params.start_age + hr["n_historic_years"]
        label = (f"RUIN at age {hr['ruin_age']}" if hr["ruined"]
                 else f"survived ₪{hr['terminal_portfolio']:,.0f}")
        data_years = f"{hr['start_year']}-{hr['end_year']}"
        hist_titles.append(
            f"{hr['name']} ({data_years}) - {label}"
            + (f" | mean return from age {boundary_age}" if boundary_age <= params.end_age else "")
        )

    subplot_titles = (["Cash-flow breakdown", "Portfolio & Property (Monte-Carlo)",
                       "Annual Portfolio Draw"] + hist_titles)

    fig = make_subplots(
        rows=3 + n, cols=1,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        vertical_spacing=0.04,
    )

    for trace in _cash_flow_bar_traces(ages_for_plotting, series, colors):
        fig.add_trace(trace, row=1, col=1)
    net = net_cash_flow(series, ages_for_plotting)
    fig.add_trace(_net_cash_flow_trace(ages_for_plotting, net), row=1, col=1)

    for trace in _portfolio_traces(ages_for_plotting, p05_port, p25_port, med_port, p75_port, p95_port, med_prop, total_med):
        fig.add_trace(trace, row=2, col=1)

    fig.add_trace(_draw_trace(ages_for_plotting, series, med_port, annual=annual), row=3, col=1)

    for k, hr in enumerate(historic_results):
        row = 4 + k
        traces, shapes = _historic_traces_and_shapes(params, hr)
        for trace in traces:
            fig.add_trace(trace, row=row, col=1)
        y_axis = "y" if row == 1 else f"y{row}"
        for shape in shapes:
            fig.add_shape(**dict(shape, yref=f"{y_axis} domain"))

    mu = params.real_return_mean
    sigma = params.real_return_sd
    title = (f"Retirement simulation — MC ruin: {ruin:.3%} [{ruin_range} {ruin_title}] "
             f"| (µ={mu:.1%} σ={sigma:.1%}) | {results['scenario_name']}")
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        barmode="relative",
        title=title,
        legend=dict(orientation="h", y=-0.04),
        xaxis_title="Age",
        yaxis_title="₪ / year (real)",
        yaxis2_title="₪ balance (real)",
        yaxis3_title="₪ / year (real)",
        height=1800 + n * 320,
    )
    for i in range(4, 4 + n):
        fig.update_yaxes(title_text="₪ balance (real)", row=i, col=1)

    return fig
