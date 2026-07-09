"""Dash callbacks for the retirement-simulator web app (PRD §5 inventory).

Callbacks #1 (hydrate_tabs), #2 (collect_edits) and #10 (render_preview) are
registered directly in this module. Callbacks #3-#9 (market info, fat-tail
slider, upload/save/load/refresh) are appended inside `register_callbacks`
below the marker comment.

IMPORTANT — hydrate_tabs / collect_edits round-trip:
    hydrate_tabs writes store → widgets.  collect_edits reads widgets → store.
    The market dropdown (dd-market) must NOT auto-overwrite the mu/sigma inputs
    because that would create a circular update: dd-market changes → inp-mu/
    inp-sigma change → collect_edits fires → store-scenario changes →
    hydrate_tabs fires → dd-market re-renders (no-op) but the snapshot guard
    would be bypassed if mu/sigma were auto-set by dd-market.  Instead, the
    user explicitly clicks "Use market default" to apply a market's mu/sigma.
"""
from __future__ import annotations

import base64
import json
import os
import re
import uuid
from collections import OrderedDict
from pathlib import Path
from tempfile import NamedTemporaryFile

import dash_bootstrap_components as dbc
import numpy as np
from dash import ALL, Input, Output, State, ctx, html, no_update

from engine.figures import (
    PLAYGROUND_COLOR, build_cash_flow_series, net_cash_flow,
    fig_cash_flow, fig_portfolio, fig_draw, fig_historic, get_ruin_explanation,
    fig_return_distribution,
)
from engine.guardrails import parse_guardrail_configs
from engine.historic_returns import historical_stress_real_factors_70_30
from engine.params import SimulationParams, validate_scenario
from engine.simulation import run_historic_scenario, run_simulation
from engine.theme import PLOTLY_TEMPLATE, SERIES_NET_CASH_FLOW, tone_for_ruin
from webapp.components import build_stat_tile, build_badge_row, build_chart_card

_GUARDRAIL_DISPLAY_NAMES = {
    "volatility_discretionary_scaling": "Spending guardrail",
    "funded_ratio_guardrail": "Funded ratio guardrail",
}
_TONE_COLOR = {"success": "var(--success)", "borderline": "var(--warning)", "danger": "var(--danger)"}

# PRD §5.1: cache raw run results by run_id, evicting the oldest once more
# than 5 accumulate (single-process/single-user app, PRD §3.2).
RESULTS_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_CACHE_SIZE = 5

# Module-level state for the store<->UI circular-update guard and the
# "unsaved changes" header dot. Single-process, single-user app (PRD §3.2) —
# no session-keyed state needed.
_last_hydrated_json = {"value": None}
_dirty = {"value": False}


def mark_clean() -> None:
    """Called by save/load/upload callbacks after a successful disk round-trip."""
    _dirty["value"] = False


def _sanitize_name(name: str) -> str:
    """Sanitize a name to be valid as a filename."""
    return re.sub(r"[^A-Za-z0-9_\-]", "_", name.strip().replace(" ", "_"))


def _scenario_options() -> list[dict]:
    """Get scenario options for the load dropdown."""
    return [
        {"label": s, "value": s}
        for s in sorted(p.stem for p in Path("scenarios").glob("*.xlsx"))
    ]


def _num(value, cast=float, default=0):
    if value in (None, ""):
        return default
    return cast(value)


def _with_annual(bands):
    """Display-only annual column; collect_edits drops it on save."""
    return [{**b, "amount_annual": _num(b.get("amount_monthly"), float, 0) * 12} for b in bands]


def _preview_figure(scenario: dict, playground_events: list | None):
    import plotly.graph_objects as go

    params = SimulationParams.from_scenario(scenario, playground_events)
    factor = 1 if params.annual else 12
    ages, series, colors = build_cash_flow_series(params, annual=params.annual)
    ages_for_plotting = ages / factor

    fig = go.Figure()
    for name, arr in series.items():
        fig.add_bar(
            x=ages_for_plotting, y=arr, name=name, marker_color=colors[name],
            hovertemplate="Age %{x}: ₪ %{y:,.0f}<extra>" + name + "</extra>",
        )
    net = net_cash_flow(series, ages_for_plotting)
    fig.add_scatter(x=ages_for_plotting, y=net, mode="lines", name="Net cash-flow",
                     line=dict(color=SERIES_NET_CASH_FLOW))

    for ev in playground_events or []:
        age, amount, label = ev["age"], ev["amount"], ev.get("label", "Playground")
        fig.add_scatter(
            x=[age], y=[amount], mode="markers", name=label,
            marker=dict(symbol="diamond", size=14, color=PLAYGROUND_COLOR),
            hovertemplate=f"Age {age}: ₪{amount:,.0f}<extra>{label}</extra>",
        )

    fig.update_layout(
        barmode="relative",
        template=PLOTLY_TEMPLATE,
        title="Cash-flow preview",
        xaxis_title="Age",
        yaxis_title="₪ / year (real)",
        legend=dict(orientation="h", y=-0.25),
        margin=dict(t=40, b=40),
        # Constant uirevision keeps the user's zoom/pan across the live
        # re-renders triggered by every table edit.
        uirevision="preview",
    )
    return fig


def _playground_chips(playground_events: list | None):
    items = []
    for ev in playground_events or []:
        sign = "+" if ev["amount"] >= 0 else "−"
        text = f"age {ev['age']} · {sign}₪{abs(ev['amount']):,.0f} · {ev.get('label', '')}"
        remove_x = html.Span(" ×", id={"type": "pg-remove", "index": ev["id"]},
                              style={"cursor": "pointer", "marginLeft": "6px"})
        items.append({"text": [text, remove_x], "color": "warning"})
    return build_badge_row(items)


def _banner_style(playground_on: bool):
    return {"display": "block"} if playground_on else {"display": "none"}


def _format_historic_name(key: str) -> str:
    parts = key.split("_")
    year = parts[0]
    name = " ".join(word.capitalize() for word in parts[1:])
    return f"{name} ({year})"


def _result_badges(badges: list[str]):
    if not badges:
        return build_badge_row([{"text": "Baseline run", "color": "secondary"}])
    return build_badge_row([{"text": b, "color": "warning"} for b in badges])


def _stat_tiles(summary: dict, guardrail_stats: dict | None = None):
    tiles = [
        build_stat_tile("Median portfolio", f"₪{summary['portfolio_median']:,.0f}"),
        build_stat_tile("Median property", f"₪{summary['property_median']:,.0f}"),
        build_stat_tile("Median estate", f"₪{summary['estate_median']:,.0f}"),
    ]
    if guardrail_stats:
        name = _GUARDRAIL_DISPLAY_NAMES.get(guardrail_stats["type"], guardrail_stats["type"])
        adj = guardrail_stats["median_adjustment"]
        sign = "-" if adj < 0 else "+"
        tiles.append(build_stat_tile(
            name,
            f"{guardrail_stats['frac_paths_triggered']:.0%} fired · {sign}₪{abs(adj):,.0f}/yr",
        ))
    else:
        tiles.append(build_stat_tile("Spending guardrail", "Off"))
    return dbc.Row([dbc.Col(t, width=3) for t in tiles], className="g-2 mb-3")


def execute_run(scenario: dict, guardrail_cfg: dict, playground_events: list[dict],
                 include_historic: bool):
    """PRD §5.1 shared run contract."""
    # Validate the scenario before proceeding
    errors = validate_scenario(scenario)
    if errors:
        raise ValueError("; ".join(errors))
    
    params = SimulationParams.from_scenario(scenario, playground_events)
    guardrails = parse_guardrail_configs(guardrail_cfg)
    results = run_simulation(params, guardrails=guardrails)

    badges = []
    if playground_events:
        badges.append(f"Includes {len(playground_events)} playground events")
    if guardrails:
        badges.append("Spending guardrail active")

    historic_figs = []
    if include_historic:
        for key, seq in historical_stress_real_factors_70_30.items():
            r = run_historic_scenario(params, seq["real_factors"], seq.get("property_factors"))
            r["name"] = _format_historic_name(key)
            r["start_year"] = int(seq["years"][0])
            r["end_age"] = int(seq["years"][-1])
            historic_figs.append((r["name"], fig_historic(results, r)))

    figures = {
        "cash_flow": fig_cash_flow(results),
        "portfolio": fig_portfolio(results),
        "draw": fig_draw(results),
        "historic": historic_figs,
    }

    run_id = str(uuid.uuid4())
    RESULTS_CACHE[run_id] = results
    if len(RESULTS_CACHE) > _CACHE_SIZE:
        RESULTS_CACHE.popitem(last=False)

    return run_id, figures, results["summary"], badges, results["guardrail_stats"]


def register_callbacks(app) -> None:
    @app.callback(
        Output("inp-initial-portfolio", "value"),
        Output("inp-start-age", "value"),
        Output("inp-end-age", "value"),
        Output("dd-market", "value"),
        Output("chk-fat-tails", "value"),
        Output("slider-df", "value"),
        Output("radio-mode", "value"),
        Output("inp-n-paths", "value"),
        Output("inp-seed", "value"),
        Output("inp-mu", "value"),
        Output("inp-sigma", "value"),
        Output("tbl-spending", "data"),
        Output("tbl-income", "data"),
        Output("tbl-lumps", "data"),
        Output("tbl-properties", "data"),
        Output("header-scenario-name", "children"),
        Input("store-scenario", "data"),
    )
    def hydrate_tabs(scenario):
        """Store -> widgets. Guarded against re-triggering on its own echo
        from collect_edits (see module docstring): compares a serialized
        snapshot of the incoming store value against the last one this
        callback itself rendered, and no-ops when they match."""
        snapshot = json.dumps(scenario, sort_keys=True)
        if snapshot == _last_hydrated_json["value"]:
            return (no_update,) * 16
        _last_hydrated_json["value"] = snapshot

        portfolio = scenario["portfolio"]
        name = scenario.get("name", "untitled")
        header = f"{name} •" if _dirty["value"] else name

        # Resolve mu/sigma: use explicit values from store, or fall back to
        # the selected market's defaults (same logic as SimulationParams.from_scenario).
        from engine.markets import MARKETS
        market = portfolio.get("market", "IL")
        mu_val = portfolio.get("mu")
        sigma_val = portfolio.get("sigma")
        if mu_val is None:
            mu_val = MARKETS[market]["mu"]
        if sigma_val is None:
            sigma_val = MARKETS[market]["sigma"]

        return (
            portfolio["initial_portfolio"],
            portfolio["start_age"],
            portfolio["end_age"],
            market,
            portfolio["fat_tails_enabled"],
            portfolio["fat_tails_df"],
            portfolio["mode"],
            portfolio["n_paths"],
            portfolio["random_seed"],
            mu_val,
            sigma_val,
            _with_annual(scenario.get("spending_bands", [])),
            _with_annual(scenario.get("income_bands", [])),
            scenario.get("lumps", []),
            scenario.get("properties", []),
            header,
        )

    TABLE_TRIGGER_IDS = {
        "tbl-spending", "tbl-income", "tbl-lumps", "tbl-properties",
        "btn-add-spending", "btn-add-income", "btn-add-lumps", "btn-add-properties",
    }

    @app.callback(
        Output("store-scenario", "data", allow_duplicate=True),
        Output("store-undo-stack", "data", allow_duplicate=True),
        Input("inp-initial-portfolio", "value"),
        Input("inp-start-age", "value"),
        Input("inp-end-age", "value"),
        Input("dd-market", "value"),
        Input("chk-fat-tails", "value"),
        Input("slider-df", "value"),
        Input("radio-mode", "value"),
        Input("inp-n-paths", "value"),
        Input("inp-seed", "value"),
        Input("inp-mu", "value"),
        Input("inp-sigma", "value"),
        Input("tbl-spending", "data"),
        Input("tbl-income", "data"),
        Input("tbl-lumps", "data"),
        Input("tbl-properties", "data"),
        Input("btn-add-spending", "n_clicks"),
        Input("btn-add-income", "n_clicks"),
        Input("btn-add-lumps", "n_clicks"),
        Input("btn-add-properties", "n_clicks"),
        Input("slider-g2-discount", "value"),
        State("store-scenario", "data"),
        State("store-undo-stack", "data"),
        prevent_initial_call=True,
    )
    def collect_edits(initial_portfolio, start_age, end_age, market, fat_tails, df,
                       mode, n_paths, seed, mu, sigma, spending_rows, income_rows, lumps_rows,
                       properties_rows, _n_sp, _n_inc, _n_lp, _n_pr, discount, prev_scenario,
                       undo_stack):
        """Widgets -> store. On an add-row button click, appends a default
        row to that band's table before serializing everything back out."""
        triggered = ctx.triggered_id
        start_age_i = _num(start_age, int, 60)
        end_age_i = _num(end_age, int, 95)

        spending_rows = list(spending_rows or [])
        income_rows = list(income_rows or [])
        lumps_rows = list(lumps_rows or [])
        properties_rows = list(properties_rows or [])

        if triggered == "btn-add-spending":
            spending_rows.append({"age_from": start_age_i, "age_to": end_age_i,
                                   "amount_monthly": 0, "label": "", "category": "lifestyle"})
        elif triggered == "btn-add-income":
            income_rows.append({"age_from": start_age_i, "age_to": end_age_i,
                                 "amount_monthly": 0, "label": ""})
        elif triggered == "btn-add-lumps":
            lumps_rows.append({"age": start_age_i, "amount": 0, "label": "", "category": "lifestyle"})
        elif triggered == "btn-add-properties":
            properties_rows.append({"start_age": start_age_i, "initial_value": 0,
                                     "rent_monthly": 0, "label": ""})

        # Persist mu/sigma: store the explicit values from inputs.
        # If the user typed a value, it's stored; if they used market default
        # via the button, that explicit value is also stored.
        mu_val = _num(mu, float) if mu not in (None, "") else None
        sigma_val = _num(sigma, float) if sigma not in (None, "") else None

        scenario = {
            "$schema": prev_scenario.get("$schema", "scenario.v1"),
            "name": prev_scenario.get("name", "untitled"),
            "portfolio": {
                "initial_portfolio": _num(initial_portfolio, float, 0),
                "start_age": start_age_i,
                "end_age": end_age_i,
                "market": market,
                "fat_tails_enabled": bool(fat_tails),
                "fat_tails_df": _num(df, int, 5),
                "mode": mode,
                "n_paths": _num(n_paths, int, 10_000),
                "random_seed": _num(seed, int, None) if seed not in (None, "") else None,
                "mu": mu_val,
                "sigma": sigma_val,
                # slider is in percent; engine wants a fraction (default 1% -> 0.01)
                "real_discount_rate": _num(discount, float, 1) / 100,
            },
            "spending_bands": [
                {
                    "id": f"sb-{i + 1}",
                    "age_from": _num(r.get("age_from"), int, start_age_i),
                    "age_to": _num(r.get("age_to"), int, end_age_i),
                    "amount_monthly": _num(r.get("amount_monthly"), float, 0),
                    "label": r.get("label") or "",
                    "category": r.get("category") or "lifestyle",
                }
                for i, r in enumerate(spending_rows)
            ],
            "income_bands": [
                {
                    "id": f"ib-{i + 1}",
                    "age_from": _num(r.get("age_from"), int, start_age_i),
                    "age_to": _num(r.get("age_to"), int, end_age_i),
                    "amount_monthly": _num(r.get("amount_monthly"), float, 0),
                    "label": r.get("label") or "",
                }
                for i, r in enumerate(income_rows)
            ],
            "lumps": [
                {
                    "id": f"lp-{i + 1}",
                    "age": _num(r.get("age"), int, start_age_i),
                    "amount": _num(r.get("amount"), float, 0),
                    "label": r.get("label") or "",
                    "category": r.get("category") or "strict",
                }
                for i, r in enumerate(lumps_rows)
            ],
            "properties": [
                {
                    "id": f"pr-{i + 1}",
                    "start_age": _num(r.get("start_age"), int, start_age_i),
                    "initial_value": _num(r.get("initial_value"), float, 0),
                    "rent_monthly": _num(r.get("rent_monthly"), float, 0),
                    "label": r.get("label") or "",
                }
                for i, r in enumerate(properties_rows)
            ],
        }
        _dirty["value"] = True

        triggered_ids = {t["prop_id"].split(".")[0] for t in ctx.triggered}
        new_undo_stack = list(undo_stack or [])
        if (triggered_ids & TABLE_TRIGGER_IDS) and scenario != prev_scenario:
            new_undo_stack = (new_undo_stack + [prev_scenario])[-20:]

        return scenario, new_undo_stack

    @app.callback(
        Output("store-scenario", "data", allow_duplicate=True),
        Output("store-undo-stack", "data", allow_duplicate=True),
        Input("btn-undo", "n_clicks"),
        State("store-undo-stack", "data"),
        prevent_initial_call=True,
    )
    def undo_last_edit(n_clicks, undo_stack):
        undo_stack = list(undo_stack or [])
        if not undo_stack:
            return no_update, no_update
        prev = undo_stack.pop()
        return prev, undo_stack

    @app.callback(
        Output("btn-undo", "disabled"),
        Input("store-undo-stack", "data"),
    )
    def toggle_undo_disabled(undo_stack):
        return not bool(undo_stack)

    @app.callback(
        Output("graph-preview", "figure"),
        Output("div-playground-chips", "children"),
        Output("banner-playground", "style"),
        Input("store-scenario", "data"),
        Input("store-playground", "data"),
        Input("switch-playground", "value"),
    )
    def render_preview(scenario, playground_events, playground_on):
        fig = _preview_figure(scenario, playground_events)
        chips = _playground_chips(playground_events)
        style = _banner_style(bool(playground_on))
        return fig, chips, style

    @app.callback(
        Output("modal-playground", "is_open"),
        Output("input-pg-age", "value"),
        Input("graph-preview", "clickData"),
        State("switch-playground", "value"),
        State("store-scenario", "data"),
        prevent_initial_call=True,
    )
    def chart_click(click_data, playground_on, scenario):
        """Callback #11. Opens the add-event modal at the clicked age."""
        if not playground_on or not click_data:
            return no_update, no_update
        age = round(click_data["points"][0]["x"])
        portfolio = scenario["portfolio"]
        age = max(portfolio["start_age"], min(portfolio["end_age"], age))
        return True, age

    @app.callback(
        Output("store-playground", "data", allow_duplicate=True),
        Output("modal-playground", "is_open", allow_duplicate=True),
        Input("btn-pg-confirm", "n_clicks"),
        State("input-pg-age", "value"),
        State("input-pg-amount", "value"),
        State("input-pg-label", "value"),
        State("store-playground", "data"),
        prevent_initial_call=True,
    )
    def confirm_playground_event(n_clicks, age, amount, label, playground_events):
        """Callback #12. PRD §4.2 schema; category fixed to 'strict'."""
        if not amount:
            return no_update, no_update
        event = {
            "id": "pg-" + uuid.uuid4().hex[:6],
            "age": int(age),
            "amount": float(amount),
            "label": label or "",
            "category": "strict",
        }
        return (playground_events or []) + [event], False

    @app.callback(
        Output("store-playground", "data", allow_duplicate=True),
        Input({"type": "pg-remove", "index": ALL}, "n_clicks"),
        State("store-playground", "data"),
        prevent_initial_call=True,
    )
    def remove_playground_event(n_clicks_list, playground_events):
        """Callback #13."""
        if not ctx.triggered_id or not any(n_clicks_list):
            return no_update
        remove_id = ctx.triggered_id["index"]
        return [ev for ev in (playground_events or []) if ev["id"] != remove_id]

    @app.callback(
        Output("store-playground", "data", allow_duplicate=True),
        Input("btn-pg-clear", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_playground(n_clicks):
        """Callback #14."""
        return []

    @app.callback(
        Output("store-guardrails", "data"),
        Input("dd-guardrail-strategy", "value"),
        Input("slider-g1-drop", "value"),
        Input("slider-g1-rise", "value"),
        Input("slider-g1-cut", "value"),
        Input("slider-g1-raise", "value"),
        Input("slider-g2-lower", "value"),
        Input("slider-g2-target", "value"),
        Input("slider-g2-upper", "value"),
    )
    def collect_guardrails(strategy, drop, rise, cut, raise_pct, fr_lower, fr_target, fr_upper):
        """Callback #15. PRD §4.3 schema, strategy-aware: exactly one guardrail
        entry is enabled — the one the dropdown selects. Sliders are in percent;
        store keeps fractions. The discount-rate slider is a SimulationParams
        field, not a guardrail option, so it feeds store-scenario (collect_edits),
        not this store."""
        return {"guardrails": [
            {"type": "volatility_discretionary_scaling",
             "enabled": strategy == "volatility_discretionary_scaling",
             "drop_threshold": drop / 100, "rise_threshold": rise / 100,
             "cut_pct": cut / 100, "raise_pct": raise_pct / 100},
            {"type": "funded_ratio_guardrail",
             "enabled": strategy == "funded_ratio_guardrail",
             "fr_lower": fr_lower / 100, "fr_target": fr_target / 100, "fr_upper": fr_upper / 100},
        ]}

    @app.callback(
        Output("collapse-guardrails", "is_open"),
        Input("btn-guardrails-header", "n_clicks"),
        State("collapse-guardrails", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_guardrail_panel(n_clicks, is_open):
        """Callback #16."""
        return not is_open

    @app.callback(
        Output("block-g1", "style"),
        Output("block-g2", "style"),
        Input("dd-guardrail-strategy", "value"),
    )
    def toggle_guardrail_blocks(strategy):
        """Show only the selected strategy's slider block; hide the other."""
        show, hide = {"display": "block"}, {"display": "none"}
        return (show if strategy == "volatility_discretionary_scaling" else hide,
                show if strategy == "funded_ratio_guardrail" else hide)

    @app.callback(
        Output("store-active-view", "data"),
        Output("div-view-dashboard", "style"),
        Output("div-view-plan", "style"),
        Input("view-toggle", "value"),
    )
    def toggle_view(view):
        """R4: Dashboard/Plan pill toggle. Mirrors _banner_style()'s display-dict pattern."""
        dashboard_style = {"display": "block"} if view == "dashboard" else {"display": "none"}
        plan_style = {"display": "block"} if view == "plan" else {"display": "none"}
        return view, dashboard_style, plan_style

    # --- callbacks #3-#9 (market info, fat-tail slider, upload/save/load/
    # refresh) are appended below this line by a separate delegated brief. ---
    
    @app.callback(
        Output("lbl-market-mu-sigma", "children"),
        Input("dd-market", "value"),
    )
    def market_info(market):
        """Display mu/sigma for the selected market."""
        if market is None:
            return ""
        from engine.markets import MARKETS
        mu = MARKETS[market]["mu"]
        sigma = MARKETS[market]["sigma"]
        return f"µ={mu:.1%} σ={sigma:.1%}"

    @app.callback(
        Output("slider-df", "disabled"),
        Input("chk-fat-tails", "value"),
    )
    def fat_tail_controls(fat_tails_enabled):
        """Disable the fat-tail slider when the checkbox is unchecked."""
        return not fat_tails_enabled

    @app.callback(
        Output("inp-mu", "value", allow_duplicate=True),
        Output("inp-sigma", "value", allow_duplicate=True),
        Input("btn-apply-market-preset", "n_clicks"),
        State("dd-market", "value"),
        prevent_initial_call=True,
    )
    def apply_market_preset(n_clicks, market):
        """Explicitly set mu/sigma from the selected market.
        
        This is a user-initiated action (button click), NOT an automatic
        overwrite from dd-market changes — see module docstring for the
        race-condition reasoning."""
        if market is None:
            return no_update, no_update
        from engine.markets import MARKETS
        m = MARKETS[market]
        return m["mu"], m["sigma"]

    @app.callback(
        Output("graph-return-distribution", "figure"),
        Input("inp-mu", "value"),
        Input("inp-sigma", "value"),
        Input("chk-fat-tails", "value"),
        Input("slider-df", "value"),
    )
    def update_return_distribution(mu, sigma, fat_tails_enabled, df):
        """Live histogram preview of the assumed return distribution.
        
        Updates whenever any of its four inputs change, so the user sees
        the effect of editing mu/sigma or toggling fat tails in real time."""
        if mu is None:
            mu = 0.05
        if sigma is None:
            sigma = 0.15
        return fig_return_distribution(
            float(mu), float(sigma), bool(fat_tails_enabled), int(df) if df else 5
        )

    @app.callback(
        Output("store-scenario", "data", allow_duplicate=True),
        Output("toast", "children"),
        Output("toast", "is_open"),
        Output("toast", "icon"),
        Input("upload-scenario", "contents"),
        State("upload-scenario", "filename"),
        prevent_initial_call=True,
    )
    def upload_xlsx(contents, filename):
        """Upload a scenario from an xlsx file. Playground events persist across loads."""
        try:
            if contents is None or filename is None:
                return no_update, "No file selected", True, "danger"

            # Decode base64 content
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)

            # Write to temporary file using the real file extension
            with NamedTemporaryFile(suffix=Path(filename).suffix.lower(), delete=False) as tmp_file:
                tmp_file.write(decoded)
                tmp_path = tmp_file.name

            try:
                # Read the scenario from file (using new dispatcher)
                from engine.params import scenario_from_file
                scenario = scenario_from_file(tmp_path)

                # Set the name from filename (without extension)
                scenario["name"] = Path(filename).stem

                # Mark as clean
                mark_clean()

                # Clean up temp file
                os.unlink(tmp_path)

                return scenario, "Scenario loaded successfully", True, "success"
            except Exception as exc:
                # Clean up temp file on error
                os.unlink(tmp_path)
                return no_update, f"Error loading scenario: {str(exc)}", True, "danger"
        except Exception as exc:
            return no_update, f"Error decoding file: {str(exc)}", True, "danger"

    @app.callback(
        Output("toast", "children", allow_duplicate=True),
        Output("toast", "is_open", allow_duplicate=True),
        Output("toast", "icon", allow_duplicate=True),
        Output("dd-load-scenario", "options", allow_duplicate=True),
        Output("modal-save", "is_open", allow_duplicate=True),
        Input("btn-save-confirm", "n_clicks"),
        State("input-save-name", "value"),
        State("chk-overwrite", "value"),
        State("store-scenario", "data"),
        prevent_initial_call=True,
    )
    def save_scenario(n_clicks, name, overwrite, scenario):
        """Save the current scenario to an xlsx file."""
        try:
            # Sanitize the name
            sanitized_name = _sanitize_name(name or "")

            if not sanitized_name:
                return "Scenario name cannot be empty", True, "danger", no_update, True

            # Build path
            save_path = Path("scenarios") / f"{sanitized_name}.xlsx"
            
            # Check if file exists and overwrite is not enabled
            if save_path.exists() and not overwrite:
                return "File already exists. Please check 'Overwrite existing file' to proceed.", True, "danger", no_update, True
            
            # Save scenario
            from engine.params import scenario_to_xlsx
            scenario_to_xlsx(scenario, save_path)
            
            # Mark as clean
            mark_clean()
            
            # Refresh the scenario list
            options = _scenario_options()
            
            return "Scenario saved successfully", True, "success", options, False
        except Exception as exc:
            return f"Error saving scenario: {str(exc)}", True, "danger", no_update, True

    @app.callback(
        Output("modal-save", "is_open", allow_duplicate=True),
        Output("input-save-name", "value"),
        Output("div-overwrite-checkbox", "style"),
        Input("btn-save", "n_clicks"),
        State("store-scenario", "data"),
        prevent_initial_call=True,
    )
    def open_save_modal(n_clicks, scenario):
        """Open the save modal with pre-filled name."""
        # Prefill the name
        name = scenario.get("name", "untitled")
        
        # Sanitize the name for filename
        sanitized_name = _sanitize_name(name)
        
        # Check if file already exists
        save_path = Path("scenarios") / f"{sanitized_name}.xlsx"
        show_overwrite = save_path.exists()
        
        # Return the modal state, name, and overwrite checkbox visibility
        return True, name, {"display": "block"} if show_overwrite else {"display": "none"}

    @app.callback(
        Output("store-scenario", "data", allow_duplicate=True),
        Output("toast", "children", allow_duplicate=True),
        Output("toast", "is_open", allow_duplicate=True),
        Output("toast", "icon", allow_duplicate=True),
        Input("btn-load", "n_clicks"),
        State("dd-load-scenario", "value"),
        prevent_initial_call=True,
    )
    def load_scenario(n_clicks, value):
        """Load a scenario from an xlsx file. Playground events persist across loads."""
        try:
            if not value:
                return no_update, "Please select a scenario to load", True, "info"
            
            # Load the scenario from xlsx
            from engine.params import scenario_from_xlsx
            save_path = Path("scenarios") / f"{value}.xlsx"
            scenario = scenario_from_xlsx(save_path)
            
            # Set the name from the value (the dropdown selection)
            scenario["name"] = value
            
            # Mark as clean
            mark_clean()
            
            return scenario, "Scenario loaded successfully", True, "success"
        except Exception as exc:
            return no_update, f"Error loading scenario: {str(exc)}", True, "danger"

    @app.callback(
        Output("dd-load-scenario", "options"),
        Input("interval-scenarios", "n_intervals"),
    )
    def refresh_scenario_list(n_intervals):
        """Refresh the list of available scenarios."""
        return _scenario_options()

    @app.callback(
        Output("graph-results", "figure"),
        Output("graph-portfolio", "figure"),
        Output("graph-draw", "figure"),
        Output("div-historic-cards", "children"),
        Output("div-chart-cards", "className"),
        Output("div-chart-placeholder", "className"),
        Output("hero-numeral", "children"),
        Output("hero-numeral", "style"),
        Output("hero-verdict", "children"),
        Output("div-hero", "className"),
        Output("div-summary", "children"),
        Output("div-result-badges", "children"),
        Output("store-run-id", "data"),
        Output("toast", "children", allow_duplicate=True),
        Output("toast", "is_open", allow_duplicate=True),
        Output("toast", "icon", allow_duplicate=True),
        Input("btn-run", "n_clicks"),
        Input("btn-run-playground", "n_clicks"),
        State("store-scenario", "data"),
        State("store-playground", "data"),
        State("store-guardrails", "data"),
        State("switch-historic", "value"),
        prevent_initial_call=True,
    )
    def run_simulation_cb(n_run, n_run_pg, scenario, playground_events, guardrail_cfg, include_historic):
        """Callback #17. btn-run ignores playground events; btn-run-playground includes them."""
        events = playground_events if ctx.triggered_id == "btn-run-playground" else []
        try:
            run_id, figures, summary, badges, guardrail_stats = execute_run(
                scenario, guardrail_cfg, events, bool(include_historic)
            )
        except Exception as exc:
            return (no_update,) * 13 + (f"Error running simulation: {exc}", True, "danger")

        ruin = summary["ruin_probability"]
        tone = tone_for_ruin(ruin)
        n_paths = scenario["portfolio"]["n_paths"]
        n_success = round((1 - ruin) * n_paths)
        historic_cards = [
            build_chart_card(name, f"graph-historic-{i}", figure=fig)
            for i, (name, fig) in enumerate(figures["historic"])
        ]
        # Constant uirevision so zoom/pan survives re-runs (matches preview).
        for key in ("cash_flow", "portfolio", "draw"):
            figures[key].update_layout(uirevision="results")

        return (
            figures["cash_flow"], figures["portfolio"], figures["draw"], historic_cards,
            "", "d-none",
            f"{1 - ruin:.1%}", {"color": _TONE_COLOR[tone]},
            f"Your plan holds in {n_success:,} of {n_paths:,} simulated futures.",
            f"wash-{tone} p-4 mb-3",
            _stat_tiles(summary, guardrail_stats), _result_badges(badges), run_id,
            "Simulation complete", True, "success",
        )

    # Figures are always rendered server-side in the light palette; repaint them
    # for the active theme client-side whenever Dash pushes new figure data
    # (window.paintCharts lives in assets/theme.js). The same push also syncs
    # each chart container to its figure's layout.height: plotly draws the SVG
    # at that height regardless of container size (absolutely positioned), so a
    # taller figure (e.g. cash flow's legend-sized panel) would otherwise bleed
    # over the row below it.
    app.clientside_callback(
        """
        function() {
            if (window.paintCharts) { window.paintCharts(); }
            requestAnimationFrame(function () {
                document.querySelectorAll(".chart-card .js-plotly-plot").forEach(function (gd) {
                    var h = gd.layout && gd.layout.height;
                    var wrap = gd.closest(".dash-graph");
                    if (h && wrap) { wrap.style.minHeight = h + "px"; }
                });
            });
            return "";
        }
        """,
        Output("theme-sync", "children"),
        Input("graph-results", "figure"),
        Input("graph-portfolio", "figure"),
        Input("graph-draw", "figure"),
        Input("graph-preview", "figure"),
        Input("div-historic-cards", "children"),
    )
