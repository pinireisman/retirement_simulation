import json

from dash import dcc, html
from dash.dash_table.Format import Format, Scheme
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from engine.markets import MARKETS
from engine.theme import PLOTLY_TEMPLATE, chart_dark_color_map
from webapp.components import build_panel, build_data_table, build_chart_card, build_stat_tile

_EMPTY_FIGURE = go.Figure(layout={"template": PLOTLY_TEMPLATE})
_MONEY_FORMAT = Format(precision=0, scheme=Scheme.fixed)

DEFAULT_SCENARIO = {
    "$schema": "scenario.v1",
    "name": "untitled",
    "portfolio": {
        "initial_portfolio": 0,
        "start_age": 60,
        "end_age": 95,
        "market": "IL",
        "fat_tails_enabled": True,
        "fat_tails_df": 5,
        "mode": "annual",
        "n_paths": 10000,
        "random_seed": 42,
        "mu": MARKETS["IL"]["mu"],
        "sigma": MARKETS["IL"]["sigma"],
    },
    "spending_bands": [],
    "income_bands": [],
    "lumps": [],
    "properties": [],
}

def build_layout():
    return dbc.Container([
        # light->dark trace-color map for assets/theme.js (generated in
        # engine/theme.py so the _shade() variants can never drift out of sync)
        html.Div(id="chart-color-map", className="d-none",
                 **{"data-map": json.dumps(chart_dark_color_map())}),
        # App bar
        dbc.Row([
            dbc.Col([
                html.H2("Retirement Simulator", className="mb-0 d-inline-block"),
                html.Span(id="header-scenario-name", children="untitled", className="ms-2"),
            ], width="auto", className="d-flex align-items-center"),
            dbc.Col(
                dbc.RadioItems(
                    id="view-toggle",
                    className="view-toggle btn-group",
                    inputClassName="btn-check",
                    labelClassName="btn btn-outline-primary",
                    labelCheckedClassName="active",
                    options=[
                        {"label": "Dashboard", "value": "dashboard"},
                        {"label": "Plan", "value": "plan"},
                    ],
                    value="dashboard",
                ),
                width="auto",
            ),
            dbc.Col([
                dbc.Button([
                    html.Span("🌙", className="icon-theme-moon"),
                    html.Span("☀️", className="icon-theme-sun"),
                ], id="btn-theme-toggle", color="outline-secondary", size="sm",
                    className="me-2", title="Toggle dark / light theme"),
                dbc.Button("Save scenario", id="btn-save", color="outline-primary", size="sm"),
                dbc.Button("Load", id="btn-load", color="outline-primary", size="sm"),
                dcc.Dropdown(id="dd-load-scenario", options=[], placeholder="Select a scenario…", className="ms-2"),
                dcc.Upload(id="upload-scenario", children=dbc.Button("Upload .xlsx/.numbers", color="outline-primary", size="sm", className="ms-2"), accept=".xlsx,.numbers"),
            ], width="auto", className="ms-auto d-flex align-items-center"),
            dbc.Toast(id="toast", header="Notice", is_open=False, dismissable=True, duration=4000, icon=None),
        ], id="app-bar", className="mb-3 align-items-center"),

        # Dashboard view (landing)
        html.Div([
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Div("Chance of success", className="overline"),
                        html.Div("—", id="hero-numeral", className="hero-numeral"),
                        html.Div("Run a simulation to see how your plan holds up.", id="hero-verdict"),
                    ], width=6),
                    dbc.Col([
                        dbc.Button("Run simulation", id="btn-run", color="primary", className="me-2"),
                        dbc.Button("Run with playground events", id="btn-run-playground", color="primary", className="me-2"),
                        html.Div(
                            dbc.Switch(id="switch-historic", label="Include historic scenarios", value=False),
                            className="mt-2",
                        ),
                    ], width=6),
                ]),
            ], id="div-hero", className="wash-neutral p-4 mb-3"),

            # Playground — what-if events layered on the plan, not part of it
            html.Div([
                dbc.Switch(id="switch-playground", label="Playground mode", value=False),
                html.Div(id="banner-playground",
                        children="Playground — click the chart to add an event",
                        style={"display": "none"}),
                html.Div([
                    html.Div(
                        dbc.Button("⤢", className="btn-maximize p-0", color="link", size="sm",
                                   title="Maximize / restore"),
                        className="d-flex justify-content-end",
                    ),
                    dcc.Graph(id="graph-preview", figure=_EMPTY_FIGURE, config={"responsive": True}),
                ], className="chart-card"),
                html.Div(id="div-playground-chips"),
                dbc.Button("Clear all", id="btn-pg-clear", color="outline-secondary", size="sm"),
                dbc.Modal([
                    dbc.ModalHeader("Add Playground Event"),
                    dbc.ModalBody([
                        dbc.Input(id="input-pg-age", type="number", placeholder="Age"),
                        dbc.Input(id="input-pg-amount", type="number", placeholder="Amount (₪)"),
                        dbc.Input(id="input-pg-label", type="text", placeholder="Label"),
                        dbc.Button("Confirm", id="btn-pg-confirm", color="primary", className="mt-2"),
                    ]),
                ], id="modal-playground", is_open=False),
            ], className="mb-3"),

            html.Div(id="div-result-badges", className="mb-2"),
            html.Div(
                id="div-summary",
                children=[
                    dbc.Row([
                        dbc.Col(build_stat_tile("Median portfolio", "—"), width=3),
                        dbc.Col(build_stat_tile("Median property", "—"), width=3),
                        dbc.Col(build_stat_tile("Median estate", "—"), width=3),
                        dbc.Col(build_stat_tile("Spending guardrail", "—"), width=3),
                    ], className="g-2 mb-3"),
                ],
            ),

            html.Div([
                dbc.Row([
                    dbc.Col(build_chart_card("Cash flow", "graph-results", figure=_EMPTY_FIGURE), width=6),
                    dbc.Col(build_chart_card("Portfolio & property", "graph-portfolio"), width=6),
                ], className="g-3 mb-3"),
                dbc.Row([
                    dbc.Col(build_chart_card("Annual draw", "graph-draw"), width=6),
                    dbc.Col(html.Div(id="div-historic-cards"), width=6),
                ], className="g-3"),
            ], id="div-chart-cards", className="d-none"),
            html.Div(
                "Charts appear here once you run a simulation.",
                id="div-chart-placeholder",
                className="text-center text-muted py-5",
            ),
        ], id="div-view-dashboard"),

        # Plan view (toggled)
        html.Div(build_panel(None, [
                # Undo button
                dbc.Button("Undo", id="btn-undo", color="outline-secondary", size="sm",
                           className="mb-2", disabled=True),

                # Tabs container
                dbc.Tabs(active_tab="tab-portfolio", children=[
                    dbc.Tab([
                        # Portfolio tab content
                        dbc.Row([
                            dbc.Col([
                                html.Label("Initial Portfolio (₪)"),
                                dbc.Input(id="inp-initial-portfolio", type="number", value=0),
                            ], width=6),
                            dbc.Col([
                                html.Label("Start Age"),
                                dbc.Input(id="inp-start-age", type="number", value=60),
                            ], width=3),
                            dbc.Col([
                                html.Label("End Age"),
                                dbc.Input(id="inp-end-age", type="number", value=95),
                            ], width=3),
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Market"),
                                dcc.Dropdown(id="dd-market", options=list(MARKETS.keys()), value="IL"),
                                html.Span(id="lbl-market-mu-sigma", className="small"),
                            ], width=6),
                            dbc.Col([
                                html.Label("Expected return µ (decimal, e.g. 0.042 = 4.2%)"),
                                dbc.Input(id="inp-mu", type="number", step=0.001, value=MARKETS["IL"]["mu"]),
                                html.Label("Volatility σ (decimal, e.g. 0.13 = 13%)", className="mt-2"),
                                dbc.Input(id="inp-sigma", type="number", step=0.001, value=MARKETS["IL"]["sigma"]),
                                dbc.Button("Use market default", id="btn-apply-market-preset", color="outline-secondary",
                                           size="sm", className="mt-2"),
                            ], width=6),
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dbc.Checkbox(id="chk-fat-tails", label="Fat tails (Student-t)", value=True),
                                dcc.Slider(id="slider-df", min=3, max=10, step=1, value=5),
                            ], width=3),
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Return distribution preview", className="small text-muted"),
                                dcc.Graph(id="graph-return-distribution", config={"displayModeBar": False}),
                            ], width=6),
                        ], className="mb-2"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Mode"),
                                dbc.RadioItems(
                                    id="radio-mode",
                                    options=[
                                        {"label": "Annual", "value": "annual"},
                                        {"label": "Monthly", "value": "monthly"}
                                    ],
                                    value="annual"
                                ),
                            ], width=6),
                        ]),
                        # Advanced section
                        html.Details([
                            html.Summary("Advanced"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Number of paths"),
                                    dbc.Input(id="inp-n-paths", type="number", value=10000),
                                ], width=6),
                                dbc.Col([
                                    html.Label("Random seed"),
                                    dbc.Input(id="inp-seed", type="number", value=42),
                                ], width=6),
                            ]),
                        ]),
                    ], label="Portfolio", tab_id="tab-portfolio"),
                    
                    dbc.Tab([
                        # Spending tab content
                        build_data_table(
                            id="tbl-spending",
                            columns=[
                                {"name": "Age From", "id": "age_from", "type": "numeric"},
                                {"name": "Age To", "id": "age_to", "type": "numeric"},
                                {"name": "Amount Monthly (₪)", "id": "amount_monthly", "type": "numeric", "format": _MONEY_FORMAT},
                                {"name": "Amount Annual (₪)", "id": "amount_annual", "type": "numeric", "format": _MONEY_FORMAT, "editable": False},
                                {"name": "Label", "id": "label", "type": "text"},
                                {"name": "Category", "id": "category", "presentation": "dropdown"}
                            ],
                            category_col="category"
                        ),
                        dbc.Button("Add band", id="btn-add-spending", color="outline-primary", size="sm"),
                    ], label="Spending", tab_id="tab-spending"),
                    
                    dbc.Tab([
                        # Income tab content
                        build_data_table(
                            id="tbl-income",
                            columns=[
                                {"name": "Age From", "id": "age_from", "type": "numeric"},
                                {"name": "Age To", "id": "age_to", "type": "numeric"},
                                {"name": "Amount Monthly (₪)", "id": "amount_monthly", "type": "numeric", "format": _MONEY_FORMAT},
                                {"name": "Amount Annual (₪)", "id": "amount_annual", "type": "numeric", "format": _MONEY_FORMAT, "editable": False},
                                {"name": "Label", "id": "label", "type": "text"}
                            ],
                            category_col=None
                        ),
                        dbc.Button("Add band", id="btn-add-income", color="outline-primary", size="sm"),
                    ], label="Income", tab_id="tab-income"),
                    
                    dbc.Tab([
                        # Lumps tab content
                        build_data_table(
                            id="tbl-lumps",
                            columns=[
                                {"name": "Age", "id": "age", "type": "numeric"},
                                {"name": "Amount (₪)", "id": "amount", "type": "numeric", "format": _MONEY_FORMAT},
                                {"name": "Label", "id": "label", "type": "text"},
                                {"name": "Category", "id": "category", "presentation": "dropdown"}
                            ],
                            category_col="category"
                        ),
                        dbc.Button("Add event", id="btn-add-lumps", color="outline-primary", size="sm"),
                    ], label="Lumps", tab_id="tab-lumps"),
                    
                    dbc.Tab([
                        # Properties tab content
                        build_data_table(
                            id="tbl-properties",
                            columns=[
                                {"name": "Start Age", "id": "start_age", "type": "numeric"},
                                {"name": "Initial Value (₪)", "id": "initial_value", "type": "numeric", "format": _MONEY_FORMAT},
                                {"name": "Rent Monthly (₪)", "id": "rent_monthly", "type": "numeric", "format": _MONEY_FORMAT},
                                {"name": "Label", "id": "label", "type": "text"}
                            ],
                            category_col=None
                        ),
                        dbc.Button("Add property", id="btn-add-properties", color="outline-primary", size="sm"),
                        html.Div("Growth µ/σ follow the scenario's selected market (see Portfolio tab).", 
                                className="small text-muted mt-1"),
                    ], label="Properties", tab_id="tab-properties"),
                ]),
                
                # Guardrails panel
                dbc.Button("Guardrails", id="btn-guardrails-header", color="outline-secondary", className="mb-2"),
                dbc.Collapse([
                    html.Div("Strategy:", className="mt-2"),
                    dcc.Dropdown(id="dd-guardrail-strategy",
                                 options=[{"label": "None", "value": "none"},
                                          {"label": "Volatility-based (G1)", "value": "volatility_discretionary_scaling"},
                                          {"label": "Funded ratio (G2)", "value": "funded_ratio_guardrail"}],
                                 value="none", clearable=False),
                    # Percent units in the UI; collect_guardrails converts to fractions.
                    html.Div([
                        html.Div("Drop threshold (%):", className="mt-2"),
                        dcc.Slider(id="slider-g1-drop", min=5, max=50, step=1, value=20,
                                   tooltip={"placement": "bottom", "template": "{value}%"}),
                        html.Div("Rise threshold (%):", className="mt-2"),
                        dcc.Slider(id="slider-g1-rise", min=5, max=50, step=1, value=20,
                                   tooltip={"placement": "bottom", "template": "{value}%"}),
                        html.Div("Cut percentage (%):", className="mt-2"),
                        dcc.Slider(id="slider-g1-cut", min=0, max=50, step=1, value=15,
                                   tooltip={"placement": "bottom", "template": "{value}%"}),
                        html.Div("Raise percentage (%):", className="mt-2"),
                        dcc.Slider(id="slider-g1-raise", min=0, max=50, step=1, value=10,
                                   tooltip={"placement": "bottom", "template": "{value}%"}),
                    ], id="block-g1", style={"display": "none"}),
                    html.Div([
                        html.Div("Lower guardrail — funded ratio (%):", className="mt-2"),
                        dcc.Slider(id="slider-g2-lower", min=50, max=100, step=1, value=85,
                                   tooltip={"placement": "bottom", "template": "{value}%"}),
                        html.Div("Target funded ratio (%):", className="mt-2"),
                        dcc.Slider(id="slider-g2-target", min=90, max=150, step=1, value=105,
                                   tooltip={"placement": "bottom", "template": "{value}%"}),
                        html.Div("Upper guardrail — funded ratio (%):", className="mt-2"),
                        dcc.Slider(id="slider-g2-upper", min=100, max=200, step=1, value=130,
                                   tooltip={"placement": "bottom", "template": "{value}%"}),
                        html.Div("Real discount rate (%):", className="mt-2"),
                        dcc.Slider(id="slider-g2-discount", min=0, max=4, step=0.25, value=1,
                                   tooltip={"placement": "bottom", "template": "{value}%"}),
                    ], id="block-g2", style={"display": "none"}),
                ], id="collapse-guardrails", is_open=False),

        ]), id="div-view-plan", style={"display": "none"}),

        # Theme sync target — clientside_callback repaints Plotly charts into
        # this dummy node's output whenever a figure changes (see callbacks.py).
        html.Div(id="theme-sync", style={"display": "none"}),

        # Hidden stores
        dcc.Store(id="store-active-view", storage_type="memory", data="dashboard"),
        dcc.Store(id="store-scenario", storage_type="session", data=DEFAULT_SCENARIO),
        # localStorage: playground events survive server restarts and plan loads
        dcc.Store(id="store-playground", storage_type="local", data=[]),
        dcc.Store(id="store-guardrails", storage_type="session", data={"guardrails": [
            {"type": "volatility_discretionary_scaling", "enabled": False, "drop_threshold": 0.20, "rise_threshold": 0.20, "cut_pct": 0.15, "raise_pct": 0.10},
            {"type": "funded_ratio_guardrail", "enabled": False, "fr_lower": 0.85, "fr_target": 1.05, "fr_upper": 1.30},
        ]}),
        dcc.Store(id="store-run-id", storage_type="memory", data=None),
        dcc.Store(id="store-undo-stack", storage_type="memory", data=[]),
        
        # Save modal
        dbc.Modal([
            dbc.ModalHeader("Save Scenario"),
            dbc.ModalBody([
                dbc.Input(id="input-save-name", type="text", placeholder="Scenario name"),
                html.Div([
                    dbc.Checkbox(id="chk-overwrite", label="Overwrite existing file"),
                ], id="div-overwrite-checkbox", style={"display": "none"}),
                dbc.Button("Save", id="btn-save-confirm", color="primary", className="mt-2"),
            ]),
        ], id="modal-save", is_open=False),
        
        # Interval for refreshing scenario list
        dcc.Interval(id="interval-scenarios", interval=30_000, n_intervals=0),
    ], fluid=True)
