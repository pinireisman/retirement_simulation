from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from engine.markets import MARKETS
from engine.figures import CATEGORY_COLORS

_EMPTY_DARK_FIGURE = go.Figure(layout={"template": "plotly_dark"})

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
    },
    "spending_bands": [],
    "income_bands": [],
    "lumps": [],
    "properties": [],
}

def build_layout():
    return dbc.Container([
        # Header row
        dbc.Row([
            dbc.Col([
                html.H2("Retirement Simulator", className="mb-0"),
                html.Span(id="header-scenario-name", children="untitled", className="ms-2"),
                dbc.Button("Save As…", id="btn-save", color="outline-primary", size="sm"),
                dbc.Button("Load", id="btn-load", color="outline-primary", size="sm"),
                dcc.Dropdown(id="dd-load-scenario", options=[], placeholder="Select a scenario…", className="ms-2"),
                dcc.Upload(id="upload-scenario", children=html.Button("Upload .xlsx", className="ms-2")),
                dbc.Toast(id="toast", header="Notice", is_open=False, dismissable=True, duration=4000, icon=None, className="position-fixed top-0 end-0 m-2"),
            ], width=12),
        ], className="mb-3"),
        
        # Main content row
        dbc.Row([
            # Left column - Builder
            dbc.Col(dbc.Card(dbc.CardBody([
                # Tabs container
                dbc.Tabs([
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
                            ], width=12),
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dbc.Checkbox(id="chk-fat-tails", label="Fat tails (Student-t)", value=True),
                                dcc.Slider(id="slider-df", min=3, max=10, step=1, value=5),
                            ], width=6),
                        ]),
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
                        dash_table.DataTable(
                            id="tbl-spending",
                            columns=[
                                {"name": "Age From", "id": "age_from", "type": "numeric"},
                                {"name": "Age To", "id": "age_to", "type": "numeric"},
                                {"name": "Amount Monthly (₪)", "id": "amount_monthly", "type": "numeric"},
                                {"name": "Label", "id": "label", "type": "text"},
                                {"name": "Category", "id": "category", "presentation": "dropdown"}
                            ],
                            data=[],
                            editable=True,
                            row_deletable=True,
                            style_header={
                                "backgroundColor": "var(--md-surface-2)", "fontWeight": "500",
                                "textTransform": "uppercase", "fontSize": "0.78rem",
                                "color": "rgba(255,255,255,.87)", "border": "none",
                            },
                            style_cell={
                                "fontFamily": "Roboto, sans-serif", "padding": "10px 12px",
                                "backgroundColor": "var(--md-surface-1)", "color": "rgba(255,255,255,.87)",
                                "border": "none",
                            },
                            style_data_conditional=[
                                {
                                    "filter_query": '{category} = "strict"',
                                    "column_id": "category",
                                    "backgroundColor": CATEGORY_COLORS["strict"] + " !important",
                                    "color": "white"
                                },
                                {
                                    "filter_query": '{category} = "lifestyle"',
                                    "column_id": "category",
                                    "backgroundColor": CATEGORY_COLORS["lifestyle"] + " !important",
                                    "color": "white"
                                },
                                {
                                    "filter_query": '{category} = "gifts"',
                                    "column_id": "category",
                                    "backgroundColor": CATEGORY_COLORS["gifts"] + " !important",
                                    "color": "white"
                                }
                            ],
                            dropdown={
                                "category": {
                                    "options": [
                                        {"label": c, "value": c} for c in ("strict","lifestyle","gifts")
                                    ]
                                }
                            }
                        ),
                        dbc.Button("+ Add band", id="btn-add-spending", color="outline-primary", size="sm"),
                    ], label="Spending", tab_id="tab-spending"),
                    
                    dbc.Tab([
                        # Income tab content
                        dash_table.DataTable(
                            id="tbl-income",
                            columns=[
                                {"name": "Age From", "id": "age_from", "type": "numeric"},
                                {"name": "Age To", "id": "age_to", "type": "numeric"},
                                {"name": "Amount Monthly (₪)", "id": "amount_monthly", "type": "numeric"},
                                {"name": "Label", "id": "label", "type": "text"}
                            ],
                            data=[],
                            editable=True,
                            row_deletable=True,
                            style_header={
                                "backgroundColor": "var(--md-surface-2)", "fontWeight": "500",
                                "textTransform": "uppercase", "fontSize": "0.78rem",
                                "color": "rgba(255,255,255,.87)", "border": "none",
                            },
                            style_cell={
                                "fontFamily": "Roboto, sans-serif", "padding": "10px 12px",
                                "backgroundColor": "var(--md-surface-1)", "color": "rgba(255,255,255,.87)",
                                "border": "none",
                            },
                        ),
                        dbc.Button("+ Add band", id="btn-add-income", color="outline-primary", size="sm"),
                    ], label="Income", tab_id="tab-income"),
                    
                    dbc.Tab([
                        # Lumps tab content
                        dash_table.DataTable(
                            id="tbl-lumps",
                            columns=[
                                {"name": "Age", "id": "age", "type": "numeric"},
                                {"name": "Amount (₪)", "id": "amount", "type": "numeric"},
                                {"name": "Label", "id": "label", "type": "text"},
                                {"name": "Category", "id": "category", "presentation": "dropdown"}
                            ],
                            data=[],
                            editable=True,
                            row_deletable=True,
                            style_header={
                                "backgroundColor": "var(--md-surface-2)", "fontWeight": "500",
                                "textTransform": "uppercase", "fontSize": "0.78rem",
                                "color": "rgba(255,255,255,.87)", "border": "none",
                            },
                            style_cell={
                                "fontFamily": "Roboto, sans-serif", "padding": "10px 12px",
                                "backgroundColor": "var(--md-surface-1)", "color": "rgba(255,255,255,.87)",
                                "border": "none",
                            },
                            style_data_conditional=[
                                {
                                    "filter_query": '{category} = "strict"',
                                    "column_id": "category",
                                    "backgroundColor": CATEGORY_COLORS["strict"] + " !important",
                                    "color": "white"
                                },
                                {
                                    "filter_query": '{category} = "lifestyle"',
                                    "column_id": "category",
                                    "backgroundColor": CATEGORY_COLORS["lifestyle"] + " !important",
                                    "color": "white"
                                },
                                {
                                    "filter_query": '{category} = "gifts"',
                                    "column_id": "category",
                                    "backgroundColor": CATEGORY_COLORS["gifts"] + " !important",
                                    "color": "white"
                                }
                            ],
                            dropdown={
                                "category": {
                                    "options": [
                                        {"label": c, "value": c} for c in ("strict","lifestyle","gifts")
                                    ]
                                }
                            }
                        ),
                        dbc.Button("+ Add event", id="btn-add-lumps", color="outline-primary", size="sm"),
                    ], label="Lumps", tab_id="tab-lumps"),
                    
                    dbc.Tab([
                        # Properties tab content
                        dash_table.DataTable(
                            id="tbl-properties",
                            columns=[
                                {"name": "Start Age", "id": "start_age", "type": "numeric"},
                                {"name": "Initial Value (₪)", "id": "initial_value", "type": "numeric"},
                                {"name": "Rent Monthly (₪)", "id": "rent_monthly", "type": "numeric"},
                                {"name": "Label", "id": "label", "type": "text"}
                            ],
                            data=[],
                            editable=True,
                            row_deletable=True,
                            style_header={
                                "backgroundColor": "var(--md-surface-2)", "fontWeight": "500",
                                "textTransform": "uppercase", "fontSize": "0.78rem",
                                "color": "rgba(255,255,255,.87)", "border": "none",
                            },
                            style_cell={
                                "fontFamily": "Roboto, sans-serif", "padding": "10px 12px",
                                "backgroundColor": "var(--md-surface-1)", "color": "rgba(255,255,255,.87)",
                                "border": "none",
                            },
                        ),
                        dbc.Button("+ Add property", id="btn-add-properties", color="outline-primary", size="sm"),
                        html.Div("Growth µ/σ follow the scenario's selected market (see Portfolio tab).", 
                                className="small text-muted mt-1"),
                    ], label="Properties", tab_id="tab-properties"),
                ]),
                
                # Playground section
                dbc.Switch(id="switch-playground", label="Playground mode", value=False),
                html.Div(id="banner-playground", 
                        children="PLAYGROUND — click the chart to add an event", 
                        style={"display": "none"}),
                dcc.Graph(id="graph-preview", figure=_EMPTY_DARK_FIGURE),
                html.Div(id="div-playground-chips"),
                
                # Playground modal
                dbc.Modal([
                    dbc.ModalHeader("Add Playground Event"),
                    dbc.ModalBody([
                        dbc.Input(id="input-pg-age", type="number", placeholder="Age"),
                        dbc.Input(id="input-pg-amount", type="number", placeholder="Amount (₪)"),
                        dbc.Input(id="input-pg-label", type="text", placeholder="Label"),
                        dbc.Button("Confirm", id="btn-pg-confirm", color="primary", className="mt-2"),
                    ]),
                ], id="modal-playground", is_open=False),
                
                dbc.Button("Clear all", id="btn-pg-clear", color="outline-secondary", size="sm"),
                
                # Guardrails panel
                dbc.Button("▸ Guardrails", id="btn-guardrails-header", color="outline-secondary", className="mb-2"),
                dbc.Collapse([
                    dbc.Checkbox(id="chk-g1-enable", label="Enable G1"),
                    html.Div("Drop threshold:", className="mt-2"),
                    dcc.Slider(id="slider-g1-drop", min=0.05, max=0.50, step=0.01, value=0.20),
                    html.Div("Rise threshold:", className="mt-2"),
                    dcc.Slider(id="slider-g1-rise", min=0.05, max=0.50, step=0.01, value=0.20),
                    html.Div("Cut percentage:", className="mt-2"),
                    dcc.Slider(id="slider-g1-cut", min=0.00, max=0.50, step=0.01, value=0.15),
                    html.Div("Raise percentage:", className="mt-2"),
                    dcc.Slider(id="slider-g1-raise", min=0.00, max=0.50, step=0.01, value=0.10),
                ], id="collapse-guardrails", is_open=False),

            ]), className="md-panel"), width=5),

            # Right column - Results
            dbc.Col(dbc.Card(dbc.CardBody([
                dbc.Button("Run Simulation ▶", id="btn-run", color="primary"),
                dbc.Button("Run with playground events ▶", id="btn-run-playground", color="primary"),
                dbc.Switch(id="switch-historic", label="Include historic scenarios", value=False),
                html.Div(id="div-result-badges"),
                html.Div(id="div-summary"),
                dcc.Loading(children=[
                    dcc.Graph(id="graph-results", figure=_EMPTY_DARK_FIGURE)
                ]),
            ]), className="md-panel"), width=7),
        ]),
        
        # Hidden stores
        dcc.Store(id="store-scenario", storage_type="session", data=DEFAULT_SCENARIO),
        dcc.Store(id="store-playground", storage_type="memory", data=[]),
        dcc.Store(id="store-guardrails", storage_type="session", data={"guardrails": [{"type": "volatility_discretionary_scaling", "enabled": False, "drop_threshold": 0.20, "rise_threshold": 0.20, "cut_pct": 0.15, "raise_pct": 0.10}]}),
        dcc.Store(id="store-run-id", storage_type="memory", data=None),
        
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