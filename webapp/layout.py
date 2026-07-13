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
                        html.Div(
                            dbc.Switch(id="switch-guardrails-enabled", label="Enable spending guardrails", value=False),
                            className="mt-2",
                        ),
                        html.Div(
                            dbc.Switch(id="switch-compare-enabled",
                                       label="Compare two-bucket strategy to single portfolio",
                                       value=True),
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
                    dbc.Col(build_chart_card("Cash flow", "graph-results", figure=_EMPTY_FIGURE,
                                              legend_toggle=True), width=6),
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
                
                # Guardrails panel — enable/disable lives on the Dashboard
                # (switch-guardrails-enabled); this panel only configures thresholds.
                dbc.Button("Guardrails", id="btn-guardrails-header", color="outline-secondary", className="mb-2"),
                dbc.Collapse([
                    # Percent units in the UI; collect_guardrails converts to fractions.
                    html.Div([
                        html.Div("What should the guardrails do for you?", className="mt-2"),
                        dcc.RadioItems(id="dd-g2-goal",
                                       options=[{"label": " Protect the plan", "value": "protect"},
                                                {"label": " Balanced", "value": "balanced"},
                                                {"label": " Enjoy the upside", "value": "upside"}],
                                       value="balanced", labelStyle={"display": "block"}),
                        html.Div("In a rough stretch, how much could you realistically trim lifestyle spending?",
                                 className="mt-2"),
                        dcc.RadioItems(id="dd-g2-tolerance",
                                       options=[{"label": " A little (~10%)", "value": "little"},
                                                {"label": " Up to a quarter", "value": "quarter"},
                                                {"label": " Up to half", "value": "half"}],
                                       value="quarter", labelStyle={"display": "block"}),
                        dbc.Checkbox(id="chk-g2-flex-lumps", value=True, className="mt-2",
                                     label="Under stress, the big optional gifts/upgrades can shrink or wait"),
                        html.Small("Thresholds are calibrated from your own plan's simulation, "
                                   "assuming planned spending: cuts are measured slightly "
                                   "conservatively, raises slightly optimistically (both bounded "
                                   "by the yearly caps).", className="text-muted d-block mt-1"),
                        dbc.Checkbox(id="chk-g2-manual", value=False, className="mt-3",
                                     label="Advanced: set thresholds manually"),
                        html.Div([
                            html.Div("Lower guardrail — funded ratio (%):", className="mt-2"),
                            dcc.Slider(id="slider-g2-lower", min=50, max=150, step=1, value=85,
                                       tooltip={"placement": "bottom", "template": "{value}%"}),
                            html.Div("Target funded ratio (%):", className="mt-2"),
                            dcc.Slider(id="slider-g2-target", min=90, max=160, step=1, value=105,
                                       tooltip={"placement": "bottom", "template": "{value}%"}),
                            html.Div("Upper guardrail — funded ratio (%):", className="mt-2"),
                            dcc.Slider(id="slider-g2-upper", min=100, max=250, step=1, value=130,
                                       tooltip={"placement": "bottom", "template": "{value}%"}),
                        ], id="block-g2-manual", style={"display": "none"}),
                        html.Div("Real discount rate (%):", className="mt-2"),
                        dcc.Slider(id="slider-g2-discount", min=0, max=4, step=0.25, value=1,
                                   tooltip={"placement": "bottom", "template": "{value}%"}),
                    ], id="block-g2"),
                ], id="collapse-guardrails", is_open=False),

                # Withdrawal strategy panel (PRD two_bucket_retirement_strategy §12.2)
                dbc.Button("Withdrawal strategy", id="btn-withdrawal-strategy-header",
                           color="outline-secondary", className="mb-2"),
                dbc.Collapse([
                    dbc.RadioItems(
                        id="radio-withdrawal-strategy",
                        options=[
                            {"label": " Single portfolio", "value": "single_portfolio"},
                            {"label": " Growth + spending reserve (two-bucket)", "value": "two_bucket"},
                        ],
                        value="single_portfolio",
                    ),
                    html.Small(
                        "Single portfolio: the whole balance is invested in one growth stream, as "
                        "elsewhere in this tool. Growth + spending reserve: splits the portfolio into "
                        "a market-exposed growth bucket and a separate reserve that funds withdrawals "
                        "during down markets, refilled from growth in good years.",
                        className="text-muted d-block mt-1",
                    ),
                    html.Small(
                        "In historic-scenario runs, growth follows the actual historic returns for "
                        "that stress period; the reserve's return is always the configured assumption "
                        "below, not historical — its historic path doesn't exist.",
                        className="text-muted d-block mt-1",
                    ),
                    html.Div([
                        dbc.Row([
                            dbc.Col(dbc.Card(dbc.CardBody([
                                html.H6("Reserve size"),
                                html.Div("Target years of spending:"),
                                html.Small(
                                    "How much of the forecast portfolio-funded gap (spending the "
                                    "portfolio must cover after income) the reserve is sized to hold, "
                                    "in years.",
                                    className="text-muted d-block",
                                ),
                                dbc.Input(id="slider-wd-target-years", type="number", step=0.5, value=4.0),
                                html.Div("Refill trigger (years):", className="mt-2"),
                                html.Small(
                                    "Reserve trigger: once the reserve falls below this many years of "
                                    "the gap, a refill from growth is considered.",
                                    className="text-muted d-block",
                                ),
                                dbc.Input(id="slider-wd-trigger-years", type="number", step=0.5, value=3.0),
                                html.Div("Coverage scope:", className="mt-2"),
                                html.Small(
                                    "Which planned cash flows count toward the reserve's forward-looking "
                                    "target size. Recurring only: everyday spending. + gifts: adds "
                                    "scheduled gift payments. All planned outflows: adds lump sums too — "
                                    "the reserve target rises before a scheduled spending increase "
                                    "arrives, since it looks ahead the full target-years window.",
                                    className="text-muted d-block",
                                ),
                                dcc.Dropdown(id="dd-wd-coverage-scope", clearable=False, options=[
                                    {"label": "Recurring spending only", "value": "recurring_gap_only"},
                                    {"label": "Recurring + scheduled gifts", "value": "recurring_plus_scheduled_gifts"},
                                    {"label": "All planned outflows", "value": "all_planned_outflows"},
                                ], value="recurring_gap_only"),
                            ])), width=3),
                            dbc.Col(dbc.Card(dbc.CardBody([
                                html.H6("Reserve investment assumptions"),
                                html.Div("Distribution:"),
                                html.Small(
                                    "How the reserve's return is sampled each period, independent of "
                                    "the growth bucket's market model. Constant: always exactly the mean "
                                    "below. Normal / Student-t: random each period; Student-t adds fatter "
                                    "tails (more extreme periods) via the degrees-of-freedom setting.",
                                    className="text-muted d-block",
                                ),
                                dcc.Dropdown(id="dd-wd-distribution", clearable=False, options=[
                                    {"label": "Constant", "value": "constant"},
                                    {"label": "Normal", "value": "normal"},
                                    {"label": "Student-t", "value": "student_t"},
                                ], value="normal"),
                                html.Div("Expected real return (%):", className="mt-2"),
                                html.Small(
                                    "Real return: the reserve's assumed growth rate net of inflation, "
                                    "sampled independently each period from the distribution above — "
                                    "not drawn from the market model used for the growth bucket.",
                                    className="text-muted d-block",
                                ),
                                dbc.Input(id="inp-wd-mean-real", type="number", step=0.1, value=1.0),
                                html.Div("Std dev, real (%):", className="mt-2"),
                                html.Small(
                                    "How much the reserve's return varies period to period around the "
                                    "mean above. Ignored when Distribution is Constant.",
                                    className="text-muted d-block",
                                ),
                                dbc.Input(id="inp-wd-std-real", type="number", step=0.1, value=3.0),
                                html.Div([
                                    html.Div("Student-t degrees of freedom:", className="mt-2"),
                                    html.Small(
                                        "Lower values (e.g. 3-5) mean fatter tails — more frequent "
                                        "extreme returns — than Normal. Higher values converge toward "
                                        "Normal.",
                                        className="text-muted d-block",
                                    ),
                                    dbc.Input(id="inp-wd-df", type="number", step=1, value=5),
                                ], id="div-wd-df-block", style={"display": "none"}),
                            ])), width=3),
                            dbc.Col(dbc.Card(dbc.CardBody([
                                html.H6("Draw policy"),
                                html.Div("Growth funds withdrawals when its trailing real return is at/above (%):"),
                                html.Small(
                                    "Below this threshold, growth is in a down market and spending "
                                    "draws from the reserve instead. If the reserve is empty when that "
                                    "happens, growth funds it anyway — a forced growth sale.",
                                    className="text-muted d-block",
                                ),
                                dbc.Input(id="inp-wd-draw-threshold", type="number", step=0.1, value=0.0),
                                html.Div("First-period funding source:", className="mt-2"),
                                html.Small(
                                    "Only matters for the very first simulated period, since there's no "
                                    "prior return yet to judge growth as favorable or not. Every later "
                                    "period is decided by the threshold above instead.",
                                    className="text-muted d-block",
                                ),
                                dcc.Dropdown(id="dd-wd-first-period-source", clearable=False, options=[
                                    {"label": "Reserve", "value": "reserve"},
                                    {"label": "Growth", "value": "growth"},
                                ], value="reserve"),
                            ])), width=3),
                            dbc.Col(dbc.Card(dbc.CardBody([
                                html.H6("Refill policy"),
                                html.Div("Eligibility:"),
                                html.Small(
                                    "When a refill from growth to the reserve is allowed (only "
                                    "considered once the reserve is below its trigger, at left). "
                                    "Threshold rule: only after a growth return at/above the % below. "
                                    "Always: refill whenever below trigger, regardless of performance. "
                                    "Never: the reserve is never topped up after its initial funding.",
                                    className="text-muted d-block",
                                ),
                                dcc.Dropdown(id="dd-wd-refill-eligibility", clearable=False, options=[
                                    {"label": "Growth return at/above threshold", "value": "growth_return_at_or_above_threshold"},
                                    {"label": "Always", "value": "always"},
                                    {"label": "Never", "value": "never"},
                                ], value="growth_return_at_or_above_threshold"),
                                html.Div("Threshold (%):", className="mt-2"),
                                html.Small(
                                    "Used only when Eligibility is the threshold rule: growth's trailing "
                                    "real return must be at/above this to allow a refill.",
                                    className="text-muted d-block",
                                ),
                                dbc.Input(id="inp-wd-refill-threshold", type="number", step=0.1, value=0.0),
                                html.Div("Amount rule:", className="mt-2"),
                                html.Small(
                                    "How much to transfer on an eligible refill. Top up to target: "
                                    "fully restores the reserve to its current target in one transfer. "
                                    "Gains only: caps the transfer at growth's own gains, never sells "
                                    "into its principal. None: eligibility is checked but nothing moves.",
                                    className="text-muted d-block",
                                ),
                                dcc.Dropdown(id="dd-wd-refill-amount-rule", clearable=False, options=[
                                    {"label": "Top up to target", "value": "to_target"},
                                    {"label": "Gains only", "value": "gains_only"},
                                    {"label": "None", "value": "none"},
                                ], value="to_target"),
                            ])), width=3),
                        ], className="g-2 mt-1"),
                    ], id="div-two-bucket-cards", style={"display": "none"}),
                ], id="collapse-withdrawal-strategy", is_open=False),

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
            {"type": "funded_ratio_guardrail", "enabled": False, "mode": "confidence",
             "c_cut": 0.85, "c_target": 0.95, "c_raise": 0.99, "c_severe": 0.80,
             "min_multiplier": 0.75, "max_cut_per_year": 0.10},
        ]}),
        # Hydration echo-guard. MUST be memory-type: its job is to track what
        # THIS page's widgets last rendered, so it has to reset on refresh (a
        # server-side guard survives refreshes and made hydrate_tabs skip the
        # first store->widgets render of a fresh page: empty tables).
        dcc.Store(id="store-hydrate-guard", storage_type="memory", data=None),
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
