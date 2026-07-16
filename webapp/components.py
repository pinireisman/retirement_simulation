from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from engine.theme import PLOTLY_TEMPLATE


def build_panel(title, children):
    """Wrap children in a dbc.Card with md-panel class and optional title."""
    card_body = [dbc.CardBody(children)]
    
    # Add title if provided
    if title:
        card_body = [html.H5(title)] + card_body
    
    return dbc.Card(card_body, className="md-panel")


def build_data_table(id, columns, category_col=None, **table_kwargs):
    """Build a shared dash_table.DataTable with consistent styling."""
    
    # Common style dicts (byte-identical across all tables)
    style_header = {
        "backgroundColor": "var(--md-surface-2)", "fontWeight": "500",
        "textTransform": "uppercase", "fontSize": "0.78rem",
        "color": "var(--ink)", "border": "none",
    }

    style_cell = {
        "fontFamily": "Roboto, sans-serif", "padding": "10px 12px",
        "backgroundColor": "var(--md-surface-1)", "color": "var(--ink)",
        "border": "none",
    }
    
    # Build style_data_conditional and dropdown if category_col is provided
    style_data_conditional = []
    dropdown = {}
    
    if category_col:
        from engine.figures import CATEGORY_COLORS
        
        # Create conditional styles for each category
        for cat in CATEGORY_COLORS.keys():
            style_data_conditional.append({
                "filter_query": '{' + category_col + '} = "' + cat + '"',
                "column_id": category_col,
                "backgroundColor": CATEGORY_COLORS[cat] + " !important",
                "color": "white"
            })
        
        # Create dropdown options
        dropdown = {
            category_col: {
                "options": [
                    {"label": c, "value": c} for c in CATEGORY_COLORS.keys()
                ]
            }
        }
    
    # Build the table with all parameters. The leading "drag-handle" column
    # holds no data — assets/row-drag.js drags rows by it, style.css renders
    # the ⠿ glyph — so it never reaches the scenario store.
    table_config = {
        "id": id,
        "columns": [{"name": "", "id": "drag-handle", "editable": False}] + columns,
        "data": [],
        "editable": True,
        "row_deletable": True,
        "style_header": style_header,
        "style_cell": style_cell,
        "style_cell_conditional": [{
            "if": {"column_id": "drag-handle"},
            "width": "28px", "minWidth": "28px", "maxWidth": "28px",
            "padding": "10px 6px",
        }],
    }
    
    if style_data_conditional:
        table_config["style_data_conditional"] = style_data_conditional
    
    if dropdown:
        table_config["dropdown"] = dropdown
        
    # Add any additional kwargs
    for key, value in table_kwargs.items():
        table_config[key] = value
    
    return dash_table.DataTable(**table_config)


def build_stat_tile(label, value, tone=None, hero=False):
    """Build a stat tile with label and value."""
    # Determine text color based on tone (CSS vars so dark mode repaints correctly)
    color = "var(--ink)"
    if tone == "success":
        color = "var(--success)"
    elif tone == "borderline":
        color = "var(--warning)"
    elif tone == "danger":
        color = "var(--danger)"
    
    # Determine font size based on hero flag
    font_size = "2rem" if hero else "1.5rem"
    
    return dbc.Card([
        dbc.CardBody([
            html.Span(label.upper(), className="small text-muted"),
            html.H3(value, style={"color": color, "fontSize": font_size})
        ])
    ])


def build_badge_row(items):
    """Build a row of badges from items list."""
    badges = []
    
    for item in items:
        if isinstance(item, str):
            # Plain string badge
            badges.append(dbc.Badge(str(item), color="secondary", className="me-1"))
        elif isinstance(item, dict):
            # Dict with text, color, and optional id
            badge_kwargs = {
                "children": item["text"],
                "color": item.get("color", "secondary"),
                "className": "me-1"
            }
            
            if "id" in item:
                badge_kwargs["id"] = item["id"]
                
            badges.append(dbc.Badge(**badge_kwargs))
    
    return badges


def build_chart_card(title, graph_id, figure=None, legend_toggle=False):
    """Build a chart card with title, a maximize button, and the graph."""
    if figure is None:
        figure = go.Figure(layout={"template": PLOTLY_TEMPLATE})

    buttons = []
    if legend_toggle:
        buttons.append(dbc.Button("Legend", className="btn-legend-toggle p-0 me-2", color="link",
                                   size="sm", title="Show / hide legend"))
    buttons.append(dbc.Button("⤢", className="btn-maximize p-0", color="link", size="sm",
                               title="Maximize / restore"))

    header = html.Div([
        html.H5(title, className="mb-0"),
        html.Div(buttons, className="d-flex align-items-center"),
    ], className="d-flex justify-content-between align-items-center mb-2")

    return dbc.Card(
        dbc.CardBody([header, dcc.Graph(id=graph_id, figure=figure, config={"responsive": True})]),
        className="md-panel chart-card",
    )
