"""Single source of truth for color tokens (PRD §3.1). Hexes are mirrored by
hand in webapp/assets/style.css :root — keep the two in sync on any change."""

import plotly.graph_objects as go

# Base palette
CANVAS = "#F6F8FC"
SURFACE = "#FFFFFF"
INK = "#1F2933"
INK_2 = "rgba(31,41,51,.64)"
PRIMARY = "#3949AB"
PRIMARY_HOVER = "#303F9F"

# Status triad — all three pass AA on white
SUCCESS = "#2E7D32"
WARNING = "#B26A00"
DANGER = "#C62828"

# Tone washes (§3.4) — dashboard backdrop tint + stat-tile tints only, never
# behind chart plot areas.
WASH_NEUTRAL = "#E7DEFF"    # lavender, shown before any run
WASH_SUCCESS = "#D7EEEC"    # mint
WASH_BORDERLINE = "#CFE8FF"  # powder
WASH_DANGER = "#FCDEDE"     # blush

TONE_WASH = {
    "neutral": WASH_NEUTRAL,
    "success": WASH_SUCCESS,
    "borderline": WASH_BORDERLINE,
    "danger": WASH_DANGER,
}

# Category colors (moved from engine/figures.py, re-exported there for compat)
CATEGORY_COLORS = {
    "strict": "#B71C1C",
    "lifestyle": "#D81B60",  # darkened from #F06292 for AA text on light
    "gifts": "#8E44AD",
}
# validate_palette.js: raw #FF9800 fails lightness band + 3:1 contrast on white,
# so the accessible "on light" shade is the only playground color now.
PLAYGROUND_COLOR = "#E65100"


def tone_for_ruin(ruin_probability: float) -> str:
    """Map a ruin probability to a wash/status tone name, reusing the
    green/orange/red thresholds already used for the in-chart ruin icon."""
    if ruin_probability < 0.03:
        return "success"
    if ruin_probability < 0.10:
        return "borderline"
    return "danger"


###############################################################################
# Chart series colors (PRD §3.1 "Chart series mapping")
###############################################################################

SERIES_PORTFOLIO_MEDIAN = PRIMARY
SERIES_PROPERTY_MEDIAN = "#AD1457"      # magenta — #8E44AD (purple) failed CVD
                                         # separation from the primary blue portfolio line
SERIES_TOTAL_ESTATE = INK_2
SERIES_INCOME_BASE = SUCCESS            # _shade()d per income band
SERIES_RENT_BASE = "#1565C0"            # _shade()d per property
SERIES_DRAW_ACTIVE = DANGER
SERIES_DRAW_NEUTRAL = "rgba(31,41,51,.15)"
SERIES_NET_CASH_FLOW = INK_2
BAND_25_75_FILL = "rgba(57,73,171,.15)"   # primary, low alpha — tighter band
BAND_5_95_FILL = "rgba(207,232,255,.55)"  # powder wash — wider band
SERIES_PERCENTILE_LINE = "rgba(57,73,171,.35)"
SERIES_POSITIVE_LUMP = "rgba(31,41,51,.45)"  # neutral ink-grey for inflow lumps (not category-colored)
HISTORIC_BOUNDARY_LINE = "rgba(31,41,51,.35)"
HISTORIC_BOUNDARY_FILL = "rgba(31,41,51,.06)"


###############################################################################
# Plotly template (R2): Roboto, white paper/plot background, light gridlines,
# hoverlabel set once here (root cause of the white-on-white hover bug).
###############################################################################

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Roboto, Helvetica Neue, Arial, sans-serif", color=INK, size=13),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        xaxis=dict(gridcolor="rgba(31,41,51,.08)", zerolinecolor="rgba(31,41,51,.15)", linecolor="rgba(31,41,51,.15)"),
        yaxis=dict(gridcolor="rgba(31,41,51,.08)", zerolinecolor="rgba(31,41,51,.15)", linecolor="rgba(31,41,51,.15)"),
        hoverlabel=dict(bgcolor=SURFACE, font=dict(color=INK, size=13)),
        legend=dict(font=dict(color=INK)),
        margin=dict(t=48, b=40, l=56, r=24),
    )
)
