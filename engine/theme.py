"""Single source of truth for color tokens (PRD §3.1). Hexes are mirrored by
hand in webapp/assets/style.css :root — keep the two in sync on any change."""

import colorsys

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


def _shade(hex_color: str, index: int, dark: bool = False) -> str:
    """Shade a hex color per index, so multiple bands of the same category
    stay distinguishable in the chart legend. Light palette alternates +-10%
    lightness around the base; the dark palette walks upward only in 5% steps,
    because its bases already sit at the lightness floor below which a fill
    loses 3:1 contrast against the dark chart surface."""
    if index == 0:
        return hex_color
    r = int(hex_color[1:3], 16) / 255
    g = int(hex_color[3:5], 16) / 255
    b = int(hex_color[5:7], 16) / 255
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    if dark:
        l = min(0.85, l + 0.05 * index)  # ponytail: collides past ~8 shades of one category, like light's clamp
    else:
        magnitude = 0.1 * ((index + 1) // 2)
        sign = 1 if index % 2 else -1
        l = min(0.78, max(0.22, l + sign * magnitude))  # stay off near-black/near-white so hue reads at a glance
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return "#{:02X}{:02X}{:02X}".format(round(r * 255), round(g * 255), round(b * 255))


###############################################################################
# Dark-mode chart palette. Figures are always baked in the light palette;
# assets/theme.js repaints traces client-side by exact string match against
# the map below, which layout.py serializes into the #chart-color-map div.
# Categorical steps validated (dataviz six checks) against surface #1B1F29.
###############################################################################

_DARK_SHADE_BASES = {
    SERIES_INCOME_BASE: "#43A047",
    SERIES_RENT_BASE: "#2196F3",
    CATEGORY_COLORS["strict"]: "#E53935",   # distinct from DANGER's dark twin so the map stays invertible
    CATEGORY_COLORS["lifestyle"]: "#EC407A",
    CATEGORY_COLORS["gifts"]: "#AB47BC",
}

_DARK_STATIC = {
    PRIMARY: "#7C93FF",                     # portfolio median
    PRIMARY_HOVER: "#93A6FF",
    WARNING: "#FFCA80",
    DANGER: "#EF5350",                      # draw active
    SERIES_PROPERTY_MEDIAN: "#F06292",
    PLAYGROUND_COLOR: "#FF9F4A",
    BAND_25_75_FILL: "rgba(124,147,255,.22)",
    BAND_5_95_FILL: "rgba(124,147,255,.12)",
    SERIES_PERCENTILE_LINE: "rgba(124,147,255,.45)",
    HISTORIC_BOUNDARY_FILL: "rgba(231,233,238,.08)",
    HISTORIC_BOUNDARY_LINE: "rgba(231,233,238,.35)",
    SERIES_DRAW_NEUTRAL: "rgba(231,233,238,.18)",
    SERIES_POSITIVE_LUMP: "rgba(231,233,238,.55)",
    INK_2: "rgba(231,233,238,.75)",         # net cash flow / total estate
}


def chart_dark_color_map() -> dict:
    """light -> dark for every color figures.py can hand to a trace, including
    all _shade() derivatives. setdefault keeps first-wins where the lightness
    clamp makes two shade indices collide."""
    m = dict(_DARK_STATIC)
    for light_base, dark_base in _DARK_SHADE_BASES.items():
        for i in range(8):
            m.setdefault(_shade(light_base, i), _shade(dark_base, i, dark=True))
    return m


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
