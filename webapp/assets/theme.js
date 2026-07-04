// Dark/light theme toggle. Bootstrap 5's [data-bs-theme] cascade + our own
// CSS token overrides (style.css) handle all Bootstrap/CSS-driven styling.
// Plotly figures are baked server-side in the light palette (engine/figures.py),
// so we remap the known trace colors client-side whenever a figure is (re)drawn.
(function () {
    // light hex/rgba -> dark equivalent, for every color engine/theme.py hands
    // to a trace (line/marker/fillcolor). Keep in sync by hand if theme.py changes.
    var LIGHT_TO_DARK = {
        "#3949AB": "#7C93FF", // primary / portfolio median
        "#303F9F": "#93A6FF", // primary hover
        "#2E7D32": "#66BB6A", // success / income base
        "#B26A00": "#FFCA80", // warning
        "#C62828": "#EF5350", // danger / draw active
        "#AD1457": "#F06292", // property median
        "#1565C0": "#64B5F6", // rent base
        "#E65100": "#FF9F4A", // playground
        "rgba(57,73,171,.15)": "rgba(124,147,255,.22)",  // band 25-75 fill
        "rgba(207,232,255,.55)": "rgba(124,147,255,.12)", // band 5-95 fill
        "rgba(57,73,171,.35)": "rgba(124,147,255,.45)",  // percentile line
        "rgba(31,41,51,.06)": "rgba(231,233,238,.08)",   // historic boundary fill
        "rgba(31,41,51,.35)": "rgba(231,233,238,.35)",   // historic boundary line
        "rgba(31,41,51,.15)": "rgba(231,233,238,.18)",   // draw neutral
        "rgba(31,41,51,.45)": "rgba(231,233,238,.55)",   // positive lump
        "rgba(31,41,51,.64)": "rgba(231,233,238,.75)",   // net cash flow / total estate
    };
    var DARK_TO_LIGHT = {};
    Object.keys(LIGHT_TO_DARK).forEach(function (k) { DARK_TO_LIGHT[LIGHT_TO_DARK[k]] = k; });

    var LIGHT_SURFACE = "#FFFFFF", DARK_SURFACE = "#1B1F29";
    var LIGHT_INK = "#1F2933", DARK_INK = "#E7E9EE";
    var LIGHT_GRID = "rgba(31,41,51,.08)", DARK_GRID = "rgba(231,233,238,.08)";
    var LIGHT_LINE = "rgba(31,41,51,.15)", DARK_LINE = "rgba(231,233,238,.2)";

    function isDark() {
        return document.documentElement.getAttribute("data-bs-theme") === "dark";
    }

    function remap(value, map) {
        if (Array.isArray(value)) return value.map(function (v) { return map[v] || v; });
        if (typeof value === "string") return map[value] || value;
        return value;
    }

    // Recolor known trace colors in place, then relayout backgrounds/fonts/grid.
    function paintCharts() {
        if (!window.Plotly) return;
        var dark = isDark();
        var map = dark ? LIGHT_TO_DARK : DARK_TO_LIGHT;
        var surface = dark ? DARK_SURFACE : LIGHT_SURFACE;
        var ink = dark ? DARK_INK : LIGHT_INK;
        var grid = dark ? DARK_GRID : LIGHT_GRID;
        var line = dark ? DARK_LINE : LIGHT_LINE;

        document.querySelectorAll(".js-plotly-plot").forEach(function (gd) {
            if (!gd.data || !gd.data.length) return;

            ["line.color", "marker.color", "fillcolor"].forEach(function (prop) {
                var parts = prop.split(".");
                var indices = [], values = [];
                gd.data.forEach(function (trace, i) {
                    var v = parts.length === 2 ? (trace[parts[0]] || {})[parts[1]] : trace[parts[0]];
                    if (v === undefined) return;
                    indices.push(i);
                    values.push(remap(v, map));
                });
                if (indices.length) Plotly.restyle(gd, { [prop]: values }, indices);
            });

            Plotly.relayout(gd, {
                "paper_bgcolor": surface,
                "plot_bgcolor": surface,
                "font.color": ink,
                "xaxis.gridcolor": grid,
                "xaxis.zerolinecolor": line,
                "xaxis.linecolor": line,
                "yaxis.gridcolor": grid,
                "yaxis.zerolinecolor": line,
                "yaxis.linecolor": line,
                "legend.font.color": ink,
                "hoverlabel.bgcolor": surface,
                "hoverlabel.font.color": ink,
                "hoverlabel.bordercolor": line,
            });
        });
    }
    window.paintCharts = paintCharts;

    document.addEventListener("click", function (e) {
        if (!e.target.closest("#btn-theme-toggle")) return;
        var next = isDark() ? "light" : "dark";
        document.documentElement.setAttribute("data-bs-theme", next);
        localStorage.setItem("theme", next);
        paintCharts();
    });
})();
