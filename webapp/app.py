"""Dash app factory + layout assembly + run."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import Dash

from webapp import callbacks
from webapp.layout import build_layout


def create_app() -> Dash:
    app = Dash(
        __name__,
        external_stylesheets=[
            dbc.themes.BOOTSTRAP,
            "https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=Roboto+Mono:wght@400;500&display=swap",
        ],
    )
    app.title = "Retirement Simulator"
    app.layout = build_layout()
    callbacks.register_callbacks(app)

    # Set data-bs-theme before first paint (persisted choice, else OS preference)
    # so the page never flashes light-then-dark on load.
    app.index_string = app.index_string.replace(
        "</head>",
        """<script>(function(){
            var t = localStorage.getItem('theme');
            if (t !== 'light' && t !== 'dark') {
                t = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
            }
            document.documentElement.setAttribute('data-bs-theme', t);
        })();</script></head>""",
    )
    return app


app = create_app()
server = app.server

if __name__ == "__main__":
    app.run(debug=True, port=8050)
