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
    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, port=8050)
