from webapp.app import create_app
from webapp.layout import DEFAULT_SCENARIO, build_layout
from webapp.callbacks import _preview_figure


def test_layout_builds():
    layout = build_layout()
    assert layout is not None


def test_app_constructs():
    app = create_app()
    assert app.layout is not None


def test_default_scenario_renders_preview():
    fig = _preview_figure(DEFAULT_SCENARIO, [])
    assert fig is not None
