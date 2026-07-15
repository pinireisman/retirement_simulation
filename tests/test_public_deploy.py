"""Acceptance tests for docs/PRD_PUBLIC_DEPLOYMENT.md phases 1-2
(save-to-download, localStorage autosave + stale-schema guard,
store-dirty, deploy plumbing)."""
import copy
import io
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _walk(node):
    """Depth-first over a Dash component tree."""
    yield node
    children = getattr(node, "children", None)
    for child in (children if isinstance(children, (list, tuple)) else [children]):
        if hasattr(child, "to_plotly_json"):
            yield from _walk(child)


def _by_id(layout, cid):
    return next((c for c in _walk(layout) if getattr(c, "id", None) == cid), None)


# --- PRD §3.1: save → browser download ---------------------------------------

def test_layout_has_download_and_no_overwrite_checkbox():
    from webapp.layout import build_layout
    layout = build_layout()
    dl = _by_id(layout, "download-scenario")
    assert dl is not None and type(dl).__name__ == "Download"
    assert _by_id(layout, "chk-overwrite") is None
    assert _by_id(layout, "div-overwrite-checkbox") is None


def test_save_callback_outputs_download_not_dropdown():
    from webapp.app import create_app
    joined = " ".join(create_app().callback_map.keys())
    assert "download-scenario.data" in joined
    assert "chk-overwrite" not in joined
    assert "div-overwrite-checkbox" not in joined


def test_scenario_xlsx_roundtrips_through_bytesio():
    from engine.params import scenario_from_xlsx, scenario_to_xlsx
    from webapp.layout import DEFAULT_SCENARIO
    s = copy.deepcopy(DEFAULT_SCENARIO)
    s["spending_bands"] = [{"id": "sb-1", "age_from": 60, "age_to": 95,
                            "amount_monthly": 1000, "label": "x",
                            "category": "lifestyle"}]
    buf = io.BytesIO()
    scenario_to_xlsx(s, buf)
    loaded = scenario_from_xlsx(io.BytesIO(buf.getvalue()))
    assert loaded["portfolio"]["start_age"] == 60
    assert loaded["spending_bands"][0]["amount_monthly"] == 1000


# --- PRD §3.3: localStorage + stale-schema guard ------------------------------

def test_store_scenario_is_local():
    from webapp.layout import build_layout
    assert _by_id(build_layout(), "store-scenario").storage_type == "local"


@pytest.mark.parametrize("blob", [
    None, "junk", 42, [], {}, {"name": "x"},
    {"$schema": "scenario.v0", "portfolio": {}},
    {"portfolio": {}},  # missing $schema
])
def test_stale_schema_guard_falls_back_to_default(blob):
    from webapp.callbacks import _valid_scenario
    from webapp.layout import DEFAULT_SCENARIO
    assert _valid_scenario(blob) == DEFAULT_SCENARIO


def test_stale_schema_guard_passes_valid_scenario():
    from webapp.callbacks import _valid_scenario
    from webapp.layout import DEFAULT_SCENARIO
    s = copy.deepcopy(DEFAULT_SCENARIO)
    s["name"] = "mine"
    assert _valid_scenario(s) == s


# --- PRD §3.4: store-dirty replaces module-level _dirty ------------------------

def test_module_dirty_global_removed():
    import webapp.callbacks as cb
    assert not hasattr(cb, "_dirty")


def test_store_dirty_wired():
    from webapp.app import create_app
    from webapp.layout import build_layout
    store = _by_id(build_layout(), "store-dirty")
    assert store is not None and store.storage_type == "memory"
    writers = [k for k in create_app().callback_map.keys()
               if "store-dirty.data" in k]
    # collect_edits (set) + save/load/upload (clear)
    assert len(writers) >= 4


# --- PRD §4: deploy plumbing ----------------------------------------------------

def test_wsgi_server_exposed():
    import webapp.app as appmod
    assert appmod.server is appmod.app.server


def test_requirements_split():
    deploy = (ROOT / "requirements.txt").read_text().lower()
    assert "gunicorn" in deploy
    for banned in ("pyqt5", "matplotlib", "pytest"):
        assert banned not in deploy
    dev = (ROOT / "requirements-dev.txt").read_text().lower()
    assert "-r requirements.txt" in dev
    for pkg in ("pyqt5", "matplotlib", "pytest", "pytest-playwright"):
        assert pkg in dev


def test_render_yaml():
    text = (ROOT / "render.yaml").read_text()
    assert "gunicorn webapp.app:server" in text
    assert "--workers 1" in text
