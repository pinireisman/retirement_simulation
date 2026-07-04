"""UI-suite plumbing: app-under-test server, console-error enforcement,
example-scenario fixtures, and the summary.json the LLM analyst consumes.

Run via scripts/run_ui_tests.sh (sets UI_RUN_DIR); a bare
`.venv/bin/pytest tests/ui -m ui` also works and defaults to a timestamped
run dir under artifacts/.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import pytest

from engine.params import scenario_from_xlsx
from tests.ui import journeys

REPO = Path(__file__).resolve().parents[2]
EXAMPLE_XLSX = REPO / "scenario_data_example.xlsx"
PORT = int(os.environ.get("UI_TEST_PORT", "8060"))

# Console messages that don't indicate an app bug (external resources when
# offline, browser chrome noise). Everything else fails the test.
ALLOWED_CONSOLE = ("fonts.googleapis.com", "fonts.gstatic.com", "favicon")


# ---------- run directory + summary.json ----------

def pytest_configure(config):
    run_dir = os.environ.get("UI_RUN_DIR")
    if not run_dir:
        run_dir = str(REPO / "artifacts" / f"ui-run-{datetime.now():%Y%m%d-%H%M%S}")
    config._ui_run_dir = Path(run_dir)
    (config._ui_run_dir / "console").mkdir(parents=True, exist_ok=True)
    config._ui_results = []


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" and not (report.when == "setup" and report.failed):
        return
    ux_marker = item.get_closest_marker("ux")
    entry = {
        "ux_id": ux_marker.args[0] if ux_marker else None,
        "nodeid": item.nodeid,
        "outcome": "error" if (report.when == "setup" and report.failed) else report.outcome,
        "duration": round(report.duration, 2),
        "failure_message": report.longreprtext[:2000] if report.failed else None,
        "artifacts": [],
    }
    console_log = item.config._ui_run_dir / "console" / f"{_safe(item.nodeid)}.log"
    if console_log.exists():
        entry["artifacts"].append(str(console_log.relative_to(item.config._ui_run_dir)))
    item.config._ui_results.append(entry)


def pytest_sessionfinish(session, exitstatus):
    config = session.config
    if not hasattr(config, "_ui_run_dir"):
        return
    results = config._ui_results
    if not results:
        return
    git_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
        capture_output=True, text=True).stdout.strip()
    summary = {
        "run_id": config._ui_run_dir.name,
        "git_sha": git_sha,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "totals": {
            k: sum(1 for r in results if r["outcome"] == k)
            for k in ("passed", "failed", "error", "skipped")
        },
        "tests": results,
    }
    (config._ui_run_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def _safe(nodeid: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in nodeid)


# ---------- app-under-test server ----------

@pytest.fixture(scope="session")
def app_url(request):
    """Dash app in a subprocess: real server.log artifact, debug=False so the
    dev toolbar never pollutes screenshots."""
    log_path = request.config._ui_run_dir / "server.log"
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            [sys.executable, "-c",
             f"from webapp.app import app; app.run(debug=False, port={PORT})"],
            cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONPATH": str(REPO)},
        )
        url = f"http://127.0.0.1:{PORT}"
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                urllib.request.urlopen(url, timeout=1)
                break
            except Exception:
                if proc.poll() is not None:
                    raise RuntimeError(f"app server died at startup — see {log_path}")
                time.sleep(0.3)
        else:
            proc.terminate()
            raise RuntimeError(f"app server not up after 15s — see {log_path}")
        yield url
        proc.terminate()
        proc.wait(timeout=5)


# ---------- console-error enforcement (covers UX-CON-01 suite-wide) ----------

@pytest.fixture(autouse=True)
def fail_on_console_errors(request):
    """Attach error collectors to the Playwright page (when the test uses one)
    and fail the test on any non-allowlisted console error or pageerror."""
    if "page" not in request.fixturenames:
        yield
        return
    page = request.getfixturevalue("page")
    errors: list[str] = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    yield
    real = [e for e in errors if not any(a in e for a in ALLOWED_CONSOLE)]
    if real:
        log = request.config._ui_run_dir / "console" / f"{_safe(request.node.nodeid)}.log"
        log.write_text("\n".join(real))
        pytest.fail(f"{len(real)} console error(s) — see {log}", pytrace=False)


# ---------- scenario fixtures ----------

@pytest.fixture(scope="session")
def example_scenario():
    """The example xlsx parsed through the engine — tests assert UI state
    against these values instead of hard-coding magic numbers."""
    return scenario_from_xlsx(EXAMPLE_XLSX)


@pytest.fixture
def loaded_page(page, app_url):
    """A page with the example scenario uploaded — the common journey start."""
    page.goto(app_url)
    journeys.upload_scenario(page, str(EXAMPLE_XLSX))
    return page
