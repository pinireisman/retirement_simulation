# PRD: Public Deployment (Render) with Private Local Scenarios

## 1. Background & goal

The retirement simulator webapp (`webapp/`, Dash 4.3) currently runs only on the developer's machine, loading/saving scenario `.xlsx` files from the server-side `scenarios/` folder. Goal: deploy it publicly with auto-deploy from GitHub, such that:

- No private scenario data is ever published or stored server-side.
- Visitors can load a bundled example scenario (`scenarios/scenario_example.xlsx`).
- Each user's own scenarios live on **their** computer: download to save, upload to load, localStorage so a refresh doesn't lose work.

**Why not GitHub Pages** (original request): Pages serves static files only; this app's engine (`engine/simulation.py`), xlsx parsing (`engine/params.py`, openpyxl/numbers-parser), and all Dash callbacks execute in Python on a live server. Decision (confirmed with user): deploy the Dash app as-is to **Render free tier** with native GitHub auto-deploy. (Alternatives considered: Pyodide-in-browser or full TS rewrite — weeks of work; rejected.)

## 2. Privacy requirements (hard)

- **P1.** Only `scenario_example.xlsx` from `scenarios/` may be committed/deployed. `.gitignore` already has `scenarios/*.xlsx`; add `!scenarios/scenario_example.xlsx` and `git add` it.
- **P2. Manual gate before committing**: open `scenario_example.xlsx`, review every sheet for personal names, real balances, identifying labels. Also re-review the already-tracked `scenario_data_example.xlsx` at repo root (it's public today). Verify with `git status --ignored scenarios/` that no other xlsx becomes tracked.
- **P3.** No server-side persistence of user scenarios: the save-to-server-disk path is removed entirely (not flag-gated). On a shared server, one user's save would be visible to all users via the load dropdown.
- **P4.** Upload path (`upload_xlsx`, `webapp/callbacks.py:844`) may keep its per-request `NamedTemporaryFile` (unique names, deleted in both success/error paths) — acceptable transient disk touch.

## 3. Functional changes

### 3.1 Save → browser download (`webapp/callbacks.py`, `webapp/layout.py`)

- Add `dcc.Download(id="download-scenario")` near the stores (`layout.py:~540`).
- Rewrite `save_scenario` (`callbacks.py:893`): replace `scenario_to_xlsx(scenario, Path("scenarios")/...)` with `buf = io.BytesIO(); scenario_to_xlsx(scenario, buf)` → `dcc.send_bytes(buf.getvalue(), f"{sanitized_name}.xlsx")`. (`scenario_to_xlsx` at `engine/params.py:240` uses `pd.ExcelWriter(..., engine="openpyxl")`, which accepts BytesIO — no engine change.)
  - Outputs: swap `Output("dd-load-scenario", "options")` for `Output("download-scenario", "data")`; keep toast + modal outputs; keep `mark_clean()`.
  - Delete the `save_path.exists()` / overwrite-checkbox logic — the browser handles filename collisions.
- Simplify `open_save_modal` (`callbacks.py:931`): drop the exists-check and the `div-overwrite-checkbox` style output.
- Layout: remove `chk-overwrite` + `div-overwrite-checkbox` (`layout.py:547-549`); keep the modal as the filename prompt; relabel button "Save" → "Download .xlsx".

### 3.2 Load dropdown: unchanged

Keep `_scenario_options()` glob (`callbacks.py:73`), `load_scenario` (`callbacks.py:955`), and the 30s `interval-scenarios` refresh as-is. The deployed checkout contains only the committed example, so the dropdown naturally shows just `scenario_example`; local dev keeps the full private dropdown for free. No APP_MODE env var — one code path. (Dev-workflow note: local "save" now downloads to ~/Downloads; move the file into `scenarios/` manually.)

### 3.3 localStorage autosave of working scenario

- `layout.py:526`: `store-scenario` `storage_type="session"` → `"local"`. Hydration already works on first render (echo-guard `store-hydrate-guard` is memory-type, per commit 03e40e9).
- **Stale-schema guard** (required): localStorage outlives deploys. At the top of `hydrate_tabs` (`callbacks.py:335`): if the stored blob is not a dict, lacks `portfolio`, or its `$schema` ≠ `"scenario.v1"` (key already in `DEFAULT_SCENARIO`, `layout.py:15`), fall back to `DEFAULT_SCENARIO` instead of raising into a blank UI. Bump the `$schema` string on future breaking scenario-schema changes.
- Multi-tab last-write-wins: accepted.

### 3.4 Shared-state fix: `_dirty`

Module-level `_dirty` (`callbacks.py:60`) is shared across all users on a server (cosmetic unsaved-dot cross-talk, no data leak). Replace with `dcc.Store(id="store-dirty", storage_type="memory")` threaded through the ~5 sites: `collect_edits` (set), `hydrate_tabs` header read (`:349`), and `save_scenario`/`load_scenario`/`upload_xlsx` (clear). `RESULTS_CACHE` (`callbacks.py:53`) is safe as-is: uuid4-keyed, read only within the same callback invocation, single worker.

## 4. Deployment setup

- `webapp/app.py`: add `server = app.server`. (`debug=True` is inside `__main__` — gunicorn never runs it.)
- `requirements.txt` split:
  - deploy: `numpy, pandas, openpyxl, plotly, dash, dash-bootstrap-components, numbers-parser` (upload accepts `.numbers`) **+ `gunicorn` (missing today)**
  - new `requirements-dev.txt`: `-r requirements.txt` + `pyqt5, matplotlib, pytest, pytest-playwright` (not imported by `webapp/`/`engine/`; pyqt5 must not go to the Linux free tier).
- New `render.yaml`:

  ```yaml
  services:
    - type: web
      name: retirement-simulator
      runtime: python
      plan: free
      buildCommand: pip install -r requirements.txt
      startCommand: gunicorn webapp.app:server --workers 1 --threads 4 --bind 0.0.0.0:$PORT
  ```

  `--workers 1` is **required** (RESULTS_CACHE is per-process), not just frugal.
- Render dashboard: connect repo, deploy branch `main` (or `webapp-port` until merged).
- Known free-tier caveats: sleeps after ~15 min idle (cold start 30–60 s); 512 MB RAM — fits (default 10k paths ≈ single-digit-MB arrays; base RSS ~250–300 MB). Escape hatch if monthly-mode OOMs later: HF Spaces (16 GB free).

## 5. Out of scope

- Pyodide/static rewrite; user accounts; server-side scenario storage of any kind; APP_MODE flag; changing `store-guardrails`/`store-playground` storage types.

## 6. Acceptance criteria / verification

Local (fresh venv, deploy requirements only):

1. `pip install -r requirements.txt` succeeds and app runs under `gunicorn webapp.app:server --workers 1 --threads 4 --bind 0.0.0.0:8050`.
2. Dropdown load of `scenario_example` hydrates tables; Run works.
3. Save modal → browser downloads `<name>.xlsx`; re-uploading that file round-trips identically (engine side covered by `tests/test_xlsx_roundtrip.py`).
4. Edit a value → hard refresh → edits persist (localStorage). Corrupt `store-scenario` in devtools → refresh → falls back to DEFAULT_SCENARIO (no blank UI).
5. Two browser profiles concurrently: runs don't cross; unsaved-dot no longer cross-talks after §3.4.
6. Existing `pytest` suite passes.

Post-deploy smoke (Render URL): page loads after cold start; dropdown shows **exactly** `scenario_example` and nothing else (privacy check); load + Run + download + upload each work once.

## 7. Implementation phases (for delegation)

1. Scenario I/O changes (§3.1–3.3) + `_dirty` store (§3.4) — webapp only, testable locally.
2. Deploy plumbing (§4): app.py, requirements split, render.yaml.
3. Privacy gate (§2): example-file review (manual, user), .gitignore change, commit example.
4. Render hookup + post-deploy smoke test.
