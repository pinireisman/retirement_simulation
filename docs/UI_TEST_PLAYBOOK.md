# UI Test Playbook — orchestration spec

**Audience**: the run orchestrator (default: **Haiku**) and its workers (the
deterministic Playwright suite + the **local LLM analyst** on LM Studio). This
playbook is mechanical on purpose: every step is a command with an expected
outcome. If a step's outcome doesn't match, follow its ✗ branch — do not improvise.

Cost model this enforces: frontier models wrote the machinery once; each test
cycle costs one Haiku session + local-LLM tokens (free). No frontier model is
needed to *run* tests, *analyze* results, or *implement* straightforward fixes.

## Roles

| Role | Who | Does |
|------|-----|------|
| Orchestrator | Haiku (Claude Code session or subagent) | runs the commands below, validates the report, dispatches fixes, escalates |
| Test executor | pytest-playwright (deterministic, no LLM) | drives the browser, emits artifacts |
| Analyst | local LLM via `scripts/analyze_ui_run.py` (LM Studio auto-picks a coder model) | reasons over artifacts → REPORT.md |
| Fix implementer | local LLM via the `delegate-local` skill | implements fixes the report suggests |
| Escalation | Sonnet | only on the triggers in §5 |

## 0. Prerequisites (once per machine — skip if already true)

```bash
cd /Users/pinir/work/retirement_simulation
.venv/bin/python -m playwright install chromium   # browser binaries
curl -s localhost:1234/v1/models | head -c 200     # LM Studio must answer
```
✗ LM Studio not answering → start LM Studio with any qwen-coder model loaded;
without it you can still run tests (skip `--analyze`) but not the analyst.

## 1. Run the suite

```bash
cd /Users/pinir/work/retirement_simulation
scripts/run_ui_tests.sh --analyze
```

- Starts the app itself (subprocess on port 8060, `debug=False`) — do **not**
  start a server manually. If port 8060 is busy: `lsof -ti:8060 | xargs kill`.
- Prints `run dir: artifacts/ui-run-<timestamp>` — call this `$RUN` below.
- Exit code = pytest's (0 all pass / 1 failures) unless the analyst failed
  (exit 2 → §5 escalation).

## 2. Expected artifacts in `$RUN`

| File | Always? | If missing |
|------|---------|-----------|
| `summary.json` | yes | the run itself broke — read pytest's terminal output and `server.log`; do not analyze |
| `junit.xml` | yes | same as above |
| `server.log` | yes | app never started — check port collision, read terminal output |
| `digest.txt` | with `--analyze` | analyst never ran — check LM Studio (§0) |
| `REPORT.md` | with `--analyze` | analyst failed validation twice → `REPORT.invalid.md` exists → §5 |
| `console/*.log` | only for tests with console errors | — |
| `pw/<test>/*.png`, `trace.zip` | only for failures | — |

## 3. Validate REPORT.md (mechanical checklist)

Run each check; all must pass before trusting the report:

1. `test -f $RUN/REPORT.md` → exists.
2. Every UX id with `"outcome": "failed"` or `"error"` in `$RUN/summary.json`
   appears in REPORT.md **both** in the Results table and as a `### <id>`
   Failures subsection:
   `python3 -c "import json,sys,re; s=json.load(open('$RUN/summary.json')); r=open('$RUN/REPORT.md').read(); bad=[t['ux_id'] for t in s['tests'] if t['outcome']!='passed' and (t['ux_id'] or '?') not in r]; print(bad or 'OK')"`
3. Spot-check 2 cited artifact paths: `ls $RUN/<path>` → exist. (The script
   already validated all of them; this guards against script regressions.)
4. Each Failures subsection has all four fields (Evidence / Suspected root
   cause / Suggested fix / Severity).

✗ any check fails → §5 escalation with the run dir.

## 4. Act on the report

For each failure, in severity order:

- **Test-bug vs app-bug triage** (orchestrator judgment, cheap): if the
  Evidence shows a selector timeout inside `tests/ui/journeys.py` while the
  screenshot shows the feature working, it's likely a journey-helper breakage
  (DOM drift) → the fix goes in `tests/ui/journeys.py` only.
- Dispatch fixes to the local LLM via the **delegate-local** skill, one
  failure per brief. The brief must contain: the REPORT.md subsection
  verbatim, the file(s) to change, and the acceptance command
  (`scripts/run_ui_tests.sh -- -k <test name>` — note tests live under
  `tests/ui`, marker `ui`; also `.venv/bin/python -m pytest tests/ -q` must
  stay green). Mark app files readonly when the fix is test-side, and vice
  versa.
- After all fixes: rerun §1. A fix is done only when its test passes AND the
  22 unit tests still pass.
- Max **2** delegate-local rounds per failure, then escalate that failure (§5).

## 5. Escalation triggers → Sonnet

Escalate (hand over `$RUN` path + REPORT.md + what you tried) when:
- `analyze_ui_run.py` exited 2 (invalid report twice), or §3 checklist fails;
- a **Blocker** failure's root cause is marked unclear in the report;
- a failure survived 2 delegate-local fix rounds;
- `summary.json` itself is missing (infrastructure breakage).

Everything else stays at the Haiku + local tier.

## 6. Definition of done (per cycle)

- `scripts/run_ui_tests.sh` exit 0, all UX ids in `summary.json` passed;
- `.venv/bin/python -m pytest tests/ -q` → 22 passed;
- REPORT.md for the final run archived in `$RUN` (leave it, artifacts/ is
  gitignored);
- One-paragraph summary posted to the user: totals, what was fixed, links to
  `$RUN`.

## 7. Initial build-out (first cycle only)

As of 2026-07-04 the machinery is proven (conftest, journeys, runner, 6 passing
exemplar tests) but five test files from `docs/UX_TEST_PLAN.md` §"Implementation
status" are still **to implement**. The orchestrator's first cycle is:

1. For each unimplemented file (`test_nav.py`, `test_run.py`, `test_io.py`,
   `test_playground.py`, `test_a11y_misc.py`, plus the missing cases in the two
   exemplar files) dispatch ONE delegate-local brief containing:
   - the relevant UX_TEST_PLAN.md section verbatim (it is written in
     journey-helper vocabulary on purpose — implementation is transcription);
   - the full text of `tests/ui/test_plan_editing.py` and
     `tests/ui/journeys.py` as the pattern to copy;
   - hard rules: `pytestmark = pytest.mark.ui`; one `@pytest.mark.ux("<ID>")`
     per test; **no selectors in test files** — if a case needs DOM knowledge
     journeys.py lacks, add a helper there with a `COUPLING:` comment;
   - acceptance: the file imports cleanly
     (`.venv/bin/python -m py_compile tests/ui/<file>`). The junior CANNOT run
     browser tests (its `run_tests` tool deselects `ui` by design) — the
     orchestrator runs `scripts/run_ui_tests.sh` itself after each merge.
   - mark `tests/ui/journeys.py` readonly ONLY if the brief needs no new
     helpers; otherwise leave it writable and review its diff extra carefully.
2. After each file lands: `scripts/run_ui_tests.sh -- -k <that file's name>`;
   triage failures per §4 (new-test bugs are usually test-side).
3. When all files pass individually, run the full §1 cycle with `--analyze`.
4. UX-VAL-01/02 and UX-IO-03 depend on exact toast wording — if a test expects
   text the app doesn't emit, the test is wrong, not the app: loosen to the
   documented expectation in UX_TEST_PLAN.md ("mentions age/spending/overwrite").

## Maintenance notes

- **Where coupling lives**: tests express user journeys via
  `tests/ui/journeys.py`; only that file may contain selectors or library
  internals (each marked with a `COUPLING:` comment). Dash component ids are
  stable API (callbacks depend on them); dash_table/react-select/Plotly
  internals are the parts that break on upgrades — expect journey-helper
  fixes, not test rewrites.
- **Adding cases**: add the case to `docs/UX_TEST_PLAN.md` first (ID, steps in
  journey vocabulary, expected, severity), then implement with
  `@pytest.mark.ux("<ID>")` in the mapped file. New-case implementation is a
  delegate-local task by default.
- The suite must stay deselected from plain `pytest` runs (`pytest.ini`
  `addopts = -m "not ui"`) so delegate-local's `run_tests` tool never launches
  browsers.
