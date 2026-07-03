# Harness hardening plan (for Sonnet to implement)

Context: Phase 1 delegation stalled — the junior rewrote byte-identical output
for 17 turns. The `run_tests` tool added afterwards is the right idea but
incomplete: it inherits the wrong Python, nothing breaks repeat-write loops,
and context saturation (the likely root cause of the stall) is unhandled.
Fable reviewed Phase 1 + the harness on 2026-07-03; this is the agreed
follow-up. Items A1–A6/B are harness/skill work — implement directly.
Items C9–C10 are project work — delegate them to the local model as the
first exercise of the hardened harness.

## A. `~/.claude/skills/delegate-local/delegate_local.py`

1. **`run_tests` must use the project venv.** It currently runs
   `sys.executable -m pytest`; the harness is usually launched with system
   `python3`, which has no pytest/numpy (verified) — the junior would get
   `No module named pytest` and flail. In `tool_run_tests`:
   prefer `<out_root>/.venv/bin/python` when it exists, else `sys.executable`.

2. **No-op write guard** (breaks the observed 17-turn loop directly). In
   `dispatch`'s `write_file` branch: if the destination exists and current
   content equals the new content, do NOT count it as progress; return
   `"no-op: file already contains exactly this content — run run_tests or, if done, reply without tool calls"`
   and print a `[wrote-noop]` line instead of `[wrote]`.

3. **Context budget** (root cause of the stall: a ~30B model saturates and
   degenerates into repetition). After appending each tool result in
   `agent_loop`, elide old tool output: keep the most recent `TOOL_KEEP = 12`
   tool-role messages intact; replace the `content` of older tool messages
   with `"(result elided to save context — call the tool again if needed)"`.

4. **LM Studio error handling.** Wrap the `_post` call in `agent_loop` with
   `try/except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError, KeyError)`;
   on HTTPError read and include the response body (LM Studio reports
   context-overflow as a 400 with a message). Print `[fail] LM Studio error: …`
   and return 1 instead of a raw traceback.

5. **`--readonly PATH` repeatable flag** (protect acceptance tests from the
   junior). Store the list; in `write_file`, if the target's repo-relative
   path matches an entry, return
   `"error: <path> is read-only acceptance material; fix the code, not the test"`.

6. **Extend `selftest`** to cover: the no-op write guard, and the readonly
   refusal. (Skip venv resolution in selftest — not worth faking a venv.)

## B. `~/.claude/skills/delegate-local/SKILL.md`

7. Launch command: use the project's venv Python when one exists —
   `.venv/bin/python ~/.claude/skills/delegate-local/delegate_local.py …` —
   so `run_tests` inherits the project's dependencies.

8. Acceptance-test ownership rule: for logic-bearing modules, the
   **architect writes the acceptance test file first** and passes it via
   `--readonly tests/test_x.py`; the junior iterates until it passes but may
   not edit it. Junior-authored tests are acceptable only for pure
   scaffolding tasks — and the reviewer must review the junior's test files
   against the spec either way (never accept "its own tests pass" as
   acceptance). Add one line noting `run_tests` executes repo code
   unsandboxed — acceptable for this local single-user setup, but the
   reviewer gate before commit is the containment.

## C. Project follow-ups (delegate to local model, webapp-port branch)

9. **Persist portfolio settings in the xlsx** (real defect, verified:
   saving market=US/mode=monthly/seed=7 and reloading silently reverts to
   IL/annual/42, violating PRD §4.4's round-trip guarantee — the PRD's own
   column list simply has nowhere to put these; this closes that spec gap).
   - Writer (`engine/params.py::scenario_to_xlsx`): add row-0 scalar columns
     after `end_age`: `market, mode, fat_tails_enabled, fat_tails_df,
     n_paths, random_seed`. The old CLI loader selects columns by name and
     ignores extras — verified, so CLI compatibility is unaffected.
   - Reader (`scenario_from_xlsx`): use these columns when present; fall
     back to today's defaults (IL/annual/true/5/10000/42) for legacy files.
   - Tests: change `test_roundtrip`'s fixture to non-default values
     (market US, mode monthly, random_seed 7, n_paths 500) so the guarantee
     is actually exercised; keep `test_legacy_file_loads` asserting the
     defaults.

10. **Add the missing §4.4 interop assertion** to
    `tests/test_xlsx_roundtrip.py`: a file written by `scenario_to_xlsx`
    loads through `cli.read_scenario_data` +
    `SimulationParams.from_legacy_scenario_data` without error, and
    `run_simulation` on both param objects (same seed) yields the same
    `ruin_probability`. (Fable verified this manually; make it a test.)

Acceptance for C: `.venv/bin/python -m pytest tests/ -q` green (now 5+
tests), and the architect passes `--readonly tests/test_xlsx_roundtrip.py`
after writing the updated test first (per rule B8).

Then: commit Phase 1 + these fixes on `webapp-port` (user reviews first),
update the `project-status` / `delegation-workflow` memories, and proceed to
Phase 2.
