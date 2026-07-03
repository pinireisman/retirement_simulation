# Delegation hardening, round 2 (for Sonnet to implement)

Context: both Phase 2 delegations produced correct-or-nearly-correct code,
then burned all 40 turns in a verification loop and exited rc=1 — a false
negative that nearly caused pointless re-delegation. Fable re-analyzed the
failure on 2026-07-03. Root causes, in order of importance:

1. **pytest exit code 5 misread.** The junior wrote a print-style script at
   repo root, pytest returned 5 ("no tests collected"), the harness passed
   the raw number through, and a 30B model has no idea what pytest exit
   codes mean. It concluded "failure" and could never reconcile that with
   the real suite passing.
2. **Advisory nudges don't break loops — proven, not suspected.** The
   "all tests pass — reply without tool calls now" nudge shipped in round 1
   was appended to every passing `run_tests` result during Phase 2, and the
   junior looped through it 30+ turns anyway. Do NOT add more nudge text as
   the fix (the report's item 4 proposes exactly that). The fix must be
   mechanical: refuse the repeated call, then take the tools away.
3. **rc semantics lie.** Turn exhaustion returns rc=1 even when the
   deliverable is complete on disk and acceptance is green. The
   orchestrator treats rc as ground truth.

Items A are `~/.claude/skills/delegate-local/delegate_local.py`; items B are
SKILL.md. Implement A/B directly (tooling). No project code changes.

## A. Harness — mechanical loop-breaking

1. **Translate pytest exit codes in `tool_run_tests`.** Prepend a
   plain-language line the model can act on:
   - 0 → `all tests passed` (keep the existing wrap-up line)
   - 1 → `some tests FAILED — read the failures below and fix the code`
   - 2 → `pytest was interrupted / internal error — not a normal failure`
   - 4 → `pytest usage error — the path or arguments are wrong`
   - 5 → `NO TESTS WERE COLLECTED at this path. This is NOT a code failure.
     Do not write your own check scripts; run the acceptance test path
     named in the brief instead.`

2. **Repeat-call breaker (mechanical).** Keep a dict of
   `hash(tool_name + canonical json of args) → (turn, result_hash)` for
   read-only tools (`run_tests`, `read_file`, `grep`, `list_files`). If the
   same call is made again and no `write_file` has succeeded since the
   previous invocation, do NOT execute it — return:
   `refused: you already made exactly this call on turn N and nothing has
   changed since. Repeating it will give the same result. If the work is
   done, reply without tool calls now.`
   Clear the seen-dict on every successful (non-no-op) `write_file` — a
   re-run after a real change is legitimate iteration and must never be
   refused. Count refusals.

3. **Forced finish (the decisive guard).** When either (a) 3 refusals from
   item 2 have accumulated, or (b) `turn >= max_turns - 2`, end the tool
   phase: send the next request **without the `tools` parameter** (omit it
   entirely — don't rely on `tool_choice: "none"` support in LM Studio),
   with an appended user message:
   `Tool access has ended. Write your final answer now: what you did, the
   files you wrote, and the last test status you observed.`
   Print `[forced-finish] reason=...` and treat the text reply as the
   normal `[done]` path. This makes a 40-turn silent burn structurally
   impossible.

4. **Honest exit codes.** rc=0: concluded normally. rc=2 (new): concluded
   via forced finish, or turns exhausted, **and at least one file was
   written** — meaning "work produced, conclusion unreliable, verify
   acceptance yourself." rc=1: only when nothing useful was produced.
   Always print a final `[files-written] <n>: <paths>` line so the
   orchestrator can see the state at a glance.

5. **Ad-hoc self-check script guard.** In `write_file`: refuse creation of
   NEW files at repo root matching `test_*.py`, `*_test.py`, `check_*.py`,
   `verify_*.py`, `debug_*.py` (existing files and anything under `tests/`
   are unaffected):
   `error: no ad-hoc self-check scripts — verify with run_tests on the
   acceptance path named in the brief; real tests belong under tests/`.
   Add one matching line to `AGENT_PROMPT`: "Never write your own
   verification scripts; run_tests is your only verification channel."
   (Second occurrence of this failure; it also left scratch files Sonnet
   had to clean up in Phase 1.)

6. **Selftest additions:** exit-5 translation text present; repeat-call
   refusal fires on an identical second call and does NOT fire after an
   intervening write; refusal counter feeds forced-finish bookkeeping;
   root-level `test_foo.py` creation refused while `tests/test_foo.py` is
   allowed. Run the full selftest with the project venv python AND system
   python3 (the round-1 regression was selftest passing under only one).

## B. SKILL.md — brief authoring + rc handling

7. **Batch-wide invariants section required in briefs** (from Sonnet's own
   report, items 1–2 — both correct): before splitting a callback/UI batch
   into a brief, build a table of every Output/id touched across the whole
   batch and flag any written by more than one callback (`allow_duplicate`
   etc.). For Output/id/property lists specifically, paste the PRD table
   rows verbatim — never paraphrase them. Bugs #1 and #3 were brief
   defects, not junior defects.

8. **rc interpretation in step 3:** "Trust the acceptance run, not the
   harness exit code. rc=2 = work was produced but the junior didn't
   conclude — read the `[files-written]` line, inspect the diff, and run
   acceptance yourself before re-delegating or escalating. Only rc=1 with
   no files written means the run produced nothing."

9. **Calibrate the fix loop in step 5:** when review findings are few,
   small, and localized (roughly ≤5 lines each — like Phase 2's three
   bugs), fix them directly and list them in the report; a fix-brief
   round-trip costs more than it saves. Reserve fix-briefs for structural
   rework or regeneration-scale problems. (Policy note: this amends the
   original "fixes always go back to the junior" rule based on measured
   cost across Phases 1–2.)

## C. Memory

10. Update the `delegation-workflow` memory: fold in the round-2 lesson —
    "advisory nudges proven insufficient; loop-breaking is mechanical
    (repeat-call refusal → forced tool-less finish); rc=2 means verify,
    don't re-delegate" — and Sonnet's brief-authoring rules (item 7).
    Keep it one memory, don't create a second one.

## Acceptance for this plan

- `delegate_local.py --selftest` green under BOTH `.venv/bin/python` and
  system `python3`.
- A live micro-delegation exercising the guards end-to-end: brief a tiny
  task ("add function X to a scratch module + make tests/test_x.py pass",
  in a throwaway dir) and confirm: junior concludes normally OR
  forced-finish fires; rc is 0 or 2, never a silent 40-turn rc=1 burn;
  `[files-written]` line present.
- No changes outside `~/.claude/skills/delegate-local/` and memory files.
