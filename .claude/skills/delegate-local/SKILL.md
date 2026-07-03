---
name: delegate-local
description: Delegate boilerplate/codegen work to a free local LM Studio model. Claude prepares a self-contained brief, launches scripts/delegate_local.py in the background, relays progress, then reviews and fixes the output. Use when the user says "delegate to local", "use the local model", "run on LM Studio", or a PRD phase is assigned to a local OSS model (Qwen3-Coder / Gemma).
---

# Delegate to local LM Studio model

You (the frontier model) are the architect and reviewer; the local model is a
one-shot code generator. It CANNOT read files, run commands, or ask questions
— the brief must be fully self-contained. Budget your effort accordingly: a
sloppy brief wastes a free run but a *vague* brief produces plausible wrong
code that costs more to review than writing it yourself.

## When to use / not use

- GOOD: repetitive CRUD/table layouts, test scaffolding from a written spec,
  file-format writers/readers with an exact schema, README sections, docstrings.
- BAD: anything needing iteration against test failures, cross-file
  refactors, subtle numpy/money math, Dash callback graphs. Do those yourself.

## Workflow

1. **Check the server**: `curl -s --max-time 3 http://localhost:1234/v1/models`.
   If it fails, tell the user to start LM Studio and load a coder model
   (qwen/qwen3-coder-30b preferred), then stop.

2. **Write the brief** to `agent-runs/local-briefs/<slug>.md`. It MUST contain:
   - The goal in 2–3 sentences.
   - The relevant spec text pasted verbatim (e.g. the PRD section) — never a
     reference like "see docs/PRD.md §4.4"; the model can't open it.
   - Full contents of every existing file the model needs to match or import
     from, in fenced blocks with their paths.
   - An explicit list of output file paths it must emit — nothing else.
   - Constraints: allowed imports/deps, style notes, "complete files only".
   - The acceptance command the output must satisfy (for your later use).

3. **Launch in background** (progress lines will surface in this chat):
   `python3 scripts/delegate_local.py agent-runs/local-briefs/<slug>.md`
   Use run_in_background=true. Relay `[progress]` / `[wrote]` lines to the
   user as they appear.

4. **Review — this is the paid-model value-add.** When the script exits:
   - `git status` + `git diff` the written files; read them fully.
   - Run the acceptance command from the brief (pytest, import check, app boot).
   - Fix defects directly yourself; do not re-delegate a fix round unless the
     output is salvageable-by-regeneration (wrong format, truncation).
   - Report to the user: what the local model produced, what you fixed, what
     the acceptance run shows. Never commit without user review.

5. On `[fail]` (exit 1): inspect the `.raw.txt` autopsy next to the brief.
   Usual causes: brief too long for the model's context (trim inlined files),
   or the model wrapped output in prose (add "output the blocks ONLY" to the
   brief and retry once). After one failed retry, do the task yourself.

## Notes

- The script auto-picks the first "coder" model; override with `--model`.
- Output paths are sandboxed under --out-root (default: CWD); the script
  rejects absolute paths and `..`.
- Playground upgrade path if one-shot proves too weak: point `aider` at
  `--openai-api-base http://localhost:1234/v1` for an agentic local loop.
