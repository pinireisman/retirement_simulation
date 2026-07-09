#!/usr/bin/env python3
"""Feed a UI-test run's artifacts to a local LLM (LM Studio) and write REPORT.md.

Usage:  .venv/bin/python scripts/analyze_ui_run.py artifacts/ui-run-<ts> \
            [--base-url http://localhost:1234/v1] [--model <id>]

The analyst is a one-shot call, not an agent: this script deterministically
pre-digests the artifacts (summary.json + failure evidence only) so a ~30B
model reasons over a small, relevant context instead of navigating files.
Screenshots are referenced by path/metadata only (no vision).

Exit codes: 0 = valid REPORT.md written; 1 = usage/IO error;
2 = model produced an invalid report twice (orchestrator: escalate).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DIGEST_CAP = 24_000          # chars; assumes >=32k ctx in LM Studio
PER_FAILURE_CAP = 2_000

REPORT_TEMPLATE = """# UI Test Report — {run_id}

## Run metadata
(run id, git sha, finished time, totals — copy from the digest)

## Results
| UX id | Test | Outcome | Duration (s) |
(one row per test in the digest, ALL of them)

## Failures
(one subsection per failed/errored test:)
### <UX id> — <one-line defect statement>
- **Evidence**: quote the failure message / console lines, citing the artifact
  path they came from (e.g. `console/tests_ui_....log`, `summary.json`).
- **Suspected root cause**: your best reasoning about WHY, referencing app
  files (webapp/layout.py, webapp/callbacks.py, webapp/components.py,
  webapp/assets/style.css, engine/figures.py) when you can.
- **Suggested fix**: concrete change + file path. If unsure, say what to
  investigate and where.
- **Severity**: Blocker / Major / Minor / Polish (per docs/UX_TEST_PLAN.md).

## Suspicious but passing
(tests that passed but whose duration, console log, or evidence hints at a
problem; empty section is fine — write "None observed.")

## Artifacts index
(bullet list: every artifact path mentioned above)
"""

SYSTEM = """You are a meticulous QA analyst for a Dash (Plotly) web app.
You receive a digest of a Playwright UI-test run. Write REPORT.md following
EXACTLY the template below. Rules:
- Every failed or errored test MUST get a Failures subsection.
- Every claim MUST cite an artifact path from the digest — never invent paths,
  line numbers, or file contents you were not shown.
- Suggested fixes must name real files (listed in the template).
- Be concise: evidence quotes over prose. No preamble before the # title.

TEMPLATE:
""" + REPORT_TEMPLATE


PREFERRED_MODELS = ["froggeric/qwen3.6-27b-mlx-8bit", "qwen/qwen3-coder-30b"]


def pick_model(base_url: str) -> str:
    with urllib.request.urlopen(f"{base_url}/models", timeout=5) as r:
        ids = [m["id"] for m in json.load(r)["data"]]
    by_lower = {i.lower(): i for i in ids}
    for pref in PREFERRED_MODELS:
        if pref in by_lower:
            return by_lower[pref]
    coders = [i for i in ids if "coder" in i.lower()]
    return (coders or ids)[0]


def chat(base_url: str, model: str, messages: list, timeout: int = 900) -> str:
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps({"model": model, "messages": messages,
                         "temperature": 0.2}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def _tail(text: str, cap: int) -> str:
    return text if len(text) <= cap else "…(truncated)…\n" + text[-cap:]


def build_digest(run_dir: Path) -> str:
    summary = json.loads((run_dir / "summary.json").read_text())
    parts = ["## summary.json\n" + json.dumps(summary, indent=1)]

    server_log = run_dir / "server.log"
    server_text = server_log.read_text() if server_log.exists() else ""

    for t in summary["tests"]:
        if t["outcome"] == "passed":
            continue
        block = [f"## FAILURE {t['ux_id'] or '(no ux id)'} — {t['nodeid']}"]
        if t.get("failure_message"):
            block.append("### failure message (from summary.json)\n"
                         + _tail(t["failure_message"], PER_FAILURE_CAP))
        for rel in t.get("artifacts", []):
            p = run_dir / rel
            if p.suffix == ".log" and p.exists():
                block.append(f"### {rel}\n" + _tail(p.read_text(), PER_FAILURE_CAP))
        # screenshots/traces: names only (no vision)
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", t["nodeid"].split("::")[-1]).strip("-")
        shots = [str(p.relative_to(run_dir))
                 for p in run_dir.glob(f"pw/*{slug}*/*") if p.suffix in (".png", ".zip")]
        if shots:
            block.append("### visual artifacts (paths only, not viewed)\n"
                         + "\n".join(shots))
        parts.append("\n".join(block))

    if any(t["outcome"] != "passed" for t in summary["tests"]) and server_text:
        parts.append("## server.log (tail)\n" + _tail(server_text, 3_000))

    digest = "\n\n".join(parts)
    if len(digest) > DIGEST_CAP:
        digest = digest[:DIGEST_CAP] + "\n…(DIGEST TRUNCATED — note partial evidence in the report)"
    return digest


def validate_report(report: str, run_dir: Path) -> list[str]:
    problems = []
    for heading in ("## Run metadata", "## Results", "## Failures",
                    "## Suspicious but passing", "## Artifacts index"):
        if heading not in report:
            problems.append(f"missing heading {heading!r}")
    summary = json.loads((run_dir / "summary.json").read_text())
    for t in summary["tests"]:
        ux = t["ux_id"]
        if ux and ux not in report:
            problems.append(f"test id {ux} absent from report")
    # every cited artifact-ish path must exist (under run dir or repo)
    for path in re.findall(r"`([\w./-]+\.(?:log|json|png|xml|zip|py|css|md))`", report):
        if not (run_dir / path).exists() and not (REPO / path).exists():
            problems.append(f"cited path does not exist: {path}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--base-url", default="http://localhost:1234/v1")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    if not (run_dir / "summary.json").exists():
        print(f"error: no summary.json in {run_dir}", file=sys.stderr)
        return 1

    model = args.model or pick_model(args.base_url)
    digest = build_digest(run_dir)
    (run_dir / "digest.txt").write_text(digest)  # audit trail
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Run directory: {run_dir.name}\n\n{digest}"}]

    for attempt in (1, 2):
        report = chat(args.base_url, model, messages)
        # strip <think> blocks some local models emit
        report = re.sub(r"<think>.*?</think>", "", report, flags=re.S).strip()
        problems = validate_report(report, run_dir)
        if not problems:
            (run_dir / "REPORT.md").write_text(report + f"\n\n---\n*analyst: {model}*\n")
            print(f"REPORT.md written ({len(report)} chars, model {model})")
            return 0
        print(f"attempt {attempt}: invalid report: {problems}", file=sys.stderr)
        messages += [{"role": "assistant", "content": report},
                     {"role": "user", "content":
                      "Your report failed validation:\n- " + "\n- ".join(problems)
                      + "\nRewrite the FULL report fixing these issues."}]

    (run_dir / "REPORT.invalid.md").write_text(report)
    print("model could not produce a valid report — escalate (exit 2)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
