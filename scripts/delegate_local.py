#!/usr/bin/env python3
"""Delegate a one-shot codegen task to a local LM Studio model (zero API cost).

Usage:
    python scripts/delegate_local.py BRIEF.md [--model ID] [--base-url URL]
                                     [--out-root DIR] [--selftest]

The brief must contain everything the model needs (task, spec excerpts, full
contents of files to read, explicit list of output paths) — the local model
cannot read files or run commands. It is instructed to answer only with:

    === FILE: relative/path/from/out-root ===
    <full file content>
    === END FILE ===

Progress lines are printed every few seconds so a supervising agent can relay
them. Exit code 0 = files written; 1 = failure (raw output saved for autopsy).
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

DEFAULT_BASE_URL = "http://localhost:1234/v1"

SYSTEM_PROMPT = """You are a code generator. Produce COMPLETE contents for every file the task brief asks for.
Output format — nothing else, no prose before, between, or after the blocks:
=== FILE: relative/path/from/repo/root ===
<full file content>
=== END FILE ===
Rules: emit every requested file; never truncate; never write placeholders like '...' or 'rest unchanged'; match the style of any example code given in the brief."""


def _strip_md_fence(lines: list[str]) -> list[str]:
    body = [l for l in lines]
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()
    if len(body) >= 2 and body[0].strip().startswith("```") \
            and body[-1].strip() == "```":
        return body[1:-1]
    return lines


def parse_files(text: str) -> dict[str, str]:
    """Tolerant of local-model quirks: missing END FILE terminators
    (a new FILE header or EOF closes the block) and file bodies wrapped
    in markdown code fences."""
    files: dict[str, str] = {}
    cur, buf = None, []

    def close():
        nonlocal cur
        if cur is not None:
            files[cur] = "\n".join(_strip_md_fence(buf)) + "\n"
            cur = None

    for line in text.splitlines():
        s = line.strip()
        if s.startswith("=== FILE:") and s.endswith("==="):
            close()
            cur = s[len("=== FILE:"):-3].strip()
            buf = []
        elif s == "=== END FILE ===":
            close()
        elif cur is not None:
            buf.append(line)
    close()
    return files


def pick_model(base_url: str) -> str:
    with urllib.request.urlopen(f"{base_url}/models", timeout=5) as r:
        ids = [m["id"] for m in json.load(r)["data"]]
    coders = [i for i in ids if "coder" in i.lower()]
    return (coders or ids)[0]


def stream_completion(base_url: str, model: str, brief: str) -> str:
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps({
            "model": model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": brief}],
            "temperature": 0.2,
            "max_tokens": -1,
            "stream": True,
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    chunks: list[str] = []
    last_report = time.time()
    with urllib.request.urlopen(req) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            delta = json.loads(payload)["choices"][0].get("delta", {})
            piece = delta.get("content") or ""
            if piece:
                chunks.append(piece)
            if time.time() - last_report > 5:
                text = "".join(chunks)
                done = text.count("=== END FILE ===")
                print(f"[progress] {len(text):,} chars generated, "
                      f"{done} file(s) completed", flush=True)
                last_report = time.time()
    return "".join(chunks)


def safe_write(out_root: Path, rel: str, content: str) -> Path:
    p = Path(rel)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"unsafe output path from model: {rel!r}")
    dest = out_root / p
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content)
    return dest


def selftest() -> None:
    sample = (
        "=== FILE: a/b.py ===\n"
        "x = 1\n"
        "\n"
        "=== END FILE ===\n"
        "junk between blocks is ignored\n"
        "=== FILE: c.txt ===\n"
        "hello\n"
        "=== END FILE ===\n"
    )
    files = parse_files(sample)
    assert files == {"a/b.py": "x = 1\n\n", "c.txt": "hello\n"}, files
    # local-model quirks: md-fenced body, missing END FILE at EOF
    quirky = (
        "=== FILE: q.py ===\n"
        "```python\n"
        "y = 2\n"
        "```\n"
        "=== FILE: r.py ===\n"
        "z = 3\n"
    )
    files = parse_files(quirky)
    assert files == {"q.py": "y = 2\n", "r.py": "z = 3\n"}, files
    for bad in ("/etc/passwd", "../up.py"):
        try:
            safe_write(Path("."), bad, "")
            raise AssertionError(f"accepted unsafe path {bad}")
        except ValueError:
            pass
    print("selftest OK")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("brief", nargs="?", help="path to the task brief markdown")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--out-root", default=".", help="repo root to write files under")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return 0
    if not args.brief:
        ap.error("brief is required (or use --selftest)")

    brief = Path(args.brief).read_text()
    out_root = Path(args.out_root).resolve()
    model = args.model or pick_model(args.base_url)
    print(f"[start] model={model} brief={args.brief} "
          f"({len(brief):,} chars) out_root={out_root}", flush=True)

    t0 = time.time()
    text = stream_completion(args.base_url, model, brief)
    files = parse_files(text)

    if not files:
        autopsy = Path(args.brief).with_suffix(".raw.txt")
        autopsy.write_text(text)
        print(f"[fail] no file blocks parsed; raw output saved to {autopsy}",
              flush=True)
        return 1

    for rel, content in files.items():
        dest = safe_write(out_root, rel, content)
        print(f"[wrote] {dest.relative_to(out_root)} "
              f"({len(content.splitlines())} lines)", flush=True)
    print(f"[done] {len(files)} file(s) in {time.time() - t0:.0f}s "
          f"— review with: git diff", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
