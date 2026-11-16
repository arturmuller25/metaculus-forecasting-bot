"""
One real call per forecaster through cc_forecaster.py, to check on a machine
(a GitHub runner, through .github/workflows/cc_check.yml) that the Claude
Code CLI works there with the subscription's credential: installed,
credential found, isolation checks passing, answer block returned. Each
call is small; nothing is published and no secret or account data is printed.

Usage: uv run python cc_check.py [spec ...]
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

import cc_forecaster as cc

DEFAULT_SPECS = (
    "claudecode/claude-sonnet-5-5@low",
    "claudecode-agent/claude-sonnet-5-5@low",
    "claudecode-agent/claude-opus-5-5@xhigh",
    "claudecode/claude-fable-5-2@low",
)

PROMPT = """You are a professional forecaster.

Question: Will the International Space Station still be crewed on December 31, 2026?

Resolution criteria: resolves Yes if at least one person is aboard the International Space Station at 00:00 UTC on December 31, 2026, according to NASA.

Write a short rationale. The last thing you write is your final answer as: "Probability: ZZ%", 0-100
"""


async def one(spec: str) -> tuple[str, bool, str]:
    start = time.monotonic()
    try:
        answer = await cc.ClaudeCodeForecaster.from_spec(spec).invoke(PROMPT)
    except Exception as exc:  # noqa: BLE001 - every failure is reported, none stops the others
        return spec, False, f"{type(exc).__name__}: {str(exc)[:400]} ({time.monotonic() - start:.0f} s)"
    last = answer.strip().splitlines()[-1] if answer.strip() else ""
    return spec, True, f"{time.monotonic() - start:.0f} s, {len(answer)} characters, last line: {last[:80]}"


async def main() -> int:
    specs = sys.argv[1:] or list(DEFAULT_SPECS)
    ok, reason = cc.available()
    print(f"available: {ok}" + (f" ({reason})" if reason else ""))
    if not ok:
        return 1
    results = await asyncio.gather(*(one(spec) for spec in specs))
    for spec, good, detail in results:
        print(f"{'OK  ' if good else 'FAIL'} {spec}: {detail}")
    # The module's own log of these calls: seconds, turns, tokens, tools used.
    try:
        with open(cc.CALL_LOG, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
    except OSError:
        rows = []
    for row in rows[-len(specs) - 2:]:
        keep = {k: v for k, v in row.items() if isinstance(v, (int, float, bool)) or (isinstance(v, str) and len(v) <= 80)}
        print("  log:", json.dumps(keep, ensure_ascii=False)[:500])
    return 0 if all(good for _, good, _ in results) else 1


if __name__ == "__main__":
    code = asyncio.run(main())
    cc.shutdown()
    sys.stdout.flush()
    os._exit(code)
