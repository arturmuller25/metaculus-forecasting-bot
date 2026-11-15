"""
Turns problems that a run hides into a GitHub issue assigned to the repository
owner, so they arrive by email. The loop in forecast.yml swallows failed
iterations on purpose and the run ends green: on 2026-11-03 the Anthropic
credit ran out, every Market Pulse and Animal Futures question failed for 11
days, and nothing was notified.

Reads the bot's console output (the workflow tees it to logs/run_output.txt)
and the coverage check's output, and opens one issue per problem found:

- the Anthropic credit is exhausted;
- the OpenRouter key is out of credit, or below ALERT_KEY_BELOW dollars (default 40);
- a tournament mode failed on every question it tried in this run;
- the agentic forecaster failed as a member at least 5 times in this run;
- the Claude Code forecasters reached the Claude subscription's usage limit;
- questions closed without a forecast (coverage_check.py).

An issue that is already open for the same problem is left alone, so a problem
that lasts for days sends one email, and closing the issue re-arms the alert.

Usage:
    uv run python alerts.py logs/run_output.txt [logs/coverage.txt] [--dry-run]
In GitHub Actions it needs GH_TOKEN and the `issues: write` permission.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

PREFIX = "Bot alert: "
SECTION = re.compile(r"Mode\s+: (\w+)")
RESULT = re.compile(r"Questions forecast: (\d+)\. Failures: (\d+)\.")
FAILED = re.compile(r"Failed or unparsable model answers this run: (.+)")
# Warnings only, so the phrase inside a logged research text does not count.
# bot.run_claude_code_shadows and cc_forecaster.py log the first form; a
# Claude Code forecaster failing as a member logs the exception's name.
CC_LIMIT = re.compile(r"\| WARNING \| (?:Claude Code usage limit|.*\bClaudeCodeLimit\b)")


def read(path: str) -> list[str]:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read().splitlines()
    except OSError:
        return []


def find_alerts(run: list[str], coverage: list[str], key_remaining: float | None, key_floor: float) -> list[tuple[str, str]]:
    """(title, body) per problem seen in this run."""
    alerts: list[tuple[str, str]] = []

    credit = [line for line in run if "credit balance is too low" in line]
    if credit:
        alerts.append((
            "Anthropic credit exhausted",
            f"The Anthropic API refused {len(credit)} calls in this run with \"credit balance is too low\".\n\n"
            "While it lasts: the agentic forecaster does not answer (the reserve model takes its seat), Claude's web "
            "search and the Claude member fall back to the OpenRouter key, and the Claude-only tournaments "
            "(Market Pulse) are not forecast at all.\n\n"
            "Fix: add credit in the Anthropic Console, or wait for the monthly renewal. Close this issue afterwards.",
        ))

    out_of_credit = any("the OpenRouter key is out of credit" in line for line in run)
    if out_of_credit or (key_remaining is not None and key_remaining < key_floor):
        left = "unknown" if key_remaining is None else f"${key_remaining:.2f}"
        alerts.append((
            "OpenRouter key out of credit" if out_of_credit else f"OpenRouter key below ${key_floor:.0f}",
            f"Credit left on the tournament key: {left}.\n\n"
            "Fix: ask Metaculus for a top-up, or set OPENROUTER_FALLBACK_KEY. Close this issue afterwards.",
        ))

    # Every process the loop starts prints "Mode : <mode>" and ends with one result line.
    mode, by_mode = None, {}
    for line in run:
        m = SECTION.search(line)
        if m:
            mode = m.group(1)
            continue
        m = RESULT.search(line)
        if m and mode:
            by_mode.setdefault(mode, []).append((int(m.group(1)), int(m.group(2))))
            mode = None
    for name, results in by_mode.items():
        tried = [(ok, bad) for ok, bad in results if ok + bad > 0]
        # Without credit the Claude-only modes fail as a consequence: one alert is enough.
        if len(tried) >= 3 and all(ok == 0 for ok, _ in tried) and not credit:
            alerts.append((
                f"{name} failed on every question in a run",
                f"In this run the mode `{name}` was tried {len(tried)} times and forecast no question "
                f"({sum(bad for _, bad in tried)} failures in total). The run still ended as a success.\n\n"
                "Read the run's log for the first error. Close this issue once the mode forecasts again.",
            ))

    agent = 0
    for line in run:
        m = FAILED.search(line)
        if m:
            for name, count in re.findall(r"(agent/\S+ \(member\)): (\d+)", m.group(1)):
                agent += int(count)
    if agent >= 5 and not credit:
        alerts.append((
            "The agent member is failing",
            f"The agentic forecaster failed as a member {agent} times in this run for a reason other than credit "
            "(rate limit, time limit, API error). The reserve model took its seat on those questions.\n\n"
            "Read the run's log for the agent's error lines. Close this issue once it answers again.",
        ))

    limit = [line for line in run if CC_LIMIT.search(line)]
    if limit:
        alerts.append((
            "Claude Code usage limit reached",
            f"The Claude Code CLI reported that the Claude subscription's usage limit was reached ({len(limit)} log "
            "line(s) in this run).\n\n"
            "While it lasts: the Claude Code shadow forecasters stop, and nothing is recorded for them until the limit "
            "resets. The published forecasts do not change (if one of them is the agent member, the reserve model "
            "takes its seat). The run's log has the CLI's message, which names the limit (session, week or model) "
            "and when it resets. If that line says requests are billing to usage credits, calls beyond the plan "
            "were charged: check Settings > Usage on claude.ai.\n\n"
            "Nothing else to fix unless it repeats: then lower CC_MAX_PARALLEL or drop a Claude Code entry from "
            "SHADOW_MODELS. Close this issue afterwards.",
        ))

    missed = [line.split("::")[-1].strip() for line in coverage if "without a forecast" in line]
    if missed:
        listing = "\n".join(f"- {line}" for line in missed[:20])
        alerts.append((
            "Questions closed without a forecast",
            f"The coverage check found {len(missed)} question(s) that closed in the last hours without a forecast:\n\n"
            f"{listing}\n\nClose this issue after checking the cause.",
        ))
    return alerts


def key_left(key: str | None) -> float | None:
    """Dollars left on an OpenRouter key, or None when it has no limit or cannot be read."""
    if not key:
        return None
    import urllib.request

    request = urllib.request.Request("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            remaining = json.load(response).get("data", {}).get("limit_remaining")
    except Exception as exc:  # noqa: BLE001 - an alert must not fail on a network error
        print(f"Could not read the OpenRouter balance: {type(exc).__name__}")
        return None
    return None if remaining is None else float(remaining)


def gh(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["gh", *args], capture_output=True, text=True, encoding="utf-8", errors="replace")


def open_titles() -> set[str] | None:
    """Titles of the alert issues already open, or None when they cannot be listed."""
    result = gh("issue", "list", "--state", "open", "--limit", "100", "--json", "title")
    if result.returncode != 0:
        print(f"Could not list issues: {result.stderr.strip()[:300]}")
        return None
    return {issue["title"] for issue in json.loads(result.stdout or "[]")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_output", help="file with the bot's console output")
    parser.add_argument("coverage", nargs="?", default="", help="file with coverage_check.py's output")
    parser.add_argument("--dry-run", action="store_true", help="print the issues instead of opening them")
    args = parser.parse_args()

    key_remaining = key_left(os.getenv("OPENROUTER_API_KEY"))
    floor = float((os.getenv("ALERT_KEY_BELOW") or "").strip() or 40)

    alerts = find_alerts(read(args.run_output), read(args.coverage) if args.coverage else [], key_remaining, floor)
    if not alerts:
        print("No alerts.")
        return
    run_url = ""
    if os.getenv("GITHUB_RUN_ID"):
        run_url = f"{os.getenv('GITHUB_SERVER_URL', 'https://github.com')}/{os.getenv('GITHUB_REPOSITORY')}/actions/runs/{os.getenv('GITHUB_RUN_ID')}"
    existing = set() if args.dry_run else open_titles()
    owner = os.getenv("GITHUB_REPOSITORY_OWNER", "")
    for title, body in alerts:
        full = PREFIX + title
        if run_url:
            body += f"\n\nRun: {run_url}"
        if args.dry_run:
            print(f"--- would open: {full}\n{body}\n")
            continue
        if existing is None:
            sys.exit(1)
        if full in existing:
            print(f"Already open: {full}")
            continue
        command = ["issue", "create", "--title", full, "--body", body]
        if owner:
            command += ["--assignee", owner]
        result = gh(*command)
        print(f"Opened: {full} {result.stdout.strip()}" if result.returncode == 0 else f"Could not open {full!r}: {result.stderr.strip()[:300]}")


if __name__ == "__main__":
    main()
