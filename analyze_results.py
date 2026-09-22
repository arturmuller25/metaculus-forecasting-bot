"""
Scores the bot's resolved forecasts, overall and per model. Uses only the
Metaculus API and local files: no LLM calls, no cost.

For every resolved question the bot forecast (seasonal tournament and
MiniBench), it compares the published forecast and each ensemble member's
forecast with the outcome. Member forecasts come from the bot's own comments
("### <model> forecast ..." lines). Shadow models, which are never published,
come from forecasts.jsonl records; --fetch-artifacts downloads them from the
GitHub Actions runs first.

Also writes logs/calibration.csv (published binary forecast, outcome) for
calibration.py.

Usage:
    uv run python analyze_results.py [--fetch-artifacts] [--include-open]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time

import dotenv
import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
dotenv.load_dotenv(".env")

API = "https://www.metaculus.com/api"
TOURNAMENTS = ["minibench", 33121]  # MiniBench and the Fall 2026 FutureEval tournament
SESSION = requests.Session()
SESSION.headers["Authorization"] = f"Token {os.environ['METACULUS_TOKEN']}"


def get(path: str, **params) -> dict:
    for attempt in range(4):
        r = SESSION.get(f"{API}{path}", params=params, timeout=60)
        if r.status_code == 429:
            time.sleep(15 * (attempt + 1))
            continue
        r.raise_for_status()
        time.sleep(1)  # stay well under the rate limit; the live bot shares it
        return r.json()
    r.raise_for_status()
    return {}


def bot_posts(user_id: int, statuses: str | list[str]) -> list[dict]:
    posts, limit = [], 100
    for tournament in TOURNAMENTS:
        offset = 0
        while True:
            page = get(
                "/posts/", tournaments=tournament, statuses=statuses,
                forecaster_id=user_id, limit=limit, offset=offset,
            )["results"]
            posts += page
            if len(page) < limit:
                break
            offset += limit
    return posts


# ---------------------------------------------------------------------------
# Reading forecasts
# ---------------------------------------------------------------------------


def members_from_comment(text: str, question: dict) -> dict[str, object]:
    """Parses the per-model lines the bot writes into its comment."""
    found: dict[str, object] = {}
    kind = question["type"]
    for line in text.splitlines():
        if not line.startswith("### "):
            continue
        if kind == "binary":
            m = re.match(r"### (\S+) (?:forecast|previu) (\d+(?:\.\d+)?)%", line)
            if m:
                found[m.group(1)] = float(m.group(2)) / 100
        elif kind == "multiple_choice":
            m = re.match(r"### (\S+) forecast: (.+)$", line)
            if m:
                probs = {}
                for option in question["options"]:
                    hit = re.search(re.escape(option) + r" (\d+(?:\.\d+)?)%", m.group(2))
                    if hit:
                        probs[option] = float(hit.group(1)) / 100
                if len(probs) == len(question["options"]):
                    total = sum(probs.values()) or 1.0
                    found[m.group(1)] = {k: v / total for k, v in probs.items()}
        else:
            m = re.match(r"### (\S+) forecast: (P\d+ .+)$", line)
            if m:
                pts = re.findall(r"P(\d+) (-?[\d,]*\.?\d+(?:e[+-]?\d+)?)", m.group(2))
                found[m.group(1)] = [(int(p) / 100, float(v.replace(",", ""))) for p, v in pts]
    return found


def shadow_records() -> dict[tuple[int, str], dict]:
    """Latest record per (post, model) from every forecasts.jsonl found locally."""
    latest: dict[tuple[int, str], dict] = {}
    files = glob.glob("logs/forecasts.jsonl") + glob.glob("logs/artifacts/**/forecasts.jsonl", recursive=True)
    for path in files:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (row.get("post_id"), f"{row.get('role')}:{row.get('model')}")
                if key not in latest or row.get("time", "") > latest[key].get("time", ""):
                    latest[key] = row
    return latest


def fetch_artifacts() -> None:
    runs = json.loads(subprocess.run(
        ["gh", "run", "list", "--workflow", "forecast.yml", "--limit", "200", "--json", "databaseId"],
        capture_output=True, text=True, check=True,
    ).stdout)
    for run in runs:
        target = os.path.join("logs", "artifacts", str(run["databaseId"]))
        if os.path.isdir(target):
            continue
        subprocess.run(
            ["gh", "run", "download", str(run["databaseId"]), "--pattern", "forecast-records-*", "--dir", target],
            capture_output=True, text=True,
        )
        os.makedirs(target, exist_ok=True)  # also marks runs without records as done


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def outcome_of(question: dict):
    res = question.get("resolution")
    kind = question["type"]
    if res in (None, "", "annulled", "ambiguous"):
        return None
    if kind == "binary":
        return {"yes": 1, "no": 0}.get(str(res).lower())
    if kind == "multiple_choice":
        return res if res in question["options"] else None
    try:
        return float(res)
    except (TypeError, ValueError):
        return str(res)  # below_lower_bound / above_upper_bound


def published(question: dict):
    values = ((question.get("my_forecasts") or {}).get("latest") or {}).get("forecast_values")
    if not values:
        return None
    if question["type"] == "binary":
        return values[1]
    if question["type"] == "multiple_choice":
        return dict(zip(question["options"], values))
    return None  # numeric CDFs are scored by Metaculus itself (score_data)


def brier(forecast, outcome, kind: str) -> float:
    if kind == "binary":
        return (forecast - outcome) ** 2
    return sum((p - (1.0 if option == outcome else 0.0)) ** 2 for option, p in forecast.items())


def covered(percentiles: list[tuple[float, float]], outcome) -> bool | None:
    """Whether the outcome falls inside the member's 10th-90th percentile range."""
    if not isinstance(outcome, float):
        return None
    values = dict(percentiles)
    if 0.1 not in values or 0.9 not in values:
        return None
    return values[0.1] <= outcome <= values[0.9]


def table(rows: dict[str, list[tuple[float, float]]], label: str) -> None:
    """rows: model -> [(model score, published score)] per question."""
    print(f"  {'model':44s} {'n':>4s} {label:>8s}  vs published (negative = better)")
    for model, pairs in rows.items():
        mine = [a for a, _ in pairs]
        line = f"  {model:44s} {len(pairs):4d} {statistics.mean(mine):8.4f}"
        if model != "published" and len(pairs) >= 2:
            diffs = [a - b for a, b in pairs]
            se = statistics.stdev(diffs) / math.sqrt(len(diffs))
            line += f"  {statistics.mean(diffs):+.4f} ± {se:.4f}"
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fetch-artifacts", action="store_true", help="download shadow records from GitHub Actions first")
    parser.add_argument("--include-open", action="store_true", help="also show parsed member forecasts on unresolved questions")
    args = parser.parse_args()

    if args.fetch_artifacts:
        fetch_artifacts()
    shadows = shadow_records()
    me = get("/users/me/")["id"]
    posts = bot_posts(me, "resolved")
    print(f"Resolved questions the bot forecast: {len(posts)}")

    rows: dict[str, dict[str, list]] = {"binary": {}, "multiple_choice": {}}
    coverage: dict[str, list[bool]] = {}
    scores: dict[str, list[float]] = {}
    calibration_rows = []
    skipped = 0

    for post in posts:
        question = get(f"/posts/{post['id']}/")["question"]
        kind = question["type"]
        outcome = outcome_of(question)
        if outcome is None:
            skipped += 1
            continue
        for name, value in ((question.get("my_forecasts") or {}).get("score_data") or {}).items():
            if isinstance(value, (int, float)):
                scores.setdefault(name, []).append(value)
        comments = get("/comments/", post=post["id"], author=me, is_private="true", limit=5)["results"]
        members = members_from_comment(comments[0]["text"], question) if comments else {}
        for (post_id, model), row in shadows.items():
            if post_id == post["id"] and model.startswith("shadow:"):
                members[model] = row["forecast"]

        final = published(question)
        if kind in rows and final is not None:
            if kind == "binary":
                calibration_rows.append((final, outcome))
            base = brier(final, outcome, kind)
            rows[kind].setdefault("published", []).append((base, base))
            for model, forecast in members.items():
                rows[kind].setdefault(model, []).append((brier(forecast, outcome, kind), base))
        elif kind in ("numeric", "discrete"):
            for model, forecast in members.items():
                hit = covered([tuple(p) for p in forecast], outcome)
                if hit is not None:
                    coverage.setdefault(model, []).append(hit)

    if skipped:
        print(f"Skipped (annulled or ambiguous): {skipped}")
    if scores:
        print("\nMetaculus scores, mean per question:")
        for name, values in sorted(scores.items()):
            print(f"  {name:28s} {statistics.mean(values):+8.2f}  (n={len(values)})")
    for kind, label in (("binary", "Brier"), ("multiple_choice", "Brier")):
        if rows[kind]:
            print(f"\n{kind.replace('_', ' ').capitalize()}:")
            table(rows[kind], label)
    if coverage:
        print("\nNumeric and discrete: outcome inside the 10th-90th percentile range (ideal 80%):")
        for model, hits in coverage.items():
            print(f"  {model:44s} {sum(hits) / len(hits):6.0%}  (n={len(hits)})")
    if rows["binary"] or rows["multiple_choice"]:
        print(
            "\nA difference smaller than about twice its ± is noise. Detecting 0.02 of Brier"
            " takes roughly 500 questions."
        )

    os.makedirs("logs", exist_ok=True)
    with open("logs/calibration.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["prediction", "outcome"])
        writer.writerows(calibration_rows)
    print(f"\nWrote logs/calibration.csv ({len(calibration_rows)} binary questions).", end=" ")
    print("Run calibration.py on it once there are at least 20; it needs about 100 to detect a real miscalibration.")

    if args.include_open:
        print("\nUnresolved questions, parsed from the bot's comments:")
        for post in bot_posts(me, ["closed", "open"]):
            question = get(f"/posts/{post['id']}/")["question"]
            comments = get("/comments/", post=post["id"], author=me, is_private="true", limit=5)["results"]
            members = members_from_comment(comments[0]["text"], question) if comments else {}
            final = published(question)
            shown = {m: (round(v, 3) if isinstance(v, float) else v) for m, v in members.items()}
            print(f"  {post['id']} {question['type'][:8]:8s} published={final if not isinstance(final, float) else round(final, 3)} members={shown}")


if __name__ == "__main__":
    main()
