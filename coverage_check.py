"""
Lists tournament questions that closed recently without a forecast from the
bot, so a gap in the schedule (a GitHub Actions run that never started, a
crash, an exhausted key) shows up instead of passing silently. Uses only the
Metaculus API. In GitHub Actions each miss becomes a warning annotation.

Usage: uv run python coverage_check.py [--hours 24]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone

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
        time.sleep(1)  # the live bot shares the rate limit
        return r.json()
    r.raise_for_status()
    return {}


def posts(**filters) -> list[dict]:
    """All pages of a post listing (the API keeps returning a `next` link, so stop on a short page)."""
    found, limit, offset = [], 100, 0
    while True:
        page = get("/posts/", limit=limit, offset=offset, **filters)["results"]
        found += page
        if len(page) < limit:
            return found
        offset += limit


def closed_at(post: dict) -> datetime | None:
    question = post.get("question") or {}
    stamp = question.get("actual_close_time") or question.get("scheduled_close_time")
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")) if stamp else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Lists recently closed questions the bot did not forecast.")
    parser.add_argument("--hours", type=float, default=24, help="look-back window in hours (default 24)")
    args = parser.parse_args()

    since = datetime.now(timezone.utc) - timedelta(hours=args.hours)
    me = get("/users/me/")["id"]
    in_actions = os.getenv("GITHUB_ACTIONS") == "true"
    checked = missed = 0
    for tournament in TOURNAMENTS:
        statuses = ["closed", "resolved"]
        mine = {p["id"] for p in posts(tournaments=tournament, statuses=statuses, forecaster_id=me)}
        for post in posts(tournaments=tournament, statuses=statuses):
            when = closed_at(post)
            if when is None or when < since:
                continue
            checked += 1
            if post["id"] in mine:
                continue
            missed += 1
            line = (
                f"closed {when:%Y-%m-%d %H:%M} UTC without a forecast: "
                f"https://www.metaculus.com/questions/{post['id']}/ {post.get('title', '')[:80]}"
            )
            print(f"::warning title=Missed question::{line}" if in_actions else f"MISSED {line}")
    print(f"Coverage, last {args.hours:g} h: {checked - missed} of {checked} closed questions forecast.")


if __name__ == "__main__":
    main()
