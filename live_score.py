"""
Scores the live shadow experiments on resolved questions, against the pool
the bot published, question by question. No model calls and no API calls:
forecasts come from the production records (logs/artifacts/<run>/**/forecasts.jsonl,
downloaded by analyze_results.py --fetch-artifacts) and questions and outcomes
from the cache that `forecast_replay.py prepare` writes for a round.

  agent3   pool of the published members and the agent shadow       enters at +1.0/q, CI90 lower > 0
  sup      the supervisor's forecast in place of the pool           enters at +1.5/q, CI90 lower > 0, n >= 50
  agent    the agent alone in place of the pool                     read only
  claude+agent  the agent in the other member's seat, beside Claude  read only
  sol56_3  pool of the published members and the GPT-5.6 Sol shadow  read only
  solo:<m> one published member alone in place of the pool          read only

Since 2026-11-15 the agent is a member beside Claude Sonnet 5.5 and GPT-6.1
Sol is the reserve, recorded as a shadow on every question. For those
questions (rules fixed on 2026-11-14, before any of them resolved, each over
at least 100 questions with the 90% interval above zero):

  trio     the published members and the GPT-6.1 Sol reserve        Sol returns as a member at +1.0/q
  oldpair  Sonnet 5.5 and the reserve, the pair of before           the agent leaves if this is above zero

Rules fixed before the results (2026-10-21 and 22, panel_score.py). The
difference in log score is the difference in spot peer score: the other
forecasters' term is the same on both sides. Members are pooled as production
does (binary in log-odds, multiple choice by the mean, numeric by the
pointwise median of the CDFs, count questions with integer steps).

The pool rebuilt from the member records is checked against the forecast
Metaculus holds for the bot; every table is shown for all questions and for
the ones where the two match.

Usage:
    uv run python live_score.py --round 33132:logs/replay_r3 --round 33135:logs/replay_r4 \
        --round 33121:logs/replay_fall --offline logs/panel_scores/33125.json --offline logs/panel_scores/33129.json
Per-question rows go to logs/live_scores/<project>.json.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import logging
import math
import os
import sys
from datetime import datetime

import numpy as np

AGENT = "agent/claude-sonnet-5-5@xhigh"
SUPERVISOR = "supervisor/claude-sonnet-5-5"
SOL56 = "openrouter/openai/gpt-5.6-sol"
SOL = "openrouter/openai/gpt-6.1-sol"
OUT = os.path.join("logs", "live_scores")
# Member records are written seconds to minutes before the forecast is
# published; shadows that wait for the publication come later in the same run.
BEFORE_S, AFTER_S = 45 * 60, 5 * 60


def boot(d: list[float]) -> str:
    a = np.array(d)
    if len(a) < 2:
        return f"{a.mean():+.2f} n={len(a)}"
    means = np.random.default_rng(0).choice(a, size=(20000, len(a)), replace=True).mean(axis=1)
    return (f"{a.mean():+.2f} [{np.percentile(means, 5):+.2f}, {np.percentile(means, 95):+.2f}]"
            f" better/worse {int((a > 1e-9).sum())}/{int((a < -1e-9).sum())} n={len(a)}")


def group_of(kind: str) -> str:
    return kind if kind in ("binary", "multiple_choice") else "numeric+discrete"


def load_records() -> dict[int, dict[str, list[dict]]]:
    """post -> run folder -> records, production runs only."""
    by_post: dict[int, dict[str, list[dict]]] = {}
    root = os.path.join("logs", "artifacts")
    for path in glob.glob(os.path.join(root, "**", "forecasts.jsonl"), recursive=True):
        run = os.path.relpath(path, root).split(os.sep)[0]
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row["t"] = datetime.fromisoformat(row["time"]).timestamp()
                by_post.setdefault(row.get("post_id"), {}).setdefault(run, []).append(row)
    return by_post


def pick(runs: dict[str, list[dict]], published_at: float) -> tuple[dict[str, object], dict[str, object], str]:
    """
    The member records behind the published forecast (the latest per model in
    the window before it) and the shadow records of the same run. Returns
    (members, shadows, note).
    """
    best, best_t = None, None
    for run, rows in runs.items():
        times = [r["t"] for r in rows if r["role"] == "member" and published_at - BEFORE_S <= r["t"] <= published_at + AFTER_S]
        if times and (best_t is None or max(times) > best_t):
            best, best_t = run, max(times)
    if best is None:
        return {}, {}, "no member record near the published forecast"
    members: dict[str, dict] = {}
    for r in runs[best]:
        if r["role"] == "member" and published_at - BEFORE_S <= r["t"] <= published_at + AFTER_S:
            if r["model"] not in members or r["t"] > members[r["model"]]["t"]:
                members[r["model"]] = r
    shadows: dict[str, dict] = {}
    for r in runs[best]:
        if r["role"] == "shadow" and r["t"] >= published_at - BEFORE_S:
            if r["model"] not in shadows or r["t"] > shadows[r["model"]]["t"]:
                shadows[r["model"]] = r
    return ({m: r["forecast"] for m, r in members.items()}, {m: r["forecast"] for m, r in shadows.items()}, best)


def load_round(folder: str) -> list[dict]:
    """Resolved single-question posts of a prepared round that the bot forecast."""
    import analyze_results as ar

    rows, groups = [], 0
    for path in sorted(glob.glob(os.path.join(folder, "cache", "post_*.json"))):
        with open(path, encoding="utf-8") as fh:
            post_json = json.load(fh)
        q_json = post_json.get("question")
        if not q_json:
            groups += 1
            continue
        latest = ((q_json.get("my_forecasts") or {}).get("latest") or {})
        outcome = ar.outcome_of(q_json)
        if not latest.get("start_time") or outcome is None:
            continue
        rows.append({
            "post_id": post_json["id"], "type": q_json["type"], "title": post_json.get("title", ""),
            "outcome": outcome, "published_at": float(latest["start_time"]),
            "published": latest.get("forecast_values"), "post_json": post_json,
        })
    if groups:
        print(f"  {folder}: {groups} group posts skipped (one question per post only)")
    return rows


async def score_round(project: str, folder: str, records: dict[int, dict[str, list[dict]]]) -> list[dict]:
    import bot
    import forecast_replay as fr
    from numeric_cdf import is_count_question
    from numeric_replay import bucket_of, pmf_at

    async def pooled(forecasts: list, row: dict, question, integer: bool):
        """(log score, the pooled forecast as published values)."""
        kind, outcome = row["type"], row["outcome"]
        if kind == "binary":
            p = bot._logodds_mean([float(f) for f in forecasts])
            return 100 * math.log(p if outcome == 1 else 1 - p), [1 - p, p]
        if kind == "multiple_choice":
            options = row["post_json"]["question"]["options"]
            aligned = [fr.clamp_options(f, options) for f in forecasts]
            mean = fr.clamp_options({o: sum(a[o] for a in aligned) / len(aligned) for o in options}, options)
            return 100 * math.log(mean[outcome]), [mean[o] for o in options]
        cdf = await fr.published_cdf([[tuple(p) for p in f] for f in forecasts], question, integer=integer)
        return 50 * math.log(pmf_at(cdf, bucket_of(question, outcome, len(cdf)))), [float(x) for x in cdf]

    out = []
    for row in load_round(folder):
        kind = row["type"]
        if kind not in ("binary", "multiple_choice", "numeric", "discrete"):
            continue
        members, shadows, note = pick(records.get(row["post_id"], {}), row["published_at"])
        item = {"project": project, "post_id": row["post_id"], "type": kind, "group": group_of(kind),
                "title": row["title"], "outcome": row["outcome"], "run": note, "members": sorted(members),
                "shadows": sorted(shadows), "diffs": {}}
        if not members:
            item["skip"] = note
            out.append(item)
            continue
        question = fr.question_object(row)
        integer = kind == "numeric" and os.getenv("NUMERIC_STEPS", "1").strip() != "0" and is_count_question(question)
        try:
            base, values = await pooled(list(members.values()), row, question, integer)
        except Exception as exc:  # noqa: BLE001 - one unreadable record must not sink the round
            item["skip"] = f"base failed: {type(exc).__name__}: {exc}"
            out.append(item)
            continue
        item["base"] = base
        item["integer"] = integer
        pub = row["published"] or []
        if len(pub) == len(values):
            gap = max(abs(a - b) for a, b in zip(pub, values))
            item["gap"] = gap
            item["match"] = gap <= (0.0051 if kind == "binary" else 0.011)
        else:
            item["gap"], item["match"] = None, False

        async def diff(name: str, forecasts: list) -> None:
            try:
                s, _ = await pooled(forecasts, row, question, integer)
            except Exception as exc:  # noqa: BLE001
                item.setdefault("errors", {})[name] = f"{type(exc).__name__}: {exc}"
                return
            item["diffs"][name] = s - base

        fed_gapfill = any(m.startswith("gapfill+") for m in shadows)
        if AGENT in shadows:
            await diff("agent3", list(members.values()) + [shadows[AGENT]])
            await diff("agent", [shadows[AGENT]])
            claude = [f for m, f in members.items() if "claude" in m]
            if claude and len(members) > 1:
                # The agent in the other member's seat, beside the Claude member.
                await diff("claude+agent", claude + [shadows[AGENT]])
        if SUPERVISOR in shadows:
            await diff("supG" if fed_gapfill else "sup", [shadows[SUPERVISOR]])
        if SOL56 in shadows and SOL56 not in members:
            await diff("sol56_3", list(members.values()) + [shadows[SOL56]])
            await diff("sol56", [shadows[SOL56]])
        if AGENT in members and SOL in shadows and SOL not in members:
            # The agent as a member, GPT-6.1 Sol standing by as the reserve.
            others = [f for m, f in members.items() if m != AGENT]
            if others:
                await diff("oldpair", others + [shadows[SOL]])
            await diff("trio", list(members.values()) + [shadows[SOL]])
        if len(members) > 1:
            for m, f in members.items():
                await diff("solo:agent" if m.startswith("agent/") else f"solo:{m.split('/')[-1]}", [f])
        out.append(item)
    return out


def report(title: str, items: list[dict], matched_only: bool) -> None:
    used = [i for i in items if "base" in i and (i.get("match") or not matched_only)]
    print(f"\n===== {title}{' (rebuilt pool matches the published forecast)' if matched_only else ''}: {len(used)} questions")
    keys: list[str] = []
    for i in used:
        keys += [k for k in i["diffs"] if k not in keys]
    for k in sorted(keys):
        rows = [(i["group"], i["diffs"][k]) for i in used if k in i["diffs"]]
        for g in ("all", "all but multiple choice", "binary", "numeric+discrete", "multiple_choice"):
            d = [v for kind, v in rows if g == "all" or kind == g or (g == "all but multiple choice" and kind != "multiple_choice")]
            if d:
                print(f"{k:28s} {g:24s} {boot(d)}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--round", action="append", required=True, help="project:prepared replay folder, e.g. 33132:logs/replay_r3")
    parser.add_argument("--offline", action="append", default=[], help="panel_score.py output to pool with (agent3 and sup)")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.disable(logging.WARNING)
    os.makedirs(OUT, exist_ok=True)
    records = load_records()
    print(f"Posts with production records: {len(records)}")

    everything: list[dict] = []
    for spec in args.round:
        project, folder = spec.split(":", 1)
        items = asyncio.run(score_round(project, folder, records))
        with open(os.path.join(OUT, f"{project}.json"), "w", encoding="utf-8") as fh:
            json.dump(items, fh, ensure_ascii=False, indent=1)
        scored = [i for i in items if "base" in i]
        skipped = [i for i in items if "skip" in i]
        gaps = [i for i in scored if not i.get("match")]
        print(f"\nRound {project}: {len(items)} resolved questions the bot forecast, {len(scored)} with member records,"
              f" {len(skipped)} without, {len(gaps)} where the rebuilt pool differs from the published forecast")
        for i in skipped:
            print(f"  skipped {i['post_id']} ({i['type']}): {i['skip']}")
        for i in gaps:
            gap = "n/a" if i.get("gap") is None else f"{i['gap']:.3f}"
            print(f"  differs {i['post_id']} ({i['type']}), max gap {gap}, members {i['members']}")
        for i in scored:
            for name, err in (i.get("errors") or {}).items():
                print(f"  error {i['post_id']} {name}: {err}")
        report(f"round {project}", items, False)
        everything += items
    if len(args.round) > 1:
        report("all live rounds", everything, False)
        report("all live rounds", everything, True)

    if args.offline:
        live_posts = {i["post_id"] for i in everything if "base" in i}
        for key in ("agent3", "sup"):
            rows = [(i["group"], i["diffs"][key]) for i in everything if key in i.get("diffs", {})]
            n_live, dropped = len(rows), 0
            for path in args.offline:
                with open(path, encoding="utf-8") as fh:
                    for post, (kind, value) in json.load(fh).get(key, {}).items():
                        if int(post) in live_posts:
                            dropped += 1
                            continue
                        rows.append((kind, value))
            print(f"\n===== {key}: live ({n_live}) + offline time-travel replay ({len(rows) - n_live}; {dropped} already live, not counted twice)")
            for g in ("all", "all but multiple choice", "binary", "numeric+discrete", "multiple_choice"):
                d = [v for kind, v in rows if g == "all" or kind == g or (g == "all but multiple choice" and kind != "multiple_choice")]
                if d:
                    print(f"{key:28s} {g:24s} {boot(d)}")


if __name__ == "__main__":
    main()
