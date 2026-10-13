"""
Backtest of market_query.py on the single-series market questions of past
MiniBench rounds (prices, index levels, yields, exchange rates, thresholds).

The Metaculus API hides most of these questions' resolutions from a bot
account, so the outcome is computed from the same market data. That checks
the forecast, not the extraction: if the LLM picks the wrong symbol, the
outcome is wrong in the same way. The extraction was reviewed by hand on 118
questions on 2026-10-13 (one doubtful token symbol, MemeCore).

Scored on every question, with data up to the day before the question opened:
the baseline alone; Claude Sonnet 5 with the bot's own prompt and a data card
(latest closes); the same with the baseline block. Claude runs on the
Anthropic API only.

Usage:
    uv run python market_query_backtest.py prepare      # Metaculus, extraction (OpenRouter, ~$1), market data
    uv run python market_query_backtest.py run [--limit N]
    uv run python market_query_backtest.py analyze
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import time
from datetime import date, datetime, timedelta, timezone

import numpy as np

import analyze_results as ar
import market_backtest as mbt  # patches the LLM calls (no OpenRouter, spending guard) and the date
import market_query as mq
from forecasting_tools import BinaryQuestion, DataOrganizer
from numeric_replay import bootstrap

OUT = os.path.join("logs", "market_query_backtest")
QUESTIONS = os.path.join(OUT, "questions.jsonl")
ANSWERS = os.path.join(OUT, "answers.jsonl")
PROJECTS = (33018, 33023, 33026, 33029, 33063, 33064, 33067, 33069, 33074, 33075, 33122, 33125)
EXTRACT_MODEL = "openrouter/openai/gpt-5.6-sol"
ARMS = ("card", "card+baseline")


def cached_post(pid: int) -> dict:
    path = os.path.join(OUT, "posts", f"{pid}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    data = ar.get(f"/posts/{pid}/")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)
    return data


def outcome(query: mq.Query):
    """What happened, in the question's unit (a value, or 1/0 for thresholds)."""
    bars = mq._bars(query, date.today())
    upto = [b for b in bars if b.day <= query.end]
    if not upto or upto[-1].day < query.end - timedelta(days=4):
        return None
    inside = [b for b in bars if (query.start or query.end) <= b.day <= query.end] or [upto[-1]]
    s = query.scale
    if query.kind == "value_on_date":
        return s * upto[-1].close
    if query.kind == "max_in_window":
        return s * max(b.high if query.intraday else b.close for b in inside)
    if query.kind == "min_in_window":
        return s * min(b.close for b in inside)
    if query.kind == "above_on_date":
        return float(s * upto[-1].close >= query.threshold)
    if query.kind == "below_on_date":
        return float(s * upto[-1].close <= query.threshold)
    if query.kind == "touch_above":
        return float(any(s * (b.high if query.intraday else b.close) >= query.threshold for b in inside))
    return float(any(s * b.close <= query.threshold for b in inside))


def card(query: mq.Query, as_of: date) -> str:
    bars = [b for b in mq._bars(query, as_of)][-10:]
    closes = ", ".join(f"{b.day.isoformat()[5:]} {query.scale * b.close:,.5g}" for b in bars)
    return (f"Market data as of the close of {as_of.isoformat()} (computed by code from {query.source}):\n"
            f"- {query.name} ({query.symbol}), last 10 closes: {closes}")


def cmd_prepare(_args) -> None:
    import main

    os.makedirs(OUT, exist_ok=True)
    mbt.ALLOW_OPENROUTER[0] = True  # the extraction runs on OpenRouter, as in production
    llm = main._thinker(EXTRACT_MODEL, 0.0, 120, effort="low")
    rows, counts = [], {"posts": 0, "hint": 0, "query": 0, "kept": 0}

    async def go() -> None:
        sem = asyncio.Semaphore(6)

        async def one(pid: int) -> None:
            post = cached_post(pid)
            q_json = post.get("question") or {}
            counts["posts"] += 1
            try:
                question = DataOrganizer.get_question_from_post_json(post)
            except Exception:  # noqa: BLE001
                return
            text = " ".join([question.question_text or "", question.resolution_criteria or ""])
            if not mq.HINT.search(text):
                return
            counts["hint"] += 1
            opened = datetime.fromisoformat(q_json["open_time"].replace("Z", "+00:00"))
            mbt._TODAY.set(opened)
            async with sem:
                try:
                    answer = await llm.invoke(mq.prompt_for(question).replace(
                        f"Today is {datetime.now(timezone.utc).date().isoformat()}", f"Today is {opened.date().isoformat()}"))
                except Exception:  # noqa: BLE001
                    return
            query = mq.parse(answer)
            if query is None or (query.kind in mq.VALUE_KINDS) == isinstance(question, BinaryQuestion):
                return
            counts["query"] += 1
            as_of = opened.date() - timedelta(days=1)
            try:
                result = await asyncio.to_thread(mq.compute, query, as_of)
                result_outcome = await asyncio.to_thread(outcome, query) if result else None
            except Exception:  # noqa: BLE001
                return
            if result is None or result_outcome is None or not mq.plausible(result, question):
                return
            counts["kept"] += 1
            rows.append({
                "post_id": pid, "type": q_json["type"], "title": post["title"], "open": opened.isoformat(),
                "as_of": as_of.isoformat(), "query": {k: (v.isoformat() if isinstance(v, date) else v) for k, v in vars(query).items()},
                "outcome": result_outcome, "percentiles": {str(k): v for k, v in result.percentiles.items()},
                "probability": result.probability, "baseline_text": result.render(), "card": card(query, as_of),
            })

        # Titles filter first, so only likely market questions cost a detail
        # call (the live bot shares the token's rate limit).
        ids = []
        for project in PROJECTS:
            listing = ar.get("/posts/", tournaments=project, limit=100)
            ids += [p["id"] for p in listing.get("results", []) if mq.HINT.search(p.get("title", ""))]
        await asyncio.gather(*(one(pid) for pid in ids))

    asyncio.run(go())
    with open(QUESTIONS, "w", encoding="utf-8") as fh:
        for r in sorted(rows, key=lambda r: r["post_id"]):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(counts)
    kinds: dict[str, int] = {}
    for r in rows:
        kinds[r["query"]["kind"]] = kinds.get(r["query"]["kind"], 0) + 1
    print(kinds)


def question_of(row: dict):
    return DataOrganizer.get_question_from_post_json(cached_post(row["post_id"]))


async def run_one(bot, row: dict, arm: str, model: str) -> None:
    question = question_of(row)
    research = row["card"] + ("\n\n" + row["baseline_text"] if arm == "card+baseline" else "")
    call: dict = {"post_id": row["post_id"], "arm": arm, "model": model}
    mbt._CALL.set(call)
    mbt._TODAY.set(datetime.fromisoformat(row["open"]))
    started = time.monotonic()
    try:
        if isinstance(question, BinaryQuestion):
            pred = await bot._run_forecast_on_binary(question, research)
            call["probability"] = float(pred.prediction_value)
        else:
            pred = await bot._run_forecast_on_numeric(question, research)
            call["percentiles"] = [[p.percentile, p.value] for p in pred.prediction_value.declared_percentiles]
        call["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        call["status"] = "error"
        call["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    call["seconds"] = round(time.monotonic() - started, 1)
    mbt.append_jsonl(ANSWERS, call)
    print(f"  {row['post_id']} {arm:14s} {call['status']} {call['seconds']}s spent ~${mbt._SPENT[0]:.2f}", flush=True)


def cmd_run(args) -> None:
    logging.basicConfig(level=logging.WARNING)
    rows = mbt.read_jsonl(QUESTIONS)[: args.limit or None]
    done = {(a["post_id"], a["arm"], a["model"]) for a in mbt.read_jsonl(ANSWERS) if a["status"] == "ok"}
    bot = mbt.build_bot(args.model)
    os.environ["MARKET_BASELINE_WEIGHT"] = "0"  # score the members' own forecast
    jobs = [(r, arm) for r in rows for arm in ARMS if (r["post_id"], arm, args.model) not in done]
    print(f"{len(jobs)} calls with {args.model}")

    async def go() -> None:
        sem = asyncio.Semaphore(mbt.IN_FLIGHT)

        async def one(r, arm):
            async with sem:
                if mbt._SPENT[0] < mbt.SPEND_CAP:
                    await run_one(bot, r, arm, args.model)

        await asyncio.gather(*(one(r, arm) for r, arm in jobs))

    asyncio.run(go())
    print(f"Estimated spend this run: ${mbt._SPENT[0]:.2f}")


def clipped(points: list[tuple[float, float]], question) -> list[tuple[float, float]]:
    """Percentiles held within a margin of the question's range, which the library requires."""
    low, high = question.lower_bound, question.upper_bound
    margin = 0.2 * (high - low)
    out: list[tuple[float, float]] = []
    for p, v in points:
        v = min(high + margin, max(low - margin, v))
        if out and v <= out[-1][1]:
            v = out[-1][1] + 1e-9 * (high - low)
        out.append((p, v))
    return out


async def safe_cdf(points, question, widen: float):
    try:
        return await mbt.cdf_from(points, question, widen)
    except ValueError:
        return await mbt.cdf_from(clipped(points, question), question, widen)


def binary_score(p: float, hit: float) -> float:
    p = min(0.99, max(0.01, p))
    return 100 * math.log(p if hit else 1 - p)


def cmd_analyze(_args) -> None:
    logging.getLogger("forecasting_tools").setLevel(logging.ERROR)
    rows = {r["post_id"]: r for r in mbt.read_jsonl(QUESTIONS)}
    answers = {(a["post_id"], a["arm"]): a for a in mbt.read_jsonl(ANSWERS) if a["status"] == "ok"}
    scores: dict[str, dict[int, float]] = {}

    async def go() -> None:
        for pid, row in rows.items():
            question = question_of(row)
            binary = isinstance(question, BinaryQuestion)
            base_p, base_cdf = row["probability"], None
            if binary:
                scores.setdefault("baseline alone", {})[pid] = binary_score(base_p, row["outcome"])
            else:
                base_cdf = await safe_cdf([(int(p) / 100, v) for p, v in row["percentiles"].items()], question, 1.0)
                scores.setdefault("baseline alone", {})[pid] = mbt.log_score(base_cdf, question, row["outcome"])
            for arm in ARMS:
                a = answers.get((pid, arm))
                if not a:
                    continue
                if binary:
                    scores.setdefault(f"claude {arm}", {})[pid] = binary_score(a["probability"], row["outcome"])
                    for w in (0.5, 0.8):
                        mix = (1 - w) * a["probability"] + w * base_p
                        scores.setdefault(f"claude {arm} | {int(100 * w)}% baseline", {})[pid] = binary_score(mix, row["outcome"])
                else:
                    cdf = await safe_cdf([tuple(x) for x in a["percentiles"]], question, 1.15)
                    scores.setdefault(f"claude {arm}", {})[pid] = mbt.log_score(cdf, question, row["outcome"])
                    for w in (0.5, 0.8):
                        mix = (1 - w) * cdf + w * base_cdf
                        scores.setdefault(f"claude {arm} | {int(100 * w)}% baseline", {})[pid] = mbt.log_score(mix, question, row["outcome"])

    asyncio.run(go())
    ref = scores["baseline alone"]
    kinds = {pid: ("binary" if r["type"] == "binary" else "numeric") for pid, r in rows.items()}
    for group in ("all", "binary", "numeric"):
        posts = set(rows) if group == "all" else {p for p, k in kinds.items() if k == group}
        print(f"\n== {group} ({len(posts)} questions)")
        for name in sorted(scores):
            common = sorted(set(scores[name]) & posts)
            if not common:
                continue
            line = f"  {name:44s} n={len(common):3d} mean {np.mean([scores[name][p] for p in common]):8.2f}"
            if name != "baseline alone":
                d = [scores[name][p] - ref[p] for p in common]
                lo, hi = bootstrap(d) if len(d) > 1 else (float("nan"), float("nan"))
                line += f"  vs baseline {np.mean(d):+.2f} (90% {lo:+.2f} to {hi:+.2f})"
            print(line)
    # The decision rule set before the run: blend on these questions only if
    # the 80% mixture beats the members' own forecast with the interval above 0.
    for arm in ARMS:
        own, mix = scores.get(f"claude {arm}", {}), scores.get(f"claude {arm} | 80% baseline", {})
        common = sorted(set(own) & set(mix))
        if len(common) > 1:
            d = [mix[p] - own[p] for p in common]
            lo, hi = bootstrap(d)
            print(f"\n80% mixture minus Claude {arm}: {np.mean(d):+.2f} per question (90% {lo:+.2f} to {hi:+.2f};"
                  f" better {sum(x > 0 for x in d)}, worse {sum(x < 0 for x in d)})")
    a, b = scores.get("claude card", {}), scores.get("claude card+baseline", {})
    common = sorted(set(a) & set(b))
    if len(common) > 1:
        d = [b[p] - a[p] for p in common]
        lo, hi = bootstrap(d)
        print(f"\nBlock effect on Claude: {np.mean(d):+.2f} per question (90% {lo:+.2f} to {hi:+.2f}; better {sum(x > 0 for x in d)}, worse {sum(x < 0 for x in d)}, n={len(d)})")
    numeric = [r for r in rows.values() if r["type"] != "binary"]
    inside = [float(r["percentiles"]["10"]) <= r["outcome"] <= float(r["percentiles"]["90"]) for r in numeric]
    print(f"Baseline P10-P90 coverage on numeric questions: {np.mean(inside):.2f} (n={len(inside)})")


def main_cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare").set_defaults(func=cmd_prepare)
    p = sub.add_parser("run")
    p.add_argument("--model", default="openrouter/anthropic/claude-sonnet-5")
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=cmd_run)
    sub.add_parser("analyze").set_defaults(func=cmd_analyze)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main_cli()
