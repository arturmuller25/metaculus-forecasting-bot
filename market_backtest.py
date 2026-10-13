"""
Backtest of market_baseline.py on resolved Market Pulse questions (26Q2 and
26Q3): relative returns of two assets, the VIX's intraday maximum, the 10-year
yield and the high-yield spread over two-week windows.

The Metaculus API hides the resolution, the rules and the community forecast of
these questions from a bot account, so the outcome is computed from the same
market data (Yahoo Finance and FRED), which can differ from the official
source by a little (futures rolls, a yield's exact fixing).

Three things are scored on every question, as of the last close before the
question closed (when Market Pulse scores forecasts):
  - the baseline alone, as a CDF;
  - the bot's numeric forecast step with a data card (the latest closes);
  - the same with the data card and the baseline block.
Both LLM arms use the bot's own prompt and parser (_run_forecast_on_numeric),
Claude through the Anthropic API only, as the Claude-only runs do; every
candidate model's training data ends before April 2026, so the windows are
unseen.

Usage:
    uv run python market_backtest.py prepare             # Metaculus and market data, no LLM
    uv run python market_backtest.py run [--model anthropic/claude-sonnet-5] [--limit N]
    uv run python market_backtest.py analyze             # offline
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import functools
import json
import logging
import math
import os
import time
from datetime import date, datetime, timedelta, timezone

import numpy as np

import analyze_results as ar
import bot as bot_module
import main
import market_baseline as mb
from forecasting_tools import DataOrganizer, NumericReport, Percentile
from forecasting_tools.ai_models import general_llm
from numeric_cdf import SmoothDistribution
from numeric_replay import bootstrap, bucket_of, pmf_at

logger = logging.getLogger("market_backtest")

PROJECTS = {33013: "Market Pulse 26Q2", 33066: "Market Pulse 26Q3"}
OUT = os.path.join("logs", "market_backtest")
QUESTIONS = os.path.join(OUT, "questions.jsonl")
ANSWERS = os.path.join(OUT, "answers.jsonl")
ARMS = ("card", "card+baseline")
SPEND_CAP = 25.0  # dollars of estimated Anthropic spend for one run
IN_FLIGHT = 4
# Dollars per million tokens (input, output), for the spending guard only.
PRICES = {"claude-sonnet-5": (3.0, 15.0), "claude-opus-5-5": (5.0, 25.0), "claude-haiku-4-5": (1.0, 5.0)}

bot_module.FORECAST_LOG = os.path.join(OUT, "forecasts.jsonl")
# Market data is downloaded once and reused, so reruns see the same series.
os.environ.setdefault("MARKET_CACHE_DIR", os.path.join(OUT, "data"))
_CALL: contextvars.ContextVar[dict | None] = contextvars.ContextVar("call", default=None)
_TODAY: contextvars.ContextVar[datetime | None] = contextvars.ContextVar("today", default=None)
_SLOTS = asyncio.Semaphore(IN_FLIGHT)
_SPENT = [0.0]
# Set by a caller that needs OpenRouter for a step other than the forecasts.
ALLOW_OPENROUTER = [False]
_real_acompletion = general_llm.acompletion


def _price(model: str) -> tuple[float, float]:
    for key, value in PRICES.items():
        if key in model.replace(".", "-"):
            return value
    return (15.0, 75.0)


@functools.wraps(_real_acompletion)
async def _guarded(*args, **kwargs):
    if _SPENT[0] >= SPEND_CAP:
        raise RuntimeError("spending cap reached")
    model = kwargs.get("model", "")
    if "openrouter" in model and not ALLOW_OPENROUTER[0]:
        raise RuntimeError(f"OpenRouter call blocked in this backtest: {model}")
    async with _SLOTS:
        response = await _real_acompletion(*args, **kwargs)
    usage = getattr(response, "usage", None)
    if usage is not None:
        pin, pout = _price(model)
        cost = (getattr(usage, "prompt_tokens", 0) or 0) * pin / 1e6 + (getattr(usage, "completion_tokens", 0) or 0) * pout / 1e6
        _SPENT[0] += cost
        call = _CALL.get()
        if call is not None:
            call.setdefault("usage", []).append({
                "model": model, "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None), "cost_estimate": round(cost, 5)})
    return response


general_llm.acompletion = _guarded


class ReplayDatetime(datetime):
    @classmethod
    def now(cls, tz=None):  # noqa: D401
        day = _TODAY.get()
        if day is None:
            return datetime.now(tz)
        return day.replace(tzinfo=None) if tz is None else day.astimezone(tz)


bot_module.datetime = ReplayDatetime


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def append_jsonl(path: str, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def sub_post(post: dict, sub: dict) -> dict:
    """A post JSON holding one subquestion, the way the library unpacks groups."""
    copy = {k: v for k, v in post.items() if k != "group_of_questions"}
    copy["question"] = sub
    return copy


def card(spec: mb.Spec, as_of: date) -> str:
    """The latest closes of every series, as the research would report them."""
    lines = [f"Market data as of the close of {as_of.isoformat()} (computed by code from Yahoo Finance and FRED):"]
    for name, (source, symbol, _) in zip(spec.names, spec.series):
        bars = [b for b in mb.bars_for(source, symbol) if b.day <= as_of][-10:]
        closes = ", ".join(f"{b.day.isoformat()[5:]} {b.close:,.4g}" for b in bars)
        lines.append(f"- {name} ({symbol}), last 10 closes: {closes}")
    return "\n".join(lines)


def cmd_prepare(_args) -> None:
    os.makedirs(OUT, exist_ok=True)
    rows, skipped = [], []
    for project in PROJECTS:
        listing = ar.get("/posts/", tournaments=project, limit=100)
        for item in listing.get("results", []):
            post = cached_post(item["id"])
            group = post.get("group_of_questions")
            if not group:
                continue
            for sub in group["questions"]:
                close = datetime.fromisoformat(sub["scheduled_close_time"].replace("Z", "+00:00"))
                spec = mb.spec_for(post["title"], sub.get("label") or "", close.year)
                if spec is None:
                    skipped.append(post["title"][:60])
                    continue
                as_of = (close - timedelta(days=1)).date()
                base = mb.compute(spec, as_of)
                result = mb.outcome(spec)
                if base is None or result is None:
                    skipped.append(f"{post['title'][:50]} {sub.get('label')}: no data")
                    continue
                rows.append({
                    "project": project, "post_id": post["id"], "question_id": sub["id"], "title": post["title"],
                    "label": sub.get("label"), "kind": spec.kind, "type": sub["type"],
                    "close": close.isoformat(), "as_of": as_of.isoformat(), "outcome": result,
                    "baseline": {str(k): v for k, v in base.percentiles.items()},
                    "baseline_text": base.render(), "card": card(spec, as_of),
                    "range": [sub["scaling"]["range_min"], sub["scaling"]["range_max"]],
                })
    with open(QUESTIONS, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    kinds: dict[str, int] = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"Questions ready: {len(rows)} {kinds}; skipped groups or windows: {len(skipped)}")
    outside = [r for r in rows if not (r["range"][0] <= r["outcome"] <= r["range"][1])]
    print(f"Outcomes outside the question's range: {len(outside)}")
    print("\nExample block:\n" + rows[0]["baseline_text"] + "\n\n" + rows[0]["card"])


def question_of(row: dict):
    post = cached_post(row["post_id"])
    sub = next(q for q in post["group_of_questions"]["questions"] if q["id"] == row["question_id"])
    return DataOrganizer.get_question_from_post_json(sub_post(post, sub))


def build_bot(model: str):
    os.environ.update(main.CLAUDE_ONLY)
    os.environ["ENSEMBLE_MODELS"] = model
    main.FORECAST_MODEL = main.CLAUDE_ONLY["FORECAST_MODEL"]
    main.RESEARCH_MODEL = main.CLAUDE_ONLY["RESEARCH_MODEL"]
    main.PARSER_MODEL = main.CLAUDE_ONLY["PARSER_MODEL"]
    bot = main.build_bot(publish=False, samples=1)
    bot._shadows = []
    return bot


async def run_one(bot, row: dict, arm: str, model: str, rep: int) -> None:
    question = question_of(row)
    research = row["card"] + ("\n\n" + row["baseline_text"] if arm == "card+baseline" else "")
    call: dict = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "question_id": row["question_id"],
                  "arm": arm, "model": model, "rep": rep}
    _CALL.set(call)
    _TODAY.set(datetime.fromisoformat(row["as_of"] + "T21:00:00+00:00"))
    started = time.monotonic()
    try:
        prediction = await bot._run_forecast_on_numeric(question, research)
        call["percentiles"] = [[p.percentile, p.value] for p in prediction.prediction_value.declared_percentiles]
        call["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        call["status"] = "error"
        call["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    call["seconds"] = round(time.monotonic() - started, 1)
    append_jsonl(ANSWERS, call)
    print(f"  {row['question_id']} {arm:14s} {call['status']} {call['seconds']}s spent ~${_SPENT[0]:.2f}", flush=True)


def cmd_run(args) -> None:
    logging.basicConfig(level=logging.WARNING)
    rows = read_jsonl(QUESTIONS)
    if args.limit:
        rows = rows[: args.limit]
    done = {(a["question_id"], a["arm"], a["model"], a["rep"]) for a in read_jsonl(ANSWERS) if a["status"] == "ok"}
    bot = build_bot(args.model)
    jobs = [(r, arm) for r in rows for arm in ARMS if (r["question_id"], arm, args.model, args.rep) not in done]
    print(f"{len(jobs)} calls to make with {args.model}")

    async def go() -> None:
        sem = asyncio.Semaphore(IN_FLIGHT)

        async def one(r, arm):
            async with sem:
                if _SPENT[0] < SPEND_CAP:
                    await run_one(bot, r, arm, args.model, args.rep)

        await asyncio.gather(*(one(r, arm) for r, arm in jobs))

    asyncio.run(go())
    print(f"Estimated spend this run: ${_SPENT[0]:.2f}")


async def cdf_from(percentiles: list[tuple[float, float]], question, widen: float) -> np.ndarray:
    dist = SmoothDistribution.build([Percentile(percentile=p, value=v) for p, v in percentiles], question,
                                    method="pchip", widen=widen)
    final = await NumericReport.aggregate_predictions([dist], question)
    return np.array([p.percentile for p in final.get_cdf()])


def log_score(cdf: np.ndarray, question, result: float) -> float:
    return 50 * math.log(max(1e-9, pmf_at(cdf, bucket_of(question, result, len(cdf)))))


def cmd_analyze(_args) -> None:
    logging.getLogger("forecasting_tools").setLevel(logging.ERROR)
    rows = {r["question_id"]: r for r in read_jsonl(QUESTIONS)}
    answers: dict[tuple, list] = {}
    for a in read_jsonl(ANSWERS):
        if a["status"] == "ok":
            answers[(a["question_id"], a["arm"], a["model"], a["rep"])] = a["percentiles"]
    scores: dict[str, dict[int, float]] = {}

    async def go() -> None:
        for qid, row in rows.items():
            question = question_of(row)
            base = [(int(p) / 100, v) for p, v in row["baseline"].items()]
            base_cdf = await cdf_from(base, question, 1.0)
            scores.setdefault("baseline alone", {})[qid] = log_score(base_cdf, question, row["outcome"])
            for (q, arm, model, rep), pts in answers.items():
                if q != qid:
                    continue
                cdf = await cdf_from([tuple(x) for x in pts], question, 1.15)
                tag = f"{model.split('/')[-1]} {arm}" + ("" if rep == 1 else f" r{rep}")
                scores.setdefault(tag, {})[qid] = log_score(cdf, question, row["outcome"])
                for w in (0.5, 0.7, 0.8, 0.9):
                    mix = (1 - w) * cdf + w * base_cdf
                    scores.setdefault(f"{tag} | {int(100 * w)}% baseline", {})[qid] = log_score(mix, question, row["outcome"])

    asyncio.run(go())
    ref = "baseline alone"
    kinds = {qid: r["kind"] for qid, r in rows.items()}
    for group in ["all"] + sorted(set(kinds.values())):
        posts = set(rows) if group == "all" else {q for q, k in kinds.items() if k == group}
        print(f"\n== {group} ({len(posts)} questions)")
        for name, by_q in sorted(scores.items()):
            common = sorted(set(by_q) & posts)
            if not common:
                continue
            line = f"  {name:58s} n={len(common):3d} mean {np.mean([by_q[q] for q in common]):7.2f}"
            if name != ref:
                diffs = [by_q[q] - scores[ref][q] for q in common if q in scores[ref]]
                lo, hi = bootstrap(diffs) if len(diffs) > 1 else (float("nan"), float("nan"))
                line += f"  vs baseline {np.mean(diffs):+.2f} (90% {lo:+.2f} to {hi:+.2f})"
            print(line)
    # The two LLM arms against each other, the decision this backtest is for.
    arms = sorted({(m, r) for (_, _, m, r) in answers})
    for model, rep in arms:
        a = scores.get(f"{model.split('/')[-1]} card" + ("" if rep == 1 else f" r{rep}"), {})
        b = scores.get(f"{model.split('/')[-1]} card+baseline" + ("" if rep == 1 else f" r{rep}"), {})
        common = sorted(set(a) & set(b))
        if len(common) > 1:
            diffs = [b[q] - a[q] for q in common]
            lo, hi = bootstrap(diffs)
            print(f"\n{model} rep {rep}: card+baseline minus card = {np.mean(diffs):+.2f} per question"
                  f" (90% {lo:+.2f} to {hi:+.2f}; better {sum(d > 0 for d in diffs)}, worse {sum(d < 0 for d in diffs)}, n={len(common)})")
    print(f"\nWidth check: share of outcomes inside the baseline's P10 to P90: "
          f"{np.mean([float(r['baseline']['10']) <= r['outcome'] <= float(r['baseline']['90']) for r in rows.values()]):.2f} (0.80 if calibrated)")


def main_cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare").set_defaults(func=cmd_prepare)
    p = sub.add_parser("run")
    p.add_argument("--model", default="openrouter/anthropic/claude-sonnet-5")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--rep", type=int, default=1)
    p.set_defaults(func=cmd_run)
    sub.add_parser("analyze").set_defaults(func=cmd_analyze)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main_cli()
