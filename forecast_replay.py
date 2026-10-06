"""
Replays the bot's forecast step on MiniBench round 1 (project 33125), to
compare forecasting models and binary prompt variants on questions that have
already resolved.

The research is frozen: each question gets the research the bot actually used
in September, rebuilt from the bot's public comment, and the date in the
prompt is set back to the day the bot forecast. Prompts and parsers are the
bot's own (_run_forecast_on_binary, _run_forecast_on_multiple_choice,
_run_forecast_on_numeric), called on a bot from main.build_bot with a single
ensemble member and no shadows, so bot.py runs unchanged. A prompt variant
edits the prompt text inside a thin wrapper around the model.

Every candidate model's knowledge cutoff (June 2026 at the latest) precedes
the questions, which resolved on 2026-10-03 and 04, so the replay is clean.

Each answer is appended to logs/replay_r1/answers.jsonl with the parsed
forecast, the raw text, and the tokens and cost of every call, so the analysis
can be redone without new calls. A run skips answers already recorded.

Usage:
    uv run python forecast_replay.py prepare                  # Metaculus only, no LLM calls
    uv run python forecast_replay.py check-prompts            # prints each variant's edited prompt, no LLM calls
    uv run python forecast_replay.py run --models sol56,sonnet5 [--variants V1,V2] [--types binary] [--limit 3] [--rep 2]
    uv run python forecast_replay.py analyze                  # scores, offline

Spending stops when the OpenRouter key has dropped $60 since the baseline in
logs/replay_r1/budget.json, or when it falls below $480 (the production bot
shares the key). The drop includes whatever the production bot spends
meanwhile, so the cap is conservative.
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import functools
import hashlib
import json
import logging
import math
import os
import re
import statistics
import time
import urllib.request
from datetime import datetime, timezone

import numpy as np

import analyze_results as ar
import bot as bot_module
import main
from forecasting_tools import (
    BinaryQuestion,
    DataOrganizer,
    MultipleChoiceQuestion,
    NumericQuestion,
    NumericReport,
    Percentile,
    PredictedOption,
    PredictedOptionList,
)
from forecasting_tools.ai_models import general_llm
from numeric_cdf import SmoothDistribution
from numeric_replay import bucket_of, members_from_comment as numeric_members_from_comment, pmf_at

logger = logging.getLogger("forecast_replay")

PROJECT = 33125  # MiniBench round 1
BOT_USER = 308418  # muller-ganhador-bot
OUT = os.path.join("logs", "replay_r1")
ANSWERS = os.path.join(OUT, "answers.jsonl")
BUDGET = os.path.join(OUT, "budget.json")

# Replay answers must never reach the bot's own forecast log, which
# numeric_replay.py and analyze_results.py read as real forecasts.
bot_module.FORECAST_LOG = os.path.join(OUT, "forecasts.jsonl")

SPEND_CAP = 60.0  # dollars of key drop since the baseline
KEY_FLOOR = 480.0  # dollars left on the key
IN_FLIGHT = 4  # LLM calls at once, parser included
# Production uses 120 s. A longer timeout here keeps a slow but valid answer
# (and its cost) instead of paying for a retry; the time of each call is
# recorded, so how often a model would hit the production timeout is known.
FORECAST_TIMEOUT = 300

MODELS = {
    "sol56": "openrouter/openai/gpt-5.6-sol",
    "sonnet5": "openrouter/anthropic/claude-sonnet-5",
    "opus55": "openrouter/anthropic/claude-opus-5.5",
    "sonnet55": "openrouter/anthropic/claude-sonnet-5.5",
    "sol61": "openrouter/openai/gpt-6.1-sol",
    "sol6": "openrouter/openai/gpt-6-sol",
}

# Two-model ensembles scored offline from the single-model answers.
PAIRS = [
    ("sol56", "sonnet5"),  # production today
    ("sol56", "opus55"),
    ("sol56", "sonnet55"),
    ("sol61", "opus55"),
    ("sol61", "sonnet5"),
    ("sol61", "sonnet55"),
]
CURRENT = "sol56+sonnet5"


# ---------------------------------------------------------------------------
# Binary prompt variants
# ---------------------------------------------------------------------------


def _words(sentence: str) -> str:
    """Pattern for a sentence whatever the line breaks between its words."""
    return r"\s+".join(re.escape(word) for word in sentence.split())


_FACE_SAVING = (
    _words("- The principal actor may have a face-saving route")
    + r".*?"
    + _words("Name that route explicitly before dismissing the outcome.")
)
_STATUS_QUO = _words(
    "You write your rationale remembering that good forecasters put extra weight on the"
    " status quo outcome since the world changes slowly most of the time."
)
_YES_RATE = _words(
    "Historically, forecasters like you have been overconfident, and only about 35% of"
    " Metaculus binary questions resolve Yes."
)

V1 = [(
    _FACE_SAVING,
    "- For each of the two outcomes, name the most plausible route by which it happens,"
    " estimate its chance, and only then decide.",
)]
V2 = [(
    _STATUS_QUO,
    "You treat the status quo as the current trajectory, not today's snapshot: for transient"
    " states (an active fire, an outbreak, a storm, a crisis) estimate the chance it persists"
    " using a daily hazard from comparable past cases and the shortest official forecast"
    " available.",
)]
V3 = [(
    _YES_RATE,
    "When the research gives a base rate for this exact kind of event, start from it and move"
    " away only with case-specific evidence.",
)]
VARIANTS = {"base": [], "V1": V1, "V2": V2, "V3": V3, "V4": V1 + V2 + V3}


class PromptEditError(RuntimeError):
    pass


def apply_edits(prompt: str, edits: list[tuple[str, str]]) -> str:
    """Applies each replacement exactly once, or fails without calling the model."""
    for pattern, new in edits:
        prompt, count = re.subn(pattern, lambda _m: new, prompt, flags=re.S)
        if count != 1:
            raise PromptEditError(f"pattern matched {count} times: {pattern[:60]}")
    return prompt


# ---------------------------------------------------------------------------
# Metaculus data (cached on disk, at most 6 calls per 10 seconds)
# ---------------------------------------------------------------------------

_last_call = [0.0]
_raw_get = ar.get


def _throttled_get(path: str, **params) -> dict:
    wait = 1.75 - (time.monotonic() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.monotonic()
    return _raw_get(path, **params)


# bot_comment_text looks `get` up in its module, so this throttles it too.
ar.get = _throttled_get


def _cache_path(name: str) -> str:
    os.makedirs(os.path.join(OUT, "cache"), exist_ok=True)
    return os.path.join(OUT, "cache", name)


def cached_json(name: str, path: str, **params) -> dict:
    file = _cache_path(name + ".json")
    if os.path.exists(file):
        with open(file, encoding="utf-8") as fh:
            return json.load(fh)
    data = ar.get(path, **params)
    with open(file, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)
    return data


def cached_comment(post_id: int) -> str:
    file = _cache_path(f"comment_{post_id}.txt")
    if os.path.exists(file):
        with open(file, encoding="utf-8") as fh:
            return fh.read()
    text = ar.bot_comment_text(post_id, BOT_USER)
    with open(file, "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def research_from_comment(text: str) -> tuple[str, str]:
    """
    The research string the bot passed to its forecasters, rebuilt from its
    comment. The library prints it under "## Report 1 Research" with every
    heading pushed one level down ("## Source: X" becomes "### Source: X"),
    or, when the headings cannot be re-leveled, with every "#" written as
    "[Hashtag]". Both are undone here; only heading depth can differ from
    the original, and only where a source skipped heading levels.
    """
    start_marker = "## Report 1 Research\n"
    start = text.find(start_marker)
    if start < 0:
        return "", "no research section (comment truncated or missing)"
    start += len(start_marker)
    end = text.find("\n# FORECASTS", start)
    block = text[start:end if end >= 0 else len(text)]
    if "[Hashtag]" in block:
        return block.replace("[Hashtag]", "#").strip(), "hashtag fallback"
    block = re.sub(r"^#(#{1,8}) ", r"\1 ", block, flags=re.M)
    return block.strip(), "headings re-leveled"


def list_posts() -> list[dict]:
    posts, offset = [], 0
    while True:
        page = cached_json(f"list_{offset}", "/posts/", tournaments=PROJECT, limit=100, offset=offset)
        posts += page["results"]
        if len(page["results"]) < 100:
            return posts
        offset += 100


def load_questions(verbose: bool = False) -> list[dict]:
    """Every post of the round the bot forecast, with research, outcome and date."""
    rows = []
    for post in list_posts():
        pid = post["id"]
        post_json = cached_json(f"post_{pid}", f"/posts/{pid}/")
        q_json = post_json["question"]
        latest = ((q_json.get("my_forecasts") or {}).get("latest") or {})
        start = latest.get("start_time")
        if not start:
            if verbose:
                print(f"  {pid}: the bot did not forecast it")
            continue
        text = cached_comment(pid)
        research, how = research_from_comment(text)
        forecast_time = datetime.fromtimestamp(float(start), tz=timezone.utc)
        rows.append({
            "post_id": pid,
            "type": q_json["type"],
            "title": post_json.get("title", ""),
            "outcome": ar.outcome_of(q_json),
            "resolution": q_json.get("resolution"),
            "forecast_time": forecast_time.isoformat(timespec="seconds"),
            "research": research,
            "research_how": how,
            "comment": text,
            "post_json": post_json,
        })
    return rows


def question_object(row: dict):
    return DataOrganizer.get_question_from_post_json(row["post_json"])


def cmd_prepare(_args) -> None:
    rows = load_questions(verbose=True)
    os.makedirs(os.path.join(OUT, "research"), exist_ok=True)
    summary = []
    for row in rows:
        with open(os.path.join(OUT, "research", f"{row['post_id']}.md"), "w", encoding="utf-8") as fh:
            fh.write(row["research"])
        labels = re.findall(r"^## Source: (.+)$", row["research"], flags=re.M)
        summary_len = 0
        m = re.search(r"### Research Summary\n(.*?)\n# RESEARCH", row["comment"], flags=re.S)
        if m:
            summary_len = len(m.group(1))
        summary.append({
            "post_id": row["post_id"], "type": row["type"], "outcome": row["outcome"],
            "forecast_time": row["forecast_time"], "research_chars": len(row["research"]),
            "summary_chars": summary_len, "comment_chars": len(row["comment"]),
            "sources": labels, "how": row["research_how"],
        })
        print(
            f"  {row['post_id']} {row['type'][:8]:8s} outcome={str(row['outcome'])[:18]:18s} "
            f"forecast {row['forecast_time'][:10]} research {len(row['research']):6d} chars "
            f"(summary {summary_len:5d}) {row['research_how']}; sources: {', '.join(labels)}"
        )
    with open(os.path.join(OUT, "questions.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    scored = [r for r in rows if r["outcome"] is not None]
    kinds: dict[str, int] = {}
    for r in scored:
        kinds[r["type"]] = kinds.get(r["type"], 0) + 1
    print(f"\nPosts the bot forecast: {len(rows)}; resolved and scored: {len(scored)} {kinds}")
    short = [r["post_id"] for r in scored if len(r["research"]) < 1000]
    if short:
        print(f"Research shorter than 1,000 characters: {short}")


# ---------------------------------------------------------------------------
# OpenRouter: balance, prices, and the guard on every call
# ---------------------------------------------------------------------------


def _openrouter(path: str) -> dict:
    req = urllib.request.Request(
        f"https://openrouter.ai/api/v1{path}",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def key_remaining() -> float:
    return float(_openrouter("/key")["data"]["limit_remaining"])


def _budget() -> dict:
    with open(BUDGET, encoding="utf-8") as fh:
        return json.load(fh)


def log_reading(label: str) -> tuple[float, float]:
    """Records the key balance; returns (remaining, drop since the baseline)."""
    remaining = key_remaining()
    data = _budget() if os.path.exists(BUDGET) else {}
    if "baseline" not in data:
        data["baseline"] = {"time": _utc_now(), "limit_remaining": remaining}
    data.setdefault("readings", []).append({"time": _utc_now(), "label": label, "limit_remaining": remaining})
    with open(BUDGET, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    return remaining, float(data["baseline"]["limit_remaining"]) - remaining


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def prices() -> dict[str, tuple[float, float]]:
    """Dollars per prompt and per completion token, by OpenRouter model id."""
    out = {}
    for m in _openrouter("/models")["data"]:
        p = m.get("pricing") or {}
        out[m["id"]] = (float(p.get("prompt") or 0), float(p.get("completion") or 0))
    return out


def load_prices() -> dict[str, tuple[float, float]]:
    try:
        with open(os.path.join(OUT, "prices.json"), encoding="utf-8") as fh:
            return {k: tuple(v) for k, v in json.load(fh).items()}
    except OSError:
        return {}


def call_cost(usage: dict, price: dict[str, tuple[float, float]]) -> float:
    """
    Dollars for one call: native tokens at OpenRouter's list price. The key
    is a BYOK key, so OpenRouter reports a cost of 0 for each generation and
    only the key's balance shows the real charge; this estimate ignores
    prompt caching discounts.
    """
    if usage.get("cost"):
        return float(usage["cost"])
    pin, pout = price.get((usage.get("model") or "").removeprefix("openrouter/"), (0.0, 0.0))
    return (usage.get("prompt_tokens") or 0) * pin + (usage.get("completion_tokens") or 0) * pout


class BudgetStop(RuntimeError):
    pass


STOP = asyncio.Event()
_SLOTS = asyncio.Semaphore(IN_FLIGHT)
_CALL: contextvars.ContextVar[dict | None] = contextvars.ContextVar("replay_call", default=None)
_TODAY: contextvars.ContextVar[datetime | None] = contextvars.ContextVar("replay_today", default=None)
_real_acompletion = general_llm.acompletion


def _usage_row(model: str, response) -> dict:
    row: dict = {"model": model, "id": getattr(response, "id", None)}
    usage = getattr(response, "usage", None)
    if usage is not None:
        row["prompt_tokens"] = getattr(usage, "prompt_tokens", None)
        row["completion_tokens"] = getattr(usage, "completion_tokens", None)
        details = getattr(usage, "completion_tokens_details", None)
        row["reasoning_tokens"] = getattr(details, "reasoning_tokens", None) if details else None
        cost = getattr(usage, "cost", None)
        if cost is None and isinstance(getattr(usage, "model_extra", None), dict):
            cost = usage.model_extra.get("cost")
        row["cost"] = cost
    hidden = getattr(response, "_hidden_params", None) or {}
    if row.get("cost") is None and hidden.get("response_cost") is not None:
        row["litellm_cost"] = hidden.get("response_cost")
    return row


@functools.wraps(_real_acompletion)
async def _guarded_acompletion(*args, **kwargs):
    """Every LLM call of the bot goes through here: cap, concurrency, usage."""
    if STOP.is_set():
        raise BudgetStop("spending cap reached")
    async with _SLOTS:
        response = await _real_acompletion(*args, **kwargs)
    call = _CALL.get()
    if call is not None:
        call.setdefault("usage", []).append(_usage_row(kwargs.get("model", ""), response))
    return response


general_llm.acompletion = _guarded_acompletion


class ReplayDatetime(datetime):
    """datetime whose now() returns the original forecast time of the question being replayed."""

    @classmethod
    def now(cls, tz=None):  # noqa: D401
        day = _TODAY.get()
        if day is None:
            return datetime.now(tz)
        return day.replace(tzinfo=None) if tz is None else day.astimezone(tz)


bot_module.datetime = ReplayDatetime


class _WarningCatcher(logging.Handler):
    """Keeps the bot's warnings (failures, parse problems) with the answer they concern."""

    def emit(self, record: logging.LogRecord) -> None:
        call = _CALL.get()
        if call is not None and record.levelno >= logging.WARNING:
            call.setdefault("warnings", []).append(record.getMessage()[:500])


bot_module.logger.addHandler(_WarningCatcher())


class ReplayLlm:
    """Stands in for a GeneralLlm: same .model, edits the prompt, keeps the raw answer."""

    def __init__(self, llm, edits: list[tuple[str, str]]) -> None:
        self.model = llm.model
        self._llm = llm
        self._edits = edits

    async def invoke(self, prompt: str) -> str:
        call = _CALL.get()
        prompt = apply_edits(prompt, self._edits)
        if call is not None:
            call["prompt_chars"] = len(prompt)
            call["prompt_sha1"] = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]
        started = time.monotonic()
        try:
            text = await self._llm.invoke(prompt)
        finally:
            if call is not None:
                call["seconds"] = round(time.monotonic() - started, 1)
        if call is not None:
            call["raw"] = text
        return text


def replay_bot(model: str, variant: str):
    """A production bot with one ensemble member, built exactly like main.py builds members."""
    bot = main.build_bot(publish=False, samples=1)
    bot._shadows = []
    bot._ensemble = [ReplayLlm(main._thinker(model, 0.3, FORECAST_TIMEOUT), VARIANTS[variant])]
    return bot


async def forecast_once(bot, question, research: str):
    if isinstance(question, BinaryQuestion):
        return await bot._run_forecast_on_binary(question, research)
    if isinstance(question, MultipleChoiceQuestion):
        return await bot._run_forecast_on_multiple_choice(question, research)
    if isinstance(question, NumericQuestion):
        return await bot._run_forecast_on_numeric(question, research)
    raise ValueError(f"unsupported question type {type(question).__name__}")


def serialize(prediction):
    if isinstance(prediction, float):
        return round(prediction, 6)
    if isinstance(prediction, PredictedOptionList):
        return {o.option_name: round(o.probability, 6) for o in prediction.predicted_options}
    return [[p.percentile, p.value] for p in prediction.declared_percentiles]


def recorded() -> list[dict]:
    if not os.path.exists(ANSWERS):
        return []
    with open(ANSWERS, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


async def _monitor(interval: float = 20.0) -> None:
    while not STOP.is_set():
        try:
            remaining, drop = await asyncio.to_thread(log_reading, "monitor")
            if drop >= SPEND_CAP or remaining < KEY_FLOOR:
                logger.warning(f"Cap reached: key dropped ${drop:.2f}, ${remaining:.2f} left. Stopping.")
                STOP.set()
                return
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not read the key balance: {exc}")
        await asyncio.sleep(interval)


async def run_jobs(jobs: list[tuple[dict, str, str, int]]) -> None:
    bots = {}
    for _, alias, variant, _rep in jobs:
        if (alias, variant) not in bots:
            bots[(alias, variant)] = replay_bot(MODELS[alias], variant)
    price = load_prices()
    queue: asyncio.Queue = asyncio.Queue()
    for job in jobs:
        queue.put_nowait(job)
    done = [0]

    async def worker() -> None:
        while not queue.empty() and not STOP.is_set():
            row, alias, variant, rep = queue.get_nowait()
            call: dict = {}
            _CALL.set(call)
            _TODAY.set(datetime.fromisoformat(row["forecast_time"]))
            question = question_object(row)
            status, prediction, error = "ok", None, None
            try:
                result = await forecast_once(bots[(alias, variant)], question, row["research"])
                prediction = serialize(result.prediction_value)
            except Exception as exc:  # noqa: BLE001
                status = "stopped" if STOP.is_set() else "failed"
                error = f"{type(exc).__name__}: {str(exc)[:300]}"
            out = {
                "time": _utc_now(), "post_id": row["post_id"], "type": row["type"],
                "alias": alias, "model": MODELS[alias], "variant": variant, "rep": rep,
                "forecast_time": row["forecast_time"], "status": status, "error": error,
                "prediction": prediction, **call,
            }
            with open(ANSWERS, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(out, ensure_ascii=False) + "\n")
            done[0] += 1
            cost = sum(call_cost(u, price) for u in call.get("usage", []))
            shown = prediction if not isinstance(prediction, list) else f"P10..P90 {prediction[0][1]}..{prediction[-1][1]}"
            if isinstance(shown, dict):
                shown = {k[:12]: round(v, 2) for k, v in shown.items()}
            logger.info(
                f"[{done[0]}/{len(jobs)}] {row['post_id']} {row['type'][:8]} {alias} {variant} r{rep}: "
                f"{status} {shown if status == 'ok' else error} ({call.get('seconds')} s, ${cost:.3f})"
            )

    monitor = asyncio.create_task(_monitor())
    await asyncio.gather(*(worker() for _ in range(IN_FLIGHT)))
    monitor.cancel()


def cmd_run(args) -> None:
    aliases = [a.strip() for a in args.models.split(",") if a.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [a for a in aliases if a not in MODELS] + [v for v in variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown model alias or variant: {unknown}")
    types = {t.strip() for t in args.types.split(",") if t.strip()}

    rows = [r for r in load_questions() if r["outcome"] is not None and r["type"] in types]
    if args.posts:
        wanted = {int(p) for p in args.posts.split(",")}
        rows = [r for r in rows if r["post_id"] in wanted]
    if args.limit:
        by_id = {r["post_id"]: r for r in rows}
        picked = main._pick([question_object(r) for r in rows], args.limit)
        rows = [by_id[q.id_of_post] for q in picked]

    done = {
        (r["post_id"], r["alias"], r["variant"], r["rep"])
        for r in recorded()
        if r["status"] in ("ok", "failed") and not (args.retry_failed and r["status"] == "failed")
    }
    jobs = [
        (row, alias, variant, args.rep)
        for row in rows
        for variant in variants
        for alias in aliases
        if (row["post_id"], alias, variant, args.rep) not in done
    ]
    remaining, drop = log_reading(f"{args.label} start")
    print(f"Key: ${remaining:.2f} left, ${drop:.2f} spent since the baseline (cap ${SPEND_CAP:.0f}, floor ${KEY_FLOOR:.0f})")
    if drop >= SPEND_CAP or remaining < KEY_FLOOR:
        raise SystemExit("Cap already reached; nothing run.")
    print(f"{len(jobs)} answers to request ({len(rows)} questions x {len(aliases)} models x {len(variants)} variants)")
    if not jobs:
        return
    with open(os.path.join(OUT, "prices.json"), "w", encoding="utf-8") as fh:
        json.dump(prices(), fh)
    asyncio.run(run_jobs(jobs))
    remaining, drop = log_reading(f"{args.label} end")
    print(f"Key: ${remaining:.2f} left, ${drop:.2f} spent since the baseline")
    if STOP.is_set():
        print("STOPPED: spending cap reached.")


def cmd_check_prompts(_args) -> None:
    """Builds the binary prompt of one question under every variant, without calling a model."""
    row = next(r for r in load_questions() if r["type"] == "binary" and r["outcome"] is not None)

    class Capture:
        model = "capture"

        async def invoke(self, prompt: str) -> str:
            self.prompt = prompt
            raise RuntimeError("captured")

    for variant, edits in VARIANTS.items():
        capture = Capture()
        bot = main.build_bot(publish=False, samples=1)
        bot._shadows = []
        bot._ensemble = [capture]
        _TODAY.set(datetime.fromisoformat(row["forecast_time"]))
        try:
            asyncio.run(bot._run_forecast_on_binary(question_object(row), row["research"]))
        except RuntimeError:
            pass
        edited = apply_edits(capture.prompt, edits)
        start = edited.find("Today is")
        end = edited.find("Two Metaculus resolution conventions")
        print(f"\n===== {variant} (post {row['post_id']}) =====")
        print(edited[start:end].rstrip())


# ---------------------------------------------------------------------------
# Scoring (offline)
# ---------------------------------------------------------------------------


def clamp_options(probs: dict[str, float], options: list[str]) -> dict[str, float]:
    """The library's own clamping to [0.01, 0.99] and renormalization."""
    total = sum(probs.get(o, 0.0) for o in options) or 1.0
    try:
        plist = PredictedOptionList(predicted_options=[
            PredictedOption(option_name=o, probability=probs.get(o, 0.0) / total) for o in options
        ])
        return {o.option_name: o.probability for o in plist.predicted_options}
    except ValueError:
        clamped = {o: min(0.99, max(0.01, probs.get(o, 0.0) / total)) for o in options}
        s = sum(clamped.values())
        return {o: v / s for o, v in clamped.items()}


async def published_cdf(members: list[list[tuple[float, float]]], question) -> np.ndarray:
    """The CDF the bot publishes from these members' percentiles: PCHIP, widened 1.15, then aggregated as in production."""
    dists = [
        SmoothDistribution.build(
            [Percentile(percentile=p, value=v) for p, v in pts], question, method="pchip", widen=1.15
        )
        for pts in members
    ]
    combined = await NumericReport.aggregate_predictions(dists, question) if len(dists) > 1 else dists[0]
    top = await NumericReport.aggregate_predictions([combined], question)
    return np.array([p.percentile for p in top.get_cdf()])


async def score(forecasts: list, row: dict, question) -> tuple[float, float | None]:
    """Log score of the forecast the bot would publish from these members, and Brier for binaries."""
    outcome, kind = row["outcome"], row["type"]
    if kind == "binary":
        p = sum(min(0.99, max(0.01, f)) for f in forecasts) / len(forecasts)
        return 100 * math.log(p if outcome == 1 else 1 - p), (p - outcome) ** 2
    if kind == "multiple_choice":
        options = row["post_json"]["question"]["options"]
        aligned = [clamp_options(f, options) for f in forecasts]
        mean = clamp_options({o: sum(a[o] for a in aligned) / len(aligned) for o in options}, options)
        return 100 * math.log(mean[outcome]), None
    question_members = [[tuple(p) for p in f] for f in forecasts]
    cdf = await published_cdf(question_members, question)
    return 50 * math.log(pmf_at(cdf, bucket_of(question, outcome, len(cdf)))), None


def paired(a: dict[int, float], b: dict[int, float], posts: set[int]) -> dict | None:
    common = sorted(set(a) & set(b) & posts)
    if not common:
        return None
    diffs = np.array([a[p] - b[p] for p in common])
    if len(diffs) >= 2:
        means = np.random.default_rng(0).choice(diffs, size=(20000, len(diffs)), replace=True).mean(axis=1)
        lo, hi = float(np.percentile(means, 5)), float(np.percentile(means, 95))
    else:
        lo = hi = float("nan")
    return {
        "n": len(common), "mean": float(diffs.mean()), "lo": lo, "hi": hi,
        "better": int((diffs > 1e-9).sum()), "worse": int((diffs < -1e-9).sum()),
    }


def cmd_analyze(args) -> None:
    # The library warns on every renormalized option list; nothing to act on here.
    logging.getLogger("forecasting_tools").setLevel(logging.ERROR)
    rows ={r["post_id"]: r for r in load_questions() if r["outcome"] is not None}
    questions = {pid: question_object(r) for pid, r in rows.items()}
    answers = recorded()
    price = load_prices()

    # Latest successful answer per (post, model, variant, rep).
    latest: dict[tuple, dict] = {}
    for a in answers:
        if a["status"] == "ok" and a["post_id"] in rows:
            latest[(a["post_id"], a["alias"], a["variant"], a["rep"])] = a

    # Members to score: config name -> {post: [member forecasts]}.
    configs: dict[str, dict[int, list]] = {}

    def single(alias: str, variant: str, rep: int) -> dict[int, object]:
        return {pid: a["prediction"] for (pid, al, v, r), a in latest.items() if al == alias and v == variant and r == rep}

    def name(core: str, variant: str, rep: int) -> str:
        return core + ("" if variant == "base" else f" {variant}") + ("" if rep == 1 else f" r{rep}")

    combos = sorted({(al, v, r) for (_, al, v, r) in latest})
    for alias, variant, rep in combos:
        configs[name(alias, variant, rep)] = {pid: [f] for pid, f in single(alias, variant, rep).items()}
    for variant, rep in sorted({(v, r) for (_, v, r) in combos}):
        for x, y in PAIRS:
            sx, sy = single(x, variant, rep), single(y, variant, rep)
            if not sx or not sy:
                continue
            # Production publishes the surviving member when one fails.
            configs[name(f"{x}+{y}", variant, rep)] = {
                pid: [f for f in (sx.get(pid), sy.get(pid)) if f is not None] for pid in set(sx) | set(sy)
            }

    # Round-1 forecasts, parsed from the bot's comments. Binaries then had two
    # members with their own web search (":online"); numeric and multiple
    # choice questions had a single model, named in the comment's LLM list,
    # whose forecast is the published one. Different prompts and research
    # setup, so this baseline mixes model, prompt and pipeline changes.
    old: dict[str, dict[int, object]] = {}
    for pid, row in rows.items():
        q_json = row["post_json"]["question"]
        default = re.search(r"'default': \{'original_model': '([^']+)'", row["comment"])
        default_model = default.group(1) if default else "unknown"
        if row["type"] in ("numeric", "discrete"):
            members, _source = numeric_members_from_comment(row["comment"])
            if "single model" in members:
                members = {default_model: members["single model"]}
        else:
            members = ar.members_from_comment(row["comment"], q_json)
            values = ((q_json.get("my_forecasts") or {}).get("latest") or {}).get("forecast_values")
            if not members and values:
                members = {default_model: values[1] if row["type"] == "binary" else dict(zip(q_json["options"], values))}
        for model, forecast in members.items():
            old.setdefault(model.split("/")[-1], {})[pid] = forecast
    for model, by_post in old.items():
        configs[f"round1 {model}"] = {pid: [f] for pid, f in by_post.items()}
    round1_members: dict[int, list] = {}
    for model, by_post in old.items():
        for pid, f in by_post.items():
            round1_members.setdefault(pid, []).append(f)
    # The same members aggregated and built (PCHIP x1.15) as the bot does today.
    configs["round1 members, today's CDF"] = round1_members

    scores: dict[str, dict[int, float]] = {}
    briers: dict[str, dict[int, float]] = {}

    async def score_all() -> None:
        for config, by_post in configs.items():
            for pid, forecasts in by_post.items():
                if not forecasts:
                    continue
                try:
                    s, b = await score(forecasts, rows[pid], questions[pid])
                except Exception as exc:  # noqa: BLE001
                    print(f"  scoring failed: {config} {pid}: {type(exc).__name__}: {str(exc)[:120]}")
                    continue
                scores.setdefault(config, {})[pid] = s
                if b is not None:
                    briers.setdefault(config, {})[pid] = b

    asyncio.run(score_all())

    # What the bot actually published in round 1.
    for pid, row in rows.items():
        q_json = row["post_json"]["question"]
        values = ((q_json.get("my_forecasts") or {}).get("latest") or {}).get("forecast_values")
        if not values:
            continue
        if row["type"] == "binary":
            p = values[1]
            scores.setdefault("round1 published", {})[pid] = 100 * math.log(p if row["outcome"] == 1 else 1 - p)
            briers.setdefault("round1 published", {})[pid] = (p - row["outcome"]) ** 2
        elif row["type"] == "multiple_choice":
            probs = dict(zip(q_json["options"], values))
            scores.setdefault("round1 published", {})[pid] = 100 * math.log(probs[row["outcome"]])
        else:
            cdf = np.array(values)
            scores.setdefault("round1 published", {})[pid] = 50 * math.log(
                pmf_at(cdf, bucket_of(questions[pid], row["outcome"], len(cdf)))
            )

    # Configurations asked twice: per question, the mean of the two runs'
    # scores, which halves the sampling part of the noise.
    for config in [c for c in scores if c.endswith(" r2")]:
        first = config[: -len(" r2")]
        for table_ in (scores, briers):
            if first in table_ and config in table_:
                both = set(table_[first]) & set(table_[config])
                table_[f"{first} (2 runs)"] = {p: (table_[first][p] + table_[config][p]) / 2 for p in both}

    groups = {
        "all": set(rows),
        "binary": {p for p, r in rows.items() if r["type"] == "binary"},
        "multiple_choice": {p for p, r in rows.items() if r["type"] == "multiple_choice"},
        "numeric+discrete": {p for p, r in rows.items() if r["type"] in ("numeric", "discrete")},
    }
    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    # Where a configuration was asked twice, its two-run mean stands in for
    # it, as configuration and as reference; the separate runs are compared
    # in the repeat section below.
    references = [f"{r} (2 runs)" if f"{r} (2 runs)" in scores else r for r in (CURRENT, "sol56", "sonnet5")]
    shown = [c for c in scores if not c.endswith(" r2") and f"{c} (2 runs)" not in scores]
    for group, posts in groups.items():
        emit(f"\n## {group} ({len(posts)} questions)\n")
        emit("| config | n | mean log score | Brier | " + " | ".join(f"vs {r}: mean [90% CI] better/worse" for r in references) + " |")
        emit("|---|---|---|---|" + "---|" * len(references))
        for config in sorted(shown, key=lambda c: (c.startswith("round1"), c)):
            mine = {p: s for p, s in scores[config].items() if p in posts}
            if not mine:
                continue
            brier = briers.get(config, {})
            b = [brier[p] for p in mine if p in brier]
            cells = []
            for ref in references:
                res = paired(scores[config], scores.get(ref, {}), posts) if ref != config else None
                cells.append(
                    f"{res['mean']:+.2f} [{res['lo']:+.2f}, {res['hi']:+.2f}] {res['better']}/{res['worse']} (n={res['n']})"
                    if res else "-"
                )
            brier_cell = f"{statistics.mean(b):.4f}" if b else "-"
            emit(
                f"| {config} | {len(mine)} | {statistics.mean(mine.values()):+.2f} | {brier_cell} | "
                + " | ".join(cells) + " |"
            )

    # Brier differences against the current pair, binaries only.
    emit("\n## Brier on binaries, paired against the current pair (negative = better)\n")
    for config in sorted(briers):
        res = paired(briers[config], briers.get(CURRENT, {}), groups["binary"])
        if res and config != CURRENT:
            emit(f"- {config}: {res['mean']:+.4f} [{res['lo']:+.4f}, {res['hi']:+.4f}] n={res['n']}")

    # Within-model noise: the same configuration asked twice.
    reps = sorted({c for c in scores if c.endswith(" r2")})
    if reps:
        emit("\n## Repeat runs (same configuration, second request)\n")
        for config in reps:
            first = config[: -len(" r2")]
            for group in ("binary", "all"):
                res = paired(scores[config], scores.get(first, {}), groups[group])
                if res:
                    absolute = statistics.mean(
                        abs(scores[config][p] - scores[first][p]) for p in set(scores[config]) & set(scores[first]) & groups[group]
                    )
                    emit(
                        f"- {config} vs {first} ({group}): {res['mean']:+.2f} [{res['lo']:+.2f}, {res['hi']:+.2f}] "
                        f"{res['better']}/{res['worse']} n={res['n']}; mean absolute change {absolute:.2f}"
                    )

    # Prompt variants against the production prompt, on binaries, with both
    # sides averaged over two runs when a second run exists.
    emit("\n## Binary prompt variants against the production prompt\n")
    emit("| configuration | runs | log score: mean [90% CI] better/worse | Brier: mean [90% CI] |")
    emit("|---|---|---|---|")
    for core in ("sol56+sonnet5", "sol56", "sonnet5"):
        for variant in ("V1", "V2", "V3", "V4"):
            for suffix, base in (("", core), (" (2 runs)", f"{core} (2 runs)")):
                config = f"{core} {variant}{suffix}"
                if config not in scores or base not in scores:
                    continue
                res = paired(scores[config], scores[base], groups["binary"])
                bres = paired(briers.get(config, {}), briers.get(base, {}), groups["binary"])
                if res:
                    emit(
                        f"| {core} {variant} vs production prompt | {'2' if suffix else '1'} | "
                        f"{res['mean']:+.2f} [{res['lo']:+.2f}, {res['hi']:+.2f}] {res['better']}/{res['worse']} (n={res['n']}) | "
                        + (f"{bres['mean']:+.4f} [{bres['lo']:+.4f}, {bres['hi']:+.4f}] |" if bres else "- |")
                    )

    # Costs, time and failures per model and variant.
    emit("\n## Cost, time and failures\n")
    emit(
        "| model | variant | rep | questions answered | failed | forecaster $/answer | parser $/answer"
        " | median s | over 120 s | completion tokens (median) | reasoning tokens (median) |"
    )
    emit("|---|---|---|---|---|---|---|---|---|---|---|")
    groups_cost: dict[tuple, list[dict]] = {}
    for a in answers:
        groups_cost.setdefault((a["alias"], a["variant"], a["rep"]), []).append(a)
    for (alias, variant, rep), items in sorted(groups_cost.items()):
        ok = [a for a in items if a["status"] == "ok"]
        failed = [a for a in items if a["status"] == "failed"]
        with_usage = [a for a in items if a.get("usage")]

        def cost_of(a: dict, forecaster: bool) -> float:
            return sum(
                call_cost(u, price) for u in a.get("usage", [])
                if ("gpt-4o-mini" not in (u.get("model") or "")) == forecaster
            )

        f_cost = statistics.mean(cost_of(a, True) for a in with_usage) if with_usage else float("nan")
        p_cost = statistics.mean(cost_of(a, False) for a in with_usage) if with_usage else float("nan")
        secs = [a["seconds"] for a in items if a.get("seconds") is not None]
        forecaster_calls = [
            u for a in items for u in a.get("usage", []) if "gpt-4o-mini" not in (u.get("model") or "")
        ]
        tokens = [u.get("completion_tokens") or 0 for u in forecaster_calls]
        thinking = [u.get("reasoning_tokens") or 0 for u in forecaster_calls]
        answered = len({a["post_id"] for a in ok})
        emit(
            f"| {alias} | {variant} | {rep} | {answered} | {len(failed)} | {f_cost:.4f} | {p_cost:.4f} | "
            f"{statistics.median(secs) if secs else float('nan'):.0f} | {sum(s > 120 for s in secs)} | "
            f"{statistics.median(tokens) if tokens else 0:.0f} | {statistics.median(thinking) if thinking else 0:.0f} |"
        )
    if os.path.exists(BUDGET):
        readings = _budget().get("readings", [])
        # A start without an end is an interrupted attempt; the next start of
        # the same label replaces it.
        pairs, open_runs = [], {}
        for r in readings:
            if r["label"].endswith(" start"):
                open_runs[r["label"][: -len(" start")]] = r
            elif r["label"].endswith(" end") and r["label"][: -len(" end")] in open_runs:
                label = r["label"][: -len(" end")]
                pairs.append((label, open_runs.pop(label), r))
        emit("\n## Key balance per run (drop includes the production bot's spending)\n")
        emit("| run | start (UTC) | end (UTC) | key drop $ | estimated replay cost $ | minutes |")
        emit("|---|---|---|---|---|---|")
        for label, start, end in pairs:
            mine = sum(
                call_cost(u, price) for a in answers if start["time"] <= a["time"] <= end["time"]
                for u in a.get("usage", [])
            )
            minutes = (datetime.fromisoformat(end["time"]) - datetime.fromisoformat(start["time"])).total_seconds() / 60
            emit(
                f"| {label} | {start['time'][11:19]} | {end['time'][11:19]} | "
                f"{start['limit_remaining'] - end['limit_remaining']:.2f} | {mine:.2f} | {minutes:.0f} |"
            )
        baseline = _budget()["baseline"]
        last = readings[-1] if readings else baseline
        total_mine = sum(call_cost(u, price) for a in answers for u in a.get("usage", []))
        emit(
            f"\nSince the baseline ({baseline['time']}): key drop ${baseline['limit_remaining'] - last['limit_remaining']:.2f}"
            f" up to {last['time']}; estimated cost of every recorded replay call ${total_mine:.2f}."
        )

    failures = [a for a in answers if a["status"] == "failed"]
    if failures:
        emit("\n## Failed answers\n")
        for a in failures:
            emit(f"- {a['post_id']} {a['type']} {a['alias']} {a['variant']} r{a['rep']}: {a['error']}")

    with open(os.path.join(OUT, "analysis.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT, "scores.json"), "w", encoding="utf-8") as fh:
        json.dump({"log": scores, "brier": briers}, fh, indent=1)
    print(f"\nWrote {OUT}/analysis.md and {OUT}/scores.json")


def main_cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="fetch posts and comments, rebuild the research (no LLM calls)")
    sub.add_parser("check-prompts", help="print the binary prompt under each variant (no LLM calls)")
    run = sub.add_parser("run", help="request forecasts")
    run.add_argument("--models", required=True, help=f"comma-separated aliases: {', '.join(MODELS)}")
    run.add_argument("--variants", default="base", help=f"comma-separated: {', '.join(VARIANTS)}")
    run.add_argument("--types", default="binary,multiple_choice,numeric,discrete")
    run.add_argument("--posts", default="", help="comma-separated post ids")
    run.add_argument("--limit", type=int, default=0, help="at most N questions, alternating types")
    run.add_argument("--rep", type=int, default=1, help="repeat number, to ask the same configuration again")
    run.add_argument("--retry-failed", action="store_true")
    run.add_argument("--label", default="run", help="label for the balance readings")
    sub.add_parser("analyze", help="score everything recorded (no LLM calls)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%H:%M:%S")
    logger.setLevel(logging.INFO)
    os.makedirs(OUT, exist_ok=True)
    {"prepare": cmd_prepare, "check-prompts": cmd_check_prompts, "run": cmd_run, "analyze": cmd_analyze}[args.command](args)


if __name__ == "__main__":
    main_cli()
