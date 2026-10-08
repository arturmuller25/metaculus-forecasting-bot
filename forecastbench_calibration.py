"""
Measures whether the bot's binary forecast step leans toward Yes (or is
otherwise miscalibrated), and how much a market price in the prompt helps, on
the ForecastBench question set of 2026-09-27 (forecastingresearch/
forecastbench-datasets, CC BY-SA 4.0) and its resolution set.

Every candidate model's knowledge cutoff (June 2026 at the latest) precedes
the questions, so the test is clean. Nothing searches the web: each resolved
binary question gets the bot's own binary prompt (bot.py,
_run_forecast_on_binary, unchanged) with the dataset's question text,
background and resolution criteria as the question fields, the forecast due
date as "today", and as research either

  V0  a note that no research is available beyond the background, or
  V1  (market questions only) the same note plus the market or crowd value
      the dataset recorded at the set's freeze, with its source and date.

Models: claude-sonnet-5 and claude-opus-5-5 at effort high through the
Anthropic API (Message Batches, half price), and openrouter/openai/gpt-6.1-sol
at reasoning high through OpenRouter, built like main._thinker. Every answer
goes through the bot's own _parse_binary with claude-haiku-4-5 on the
Anthropic key (as in claude_direct_replay.py), so the OpenRouter key pays for
GPT-6.1 Sol only; any other model call is refused.

main.py is not imported (it may be mid-edit); the bot is built here with the
same member settings. The bot's forecast log goes to logs/forecastbench/.

Data (download into logs/forecastbench/data/ first):
  https://raw.githubusercontent.com/forecastingresearch/forecastbench-datasets/main/datasets/question_sets/2026-09-27-llm.json
  https://raw.githubusercontent.com/forecastingresearch/forecastbench-datasets/main/datasets/resolution_sets/2026-09-27_resolution_set.json

Usage:
    uv run python forecastbench_calibration.py prepare         # prompts, no LLM calls
    uv run python forecastbench_calibration.py smoke           # 5 questions, every model, plain calls
    uv run python forecastbench_calibration.py submit          # Claude batches
    uv run python forecastbench_calibration.py collect [--wait]
    uv run python forecastbench_calibration.py run-gpt         # GPT-6.1 Sol, capped on the key balance
    uv run python forecastbench_calibration.py reparse         # failed parses, from recorded text
    uv run python forecastbench_calibration.py analyze         # offline
    uv run python forecastbench_calibration.py spend
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
import sys
import time
import urllib.request
from datetime import datetime, timezone

import dotenv

dotenv.load_dotenv(".env")

import anthropic  # noqa: E402
import numpy as np  # noqa: E402

import bot as bot_module  # noqa: E402
import calibration  # noqa: E402
from bot import ForecasterBot  # noqa: E402
from forecasting_tools import BinaryQuestion, GeneralLlm  # noqa: E402
from forecasting_tools.ai_models import general_llm  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
logger = logging.getLogger("forecastbench_calibration")

SET_DATE = "2026-09-27"
TODAY = datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc)
OUT = os.path.join("logs", "forecastbench")
DATA = os.path.join(OUT, "data")
QSET = os.path.join(DATA, f"{SET_DATE}-llm.json")
RSET = os.path.join(DATA, f"{SET_DATE}_resolution_set.json")
PROMPTS = os.path.join(OUT, "prompts.jsonl")
RAW = os.path.join(OUT, "raw.jsonl")
# GPT answers (and the smoke run) go to answers.jsonl, Claude batch answers to
# answers_claude.jsonl, so two processes never append to the same file.
ANSWERS = os.path.join(OUT, "answers.jsonl")
ANSWERS_CLAUDE = os.path.join(OUT, "answers_claude.jsonl")
ANSWERS_GPT2 = os.path.join(OUT, "answers_gpt2.jsonl")  # a second GPT process (the supplement)
GPT_OUT = [ANSWERS]

# Supplement: resolved market questions from the earlier sets whose questions
# postdate 2026-06-30 (freezes 2026-07-09 to 2026-09-03), not already in the
# 2026-09-27 set and not from Metaculus (whose terms restrict using its data
# to evaluate AI models). A seeded sample keeps the spend inside the caps.
PREV = os.path.join(OUT, "data_prev")
PREV_SETS = ["2026-07-19", "2026-08-02", "2026-08-16", "2026-08-30", "2026-09-13"]
SUPP_N = 120
SUPP_SEED = 20261008
BATCHES = os.path.join(OUT, "batches.json")
OR_BUDGET = os.path.join(OUT, "openrouter_budget.json")
RESULTS = os.path.join(OUT, "results.json")
TABLES = os.path.join(OUT, "results.md")

# Replay answers must never reach the bot's own forecast log.
bot_module.FORECAST_LOG = os.path.join(OUT, "forecasts.jsonl")

MARKET_SOURCES = {"manifold", "metaculus", "polymarket", "kalshi", "infer"}
SOURCE_NAMES = {"manifold": "Manifold", "metaculus": "Metaculus", "polymarket": "Polymarket",
                "kalshi": "Kalshi", "infer": "INFER"}

PARSER = "anthropic/claude-haiku-4-5"
CLAUDE = {"s5": "claude-sonnet-5", "o55": "claude-opus-5-5"}
GPT_ALIAS, GPT_MODEL = "sol61", "openrouter/openai/gpt-6.1-sol"
EFFORT = "high"
MAX_TOKENS = 16000  # thinking plus answer, as in claude_direct_replay.py
GPT_TIMEOUT = 300  # production uses 120; a longer one keeps slow valid answers instead of paying a retry
IN_FLIGHT = 6

ANTHROPIC_CAP = 25.0  # dollars, estimated from token usage at batch prices
OPENROUTER_CAP = 10.0  # dollars of drop in the key's limit_remaining
OPENROUTER_STOP = 9.3  # no new GPT call past this drop; calls in flight finish under the cap

# Anthropic list prices in dollars per million tokens (input, output), as in
# claude_direct_replay.py (2026-10-08). Batches cost half.
PRICES = {
    "claude-sonnet-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
BATCH_SHARE = 0.5
OUTPUT_GUESS = 4000  # output tokens per answer assumed before any is measured
GPT_PRICE_FALLBACK = (2.0, 10.0)  # OpenRouter list price per million, logs/replay_r1/prices.json

NO_RESEARCH = "No research is available for this question beyond the background above."


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def append_jsonl(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def read_answers() -> list[dict]:
    return read_jsonl(ANSWERS) + read_jsonl(ANSWERS_CLAUDE) + read_jsonl(ANSWERS_GPT2)


# ---------------------------------------------------------------------------
# Questions and prompts
# ---------------------------------------------------------------------------


def _set_rows(qpath: str, rpath: str, set_date: str, markets_only: bool = False) -> list[dict]:
    """Every resolved binary question of one set (single questions only), with its outcome."""
    with open(qpath, encoding="utf-8") as fh:
        qset = json.load(fh)
    with open(rpath, encoding="utf-8") as fh:
        rset = json.load(fh)
    if qset.get("forecast_due_date") != set_date or rset.get("forecast_due_date") != set_date:
        raise SystemExit(f"question or resolution set is not the {set_date} one")
    by_key = {(q["source"], str(q["id"])): q for q in qset["questions"]}
    rows = []
    for res in rset["resolutions"]:
        if not res.get("resolved") or isinstance(res["id"], list) or res.get("direction"):
            continue
        if markets_only and res["source"] not in MARKET_SOURCES:
            continue
        y = float(res["resolved_to"])
        if y not in (0.0, 1.0):
            continue
        q = by_key.get((res["source"], str(res["id"])))
        if q is None:
            print(f"  no question for resolution {res['source']} {res['id']}")
            continue
        date = res["resolution_date"]

        def fill(text: str, date: str = date) -> str:
            return (text or "").replace("{resolution_date}", date).replace("{forecast_due_date}", set_date)

        kind = "market" if res["source"] in MARKET_SOURCES else "dataset"
        criteria = fill(q["resolution_criteria"])
        extra = (q.get("market_info_resolution_criteria") or "N/A").strip()
        if extra and extra != "N/A":
            criteria += "\n\n" + fill(extra)
        fine_print = ""
        market = None
        if kind == "market":
            close = (q.get("market_info_close_datetime") or "N/A").strip()
            if close and close != "N/A":
                fine_print = f"The market closes on {close[:10]}."
            what = (q.get("freeze_datetime_value_explanation") or "The market value.").strip().rstrip(".")
            market = {
                "value": float(q["freeze_datetime_value"]),
                "date": q["freeze_datetime"][:10],
                "what": what[0].lower() + what[1:],
                "source_name": SOURCE_NAMES.get(res["source"], res["source"]),
            }
        key = f"{res['source']}:{res['id']}:{date}"
        if set_date != SET_DATE:
            key = f"{set_date}:{key}"
        rows.append({
            "qid": hashlib.sha1(key.encode("utf-8")).hexdigest()[:10],
            "key": key,
            "set": set_date,
            "group": "main" if set_date == SET_DATE else "supp",
            "id": str(res["id"]),
            "source": res["source"],
            "kind": kind,
            "resolution_date": date,
            "outcome": int(y),
            "question": fill(q["question"]),
            "background": fill(q.get("background") or ""),
            "criteria": criteria,
            "fine_print": fine_print,
            "url": q.get("url") or "",
            "market": market,
        })
    rows.sort(key=lambda r: (r["kind"], r["source"], r["qid"]))
    return rows


def load_rows() -> list[dict]:
    """The 2026-09-27 set's resolved questions."""
    rows = _set_rows(QSET, RSET, SET_DATE)
    for i, r in enumerate(rows, start=1):
        r["index"] = i
    return rows


def load_supp_rows(main_rows: list[dict]) -> list[dict]:
    """The supplement: a seeded sample of SUPP_N resolved market questions from the earlier clean sets,
    each market once (its latest set), none already resolved in the 2026-09-27 set, none from Metaculus."""
    taken = {(r["source"], r["id"]) for r in main_rows if r["kind"] == "market"}
    latest: dict[tuple[str, str], dict] = {}
    for d in PREV_SETS:
        for r in _set_rows(os.path.join(PREV, f"{d}-llm.json"), os.path.join(PREV, f"{d}_resolution_set.json"),
                           d, markets_only=True):
            mid = (r["source"], r["id"])
            if r["source"] == "metaculus" or mid in taken:
                continue
            latest[mid] = r  # sets are read oldest first, so the newest version wins
    pool = sorted(latest.values(), key=lambda r: r["qid"])
    rng = np.random.default_rng(SUPP_SEED)
    picked = sorted((pool[i] for i in rng.choice(len(pool), size=min(SUPP_N, len(pool)), replace=False)),
                    key=lambda r: (r["source"], r["qid"]))
    for i, r in enumerate(picked, start=10001):
        r["index"] = i
    return picked


def all_rows() -> list[dict]:
    main = load_rows()
    if not os.path.exists(PREV):
        return main
    return main + load_supp_rows(main)


def today_of(row: dict) -> datetime:
    y, m, d = (int(x) for x in row.get("set", SET_DATE).split("-"))
    return datetime(y, m, d, 12, 0, tzinfo=timezone.utc)


def research_text(row: dict, variant: str) -> str:
    if variant == "V0":
        return NO_RESEARCH
    m = row["market"]
    return (
        "No research is available for this question beyond the background above, except one data point:\n"
        f"{m['source_name']}, {m['what']} on {m['date']}: {100 * m['value']:.1f}% Yes "
        f"(recorded by the ForecastBench question set at its freeze on {m['date']}; source page {row['url']})."
    )


def variants_of(row: dict) -> list[str]:
    return ["V0", "V1"] if row["kind"] == "market" else ["V0"]


def question_object(row: dict) -> BinaryQuestion:
    return BinaryQuestion(
        question_text=row["question"],
        id_of_post=row["index"],
        id_of_question=row["index"],
        page_url=row["url"] or f"forecastbench:{row['key']}",
        background_info=row["background"],
        resolution_criteria=row["criteria"],
        fine_print=row["fine_print"],
    )


_TODAY: contextvars.ContextVar[datetime] = contextvars.ContextVar("fb_today", default=TODAY)


class _FixedDatetime(datetime):
    """datetime whose now() is the question's set forecast due date, so the prompt says 'Today is <that date>'."""

    @classmethod
    def now(cls, tz=None):  # noqa: D401
        day = _TODAY.get()
        return day.replace(tzinfo=None) if tz is None else day.astimezone(tz)


bot_module.datetime = _FixedDatetime


def make_bot(member) -> ForecasterBot:
    """A bot with one ensemble member, no shadows, no research, and the Haiku parser."""
    parser = GeneralLlm(model=PARSER, temperature=0.0, timeout=60, allowed_tries=2)
    bot = ForecasterBot(
        ensemble=[member],
        shadows=[],
        research_reports_per_question=1,
        predictions_per_research_report=1,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=False,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=False,
        extra_metadata_in_explanation=True,
        llms={"default": parser, "researcher": parser, "parser": parser, "summarizer": parser},
    )
    assert bot.get_llm("parser", "llm").model == PARSER
    return bot


class Capture:
    """Stands in for a model to read the prompt the bot builds; no call is made."""

    model = "capture"

    async def invoke(self, prompt: str) -> str:
        self.prompt = prompt
        raise RuntimeError("captured")


async def build_prompt(row: dict, variant: str) -> str:
    capture = Capture()
    _TODAY.set(today_of(row))
    try:
        await make_bot(capture)._run_forecast_on_binary(question_object(row), research_text(row, variant))
    except RuntimeError:
        pass
    return capture.prompt


def load_prompts() -> dict[tuple[str, str], dict]:
    prompts = {(p["qid"], p["variant"]): p for p in read_jsonl(PROMPTS)}
    if not prompts:
        raise SystemExit(f"{PROMPTS} is missing: run prepare first")
    return prompts


def cmd_prepare(_args) -> None:
    rows = all_rows()
    level = bot_module.logger.level
    bot_module.logger.setLevel(logging.ERROR)  # every capture logs a failed member

    async def go() -> list[tuple[dict, str, str]]:
        return [(r, v, await build_prompt(r, v)) for r in rows for v in variants_of(r)]

    built = asyncio.run(go())
    bot_module.logger.setLevel(level)
    if os.path.exists(PROMPTS):
        os.remove(PROMPTS)
    for row, variant, prompt in built:
        append_jsonl(PROMPTS, {"qid": row["qid"], "variant": variant, "sha1": sha1(prompt),
                               "chars": len(prompt), "prompt": prompt})
    with open(os.path.join(OUT, "questions.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1)
    kinds: dict[str, list[int]] = {}
    for r in rows:
        kinds.setdefault(r["source"], []).append(r["outcome"])
    print(f"Resolved binary questions: {len(rows)} "
          f"({sum(r['kind'] == 'market' for r in rows)} market, {sum(r['kind'] == 'dataset' for r in rows)} dataset); "
          f"Yes rate {statistics.mean(r['outcome'] for r in rows):.3f}")
    for s, ys in sorted(kinds.items()):
        print(f"  {s:10s} n={len(ys):3d} Yes={sum(ys):3d}")
    chars = [len(p) for _, _, p in built]
    print(f"Prompts: {len(built)}, chars median {statistics.median(chars):.0f}, max {max(chars)}")
    sample = next(p for r, v, p in built if r["kind"] == "market" and v == "V1")
    start = sample.find("Your research assistant says:")
    print("\n--- V1 research block of one market prompt ---\n" + sample[start:start + 600])


# ---------------------------------------------------------------------------
# LLM calls: guard, usage, parse
# ---------------------------------------------------------------------------


class BudgetStop(RuntimeError):
    pass


GPT_STOP = asyncio.Event()
_CALL: contextvars.ContextVar[dict | None] = contextvars.ContextVar("fb_call", default=None)
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
    return row


@functools.wraps(_real_acompletion)
async def _guarded_acompletion(*args, **kwargs):
    """Every LLM call the bot makes goes through here: only the Haiku parser and GPT-6.1 Sol pass."""
    model = str(kwargs.get("model") or (args[0] if args else ""))
    if model != GPT_MODEL and not model.startswith(PARSER):
        raise RuntimeError(f"refused a call to {model}: not the parser or GPT-6.1 Sol")
    if model == GPT_MODEL and GPT_STOP.is_set():
        raise BudgetStop("OpenRouter cap reached")
    response = await _real_acompletion(*args, **kwargs)
    call = _CALL.get()
    if call is not None:
        call.setdefault("usage", []).append(_usage_row(model, response))
    return response


general_llm.acompletion = _guarded_acompletion


class _WarningCatcher(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        call = _CALL.get()
        if call is not None and record.levelno >= logging.WARNING:
            call.setdefault("warnings", []).append(record.getMessage()[:300])


bot_module.logger.addHandler(_WarningCatcher())

_PROB_RE = re.compile(r"Probability:\s*\**\s*([0-9]+(?:\.[0-9]+)?)\s*%", re.IGNORECASE)


def regex_probability(text: str) -> float | None:
    """The last 'Probability: ZZ%' in the answer, clamped like the bot; a cross-check on the parser."""
    found = _PROB_RE.findall(text or "")
    if not found:
        return None
    return max(0.01, min(0.99, float(found[-1]) / 100))


class CannedLlm:
    """Returns an answer already obtained from the model, after checking the bot built the same prompt."""

    def __init__(self, model: str, text: str, expected_sha1: str) -> None:
        self.model = model
        self._text = text
        self._expected = expected_sha1

    async def invoke(self, prompt: str) -> str:
        got = sha1(prompt)
        if got != self._expected:
            raise RuntimeError(f"prompt mismatch: sent {self._expected}, rebuilt {got}")
        return self._text


class GptMember:
    """GPT-6.1 Sol built like main._thinker (temperature 0.3, reasoning high), keeping the raw answer."""

    model = GPT_MODEL

    def __init__(self, expected_sha1: str) -> None:
        self._llm = GeneralLlm(model=GPT_MODEL, temperature=0.3, timeout=GPT_TIMEOUT, allowed_tries=2,
                               reasoning_effort=EFFORT)
        self._expected = expected_sha1

    async def invoke(self, prompt: str) -> str:
        if sha1(prompt) != self._expected:
            raise RuntimeError(f"prompt mismatch: expected {self._expected}, built {sha1(prompt)}")
        call = _CALL.get()
        started = time.monotonic()
        try:
            text = await self._llm.invoke(prompt)
        finally:
            if call is not None:
                call["seconds"] = round(time.monotonic() - started, 1)
        if call is not None:
            call["raw"] = text
        return text


async def run_bot(row: dict, variant: str, member) -> tuple[str, float | None, str | None]:
    """The bot's binary step with one member; returns (status, prediction, error)."""
    _TODAY.set(today_of(row))
    try:
        result = await make_bot(member)._run_forecast_on_binary(question_object(row), research_text(row, variant))
        return "ok", float(result.prediction_value), None
    except Exception as exc:  # noqa: BLE001
        return "failed", None, f"{type(exc).__name__}: {str(exc)[:300]}"


async def finish_claude(row: dict, alias: str, variant: str, run: str, route: str, expected: str,
                        resp: dict | None, error: str | None) -> dict:
    """Parses a Claude answer with the bot's own parser and records it."""
    call: dict = {"usage": [resp["usage"]] if resp and resp.get("usage") else []}
    _CALL.set(call)
    status, prediction = "failed", None
    if error is None and resp is not None and not resp["text"].strip():
        error = f"empty answer (stop_reason {resp['stop_reason']})"
    if error is None:
        status, prediction, error = await run_bot(row, variant, CannedLlm(f"anthropic/{CLAUDE[alias]}", resp["text"], expected))
    out = {
        "time": _utc_now(), "qid": row["qid"], "source": row["source"], "kind": row["kind"],
        "alias": alias, "model": CLAUDE[alias], "variant": variant, "run": run, "route": route,
        "status": status, "error": error, "prediction": prediction,
        "regex": regex_probability(resp["text"]) if resp else None,
        "stop_reason": resp.get("stop_reason") if resp else None,
        "prompt_sha1": expected, "raw": resp["text"] if resp else None,
        "usage": call.get("usage", []), "warnings": call.get("warnings"),
    }
    append_jsonl(ANSWERS_CLAUDE, out)
    print(f"  {row['qid']} {row['source']:10s} {alias} {variant} {run}: {status} "
          f"{prediction if status == 'ok' else error} (regex {out['regex']})")
    return out


# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------


def anthropic_cost(u: dict) -> float:
    model = (u.get("model") or "").removeprefix("anthropic/")
    if model not in PRICES:
        return 0.0
    pin, pout = PRICES[model]
    share = BATCH_SHARE if u.get("batch") else 1.0
    return share * ((u.get("prompt_tokens") or 0) * pin + (u.get("completion_tokens") or 0) * pout) / 1e6


def anthropic_spent() -> float:
    """Every Claude response (raw.jsonl) plus every Haiku parser call recorded."""
    total = sum(anthropic_cost(r["usage"]) for r in read_jsonl(RAW) if r.get("usage") and r.get("alias") in CLAUDE)
    for a in read_answers():
        total += sum(anthropic_cost(u) for u in a.get("usage", []) if (u.get("model") or "").startswith(PARSER))
    return total


def gpt_price() -> tuple[float, float]:
    path = os.path.join(OUT, "gpt_price.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return tuple(json.load(fh))
    return GPT_PRICE_FALLBACK


def gpt_estimated_spent() -> float:
    pin, pout = gpt_price()
    total = 0.0
    for a in read_answers():
        for u in a.get("usage", []):
            if u.get("model") == GPT_MODEL:
                total += ((u.get("prompt_tokens") or 0) * pin + (u.get("completion_tokens") or 0) * pout) / 1e6
    return total


def load_batches() -> list[dict]:
    if not os.path.exists(BATCHES):
        return []
    with open(BATCHES, encoding="utf-8") as fh:
        return json.load(fh)


def save_batches(batches: list[dict]) -> None:
    with open(BATCHES, "w", encoding="utf-8") as fh:
        json.dump(batches, fh, indent=1)


def pending_estimate() -> float:
    return sum(b["estimate"] for b in load_batches() if not b.get("collected"))


def claude_estimate(model: str, prompt: str, batch: bool) -> float:
    measured = [r["usage"]["completion_tokens"] for r in read_jsonl(RAW)
                if r.get("usage") and r.get("model") == model]
    out = 1.5 * statistics.mean(measured) if measured else OUTPUT_GUESS
    pin, pout = PRICES[model]
    parser = 2 * (1500 * 1.0 + 40 * 5.0) / 1e6  # two Haiku calls per parse
    return (BATCH_SHARE if batch else 1.0) * (len(prompt) / 3 * pin + out * pout) / 1e6 + parser


def check_anthropic_cap(extra: float, what: str) -> None:
    done, pending = anthropic_spent(), pending_estimate()
    print(f"Anthropic: spent ${done:.2f}, pending batches up to ${pending:.2f}, {what} up to ${extra:.2f} (cap ${ANTHROPIC_CAP:.0f})")
    if done + pending + extra > ANTHROPIC_CAP:
        raise SystemExit("Refused: this could take the Anthropic estimate past the cap.")


def _openrouter(path: str) -> dict:
    req = urllib.request.Request(
        f"https://openrouter.ai/api/v1{path}",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def key_reading(label: str) -> tuple[float, float]:
    """Records the key's limit_remaining; returns (remaining, drop since the baseline)."""
    remaining = float(_openrouter("/key")["data"]["limit_remaining"])
    data = {}
    if os.path.exists(OR_BUDGET):
        with open(OR_BUDGET, encoding="utf-8") as fh:
            data = json.load(fh)
    if "baseline" not in data:
        data["baseline"] = {"time": _utc_now(), "limit_remaining": remaining}
    data.setdefault("readings", []).append({"time": _utc_now(), "label": label, "limit_remaining": remaining})
    with open(OR_BUDGET, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    return remaining, float(data["baseline"]["limit_remaining"]) - remaining


def save_gpt_price() -> None:
    for m in _openrouter("/models")["data"]:
        if m["id"] == GPT_MODEL.removeprefix("openrouter/"):
            p = m.get("pricing") or {}
            price = (float(p.get("prompt") or 0) * 1e6, float(p.get("completion") or 0) * 1e6)
            with open(os.path.join(OUT, "gpt_price.json"), "w", encoding="utf-8") as fh:
                json.dump(price, fh)
            print(f"GPT-6.1 Sol list price: ${price[0]:.2f} in, ${price[1]:.2f} out per million tokens")
            return
    print("GPT-6.1 Sol not in the model list; using the fallback price")


# ---------------------------------------------------------------------------
# Claude: plain calls (smoke) and batches
# ---------------------------------------------------------------------------


def request_params(model: str, prompt: str) -> dict:
    """Adaptive thinking plus output_config.effort, as claude_direct_replay.py sends (no temperature)."""
    return {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": EFFORT},
        "messages": [{"role": "user", "content": prompt}],
    }


def response_row(msg, model: str, batch: bool, seconds: float | None = None) -> dict:
    usage = msg.usage.model_dump()
    thinking = (usage.get("output_tokens_details") or {}).get("thinking_tokens")
    return {
        "served_model": msg.model,
        "stop_reason": msg.stop_reason,
        "text": "".join(b.text for b in msg.content if b.type == "text"),
        "seconds": None if seconds is None else round(seconds, 1),
        "usage": {
            "model": f"anthropic/{model}",
            "id": msg.id,
            "prompt_tokens": (usage.get("input_tokens") or 0)
            + (usage.get("cache_creation_input_tokens") or 0)
            + (usage.get("cache_read_input_tokens") or 0),
            "completion_tokens": usage.get("output_tokens") or 0,
            "reasoning_tokens": thinking,
            "batch": batch,
        },
    }


def jobs_for(rows: list[dict], alias: str, run: str) -> list[tuple[dict, str]]:
    done = {(a["qid"], a["variant"]) for a in read_answers()
            if a["alias"] == alias and a["run"] == run and a["status"] == "ok"}
    return [(r, v) for r in rows for v in variants_of(r) if (r["qid"], v) not in done]


def smoke_rows(rows: list[dict]) -> list[dict]:
    """Two market questions (one Yes, one No) and three dataset questions from different sources."""
    picked: list[dict] = []
    for want in (("market", 1), ("market", 0)):
        picked.append(next(r for r in rows if (r["kind"], r["outcome"]) == want and r not in picked))
    for source in ("fred", "yfinance", "wikipedia"):
        picked.append(next(r for r in rows if r["source"] == source))
    return picked


async def claude_plain(rows: list[dict], run: str) -> None:
    prompts = load_prompts()
    client = anthropic.AsyncAnthropic(max_retries=3, timeout=900)
    sem = asyncio.Semaphore(IN_FLIGHT)

    async def one(row: dict, alias: str, variant: str) -> None:
        model = CLAUDE[alias]
        p = prompts[(row["qid"], variant)]
        resp, error = None, None
        async with sem:
            try:
                started = time.monotonic()
                msg = await client.messages.create(**request_params(model, p["prompt"]))
                resp = response_row(msg, model, batch=False, seconds=time.monotonic() - started)
            except anthropic.APIError as exc:
                error = f"{type(exc).__name__}: {str(exc)[:300]}"
            append_jsonl(RAW, {"time": _utc_now(), "custom_id": f"{alias}_{variant}_{row['qid']}", "batch_id": None,
                               "alias": alias, "model": model, "variant": variant, "run": run, "qid": row["qid"],
                               "prompt_sha1": p["sha1"], "error": error, **(resp or {})})
            await finish_claude(row, alias, variant, run, "direct", p["sha1"], resp, error)

    await asyncio.gather(*(one(r, a, v) for a in CLAUDE for r, v in
                           [(r, v) for r in rows for v in variants_of(r)]))


def cmd_submit(args) -> None:
    rows = all_rows()
    prompts = load_prompts()
    aliases = [a.strip() for a in args.models.split(",") if a.strip()]
    batches = load_batches()
    pending = {(meta["qid"], meta["variant"], b["alias"]) for b in batches if not b.get("collected")
               for meta in b["requests"].values()}
    client = anthropic.Anthropic(max_retries=3)
    for alias in aliases:
        model = CLAUDE[alias]
        todo = [(r, v) for r, v in jobs_for(rows, alias, "main") if (r["qid"], v, alias) not in pending]
        if args.limit:
            todo = todo[: args.limit]
        if not todo:
            print(f"{alias}: nothing to request")
            continue
        extra = sum(claude_estimate(model, prompts[(r["qid"], v)]["prompt"], batch=True) for r, v in todo)
        check_anthropic_cap(extra, f"batch {alias} ({len(todo)} requests)")
        requests = [{"custom_id": f"{alias}_{v}_{r['qid']}", "params": request_params(model, prompts[(r["qid"], v)]["prompt"])}
                    for r, v in todo]
        batch = client.messages.batches.create(requests=requests)
        batches.append({
            "id": batch.id, "alias": alias, "created": _utc_now(), "estimate": round(extra, 4),
            "requests": {req["custom_id"]: {"qid": r["qid"], "variant": v, "sha1": prompts[(r["qid"], v)]["sha1"]}
                         for req, (r, v) in zip(requests, todo)},
            "collected": False,
        })
        save_batches(batches)
        print(f"{alias}: batch {batch.id} with {len(requests)} requests ({batch.processing_status})")


def cmd_collect(args) -> None:
    client = anthropic.Anthropic(max_retries=3)
    rows = {r["qid"]: r for r in all_rows()}
    while True:
        batches = load_batches()
        open_batches = [b for b in batches if not b.get("collected")]
        if not open_batches:
            print("Every batch is collected.")
            break
        for b in open_batches:
            info = client.messages.batches.retrieve(b["id"])
            counts = info.request_counts
            if info.processing_status != "ended":
                print(f"{_utc_now()} {b['alias']} {b['id']}: {info.processing_status}, {counts.processing} processing, "
                      f"{counts.succeeded} succeeded, {counts.errored} errored")
                continue
            model = CLAUDE[b["alias"]]
            items = []
            for result in client.messages.batches.results(b["id"]):
                meta = b["requests"][result.custom_id]
                resp, error = None, None
                if result.result.type == "succeeded":
                    resp = response_row(result.result.message, model, batch=True)
                else:
                    error = f"batch result {result.result.type}: {str(getattr(result.result, 'error', None))[:300]}"
                append_jsonl(RAW, {"time": _utc_now(), "custom_id": result.custom_id, "batch_id": b["id"],
                                   "alias": b["alias"], "model": model, "variant": meta["variant"], "run": "main",
                                   "qid": meta["qid"], "prompt_sha1": meta["sha1"], "error": error, **(resp or {})})
                items.append((rows[meta["qid"]], meta["variant"], meta["sha1"], resp, error))

            async def parse_all(items=items, b=b) -> None:
                sem = asyncio.Semaphore(IN_FLIGHT)

                async def one(row, variant, expected, resp, error):
                    async with sem:
                        await finish_claude(row, b["alias"], variant, "main", "batch", expected, resp, error)

                await asyncio.gather(*(one(*item) for item in items))

            asyncio.run(parse_all())
            b["collected"] = True
            b["ended"] = info.ended_at.isoformat() if info.ended_at else None
            b["counts"] = {"succeeded": counts.succeeded, "errored": counts.errored,
                           "expired": counts.expired, "canceled": counts.canceled}
            save_batches(batches)
            print(f"{b['alias']}: collected {len(items)} results; Anthropic spent so far ${anthropic_spent():.2f}")
        if not args.wait:
            break
        if not all(b.get("collected") for b in load_batches()):
            time.sleep(60)


# ---------------------------------------------------------------------------
# GPT-6.1 Sol through OpenRouter
# ---------------------------------------------------------------------------


async def _monitor(interval: float = 20.0) -> None:
    while not GPT_STOP.is_set():
        try:
            remaining, drop = await asyncio.to_thread(key_reading, "monitor")
            est = gpt_estimated_spent()
            if drop >= OPENROUTER_STOP or est >= OPENROUTER_STOP:
                print(f"OpenRouter cap reached: key dropped ${drop:.2f} (estimate ${est:.2f}). Stopping.")
                GPT_STOP.set()
                return
        except Exception as exc:  # noqa: BLE001
            print(f"Could not read the key balance: {exc}")
        await asyncio.sleep(interval)


async def gpt_jobs(jobs: list[tuple[dict, str]], run: str) -> None:
    prompts = load_prompts()
    queue: asyncio.Queue = asyncio.Queue()
    for job in jobs:
        queue.put_nowait(job)
    done = [0]

    async def worker() -> None:
        while not queue.empty() and not GPT_STOP.is_set():
            row, variant = queue.get_nowait()
            p = prompts[(row["qid"], variant)]
            call: dict = {}
            _CALL.set(call)
            status, prediction, error = await run_bot(row, variant, GptMember(p["sha1"]))
            if status != "ok" and GPT_STOP.is_set() and "raw" not in call:
                status = "stopped"
            out = {
                "time": _utc_now(), "qid": row["qid"], "source": row["source"], "kind": row["kind"],
                "alias": GPT_ALIAS, "model": GPT_MODEL, "variant": variant, "run": run, "route": "openrouter",
                "status": status, "error": error, "prediction": prediction,
                "regex": regex_probability(call.get("raw") or ""), "seconds": call.get("seconds"),
                "prompt_sha1": p["sha1"], "raw": call.get("raw"),
                "usage": call.get("usage", []), "warnings": call.get("warnings"),
            }
            append_jsonl(GPT_OUT[0], out)
            done[0] += 1
            gpt_usage = [u for u in out["usage"] if u.get("model") == GPT_MODEL]
            toks = f"{sum(u.get('prompt_tokens') or 0 for u in gpt_usage)}/{sum(u.get('completion_tokens') or 0 for u in gpt_usage)}"
            print(f"  [{done[0]}/{len(jobs)}] {row['qid']} {row['source']:10s} sol61 {variant}: {status} "
                  f"{prediction if status == 'ok' else error} (regex {out['regex']}, {out['seconds']} s, tokens {toks})")

    monitor = asyncio.create_task(_monitor())
    await asyncio.gather(*(worker() for _ in range(IN_FLIGHT)))
    monitor.cancel()


def run_gpt(rows: list[dict]) -> None:
    jobs = jobs_for(rows, GPT_ALIAS, "main")
    remaining, drop = key_reading("run-gpt start")
    est = gpt_estimated_spent()
    print(f"OpenRouter key: ${drop:.2f} dropped since the baseline (stop at ${OPENROUTER_STOP:.2f}, cap ${OPENROUTER_CAP:.0f}); "
          f"estimate from tokens ${est:.2f}; {len(jobs)} GPT answers to request")
    if drop >= OPENROUTER_STOP or est >= OPENROUTER_STOP:
        raise SystemExit("OpenRouter cap already reached; nothing run.")
    if not jobs:
        return
    save_gpt_price()
    asyncio.run(gpt_jobs(jobs, "main"))
    remaining, drop = key_reading("run-gpt end")
    print(f"OpenRouter key: ${drop:.2f} dropped since the baseline; estimate from tokens ${gpt_estimated_spent():.2f}")


def cmd_run_gpt(args) -> None:
    rows = all_rows()
    if args.group:
        rows = [r for r in rows if r["group"] == args.group]
    if args.limit:
        rows = rows[: args.limit]
    if args.second:
        # A second process at the same time: its own answers file, disjoint rows (use --group).
        GPT_OUT[0] = ANSWERS_GPT2
    run_gpt(rows)


def cmd_smoke(_args) -> None:
    rows = smoke_rows(load_rows())
    prompts = load_prompts()
    extra = sum(claude_estimate(CLAUDE[a], prompts[(r["qid"], v)]["prompt"], batch=False)
                for a in CLAUDE for r in rows for v in variants_of(r))
    check_anthropic_cap(extra, "smoke calls")
    print("Smoke questions: " + ", ".join(f"{r['qid']} {r['source']} y={r['outcome']}" for r in rows))
    asyncio.run(claude_plain(rows, "smoke"))
    run_gpt(rows)
    cmd_spend(None)


def cmd_reparse(_args) -> None:
    """Re-runs the bot's parse on recorded answers whose parse failed (no new forecast calls)."""
    rows = {r["qid"]: r for r in all_rows()}
    answers = read_answers()
    ok = {(a["qid"], a["alias"], a["variant"], a["run"]) for a in answers if a["status"] == "ok"}
    todo = {}
    for a in answers:
        k = (a["qid"], a["alias"], a["variant"], a["run"])
        if a["status"] == "failed" and a.get("raw") and k not in ok:
            todo[k] = a
    print(f"{len(todo)} answers to re-parse")

    async def go() -> None:
        for (qid, alias, variant, run), a in todo.items():
            call: dict = {}
            _CALL.set(call)
            model = GPT_MODEL if alias == GPT_ALIAS else f"anthropic/{CLAUDE[alias]}"
            status, prediction, error = await run_bot(rows[qid], variant, CannedLlm(model, a["raw"], a["prompt_sha1"]))
            new = {**a, "time": _utc_now(), "status": status, "prediction": prediction, "error": error,
                   "reparsed": True, "usage": call.get("usage", []), "warnings": call.get("warnings")}
            append_jsonl(ANSWERS if alias == GPT_ALIAS else ANSWERS_CLAUDE, new)
            print(f"  {qid} {alias} {variant}: {status} {prediction if status == 'ok' else error}")

    asyncio.run(go())


def cmd_spend(_args) -> None:
    raws = [r for r in read_jsonl(RAW) if r.get("usage")]
    by: dict[str, float] = {}
    for r in raws:
        k = f"{r['alias']} {r['run']} {'batch' if r['usage'].get('batch') else 'direct'}"
        by[k] = by.get(k, 0.0) + anthropic_cost(r["usage"])
    parser = sum(anthropic_cost(u) for a in read_answers() for u in a.get("usage", [])
                 if (u.get("model") or "").startswith(PARSER))
    for k, v in sorted(by.items()):
        print(f"  {k}: ${v:.3f}")
    print(f"  Haiku parser: ${parser:.3f}")
    print(f"Anthropic estimate: ${anthropic_spent():.2f} (pending batches up to ${pending_estimate():.2f})")
    if os.path.exists(OR_BUDGET):
        with open(OR_BUDGET, encoding="utf-8") as fh:
            data = json.load(fh)
        last = data["readings"][-1]
        drop = data["baseline"]["limit_remaining"] - last["limit_remaining"]
        print(f"OpenRouter: key dropped ${drop:.2f} since {data['baseline']['time']} (last reading {last['time']}); "
              f"estimate from GPT tokens ${gpt_estimated_spent():.2f}")


# ---------------------------------------------------------------------------
# Analysis (offline)
# ---------------------------------------------------------------------------

LABELS = {"s5": "Claude Sonnet 5", "o55": "Claude Opus 5.5", "sol61": "GPT-6.1 Sol",
          "pair": "Par Sol 6.1 + Sonnet 5", "pair_o55": "Par Sol 6.1 + Opus 5.5"}
PAIRS = {"pair": ("sol61", "s5"), "pair_o55": ("sol61", "o55")}
ORDER = ["s5", "o55", "sol61", "pair", "pair_o55"]
BOOT = 4000
BOOT_SEED = 20261008  # each bootstrap starts from this seed, so a CI does not depend on what ran before it


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(p, 0.01, 0.99)


def _logit(p: np.ndarray) -> np.ndarray:
    p = _clip(p)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def brier(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (p - y) ** 2


def logscore(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Natural log of the probability given to what happened (0 is perfect, higher is better)."""
    p = _clip(p)
    return np.where(y == 1, np.log(p), np.log(1 - p))


def fit_ab(p: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    """Maximum-likelihood Platt fit y ~ sigmoid(A*logit(p) + B), Newton steps; None if it diverges."""
    x = _logit(p)
    X = np.column_stack([x, np.ones_like(x)])

    def loglik(w: np.ndarray) -> float:
        z = X @ w
        return float(np.sum(y * z - np.logaddexp(0, z)))

    w = np.array([1.0, 0.0])
    cur = loglik(w)
    for _ in range(200):
        q = _sigmoid(X @ w)
        g = X.T @ (y - q)
        H = (X.T * (q * (1 - q))) @ X + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, g)
        t = 1.0
        while True:  # damped Newton: halve the step until the likelihood does not fall
            new_w = w + t * step
            new = loglik(new_w)
            if new >= cur - 1e-12 or t < 1e-8:
                break
            t /= 2
        moved = np.max(np.abs(new_w - w))
        w, cur = new_w, new
        if moved < 1e-9:
            break
        if np.max(np.abs(w)) > 100:
            return None
    return float(w[0]), float(w[1])


def fit_shift(p: np.ndarray, y: np.ndarray) -> float | None:
    """Log-odds shift c with the slope held at 1: y ~ sigmoid(logit(p) + c). Negative c = forecasts lean Yes."""
    x = _logit(p)
    c = 0.0
    for _ in range(100):
        q = _sigmoid(x + c)
        step = np.sum(y - q) / max(np.sum(q * (1 - q)), 1e-9)
        c += step
        if abs(step) < 1e-10:
            break
        if abs(c) > 50:
            return None
    return float(c)


def boot_ci(stat, n: int, level: float = 0.90) -> tuple[float, float]:
    vals = []
    rng = np.random.default_rng(BOOT_SEED)
    for _ in range(BOOT):
        idx = rng.integers(0, n, n)
        v = stat(idx)
        if v is not None and np.isfinite(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [50 * (1 - level), 100 - 50 * (1 - level)])
    return float(lo), float(hi)


def predictions() -> dict[tuple[str, str], dict[str, float]]:
    """(alias, variant) -> qid -> forecast, main runs only, the latest successful answer of each."""
    out: dict[tuple[str, str], dict[str, float]] = {}
    for a in read_answers():
        if a["run"] == "main" and a["status"] == "ok":
            out.setdefault((a["alias"], a["variant"]), {})[a["qid"]] = float(a["prediction"])
    for name, (m1, m2) in PAIRS.items():
        for v in ("V0", "V1"):
            d1, d2 = out.get((m1, v), {}), out.get((m2, v), {})
            out[(name, v)] = {q: (d1[q] + d2[q]) / 2 for q in set(d1) & set(d2)}
    return out


def cmd_analyze(_args) -> None:
    rows = {r["qid"]: r for r in all_rows()}
    preds = predictions()
    res: dict = {"set": SET_DATE, "n_rows": len(rows)}
    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    present = [m for m in ORDER if preds.get((m, "V0"))]
    singles = [m for m in present if m not in PAIRS]
    # Questions every single model answered under V0; every table uses these.
    main_q = {q for q in rows if rows[q]["group"] == "main"}
    supp_q = {q for q in rows if rows[q]["group"] == "supp"}
    common = set(main_q)
    common_supp = set(supp_q)
    for m in singles:
        common &= set(preds[(m, "V0")])
        common_supp &= set(preds[(m, "V0")])
    missing = {m: sorted((main_q | supp_q) - set(preds[(m, "V0")])) for m in singles}
    emit(f"# ForecastBench {SET_DATE}: {len(main_q)} resolved binary questions; {len(common)} answered by every model under V0")
    emit(f"- Suplemento (perguntas de mercado de conjuntos anteriores): {len(supp_q)}; {len(common_supp)} respondidas por todos os modelos em V0")
    for m, miss in missing.items():
        if miss:
            emit(f"- {LABELS[m]} missing V0 answers on {len(miss)}: {', '.join(miss[:8])}{' ...' if len(miss) > 8 else ''}")
    res["common_v0"] = len(common)

    # Parser cross-check: the bot's Haiku parse against the last 'Probability: ZZ%' of each answer.
    diffs = [abs(a["prediction"] - a["regex"]) for a in read_answers()
             if a["run"] == "main" and a["status"] == "ok" and a.get("regex") is not None]
    no_regex = sum(1 for a in read_answers() if a["run"] == "main" and a["status"] == "ok" and a.get("regex") is None)
    if diffs:
        emit(f"- Parser against regex: {len(diffs)} answers, {sum(d < 0.005 for d in diffs)} identical, "
             f"largest difference {max(diffs):.3f}; {no_regex} answers with no 'Probability: ZZ%'")

    groups = {
        "todas": sorted(common),
        "mercado": sorted(q for q in common if rows[q]["kind"] == "market"),
        "dataset": sorted(q for q in common if rows[q]["kind"] == "dataset"),
    }
    for s in sorted({rows[q]["source"] for q in main_q}):
        groups[s] = sorted(q for q in common if rows[q]["source"] == s)
    groups["todas sem yfinance"] = sorted(q for q in common if rows[q]["source"] != "yfinance")
    groups["suplemento"] = sorted(common_supp)
    groups["mercado + suplemento"] = sorted(groups["mercado"] + groups["suplemento"])
    AGGREGATES = ("todas", "mercado", "dataset", "todas sem yfinance", "suplemento", "mercado + suplemento")

    def arrays(m: str, v: str, qs: list[str]) -> tuple[np.ndarray, np.ndarray]:
        return (np.array([preds[(m, v)][q] for q in qs]), np.array([rows[q]["outcome"] for q in qs], dtype=float))

    # 1. Scores under V0
    emit("\n## Brier e log score, V0 (sem pesquisa)\n")
    emit("| grupo | n | taxa de Sim | " + " | ".join(f"{LABELS[m]} Brier | log" for m in present) + " |")
    emit("|---|---|---|" + "---|---|" * len(present))
    res["scores_v0"] = {}
    for g, qs in groups.items():
        if not qs:
            continue
        y = np.array([rows[q]["outcome"] for q in qs], dtype=float)
        cells = []
        for m in present:
            p, _ = arrays(m, "V0", qs)
            b, ls = float(brier(p, y).mean()), float(logscore(p, y).mean())
            res["scores_v0"].setdefault(g, {})[m] = {"n": len(qs), "brier": b, "log": ls}
            cells.append(f"{b:.3f} | {ls:.3f}")
        emit(f"| {g} | {len(qs)} | {y.mean():.2f} | " + " | ".join(cells) + " |")
    emit("\nReferencias: prever sempre 50% da Brier 0,250 e log -0,693; prever sempre a taxa de Sim do proprio grupo "
         "da Brier = taxa*(1-taxa).")

    # 2. Mean forecast vs base rate, and the Platt fit
    emit("\n## Media prevista contra a taxa de Sim, e ajuste de Platt (V0)\n")
    emit("| modelo | grupo | n | media prevista | taxa de Sim | diferenca (IC 90%) | A (inclinacao, IC 90%) | B (intercepto, IC 90%) | desvio de log-odds c com A=1 (IC 90%) | Brier LOO com Platt |")
    emit("|---|---|---|---|---|---|---|---|---|---|")
    res["lean"] = {}
    for m in present:
        for g in ("todas", "mercado", "dataset", "todas sem yfinance", "mercado + suplemento"):
            qs = groups[g]
            if len(qs) < 20:
                continue
            p, y = arrays(m, "V0", qs)
            n = len(qs)
            diff = float(p.mean() - y.mean())
            diff_ci = boot_ci(lambda i: float(p[i].mean() - y[i].mean()), n)
            ab = fit_ab(p, y)
            a_ci = boot_ci(lambda i: (fit_ab(p[i], y[i]) or (np.nan, np.nan))[0], n)
            b_ci = boot_ci(lambda i: (fit_ab(p[i], y[i]) or (np.nan, np.nan))[1], n)
            c = fit_shift(p, y)
            c_ci = boot_ci(lambda i: fit_shift(p[i], y[i]), n)
            loo = None
            try:
                fr = calibration.fit_platt([(float(pp), int(yy)) for pp, yy in zip(p, y)])
                loo = (fr.brier_before, fr.brier_held_out, fr.a, fr.b)
            except ValueError:
                pass
            res["lean"].setdefault(m, {})[g] = {
                "n": n, "mean_p": float(p.mean()), "base_rate": float(y.mean()), "diff": diff, "diff_ci": diff_ci,
                "A": ab[0] if ab else None, "A_ci": a_ci, "B": ab[1] if ab else None, "B_ci": b_ci,
                "shift": c, "shift_ci": c_ci, "loo": loo,
            }
            a_txt = f"{ab[0]:.2f} ({a_ci[0]:.2f} a {a_ci[1]:.2f})" if ab else "n/d"
            b_txt = f"{ab[1]:+.2f} ({b_ci[0]:+.2f} a {b_ci[1]:+.2f})" if ab else "n/d"
            c_txt = f"{c:+.2f} ({c_ci[0]:+.2f} a {c_ci[1]:+.2f})" if c is not None else "n/d"
            emit(f"| {LABELS[m]} | {g} | {n} | {p.mean():.3f} | {y.mean():.3f} | {diff:+.3f} ({diff_ci[0]:+.3f} a {diff_ci[1]:+.3f}) | "
                 f"{a_txt} | {b_txt} | {c_txt} | "
                 + (f"{loo[0]:.4f} -> {loo[1]:.4f}" if loo else "") + " |")
    emit("\nA<1: confiante demais (puxar para 50%); A>1: timido demais. B ou c negativos: as previsoes tem log-odds "
         "altas demais, isto e, pendem para o Sim. Mesma forma de calibration.py: p' = sigmoid(A*logit(p) + B).")

    # Discrimination: does the forecast separate the questions that resolved Yes from those that resolved No?
    emit("\n## Separacao entre Sim e Nao, V0\n")
    emit("| modelo | grupo | media prevista nas que deram Sim | media prevista nas que deram Nao | AUC (IC 90%) |")
    emit("|---|---|---|---|---|")

    def auc(p: np.ndarray, y: np.ndarray) -> float | None:
        pos, neg = p[y == 1], p[y == 0]
        if not len(pos) or not len(neg):
            return None
        return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))

    res["discrimination"] = {}
    for m in present:
        for g in ("todas", "mercado", "dataset", "mercado + suplemento"):
            qs = groups[g]
            if not qs:
                continue
            p, y = arrays(m, "V0", qs)
            a = auc(p, y)
            ci = boot_ci(lambda i: auc(p[i], y[i]), len(qs))
            res["discrimination"].setdefault(m, {})[g] = {"mean_yes": float(p[y == 1].mean()), "mean_no": float(p[y == 0].mean()),
                                                          "auc": a, "auc_ci": ci}
            emit(f"| {LABELS[m]} | {g} | {p[y == 1].mean():.3f} | {p[y == 0].mean():.3f} | {a:.3f} ({ci[0]:.3f} a {ci[1]:.3f}) |")
    emit("\nAUC 0,5 = nao separa; 1,0 = separa perfeitamente.")

    # 3. Reliability tables
    edges = np.linspace(0, 1, 11)
    res["reliability"] = {}
    for g, m in [(g, m) for g in ("todas", "mercado + suplemento") for m in present]:
        if not groups[g]:
            continue
        p, y = arrays(m, "V0", groups[g])
        emit(f"\n### Confiabilidade, V0, {LABELS[m]}, grupo {g} (n={len(p)})\n")
        emit("| faixa prevista | n | media prevista | frequencia observada de Sim |")
        emit("|---|---|---|---|")
        table = []
        for i in range(10):
            lo, hi = edges[i], edges[i + 1]
            mask = (p >= lo) & ((p < hi) if i < 9 else (p <= hi))
            n = int(mask.sum())
            if n:
                table.append({"bin": [lo, hi], "n": n, "mean_p": float(p[mask].mean()), "freq": float(y[mask].mean())})
                emit(f"| {100 * lo:.0f}-{100 * hi:.0f}% | {n} | {p[mask].mean():.3f} | {y[mask].mean():.3f} |")
            else:
                table.append({"bin": [lo, hi], "n": 0})
                emit(f"| {100 * lo:.0f}-{100 * hi:.0f}% | 0 | | |")
        res["reliability"].setdefault(g, {})[m] = table

    # Per source: mean forecast against the Yes rate
    emit("\n## Media prevista por fonte, V0\n")
    emit("| fonte | n | taxa de Sim | " + " | ".join(LABELS[m] for m in present) + " |")
    emit("|---|---|---|" + "---|" * len(present))
    for g, qs in groups.items():
        if g in AGGREGATES or not qs:
            continue
        y = np.array([rows[q]["outcome"] for q in qs], dtype=float)
        emit(f"| {g} | {len(qs)} | {y.mean():.2f} | " + " | ".join(f"{arrays(m, 'V0', qs)[0].mean():.3f}" for m in present) + " |")

    # 4. V1 against V0 on market questions, and the market value alone
    res["v1"] = {}
    for label, market_qs in (("mercado 2026-09-27", sorted(q for q in main_q if rows[q]["kind"] == "market")),
                             ("mercado + suplemento", sorted(q for q in rows if rows[q]["kind"] == "market"))):
        v1_section(label, market_qs, rows, preds, present, res, emit)

    # Per-question list of the market questions of the 2026-09-27 set
    market_qs = sorted(q for q in main_q if rows[q]["kind"] == "market")
    emit("\n## Perguntas de mercado resolvidas do conjunto 2026-09-27, uma a uma\n")
    head = [m for m in present]
    emit("| fonte | pergunta | mercado | resultado | " + " | ".join(f"{LABELS[m]} V0 / V1" for m in head) + " |")
    emit("|---|---|---|---|" + "---|" * len(head))
    for q in market_qs:
        r = rows[q]
        cells = []
        for m in head:
            a, b = preds.get((m, "V0"), {}).get(q), preds.get((m, "V1"), {}).get(q)
            cells.append(f"{'' if a is None else f'{a:.2f}'} / {'' if b is None else f'{b:.2f}'}")
        title = r["question"].split("\n")[0][:90].replace("|", "/")
        emit(f"| {r['source']} | {title} | {r['market']['value']:.2f} | {'Sim' if r['outcome'] else 'Nao'} | " + " | ".join(cells) + " |")

    # Answers per model, thinking and stops
    emit("\n## Respostas e tokens\n")
    for m in singles:
        ans = [a for a in read_answers() if a["alias"] == m and a["run"] == "main"]
        ok = [a for a in ans if a["status"] == "ok"]
        outs = [u.get("completion_tokens") or 0 for a in ok for u in a.get("usage", []) if not (u.get("model") or "").startswith(PARSER)]
        stops = [a.get("stop_reason") for a in ok if a.get("stop_reason") not in (None, "end_turn")]
        emit(f"- {LABELS[m]}: {len(ok)} respostas ok de {len(ans)} registradas; tokens de saida mediana "
             f"{statistics.median(outs) if outs else 0:.0f}, maximo {max(outs) if outs else 0}; paradas fora de end_turn: {len(stops)}")

    with open(RESULTS, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    with open(TABLES, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nWrote {RESULTS} and {TABLES}")


def v1_section(label: str, market_qs: list[str], rows: dict, preds: dict, present: list[str], res: dict, emit) -> None:
    """V1 against V0, paired, and both against the market value alone, on the given market questions."""
    if not market_qs:
        return
    emit(f"\n## Valor de mercado no prompt (V1) contra V0: {label}, {len(market_qs)} perguntas de mercado resolvidas\n")
    mv = np.array([rows[q]["market"]["value"] for q in market_qs])
    ym = np.array([rows[q]["outcome"] for q in market_qs], dtype=float)
    emit(f"Mercado sozinho (valor no congelamento): Brier {brier(_clip(mv), ym).mean():.3f}, "
         f"log {logscore(mv, ym).mean():.3f}, media {mv.mean():.3f} (n={len(market_qs)}, taxa de Sim {ym.mean():.2f})\n")
    emit("| modelo | n | Brier V0 | Brier V1 | V1-V0 Brier (IC 90%) | V1-V0 log (IC 90%) | V1 melhor/pior/empate (Brier) | V1-mercado Brier (IC 90%) | V0-mercado Brier (IC 90%) | distancia media ao mercado V0 -> V1 | media prevista V0 / V1 |")
    emit("|---|---|---|---|---|---|---|---|---|---|---|")
    res["v1"][label] = {"market_alone": {"n": len(market_qs), "brier": float(brier(_clip(mv), ym).mean()),
                                         "log": float(logscore(mv, ym).mean()), "mean": float(mv.mean()),
                                         "base_rate": float(ym.mean())}}
    for m in present:
        d0, d1 = preds.get((m, "V0"), {}), preds.get((m, "V1"), {})
        qs = [q for q in market_qs if q in d0 and q in d1]
        if not qs:
            continue
        p0 = np.array([d0[q] for q in qs])
        p1 = np.array([d1[q] for q in qs])
        y = np.array([rows[q]["outcome"] for q in qs], dtype=float)
        mk = _clip(np.array([rows[q]["market"]["value"] for q in qs]))
        n = len(qs)
        db = brier(p1, y) - brier(p0, y)
        dl = logscore(p1, y) - logscore(p0, y)
        dm1 = brier(p1, y) - brier(mk, y)
        dm0 = brier(p0, y) - brier(mk, y)
        ci_b, ci_l = boot_ci(lambda i: float(db[i].mean()), n), boot_ci(lambda i: float(dl[i].mean()), n)
        ci_m1, ci_m0 = boot_ci(lambda i: float(dm1[i].mean()), n), boot_ci(lambda i: float(dm0[i].mean()), n)
        better, worse = int((db < -1e-12).sum()), int((db > 1e-12).sum())
        dist0, dist1 = float(np.abs(p0 - mk).mean()), float(np.abs(p1 - mk).mean())
        res["v1"][label][m] = {"n": n, "brier_v0": float(brier(p0, y).mean()), "brier_v1": float(brier(p1, y).mean()),
                        "log_v0": float(logscore(p0, y).mean()), "log_v1": float(logscore(p1, y).mean()),
                        "d_brier": float(db.mean()), "d_brier_ci": ci_b, "d_log": float(dl.mean()), "d_log_ci": ci_l,
                        "v1_minus_market": float(dm1.mean()), "v1_minus_market_ci": ci_m1,
                        "v0_minus_market": float(dm0.mean()), "v0_minus_market_ci": ci_m0,
                        "better": better, "worse": worse, "dist_v0": dist0, "dist_v1": dist1}
        emit(f"| {LABELS[m]} | {n} | {brier(p0, y).mean():.3f} | {brier(p1, y).mean():.3f} | {db.mean():+.3f} ({ci_b[0]:+.3f} a {ci_b[1]:+.3f}) | "
             f"{dl.mean():+.3f} ({ci_l[0]:+.3f} a {ci_l[1]:+.3f}) | {better}/{worse}/{n - better - worse} | "
             f"{dm1.mean():+.3f} ({ci_m1[0]:+.3f} a {ci_m1[1]:+.3f}) | {dm0.mean():+.3f} ({ci_m0[0]:+.3f} a {ci_m0[1]:+.3f}) | "
             f"{dist0:.3f} -> {dist1:.3f} | {p0.mean():.3f} / {p1.mean():.3f} |")
    emit("\nDiferencas negativas de Brier (positivas de log) favorecem V1 ou o modelo.")


# ---------------------------------------------------------------------------


def main_cli() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("forecasting_tools").setLevel(logging.ERROR)
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare").set_defaults(func=cmd_prepare)
    sub.add_parser("smoke").set_defaults(func=cmd_smoke)
    p = sub.add_parser("submit")
    p.add_argument("--models", default="s5,o55")
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=cmd_submit)
    p = sub.add_parser("collect")
    p.add_argument("--wait", action="store_true")
    p.set_defaults(func=cmd_collect)
    p = sub.add_parser("run-gpt")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--group", choices=["main", "supp"])
    p.add_argument("--second", action="store_true")
    p.set_defaults(func=cmd_run_gpt)
    sub.add_parser("reparse").set_defaults(func=cmd_reparse)
    sub.add_parser("analyze").set_defaults(func=cmd_analyze)
    sub.add_parser("spend").set_defaults(func=cmd_spend)
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main_cli()
