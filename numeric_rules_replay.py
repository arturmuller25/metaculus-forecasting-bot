"""
Replays the numeric forecast step of MiniBench round 1 with two prompt rules
for numeric and discrete questions, to decide whether to adopt them:

  N1  floors and zero: counts that only grow keep every percentile at or
      above the last confirmed value, and a plausible exact zero (or lowest
      value) gets the lowest percentiles placed on it
  N2  13 percentiles (1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99)
      instead of 6 (10, 20, 40, 60, 80, 90)
  N3  both

Everything else comes from forecast_replay.py and claude_direct_replay.py:
the frozen research rebuilt from the bot's comments, the original dates, the
bot's own numeric prompt and parse (bot._run_forecast_on_numeric, unchanged;
a wrapper around the member edits the prompt), and the scoring (50 ln of the
mass the published CDF gives the outcome's bucket, PCHIP widened 1.15,
members aggregated as production does). The parser is claude-haiku-4-5 on the
Anthropic key for both models, as in claude_direct_replay.py.

Models:
- claude-sonnet-5 through the Anthropic Message Batches API, adaptive
  thinking at effort high (what production's direct-first member sends);
- openrouter/openai/gpt-6.1-sol through OpenRouter at reasoning high, built
  like main._thinker. Spending stops when the key's limit_remaining has
  dropped SPEND_CAP_OPENROUTER dollars since this script's first reading.

Questions: the 24 scored numeric and discrete questions of the earlier
replays, plus 45780 (Strait of Hormuz transits), which resolved 0 on
2026-10-07, after those replays cached it. The `refresh` command fetches the
posts that had no outcome into logs/replay_numeric/cache/ (the earlier
replay's cache is left alone).

Usage:
    uv run python numeric_rules_replay.py refresh                # Metaculus only
    uv run python numeric_rules_replay.py prompts                # builds, checks and prints the edited prompts
    uv run python numeric_rules_replay.py run-gpt --variants base,N1,N2,N3 --reps 1,2 [--posts ...]
    uv run python numeric_rules_replay.py submit --variants base,N1,N2,N3 --reps 1,2
    uv run python numeric_rules_replay.py collect [--wait]
    uv run python numeric_rules_replay.py analyze                # offline
    uv run python numeric_rules_replay.py spend
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
import re
import statistics
import time
from datetime import datetime, timezone

import numpy as np

import anthropic

import analyze_results as ar
import bot as bot_module
import forecast_replay as fr
import main
from forecasting_tools import GeneralLlm

logger = logging.getLogger("numeric_rules_replay")

OUT = os.path.join("logs", "replay_numeric")
ANSWERS = os.path.join(OUT, "answers.jsonl")
RAW = os.path.join(OUT, "raw.jsonl")
PROMPTS = os.path.join(OUT, "prompts.jsonl")
BATCHES = os.path.join(OUT, "batches.json")
BUDGET = os.path.join(OUT, "budget.json")
CACHE = os.path.join(OUT, "cache")

# forecast_replay points the bot's own log at logs/replay_r1/; these runs get theirs.
bot_module.FORECAST_LOG = os.path.join(OUT, "forecasts.jsonl")

PARSER = "anthropic/claude-haiku-4-5"
SPEND_CAP_OPENROUTER = 6.0  # dollars of key drop (limit_remaining) since the first reading
SPEND_CAP_ANTHROPIC = 20.0  # dollars, estimated from token usage
IN_FLIGHT = 6
FORECAST_TIMEOUT = 300  # as in forecast_replay.py
MAX_TOKENS = 16000

GPT_MODEL = "openrouter/openai/gpt-6.1-sol"
CLAUDE_MODEL = "claude-sonnet-5"
ALIASES = {"sol61": GPT_MODEL, "s5": f"anthropic/{CLAUDE_MODEL}"}

# Anthropic list prices per million tokens (input, output), as in
# claude_direct_replay.py; batches cost half.
PRICES = {"claude-sonnet-5": (2.0, 10.0), "claude-haiku-4-5": (1.0, 5.0)}
OPENROUTER_PRICE = {"openai/gpt-6.1-sol": (2.0, 10.0)}

# Set before any answer was requested: the questions where N1 is meant to act
# (a count that only grows, or a count whose range includes zero or its lowest
# possible value), from the titles and bounds alone.
N1_TARGET = {
    45739, 45748, 45749, 45754, 45764, 45775, 45778, 45779,  # cumulative counts
    45742, 45759, 45766, 45780, 45794, 45796,  # counts where zero or the lowest value is plausible
}
# Known floors (last officially confirmed value when the bot forecast), from
# the post-mortem: 5 people charged in 45778, 167 cases on the DHS page in 45754.
FLOORS = {45778: 5.0, 45754: 167.0}
NAMED = [45778, 45780, 45766, 45754]


# ---------------------------------------------------------------------------
# Prompt variants
# ---------------------------------------------------------------------------

_words = fr._words

N1_TEXT = (
    "For counts that only grow (cases, arrests, candidates withdrawn, launches so far), no percentile"
    " may be below the last officially confirmed value. When an outcome of exactly zero (or the lowest"
    " possible value) is plausible, give it explicit probability: put your lowest percentiles on that"
    " value rather than spreading them below it."
)
_HUMBLE = _words(
    "You remind yourself that good forecasters are humble and set wide 90/10"
    " confidence intervals to account for unknown unknowns."
)
LEVELS_6 = [10, 20, 40, 60, 80, 90]
LEVELS_13 = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]


def _answer_block(levels: list[int]) -> str:
    lines = []
    for i, p in enumerate(levels):
        note = " (lowest number value)" if i == 0 else " (highest number value)" if i == len(levels) - 1 else ""
        lines.append(f"Percentile {p}: XX{note}")
    return "\n".join(lines)


_BLOCK_6 = _words(_answer_block(LEVELS_6))

N1 = [(_HUMBLE, lambda m: m.group(0) + "\n\n" + N1_TEXT)]
N2 = [(_BLOCK_6, lambda _m: _answer_block(LEVELS_13))]
VARIANTS = {"base": [], "N1": N1, "N2": N2, "N3": N1 + N2}


def levels_of(variant: str) -> list[int]:
    return LEVELS_13 if variant in ("N2", "N3") else LEVELS_6


class PromptEditError(RuntimeError):
    pass


def apply_edits(prompt: str, edits) -> str:
    """Applies each replacement exactly once, or fails without calling the model."""
    for pattern, new in edits:
        prompt, count = re.subn(pattern, new, prompt, flags=re.S)
        if count != 1:
            raise PromptEditError(f"pattern matched {count} times: {pattern[:60]}")
    return prompt


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Files and costs
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def append_jsonl(path: str, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def usage_cost(u: dict) -> float:
    """Dollars for one call, at list price (half for batches)."""
    model = (u.get("model") or "")
    pin = pout = 0.0
    if model.startswith("anthropic/"):
        pin, pout = PRICES[model.removeprefix("anthropic/")]
    elif model.startswith("openrouter/"):
        pin, pout = OPENROUTER_PRICE.get(model.removeprefix("openrouter/"), (0.0, 0.0))
    share = 0.5 if u.get("batch") else 1.0
    return share * ((u.get("prompt_tokens") or 0) * pin + (u.get("completion_tokens") or 0) * pout) / 1e6


def is_anthropic(u: dict) -> bool:
    return (u.get("model") or "").startswith("anthropic/")


def spent_anthropic() -> float:
    total = sum(usage_cost(r["usage"]) for r in read_jsonl(RAW) if r.get("usage"))
    for a in read_jsonl(ANSWERS):
        total += sum(usage_cost(u) for u in a.get("usage", []) if is_anthropic(u) and not u.get("batch"))
    return total


def spent_openrouter_tokens() -> float:
    return sum(usage_cost(u) for a in read_jsonl(ANSWERS) for u in a.get("usage", []) if not is_anthropic(u))


def load_batches() -> list[dict]:
    if not os.path.exists(BATCHES):
        return []
    with open(BATCHES, encoding="utf-8") as fh:
        return json.load(fh)


def save_batches(batches: list[dict]) -> None:
    with open(BATCHES, "w", encoding="utf-8") as fh:
        json.dump(batches, fh, indent=1)


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------


def _refreshed(post_id: int) -> dict | None:
    path = os.path.join(CACHE, f"post_{post_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def numeric_rows(scored_only: bool = True) -> list[dict]:
    """The round's numeric and discrete questions, with the outcome of any post refreshed since."""
    rows = []
    for r in fr.load_questions():
        if r["type"] not in ("numeric", "discrete"):
            continue
        fresh = _refreshed(r["post_id"])
        if fresh is not None:
            r = {**r, "post_json": fresh, "outcome": ar.outcome_of(fresh["question"]),
                 "resolution": fresh["question"].get("resolution")}
        if scored_only and r["outcome"] is None:
            continue
        rows.append(r)
    return rows


def cmd_refresh(_args) -> None:
    os.makedirs(CACHE, exist_ok=True)
    for r in fr.load_questions():
        if r["type"] in ("numeric", "discrete") and r["outcome"] is None:
            data = fr._throttled_get(f"/posts/{r['post_id']}/")
            with open(os.path.join(CACHE, f"post_{r['post_id']}.json"), "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False)
            q = data["question"]
            print(f"  {r['post_id']}: resolution {q.get('resolution')!r}, outcome {ar.outcome_of(q)!r}")


# ---------------------------------------------------------------------------
# The bot and its members
# ---------------------------------------------------------------------------


def replay_bot(member) -> object:
    """A production bot with one ensemble member, no shadows, and the Anthropic parser."""
    bot = main.build_bot(publish=False, samples=1)
    bot._shadows = []
    bot._ensemble = [member]
    bot._llms["parser"] = GeneralLlm(model=PARSER, temperature=0.0, timeout=60, allowed_tries=2)
    assert bot.get_llm("parser", "llm").model == PARSER
    return bot


class Capture:
    """Stands in for a model to read the prompt the bot builds; no call is made."""

    model = "capture"

    async def invoke(self, prompt: str) -> str:
        self.prompt = prompt
        raise RuntimeError("captured")


async def build_prompt(row: dict) -> str:
    capture = Capture()
    fr._TODAY.set(datetime.fromisoformat(row["forecast_time"]))
    try:
        await fr.forecast_once(replay_bot(capture), fr.question_object(row), row["research"])
    except RuntimeError:
        pass
    return capture.prompt


class VariantLlm:
    """Edits the production prompt for a variant and calls the model (or returns a
    canned answer after checking the edited prompt is the one that was sent)."""

    def __init__(self, model: str, edits, llm=None, canned: str | None = None, expected_sha1: str | None = None) -> None:
        self.model = model
        self._llm = llm
        self._edits = edits
        self._canned = canned
        self._expected = expected_sha1

    async def invoke(self, prompt: str) -> str:
        prompt = apply_edits(prompt, self._edits)
        got = sha1(prompt)
        if self._expected is not None and got != self._expected:
            raise RuntimeError(f"prompt mismatch: sent {self._expected}, rebuilt {got}")
        call = fr._CALL.get()
        if call is not None:
            call["prompt_chars"] = len(prompt)
            call["prompt_sha1"] = got
        if self._canned is not None:
            text = self._canned
        else:
            started = time.monotonic()
            try:
                text = await self._llm.invoke(prompt)
            finally:
                if call is not None:
                    call["seconds"] = round(time.monotonic() - started, 1)
        if call is not None:
            call["raw"] = text
        return text


_PCT_RE = re.compile(r"Percentile\s+(\d{1,2})\s*[:=]\s*[$€£₽]?\s*(-?[\d,]*\.?\d+)")


def regex_percentiles(text: str, levels: list[int]) -> list[tuple[float, float]] | None:
    """The last complete answer block with exactly these levels, read by regex."""
    found = _PCT_RE.findall(text or "")
    for start in range(len(found) - len(levels), -1, -1):
        block = found[start : start + len(levels)]
        if [int(p) for p, _ in block] == levels:
            return [(int(p) / 100, float(v.replace(",", ""))) for p, v in block]
    return None


def parse_check(prediction, raw: str, variant: str) -> str:
    """Whether the parser returned every requested percentile with the values in the text."""
    if not isinstance(prediction, list):
        return "no prediction"
    levels = levels_of(variant)
    parsed_levels = [round(p * 100) for p, _ in prediction]
    if parsed_levels != levels:
        return f"levels {parsed_levels}"
    rx = regex_percentiles(raw, levels)
    if rx is None:
        return "ok (no regex block)"
    for (_, v1), (_, v2) in zip(prediction, rx):
        if abs(v1 - v2) > max(1e-3, 1e-4 * abs(v2)):
            return "values differ from the text"
    return "ok"


async def run_one(row: dict, member: VariantLlm, alias: str, variant: str, rep: int, route: str,
                  call: dict, pre_error: str | None = None, extra: dict | None = None) -> dict:
    """Runs the bot's numeric forecast with this member and records the answer."""
    fr._CALL.set(call)
    fr._TODAY.set(datetime.fromisoformat(row["forecast_time"]))
    status, prediction, error = "ok", None, None
    try:
        if pre_error:
            raise RuntimeError(pre_error)
        result = await fr.forecast_once(replay_bot(member), fr.question_object(row), row["research"])
        prediction = fr.serialize(result.prediction_value)
    except Exception as exc:  # noqa: BLE001
        status = "stopped" if fr.STOP.is_set() else "failed"
        error = f"{type(exc).__name__}: {str(exc)[:300]}"
    check = parse_check(prediction, call.get("raw", ""), variant) if status == "ok" else None
    out = {
        "time": _utc_now(), "post_id": row["post_id"], "type": row["type"], "alias": alias,
        "model": ALIASES[alias], "variant": variant, "rep": rep, "route": route,
        "forecast_time": row["forecast_time"], "status": status, "error": error,
        "prediction": prediction, "parse_check": check, **(extra or {}), **call,
    }
    append_jsonl(ANSWERS, out)
    shown = f"P{round(prediction[0][0] * 100)} {prediction[0][1]:.4g} .. P{round(prediction[-1][0] * 100)} {prediction[-1][1]:.4g}" if prediction else error
    logger.info(f"{row['post_id']} {alias} {variant} r{rep}: {status} {shown} [{check}] ({call.get('seconds')} s)")
    return out


# ---------------------------------------------------------------------------
# Commands: prompts
# ---------------------------------------------------------------------------


def cmd_prompts(args) -> None:
    rows = numeric_rows()
    level = bot_module.logger.level
    bot_module.logger.setLevel(logging.ERROR)  # every capture logs a failed member

    async def go() -> list[str]:
        return [await build_prompt(r) for r in rows]

    prompts = asyncio.run(go())
    bot_module.logger.setLevel(level)

    # The production prompt must be the one the earlier replays sent (numeric prompt unchanged).
    recorded = {}
    for a in fr.recorded():
        if a["alias"] in ("sol61", "sonnet5") and a["variant"] == "base" and a.get("prompt_sha1"):
            recorded.setdefault(a["post_id"], set()).add(a["prompt_sha1"])
    same = sum(sha1(p) in recorded.get(r["post_id"], set()) for r, p in zip(rows, prompts))
    print(f"{len(rows)} questions; production prompt identical to the earlier replay's: {same} of "
          f"{sum(r['post_id'] in recorded for r in rows)} with an earlier answer")

    with open(PROMPTS, "w", encoding="utf-8") as fh:
        for row, prompt in zip(rows, prompts):
            for variant, edits in VARIANTS.items():
                edited = apply_edits(prompt, edits)
                fh.write(json.dumps({"post_id": row["post_id"], "variant": variant, "sha1": sha1(edited),
                                     "chars": len(edited), "prompt": edited}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows) * len(VARIANTS)} prompts to {PROMPTS}")

    # One edited prompt per variant, from the question's date to the end.
    row, prompt = rows[0], prompts[0]
    lines = []
    for variant, edits in VARIANTS.items():
        edited = apply_edits(prompt, edits)
        start = edited.find("Today is")
        lines.append(f"\n===== {variant} (post {row['post_id']}) =====\n{edited[start:].rstrip()}")
    text = "\n".join(lines)
    with open(os.path.join(OUT, "check_prompts.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)


def load_prompts() -> dict[tuple[int, str], dict]:
    prompts = {(p["post_id"], p["variant"]): p for p in read_jsonl(PROMPTS)}
    if not prompts:
        raise SystemExit(f"{PROMPTS} is missing: run the prompts command first")
    return prompts


# ---------------------------------------------------------------------------
# GPT-6.1 Sol through OpenRouter
# ---------------------------------------------------------------------------


def _budget() -> dict:
    if not os.path.exists(BUDGET):
        return {}
    with open(BUDGET, encoding="utf-8") as fh:
        return json.load(fh)


def log_reading(label: str) -> tuple[float, float]:
    """Records the key balance; returns (remaining, drop since the first reading)."""
    remaining = fr.key_remaining()
    data = _budget()
    if "baseline" not in data:
        data["baseline"] = {"time": _utc_now(), "limit_remaining": remaining}
    data.setdefault("readings", []).append({"time": _utc_now(), "label": label, "limit_remaining": remaining})
    with open(BUDGET, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    return remaining, float(data["baseline"]["limit_remaining"]) - remaining


async def _monitor(interval: float = 20.0) -> None:
    while not fr.STOP.is_set():
        try:
            remaining, drop = await asyncio.to_thread(log_reading, "monitor")
            if drop >= SPEND_CAP_OPENROUTER:
                logger.warning(f"Cap reached: key dropped ${drop:.2f} (${remaining:.2f} left). Stopping.")
                fr.STOP.set()
                return
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not read the key balance: {exc}")
        await asyncio.sleep(interval)


def parse_list(text: str, kind=str) -> list:
    return [kind(x.strip()) for x in (text or "").split(",") if x.strip()]


def cmd_run_gpt(args) -> None:
    variants, reps = parse_list(args.variants), parse_list(args.reps, int)
    rows = numeric_rows()
    if args.posts:
        wanted = set(parse_list(args.posts, int))
        rows = [r for r in rows if r["post_id"] in wanted]
    prompts = load_prompts()
    done = {(a["post_id"], a["variant"], a["rep"]) for a in read_jsonl(ANSWERS)
            if a["alias"] == "sol61" and a["status"] in ("ok", "failed")}
    # Rep by rep, and within a rep all variants of a question together, so a stop
    # at the cap leaves complete pairs.
    jobs = [(row, v, rep) for rep in reps for row in rows for v in variants
            if (row["post_id"], v, rep) not in done]
    remaining, drop = log_reading(f"{args.label} start")
    estimate = sum(len(prompts[(r["post_id"], v)]["prompt"]) / 3 * 2e-6 + 2500 * 10e-6 for r, v, _ in jobs)
    print(f"Key: ${remaining:.2f} left, ${drop:.2f} dropped since the first reading (cap ${SPEND_CAP_OPENROUTER:.2f}).")
    print(f"{len(jobs)} answers to request; token estimate up to ${estimate:.2f}.")
    if drop >= SPEND_CAP_OPENROUTER:
        raise SystemExit("Cap already reached; nothing run.")
    if not jobs:
        return

    async def go() -> None:
        fr._SLOTS = asyncio.Semaphore(IN_FLIGHT)
        queue: asyncio.Queue = asyncio.Queue()
        for job in jobs:
            queue.put_nowait(job)

        async def worker() -> None:
            while not queue.empty() and not fr.STOP.is_set():
                row, variant, rep = queue.get_nowait()
                member = VariantLlm(
                    GPT_MODEL, VARIANTS[variant], llm=main._thinker(GPT_MODEL, 0.3, FORECAST_TIMEOUT),
                    expected_sha1=prompts[(row["post_id"], variant)]["sha1"],
                )
                await run_one(row, member, "sol61", variant, rep, "openrouter", {})

        monitor = asyncio.create_task(_monitor())
        await asyncio.gather(*(worker() for _ in range(IN_FLIGHT)))
        monitor.cancel()

    asyncio.run(go())
    remaining, drop = log_reading(f"{args.label} end")
    print(f"Key: ${remaining:.2f} left, ${drop:.2f} dropped since the first reading; "
          f"token estimate of every GPT call recorded ${spent_openrouter_tokens():.2f}")
    if fr.STOP.is_set():
        print("STOPPED: spending cap reached.")


# ---------------------------------------------------------------------------
# Claude Sonnet 5 through Message Batches
# ---------------------------------------------------------------------------


def request_params(prompt: str) -> dict:
    """What GeneralLlm sends through litellm for reasoning_effort="high" on Claude Sonnet 5
    (adaptive thinking plus output_config.effort), as in claude_direct_replay.py."""
    return {
        "model": CLAUDE_MODEL,
        "max_tokens": MAX_TOKENS,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
        "messages": [{"role": "user", "content": prompt}],
    }


def response_row(msg, batch: bool) -> dict:
    usage = msg.usage.model_dump()
    thinking = (usage.get("output_tokens_details") or {}).get("thinking_tokens")
    return {
        "served_model": msg.model,
        "stop_reason": msg.stop_reason,
        "text": "".join(b.text for b in msg.content if b.type == "text"),
        "usage": {
            "model": f"anthropic/{CLAUDE_MODEL}",
            "id": msg.id,
            "prompt_tokens": (usage.get("input_tokens") or 0)
            + (usage.get("cache_creation_input_tokens") or 0)
            + (usage.get("cache_read_input_tokens") or 0),
            "completion_tokens": usage.get("output_tokens") or 0,
            "reasoning_tokens": thinking,
            "batch": batch,
        },
    }


def cmd_submit(args) -> None:
    variants, reps = parse_list(args.variants), parse_list(args.reps, int)
    rows = numeric_rows()
    if args.posts:
        wanted = set(parse_list(args.posts, int))
        rows = [r for r in rows if r["post_id"] in wanted]
    prompts = load_prompts()
    done = {(a["post_id"], a["variant"], a["rep"]) for a in read_jsonl(ANSWERS)
            if a["alias"] == "s5" and a["status"] == "ok"}
    batches = load_batches()
    pending = {(m["post_id"], b["variant"], b["rep"]) for b in batches if not b.get("collected")
               for m in b["requests"].values()}
    client = anthropic.Anthropic(max_retries=3)
    for variant in variants:
        for rep in reps:
            todo = [r for r in rows if (r["post_id"], variant, rep) not in done | pending]
            if not todo:
                print(f"{variant} r{rep}: nothing to request")
                continue
            # Upper estimate: 1 token per 3 characters, 6,000 output tokens, half price.
            extra = sum(0.5 * (len(prompts[(r["post_id"], variant)]["prompt"]) / 3 * 2 + 6000 * 10) / 1e6 for r in todo)
            pend = sum(b["estimate"] for b in batches if not b.get("collected"))
            total = spent_anthropic() + pend + extra
            print(f"Anthropic: spent ${spent_anthropic():.2f}, pending up to ${pend:.2f}, this batch up to ${extra:.2f} (cap ${SPEND_CAP_ANTHROPIC:.0f})")
            if total > SPEND_CAP_ANTHROPIC:
                raise SystemExit("Refused: this could take the estimated spend past the cap.")
            requests = [
                {"custom_id": f"s5_{variant}_r{rep}_{r['post_id']}",
                 "params": request_params(prompts[(r["post_id"], variant)]["prompt"])}
                for r in todo
            ]
            batch = client.messages.batches.create(requests=requests)
            batches.append({
                "id": batch.id, "variant": variant, "rep": rep, "created": _utc_now(), "estimate": round(extra, 4),
                "requests": {req["custom_id"]: {"post_id": r["post_id"], "sha1": prompts[(r["post_id"], variant)]["sha1"]}
                             for req, r in zip(requests, todo)},
                "collected": False,
            })
            save_batches(batches)
            print(f"{variant} r{rep}: batch {batch.id} with {len(requests)} requests ({batch.processing_status})")


def cmd_collect(args) -> None:
    client = anthropic.Anthropic(max_retries=3)
    rows = {r["post_id"]: r for r in numeric_rows()}
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
                print(f"{_utc_now()} {b['variant']} r{b['rep']} {b['id']}: {info.processing_status}, "
                      f"{counts.processing} processing, {counts.succeeded} succeeded, {counts.errored} errored")
                continue
            items = []
            for result in client.messages.batches.results(b["id"]):
                meta = b["requests"][result.custom_id]
                resp, error = None, None
                if result.result.type == "succeeded":
                    resp = response_row(result.result.message, batch=True)
                else:
                    error = f"batch result {result.result.type}: {str(getattr(result.result, 'error', None))[:300]}"
                append_jsonl(RAW, {
                    "time": _utc_now(), "custom_id": result.custom_id, "batch_id": b["id"],
                    "variant": b["variant"], "rep": b["rep"], "post_id": meta["post_id"],
                    "prompt_sha1": meta["sha1"], "error": error, **(resp or {}),
                })
                items.append((rows[meta["post_id"]], meta["sha1"], resp, error))

            async def parse_all(items=items, b=b) -> None:
                fr._SLOTS = asyncio.Semaphore(IN_FLIGHT)
                sem = asyncio.Semaphore(IN_FLIGHT)

                async def one(row, expected, resp, error):
                    async with sem:
                        if error is None and not (resp["text"] or "").strip():
                            error = f"empty answer (stop_reason {resp['stop_reason']})"
                        member = VariantLlm(ALIASES["s5"], VARIANTS[b["variant"]],
                                            canned=(resp or {}).get("text", ""), expected_sha1=expected)
                        call = {"usage": [resp["usage"]] if resp and resp.get("usage") else []}
                        extra = {"stop_reason": (resp or {}).get("stop_reason")}
                        await run_one(row, member, "s5", b["variant"], b["rep"], "batch", call, error, extra)

                await asyncio.gather(*(one(*item) for item in items))

            asyncio.run(parse_all())
            b["collected"] = True
            b["ended"] = info.ended_at.isoformat() if info.ended_at else None
            b["counts"] = {"succeeded": counts.succeeded, "errored": counts.errored,
                           "expired": counts.expired, "canceled": counts.canceled}
            save_batches(batches)
            print(f"{b['variant']} r{b['rep']}: collected {len(items)} results; Anthropic spend so far ${spent_anthropic():.2f}")
        if not args.wait:
            break
        if not all(b.get("collected") for b in load_batches()):
            time.sleep(60)


def cmd_parse_stopped(_args) -> None:
    """Parses GPT answers that arrived (and were paid) but whose parser call was
    refused by the spending stop. Only the Anthropic parser is called."""
    rows = {r["post_id"]: r for r in numeric_rows()}
    answers = read_jsonl(ANSWERS)
    ok = {(a["post_id"], a["alias"], a["variant"], a["rep"]) for a in answers if a["status"] == "ok"}
    todo = [a for a in answers if a["status"] == "stopped" and a.get("raw")
            and (a["post_id"], a["alias"], a["variant"], a["rep"]) not in ok]

    async def go() -> None:
        fr._SLOTS = asyncio.Semaphore(IN_FLIGHT)
        for a in todo:
            member = VariantLlm(a["model"], VARIANTS[a["variant"]], canned=a["raw"], expected_sha1=a["prompt_sha1"])
            # The forecaster's usage stays on the stopped row; this row carries the parser's.
            await run_one(rows[a["post_id"]], member, a["alias"], a["variant"], a["rep"],
                          "openrouter (parsed after the stop)", {})

    asyncio.run(go())
    print(f"Parsed {len(todo)} answers.")


def cmd_spend(_args) -> None:
    data = _budget()
    if data:
        base, last = data["baseline"], data["readings"][-1]
        print(f"OpenRouter key: drop ${base['limit_remaining'] - last['limit_remaining']:.2f} from {base['time']} to {last['time']}"
              f" (includes anything else spending the key meanwhile)")
    print(f"OpenRouter, token estimate of the GPT calls: ${spent_openrouter_tokens():.2f}")
    raw = sum(usage_cost(r["usage"]) for r in read_jsonl(RAW) if r.get("usage"))
    parser = sum(usage_cost(u) for a in read_jsonl(ANSWERS) for u in a.get("usage", []) if (u.get("model") or "") == PARSER)
    print(f"Anthropic: Sonnet 5 batches ${raw:.2f}, Haiku parser ${parser:.2f}, total ${spent_anthropic():.2f}")


# ---------------------------------------------------------------------------
# Scoring (offline)
# ---------------------------------------------------------------------------


def _quantile_location(cdf: np.ndarray, q: float) -> float:
    """Where on the 0-1 axis the CDF crosses q (linear between grid points)."""
    x = np.linspace(0, 1, len(cdf))
    idx = int(np.searchsorted(cdf, q))
    if idx <= 0:
        return 0.0
    if idx >= len(cdf):
        return 1.0
    x0, x1, y0, y1 = x[idx - 1], x[idx], cdf[idx - 1], cdf[idx]
    return float(x0 if y1 == y0 else x0 + (x1 - x0) * (q - y0) / (y1 - y0))


def cmd_analyze(_args) -> None:
    logging.getLogger("forecasting_tools").setLevel(logging.ERROR)
    rows = {r["post_id"]: r for r in numeric_rows()}
    questions = {pid: fr.question_object(r) for pid, r in rows.items()}
    answers = read_jsonl(ANSWERS)

    def latest(source: list[dict], alias: str, variant: str, rep: int) -> dict[int, list]:
        out = {}
        for a in source:
            if a["status"] == "ok" and a["alias"] == alias and a["variant"] == variant and a["rep"] == rep and a["post_id"] in rows:
                out[a["post_id"]] = a["prediction"]
        return out

    old_r1 = fr.recorded()
    old_claude = read_jsonl(os.path.join("logs", "replay_claude", "answers.jsonl"))
    # name -> rep -> post -> list of member forecasts
    configs: dict[str, dict[int, dict[int, list]]] = {}
    for rep in (1, 2):
        for variant in VARIANTS:
            sol, s5 = latest(answers, "sol61", variant, rep), latest(answers, "s5", variant, rep)
            configs.setdefault(f"sol61 {variant}", {})[rep] = {p: [f] for p, f in sol.items()}
            configs.setdefault(f"s5 {variant}", {})[rep] = {p: [f] for p, f in s5.items()}
            # Production publishes the surviving member when one fails.
            configs.setdefault(f"pair {variant}", {})[rep] = {
                p: [f for f in (sol.get(p), s5.get(p)) if f is not None] for p in set(sol) | set(s5)
            }
        # Earlier answers with the production prompt (no 45780): GPT-6.1 Sol through
        # OpenRouter (forecast_replay.py) and Sonnet 5 direct, batch, high (claude_direct_replay.py).
        osol = latest(old_r1, "sol61", "base", rep)
        os5 = latest(old_claude, "s5-high", "base", rep)
        configs.setdefault("sol61 base (earlier)", {})[rep] = {p: [f] for p, f in osol.items()}
        configs.setdefault("s5 base (earlier)", {})[rep] = {p: [f] for p, f in os5.items()}
        configs.setdefault("pair base (earlier)", {})[rep] = {
            p: [f for f in (osol.get(p), os5.get(p)) if f is not None] for p in set(osol) | set(os5)
        }

    scores: dict[str, dict[int, float]] = {}  # "name r1" -> post -> score
    cdfs: dict[str, dict[int, np.ndarray]] = {}

    async def score_all() -> None:
        for name, by_rep in configs.items():
            for rep, by_post in by_rep.items():
                key = f"{name} r{rep}"
                for pid, forecasts in by_post.items():
                    if not forecasts:
                        continue
                    try:
                        members = [[tuple(p) for p in f] for f in forecasts]
                        cdf = await fr.published_cdf(members, questions[pid])
                        bucket = fr.bucket_of(questions[pid], rows[pid]["outcome"], len(cdf))
                        s = 50 * math.log(fr.pmf_at(cdf, bucket))
                    except Exception as exc:  # noqa: BLE001
                        print(f"  scoring failed: {key} {pid}: {type(exc).__name__}: {str(exc)[:120]}")
                        continue
                    scores.setdefault(key, {})[pid] = s
                    cdfs.setdefault(key, {})[pid] = cdf

    asyncio.run(score_all())

    # Balanced runs: a question counts a run for a model only if all four
    # variants were answered in that run (GPT-6.1 Sol's run 2 stopped at the
    # spending cap), and for the pair only if both members were.
    def complete(who: str, rep: int) -> set[int]:
        if who == "pair":
            return complete("sol61", rep) & complete("s5", rep)
        return set.intersection(*(set(scores.get(f"{who} {v} r{rep}", {})) for v in VARIANTS))

    runs_used: dict[str, dict[int, list[int]]] = {}
    for who in ("pair", "sol61", "s5"):
        for pid in rows:
            reps_ok = [rep for rep in (1, 2) if pid in complete(who, rep)]
            runs_used.setdefault(who, {})[pid] = reps_ok
            for v in VARIANTS:
                if reps_ok:
                    scores.setdefault(f"{who} {v}", {})[pid] = statistics.mean(scores[f"{who} {v} r{rep}"][pid] for rep in reps_ok)
    # Earlier answers: the mean of the runs available.
    for name in configs:
        if name.endswith("(earlier)"):
            r1, r2 = scores.get(f"{name} r1", {}), scores.get(f"{name} r2", {})
            scores[name] = {p: statistics.mean([t[p] for t in (r1, r2) if p in t]) for p in set(r1) | set(r2)}

    all_posts = set(rows)
    groups = {
        f"todas ({len(all_posts)})": all_posts,
        f"sem 45780 ({len(all_posts - {45780})})": all_posts - {45780},
        f"alvo da N1 ({len(all_posts & N1_TARGET)})": all_posts & N1_TARGET,
        f"fora do alvo ({len(all_posts - N1_TARGET)})": all_posts - N1_TARGET,
    }
    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    def fmt(res: dict | None) -> str:
        if not res:
            return "-"
        return f"{res['mean']:+.2f} [{res['lo']:+.2f}, {res['hi']:+.2f}] {res['better']}/{res['worse']}"

    def diff_table(table: dict[str, dict[int, float]], a: str, b: str, posts: set[int]) -> dict | None:
        return fr.paired(table.get(a, {}), table.get(b, {}), posts)

    emit("Scores: 50 ln(mass on the outcome's bucket), production CDF (PCHIP x1.15). Paired difference per question,")
    emit("mean [90% bootstrap CI] better/worse; each side is the mean of its 2 runs unless marked r1/r2.")

    emit("\n## Variants against the fresh baseline (same day, same route, production prompt)\n")
    emit("| configuration | mean score, base | " + " | ".join(groups) + " |")
    emit("|---|---|" + "---|" * len(groups))
    for who in ("pair", "sol61", "s5"):
        base = f"{who} base"
        for variant in ("N1", "N2", "N3"):
            name = f"{who} {variant}"
            mean_base = statistics.mean(scores[base].values()) if scores.get(base) else float("nan")
            emit(f"| {name} vs {base} | {mean_base:+.2f} | " + " | ".join(
                f"{fmt(diff_table(scores, name, base, posts))} (n={len(set(scores.get(name, {})) & set(scores.get(base, {})) & posts)})"
                for posts in groups.values()) + " |")

    # 2x2 factorial: each main effect uses all four cells.
    emit("\n## Main effects (2x2 factorial: N1 = mean of N1-base and N3-N2; N2 = mean of N2-base and N3-N1)\n")
    emit("| effect | " + " | ".join(groups) + " |")
    emit("|---|" + "---|" * len(groups))
    effects: dict[str, dict[int, float]] = {}
    for who in ("pair", "sol61", "s5"):
        s = {v: scores.get(f"{who} {v}", {}) for v in VARIANTS}
        common = set.intersection(*(set(t) for t in s.values())) if all(s.values()) else set()
        effects[f"{who} N1"] = {p: 0.5 * ((s["N1"][p] - s["base"][p]) + (s["N3"][p] - s["N2"][p])) for p in common}
        effects[f"{who} N2"] = {p: 0.5 * ((s["N2"][p] - s["base"][p]) + (s["N3"][p] - s["N1"][p])) for p in common}
        effects[f"{who} N1xN2"] = {p: (s["N3"][p] - s["N2"][p]) - (s["N1"][p] - s["base"][p]) for p in common}
    zero = {p: 0.0 for p in rows}
    for name in effects:
        emit(f"| {name} | " + " | ".join(fmt(fr.paired(effects[name], zero, posts)) for posts in groups.values()) + " |")

    # Each run separately: a real effect should show in both.
    emit("\n## Each run separately (variant rK against base rK, questions complete in that run)\n")
    emit("| configuration | run 1, all | run 2, all | run 1, N1 target | run 2, N1 target |")
    emit("|---|---|---|---|---|")
    for who in ("pair", "sol61", "s5"):
        for variant in ("N1", "N2", "N3"):
            cells = []
            for posts in (all_posts, all_posts & N1_TARGET):
                for rep in (1, 2):
                    res = diff_table(scores, f"{who} {variant} r{rep}", f"{who} base r{rep}", posts & complete(who, rep))
                    cells.append(f"{fmt(res)} (n={res['n'] if res else 0})")
            emit(f"| {who} {variant} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    emit("\nRuns used per question: " + ", ".join(
        f"{who} {sum(len(r) == 2 for r in runs_used[who].values())} with 2 runs, {sum(len(r) == 1 for r in runs_used[who].values())} with 1"
        for who in ("pair", "sol61", "s5")))

    # Selection: pick the best variant on one run, measure it on the other.
    emit("\n## Choosing the best variant on one run and measuring it on the other (pair, all questions)\n")
    for pick, test in ((1, 2), (2, 1)):
        ranked = []
        for variant in ("N1", "N2", "N3"):
            res = diff_table(scores, f"pair {variant} r{pick}", f"pair base r{pick}", complete("pair", pick))
            if res:
                ranked.append((res["mean"], variant))
        if not ranked:
            continue
        best_mean, best = max(ranked)
        res = diff_table(scores, f"pair {best} r{test}", f"pair base r{test}", complete("pair", test))
        emit(f"- Best on run {pick}: {best} ({best_mean:+.2f}); the same variant on run {test}: {fmt(res)} (n={res['n'] if res else 0})")

    # Noise.
    emit("\n## Noise: the production prompt asked again\n")
    emit("| comparison | all available | N1 target | other |")
    emit("|---|---|---|---|")
    for who in ("pair", "sol61", "s5"):
        for a, b, label in (
            (f"{who} base r2", f"{who} base r1", "fresh base, run 2 vs run 1"),
            (f"{who} base", f"{who} base (earlier)", "fresh base (2 runs) vs earlier base (2 runs)"),
            (f"{who} base (earlier) r2", f"{who} base (earlier) r1", "earlier base, run 2 vs run 1"),
        ):
            emit(f"| {who}: {label} | " + " | ".join(
                fmt(diff_table(scores, a, b, posts)) for posts in (all_posts, all_posts & N1_TARGET, all_posts - N1_TARGET)
            ) + " |")
            ab = set(scores.get(a, {})) & set(scores.get(b, {}))
            if ab:
                lines[-1] += f" mean |change| {statistics.mean(abs(scores[a][p] - scores[b][p]) for p in ab):.1f}"
                print(f"   mean |change| {statistics.mean(abs(scores[a][p] - scores[b][p]) for p in ab):.1f}")

    # Robustness: the pair's difference without the questions that moved most.
    emit("\n## Robustness: pair difference without the largest movers\n")
    for variant in ("N1", "N2", "N3"):
        a, b = scores.get(f"pair {variant}", {}), scores.get("pair base", {})
        diffs = sorted(((abs(a[p] - b[p]), p) for p in set(a) & set(b)), reverse=True)
        for k in (1, 2):
            drop = {p for _, p in diffs[:k]}
            emit(f"- pair {variant} without {sorted(drop)}: {fmt(diff_table(scores, f'pair {variant}', 'pair base', all_posts - drop))}")

    # Sensitivity to the CDF construction: the same answers built three ways.
    emit("\n## The same answers under other CDF constructions (pair, variant against base)\n")
    from numeric_replay import final_cdf
    alt: dict[str, dict[int, float]] = {}

    async def score_alt() -> None:
        for label, method, widen in (("pchip x1.0", "pchip", 1.0), ("linear x1.0 (library)", "linear", 1.0)):
            for v in VARIANTS:
                for pid in rows:
                    reps_ok = runs_used["pair"][pid]
                    vals = []
                    for rep in reps_ok:
                        members = [[tuple(p) for p in f] for f in configs[f"pair {v}"][rep][pid]]
                        cdf = await final_cdf(members, questions[pid], method, widen)
                        vals.append(50 * math.log(fr.pmf_at(cdf, fr.bucket_of(questions[pid], rows[pid]["outcome"], len(cdf)))))
                    if vals:
                        alt.setdefault(f"{label} {v}", {})[pid] = statistics.mean(vals)

    asyncio.run(score_alt())
    emit("| construction | N1 | N2 | N3 | base mean score |")
    emit("|---|---|---|---|---|")
    emit("| pchip x1.15 (production) | " + " | ".join(fmt(diff_table(scores, f"pair {v}", "pair base", all_posts)) for v in ("N1", "N2", "N3"))
         + f" | {statistics.mean(scores['pair base'].values()):+.2f} |")
    for label in ("pchip x1.0", "linear x1.0 (library)"):
        emit(f"| {label} | " + " | ".join(fmt(diff_table(alt, f"{label} {v}", f"{label} base", all_posts)) for v in ("N1", "N2", "N3"))
             + f" | {statistics.mean(alt[f'{label} base'].values()):+.2f} |")

    # Per-question table.
    emit("\n## Per question (pair, mean of 2 runs): base score and each variant's difference\n")
    emit("| post | type | outcome | N1 target | base | N1 | N2 | N3 | sol61 N1 / N2 / N3 | s5 N1 / N2 / N3 |")
    emit("|---|---|---|---|---|---|---|---|---|---|")
    order = sorted(rows, key=lambda p: -max(abs(scores.get(f"pair {v}", {}).get(p, 0) - scores.get("pair base", {}).get(p, 0)) for v in ("N1", "N2", "N3")))
    for pid in order:
        base = scores.get("pair base", {}).get(pid)
        if base is None:
            continue

        def d(who: str, v: str) -> str:
            a, b = scores.get(f"{who} {v}", {}).get(pid), scores.get(f"{who} base", {}).get(pid)
            return "-" if a is None or b is None else f"{a - b:+.1f}"

        emit(f"| {pid}{' *' if pid in NAMED else ''} | {rows[pid]['type'][:4]} | {rows[pid]['outcome']} | {'yes' if pid in N1_TARGET else ''} | "
             f"{base:+.1f} | {d('pair', 'N1')} | {d('pair', 'N2')} | {d('pair', 'N3')} | "
             f"{d('sol61', 'N1')} / {d('sol61', 'N2')} / {d('sol61', 'N3')} | {d('s5', 'N1')} / {d('s5', 'N2')} / {d('s5', 'N3')} |")

    # Mechanism checks.
    emit("\n## Floors and zeros: what each variant did on the named questions\n")
    emit("Member answers below the floor = share of answers with any declared percentile under the last confirmed value.")
    emit("Pair CDF mass = probability the published pair CDF (mean of 2 runs) puts below the floor, or on the outcome's bucket for zero outcomes.\n")
    emit("| post | quantity | " + " | ".join(VARIANTS) + " |")
    emit("|---|---|" + "---|" * len(VARIANTS))
    for pid in NAMED:
        if pid not in rows:
            continue
        q = questions[pid]
        if pid in FLOORS:
            floor = FLOORS[pid]
            cells = []
            for v in VARIANTS:
                preds = [a["prediction"] for a in answers if a["status"] == "ok" and a["post_id"] == pid and a["variant"] == v]
                below = sum(any(val < floor - 1e-6 for _, val in p) for p in preds)
                cells.append(f"{below}/{len(preds)}")
            emit(f"| {pid} | member answers below the floor {floor:g} | " + " | ".join(cells) + " |")
            cells = []
            for v in VARIANTS:
                vals = []
                for rep in (1, 2):
                    cdf = cdfs.get(f"pair {v} r{rep}", {}).get(pid)
                    if cdf is not None:
                        # Mass strictly below the floor's bucket.
                        b = fr.bucket_of(q, floor, len(cdf))
                        vals.append(float(np.concatenate([[0.0], cdf])[b]))
                cells.append(f"{statistics.mean(vals):.3f}" if vals else "-")
            emit(f"| {pid} | pair CDF mass below the floor | " + " | ".join(cells) + " |")
        cells = []
        for v in VARIANTS:
            vals = []
            for rep in (1, 2):
                cdf = cdfs.get(f"pair {v} r{rep}", {}).get(pid)
                if cdf is not None:
                    vals.append(fr.pmf_at(cdf, fr.bucket_of(q, rows[pid]["outcome"], len(cdf))))
            cells.append(f"{statistics.mean(vals):.3f}" if vals else "-")
        emit(f"| {pid} | pair CDF mass on the outcome ({rows[pid]['outcome']:g}) | " + " | ".join(cells) + " |")
        for who in ("sol61", "s5"):
            cells = []
            for v in VARIANTS:
                preds = [a["prediction"] for a in answers if a["status"] == "ok" and a["post_id"] == pid and a["variant"] == v and a["alias"] == who]
                cells.append("; ".join(f"{p[0][1]:g}..{p[-1][1]:g}" for p in preds) or "-")
            emit(f"| {pid} | {who} lowest..highest declared value | " + " | ".join(cells) + " |")

    # Spread: P10 to P90 width of the published pair CDF, on the question axis.
    emit("\n## Spread of the published CDF (P10 to P90 width as a share of the question range; median over questions)\n")
    emit("| who | " + " | ".join(VARIANTS) + " |")
    emit("|---|" + "---|" * len(VARIANTS))
    for who in ("pair", "sol61", "s5"):
        cells = []
        for v in VARIANTS:
            widths = []
            for rep in (1, 2):
                for pid, cdf in cdfs.get(f"{who} {v} r{rep}", {}).items():
                    widths.append(_quantile_location(cdf, 0.9) - _quantile_location(cdf, 0.1))
            cells.append(f"{statistics.median(widths):.3f}" if widths else "-")
        emit(f"| {who} | " + " | ".join(cells) + " |")

    # Parsing, failures, tokens and cost.
    emit("\n## Answers, parsing, tokens and cost\n")
    emit("| model | variant | ok | requested without an answer | parse check not ok | output tokens (median) | reasoning tokens (median) | $ per answer, forecaster | $ per answer, parser |")
    emit("|---|---|---|---|---|---|---|---|---|")
    for who in ("sol61", "s5"):
        for v in VARIANTS:
            items = [a for a in answers if a["alias"] == who and a["variant"] == v]
            ok = [a for a in items if a["status"] == "ok"]
            missing = {(a["post_id"], a["rep"]) for a in items} - {(a["post_id"], a["rep"]) for a in ok}
            bad_parse = [a for a in ok if not str(a.get("parse_check", "")).startswith("ok")]
            fc = [u for a in items for u in a.get("usage", []) if (u.get("model") or "") != PARSER]
            pc = [sum(usage_cost(u) for u in a.get("usage", []) if (u.get("model") or "") == PARSER) for a in ok]
            emit(
                f"| {who} | {v} | {len(ok)} | {len(missing)} | {len(bad_parse)} | "
                f"{statistics.median([u.get('completion_tokens') or 0 for u in fc]) if fc else 0:.0f} | "
                f"{statistics.median([u.get('reasoning_tokens') or 0 for u in fc]) if fc else 0:.0f} | "
                f"{statistics.mean([usage_cost(u) for u in fc]) if fc else float('nan'):.4f} | "
                f"{statistics.mean(pc) if pc else float('nan'):.4f} |"
            )
    answered = {(a["post_id"], a["alias"], a["variant"], a["rep"]) for a in answers if a["status"] == "ok"}
    failures = [a for a in answers if a["status"] != "ok" and (a["post_id"], a["alias"], a["variant"], a["rep"]) not in answered]
    bad = [a for a in answers if a["status"] == "ok" and not str(a.get("parse_check", "")).startswith("ok")]
    if failures or bad:
        emit("\n## Failed answers and parse warnings\n")
        for a in failures:
            emit(f"- {a['post_id']} {a['alias']} {a['variant']} r{a['rep']}: {a['status']} {a['error']}")
        for a in bad:
            emit(f"- {a['post_id']} {a['alias']} {a['variant']} r{a['rep']}: parse check {a['parse_check']}")

    data = _budget()
    if data:
        base, last = data["baseline"], data["readings"][-1]
        emit(f"\nOpenRouter key drop since the first reading: ${base['limit_remaining'] - last['limit_remaining']:.2f} "
             f"({base['time']} to {last['time']}; includes anything else spending the key meanwhile).")
    emit(f"OpenRouter token estimate of the GPT calls: ${spent_openrouter_tokens():.2f}. Anthropic estimate: ${spent_anthropic():.2f}.")

    with open(os.path.join(OUT, "analysis.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT, "scores.json"), "w", encoding="utf-8") as fh:
        json.dump({"log": scores, "effects": effects}, fh, indent=1)
    print(f"\nWrote {OUT}/analysis.md and {OUT}/scores.json")


def main_cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("refresh", help="fetch the posts that had no outcome (Metaculus only)")
    sub.add_parser("prompts", help="build, check and print the edited prompts (no LLM calls)")
    run = sub.add_parser("run-gpt", help="GPT-6.1 Sol through OpenRouter, capped on the key balance")
    run.add_argument("--variants", default="base,N1,N2,N3")
    run.add_argument("--reps", default="1,2")
    run.add_argument("--posts", default="")
    run.add_argument("--label", default="run")
    submit = sub.add_parser("submit", help="one Claude Sonnet 5 batch per variant and run")
    submit.add_argument("--variants", default="base,N1,N2,N3")
    submit.add_argument("--reps", default="1,2")
    submit.add_argument("--posts", default="")
    collect = sub.add_parser("collect", help="fetch finished batches and parse their answers")
    collect.add_argument("--wait", action="store_true")
    sub.add_parser("parse-stopped", help="parse GPT answers received before the spending stop (Anthropic parser only)")
    sub.add_parser("analyze", help="score everything recorded (no LLM calls)")
    sub.add_parser("spend", help="spend so far")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%H:%M:%S")
    logger.setLevel(logging.INFO)
    os.makedirs(OUT, exist_ok=True)
    {
        "refresh": cmd_refresh, "prompts": cmd_prompts, "run-gpt": cmd_run_gpt, "submit": cmd_submit,
        "collect": cmd_collect, "parse-stopped": cmd_parse_stopped, "analyze": cmd_analyze, "spend": cmd_spend,
    }[args.command](args)


if __name__ == "__main__":
    main_cli()
