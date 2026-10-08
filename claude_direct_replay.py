"""
Replays the forecast step of MiniBench round 1 with Claude models called
through the Anthropic API directly (ANTHROPIC_API_KEY), to choose the model
and thinking setting for the Anthropic seat of the ensemble.

Everything but the route comes from forecast_replay.py: the 56 scored
questions, the frozen research, the original dates, the bot's own prompts and
parsers, and the scoring. The differences:

- Forecasts are requested through the Message Batches API (half the list
  price), or with plain calls for the calibration, with adaptive thinking and
  an explicit effort level. Claude Sonnet 5, Sonnet 5.5 and Opus 5.5 take no
  thinking budget: the Models API lists thinking type "enabled" as
  unsupported for all three, and litellm silently turns a budget_tokens value
  into an effort level (under 2,048 low, under 4,096 medium, under 8,192
  high, else xhigh). Effort is therefore the only control.
- The parser is claude-haiku-4-5 on the same Anthropic key instead of
  gpt-4o-mini on OpenRouter, so nothing here spends the OpenRouter key; a
  call to any other provider is refused. check-parser compares it with
  gpt-4o-mini's parses of answers already recorded.
- The binary prompt is today's, which since 2026-10-06 is the replay's
  variant V2 (status quo as a trajectory). The Claude Sonnet 5 reference is
  therefore its OpenRouter answers with V2 on binaries and the production
  prompt elsewhere. GPT-6.1 Sol is used as recorded, so its binaries predate
  V2; it is the same member in every pair, so pair differences come from the
  Claude member alone.

Each answer is appended to logs/replay_claude/answers.jsonl in the format of
forecast_replay.py, plus the effort, route and stop reason; every model
response is kept in raw.jsonl, so parsing and scoring can be redone without
new calls. Costs are estimated from the token usage of every call at
Anthropic list prices, halved for batches; a submission that could take the
total past SPEND_CAP is refused.

Usage (the anthropic SDK is not a project dependency, hence --with):
    uv run --with anthropic python claude_direct_replay.py prompts
    uv run --with anthropic python claude_direct_replay.py check-parser
    uv run --with anthropic python claude_direct_replay.py calibrate --configs s5-high,s5-xhigh [--samples 2]
    uv run --with anthropic python claude_direct_replay.py submit --configs s5-xhigh,s55-xhigh --reps 1,2
    uv run --with anthropic python claude_direct_replay.py collect [--wait]
    uv run --with anthropic python claude_direct_replay.py analyze
    uv run --with anthropic python claude_direct_replay.py spend

A configuration is <model>-<effort>, with model s5 (claude-sonnet-5), s55
(claude-sonnet-5-5) or o55 (claude-opus-5-5) and effort low, medium, high,
xhigh or max.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import statistics
import time
from datetime import datetime, timezone

PARSER = "anthropic/claude-haiku-4-5"
# main.py reads PARSER_MODEL once, when forecast_replay imports it.
os.environ["PARSER_MODEL"] = PARSER

import anthropic  # noqa: E402

import bot as bot_module  # noqa: E402
import forecast_replay as fr  # noqa: E402
import main  # noqa: E402
from forecasting_tools import GeneralLlm  # noqa: E402
from forecasting_tools.ai_models import general_llm  # noqa: E402

logger = logging.getLogger("claude_direct_replay")

OUT = os.path.join("logs", "replay_claude")
ANSWERS = os.path.join(OUT, "answers.jsonl")
RAW = os.path.join(OUT, "raw.jsonl")
PROMPTS = os.path.join(OUT, "prompts.jsonl")
BATCHES = os.path.join(OUT, "batches.json")
PARSER_CHECK = os.path.join(OUT, "parser_check.jsonl")

# forecast_replay points the bot's own log at logs/replay_r1/; these runs get theirs.
bot_module.FORECAST_LOG = os.path.join(OUT, "forecasts.jsonl")

SPEND_CAP = 120.0  # dollars, estimated from token usage
MAX_TOKENS = 16000  # thinking plus answer; a stop at the cap is recorded
IN_FLIGHT = 4  # plain calls at once, parser included

MODELS = {"s5": "claude-sonnet-5", "s55": "claude-sonnet-5-5", "o55": "claude-opus-5-5"}
EFFORTS = ("low", "medium", "high", "xhigh", "max")

# Anthropic list prices in dollars per million tokens (input, output), as of
# 2026-10-08. Batches cost half.
PRICES = {
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
BATCH_SHARE = 0.5

# Output tokens assumed per answer when nothing has been measured yet, for
# the spending guard only. Deliberately high.
OUTPUT_GUESS = {"low": 3000, "medium": 4000, "high": 6000, "xhigh": 9000, "max": 14000}


# ---------------------------------------------------------------------------
# Guard: only the Anthropic key is spent
# ---------------------------------------------------------------------------

_fr_acompletion = general_llm.acompletion


async def _anthropic_only(*args, **kwargs):
    model = str(kwargs.get("model") or (args[0] if args else ""))
    if not model.startswith("anthropic/"):
        raise RuntimeError(f"refused a call to {model}: this script only spends the Anthropic key")
    return await _fr_acompletion(*args, **kwargs)


general_llm.acompletion = _anthropic_only


# ---------------------------------------------------------------------------
# Files, configurations and costs
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


def parse_config(name: str) -> tuple[str, str]:
    alias, _, effort = name.rpartition("-")
    if alias not in MODELS or effort not in EFFORTS:
        raise SystemExit(f"unknown configuration {name!r}: use <{'|'.join(MODELS)}>-<{'|'.join(EFFORTS)}>")
    return MODELS[alias], effort


def usage_cost(u: dict) -> float:
    model = (u.get("model") or "").removeprefix("anthropic/")
    if model not in PRICES:
        raise KeyError(f"no price for {model!r}")
    pin, pout = PRICES[model]
    share = BATCH_SHARE if u.get("batch") else 1.0
    return share * ((u.get("prompt_tokens") or 0) * pin + (u.get("completion_tokens") or 0) * pout) / 1e6


def list_cost(u: dict) -> float:
    """The same call at the standard (not batch) price, which is what production would pay."""
    return usage_cost({**u, "batch": False})


def is_parser(u: dict) -> bool:
    return (u.get("model") or "") == PARSER


def spent() -> float:
    """Every forecaster response (raw.jsonl) plus every parser call recorded."""
    total = sum(usage_cost(r["usage"]) for r in read_jsonl(RAW) if r.get("usage"))
    for row in read_jsonl(ANSWERS) + read_jsonl(PARSER_CHECK):
        total += sum(usage_cost(u) for u in row.get("usage", []) if is_parser(u))
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


def estimate(model: str, effort: str, prompt: str, batch: bool) -> float:
    """Upper estimate of one request: 1 token per 3 characters of prompt, and
    1.5 times the mean output measured for this model and effort (or a high
    guess when none was measured)."""
    measured = [
        r["usage"]["completion_tokens"] for r in read_jsonl(RAW)
        if r.get("usage") and r.get("requested_model") == model and r.get("effort") == effort
    ]
    out = 1.5 * statistics.mean(measured) if measured else OUTPUT_GUESS[effort]
    pin, pout = PRICES[model]
    share = BATCH_SHARE if batch else 1.0
    return share * (len(prompt) / 3 * pin + out * pout) / 1e6


def check_cap(extra: float, what: str) -> None:
    done, pending = spent(), pending_estimate()
    print(f"Spent so far ${done:.2f}, pending batches up to ${pending:.2f}, {what} up to ${extra:.2f} (cap ${SPEND_CAP:.0f})")
    if done + pending + extra > SPEND_CAP:
        raise SystemExit("Refused: this could take the estimated spend past the cap.")


# ---------------------------------------------------------------------------
# Questions, prompts and the bot
# ---------------------------------------------------------------------------


def scored_rows() -> list[dict]:
    return [r for r in fr.load_questions() if r["outcome"] is not None]


def calibration_rows(rows: list[dict]) -> list[dict]:
    """The same 3 questions as the previous smoke test: one of each kind, alternating types."""
    by_id = {r["post_id"]: r for r in rows}
    picked = main._pick([fr.question_object(r) for r in rows], 3)
    return [by_id[q.id_of_post] for q in picked]


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def replay_bot(members: list) -> object:
    """A production bot with the given members, no shadows, and the Anthropic parser."""
    bot = main.build_bot(publish=False, samples=1)
    bot._shadows = []
    bot._ensemble = list(members)
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
        await fr.forecast_once(replay_bot([capture]), fr.question_object(row), row["research"])
    except RuntimeError:
        pass
    return capture.prompt


class CannedLlm:
    """Returns an answer already obtained from the model, after checking that
    the bot built the same prompt that was sent."""

    def __init__(self, model: str, text: str, expected_sha1: str | None) -> None:
        self.model = model
        self._text = text
        self._expected = expected_sha1

    async def invoke(self, prompt: str) -> str:
        got = sha1(prompt)
        if self._expected is not None and got != self._expected:
            raise RuntimeError(f"prompt mismatch: sent {self._expected}, rebuilt {got}")
        call = fr._CALL.get()
        if call is not None:
            call["prompt_chars"] = len(prompt)
            call["prompt_sha1"] = got
            call["raw"] = self._text
        return self._text


def load_prompts() -> dict[int, dict]:
    prompts = {p["post_id"]: p for p in read_jsonl(PROMPTS)}
    if not prompts:
        raise SystemExit(f"{PROMPTS} is missing: run the prompts command first")
    return prompts


def request_params(config: str, prompt: str) -> dict:
    """What GeneralLlm sends through litellm for reasoning_effort=<effort> on
    these models (adaptive thinking plus output_config.effort), without the
    temperature, which these models reject with thinking on."""
    model, effort = parse_config(config)
    return {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": effort},
        "messages": [{"role": "user", "content": prompt}],
    }


def response_row(msg, model: str, batch: bool, seconds: float | None = None) -> dict:
    usage = msg.usage.model_dump()
    thinking = (usage.get("output_tokens_details") or {}).get("thinking_tokens")
    stop_details = getattr(msg, "stop_details", None)
    return {
        "served_model": msg.model,
        "stop_reason": msg.stop_reason,
        "stop_details": stop_details.model_dump() if hasattr(stop_details, "model_dump") else stop_details,
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


async def finish(row: dict, config: str, variant: str, rep: int, expected_sha1: str | None,
                 resp: dict | None, route: str, error: str | None = None) -> dict:
    """Runs the bot's own parse on a model answer and records the result."""
    model, effort = parse_config(config)
    call: dict = {"usage": [resp["usage"]] if resp and resp.get("usage") else []}
    fr._CALL.set(call)
    fr._TODAY.set(datetime.fromisoformat(row["forecast_time"]))
    status, prediction = "ok", None
    try:
        if error:
            raise RuntimeError(error)
        if not resp["text"].strip():
            raise RuntimeError(f"empty answer (stop_reason {resp['stop_reason']})")
        bot = replay_bot([CannedLlm(f"anthropic/{model}", resp["text"], expected_sha1)])
        result = await fr.forecast_once(bot, fr.question_object(row), row["research"])
        prediction = fr.serialize(result.prediction_value)
    except Exception as exc:  # noqa: BLE001
        status, error = "failed", f"{type(exc).__name__}: {str(exc)[:300]}"
    out = {
        "time": _utc_now(), "post_id": row["post_id"], "type": row["type"],
        "alias": config, "model": model, "effort": effort, "variant": variant, "rep": rep,
        "route": route, "forecast_time": row["forecast_time"], "status": status, "error": error,
        "prediction": prediction,
        "stop_reason": resp.get("stop_reason") if resp else None,
        "seconds": resp.get("seconds") if resp else None,
        **call,
    }
    append_jsonl(ANSWERS, out)
    thinking = (resp or {}).get("usage", {}).get("reasoning_tokens")
    shown = prediction if not isinstance(prediction, list) else f"P10..P90 {prediction[0][1]}..{prediction[-1][1]}"
    if isinstance(shown, dict):
        shown = {k[:12]: round(v, 2) for k, v in shown.items()}
    logger.info(
        f"{row['post_id']} {row['type'][:8]} {config} {variant} r{rep}: {status} "
        f"{shown if status == 'ok' else error} (thinking {thinking}, {(resp or {}).get('seconds')} s)"
    )
    return out


# ---------------------------------------------------------------------------
# Commands: prompts, check-parser, calibrate
# ---------------------------------------------------------------------------


def cmd_prompts(_args) -> None:
    """Builds and saves the prompt of every scored question, without calling a model,
    and checks it against the prompts the recorded Sonnet 5 answers were given."""
    rows = scored_rows()
    level = bot_module.logger.level
    bot_module.logger.setLevel(logging.ERROR)  # every capture logs a failed member

    async def go() -> list[str]:
        return [await build_prompt(r) for r in rows]

    prompts = asyncio.run(go())
    bot_module.logger.setLevel(level)

    # The replay's V2 edit wrote the new sentence on one line; bot.py wraps it.
    v2 = fr.V2[0][1]
    recorded: dict[tuple[int, str], set[str]] = {}
    for a in fr.recorded():
        if a["alias"] == "sonnet5" and a["status"] == "ok":
            recorded.setdefault((a["post_id"], a["variant"]), set()).add(a.get("prompt_sha1"))
    same, different = 0, []
    with open(PROMPTS, "w", encoding="utf-8") as fh:
        for row, prompt in zip(rows, prompts):
            fh.write(json.dumps({"post_id": row["post_id"], "type": row["type"], "sha1": sha1(prompt),
                                 "chars": len(prompt), "prompt": prompt}, ensure_ascii=False) + "\n")
            if row["type"] == "binary":
                as_v2, n = re.subn(fr._words(v2), lambda _m: v2, prompt, flags=re.S)
                ok = n == 1 and sha1(as_v2) in recorded.get((row["post_id"], "V2"), set())
            else:
                ok = sha1(prompt) in recorded.get((row["post_id"], "base"), set())
            same += ok
            if not ok:
                different.append(row["post_id"])
    print(f"Wrote {len(prompts)} prompts to {PROMPTS}.")
    print(f"Identical to the prompt of the recorded Sonnet 5 answers (V2 on binaries): {same} of {len(rows)}")
    if different:
        print(f"Different: {different}")


def cmd_check_parser(args) -> None:
    """Re-parses recorded answers with the Anthropic parser and compares with gpt-4o-mini's parse."""
    rows = {r["post_id"]: r for r in scored_rows()}
    wanted = [a.strip() for a in args.aliases.split(",") if a.strip()]
    done = {(c["post_id"], c["alias"], c["rep"]) for c in read_jsonl(PARSER_CHECK)}
    items = [
        a for a in fr.recorded()
        if a["status"] == "ok" and a["alias"] in wanted and a["variant"] == "base" and a["rep"] == 1
        and a["post_id"] in rows and (a["post_id"], a["alias"], a["rep"]) not in done
    ]
    # One of each post and alias (the latest), then the first --n.
    latest = {(a["post_id"], a["alias"]): a for a in items}
    items = list(latest.values())[: args.n] if args.n else list(latest.values())
    check_cap(len(items) * 2 * (3000 * 1.0 + 300 * 5.0) / 1e6, f"{len(items)} parser checks")

    async def one(a: dict) -> None:
        row = rows[a["post_id"]]
        call: dict = {}
        fr._CALL.set(call)
        fr._TODAY.set(datetime.fromisoformat(row["forecast_time"]))
        status, prediction, error = "ok", None, None
        try:
            bot = replay_bot([CannedLlm(a["model"], a["raw"], None)])
            result = await fr.forecast_once(bot, fr.question_object(row), row["research"])
            prediction = fr.serialize(result.prediction_value)
        except Exception as exc:  # noqa: BLE001
            status, error = "failed", f"{type(exc).__name__}: {str(exc)[:300]}"
        append_jsonl(PARSER_CHECK, {
            "time": _utc_now(), "post_id": a["post_id"], "type": a["type"], "alias": a["alias"],
            "rep": a["rep"], "status": status, "error": error, "old": a["prediction"], "new": prediction,
            "usage": call.get("usage", []), "warnings": call.get("warnings"),
        })

    async def go() -> None:
        fr._SLOTS = asyncio.Semaphore(IN_FLIGHT)
        await asyncio.gather(*(one(a) for a in items))

    asyncio.run(go())
    report_parser_check()


def _differences(old, new) -> float:
    """Largest absolute difference between two parsed forecasts (relative for numeric values)."""
    if isinstance(old, (int, float)):
        return abs(old - new)
    if isinstance(old, dict):
        return max(abs(old[k] - new.get(k, 0.0)) for k in old)
    worst = 0.0
    for (p1, v1), (p2, v2) in zip(old, new):
        scale = max(abs(v1), abs(v2), 1e-9)
        worst = max(worst, abs(p1 - p2), abs(v1 - v2) / scale)
    return worst if len(old) == len(new) else float("inf")


def report_parser_check() -> None:
    checks = read_jsonl(PARSER_CHECK)
    by_type: dict[str, list[float]] = {}
    failures = []
    for c in checks:
        if c["status"] != "ok":
            failures.append(c)
            continue
        by_type.setdefault(c["type"], []).append(_differences(c["old"], c["new"]))
    print("\nAnthropic parser against gpt-4o-mini on recorded answers:")
    for kind, diffs in sorted(by_type.items()):
        exact = sum(d < 1e-6 for d in diffs)
        print(f"  {kind}: {len(diffs)} answers, {exact} identical, largest difference {max(diffs):.4f}")
    for c in failures:
        print(f"  FAILED {c['post_id']} {c['alias']}: {c['error']} {c.get('warnings')}")
    cost = sum(usage_cost(u) for c in checks for u in c.get("usage", []))
    print(f"  parser cost ${cost:.3f}")


def cmd_calibrate(args) -> None:
    """Plain calls on 3 questions, to measure thinking tokens per effort level."""
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    for c in configs:
        parse_config(c)
    rows = calibration_rows(scored_rows())
    prompts = load_prompts()
    done = {
        (a["post_id"], a["alias"], a["rep"]) for a in read_jsonl(ANSWERS)
        if a["variant"] == "calib" and a["status"] == "ok"
    }
    jobs = [
        (row, config, sample) for config in configs for sample in range(1, args.samples + 1)
        for row in rows if (row["post_id"], config, sample) not in done
    ]
    extra = sum(estimate(*parse_config(c), prompts[r["post_id"]]["prompt"], batch=False) for r, c, _ in jobs)
    check_cap(extra, f"{len(jobs)} calibration calls")
    client = anthropic.AsyncAnthropic(max_retries=3, timeout=900)

    async def one(row: dict, config: str, sample: int) -> None:
        model, effort = parse_config(config)
        prompt = prompts[row["post_id"]]
        resp, error = None, None
        try:
            async with fr._SLOTS:
                started = time.monotonic()
                msg = await client.messages.create(**request_params(config, prompt["prompt"]))
                resp = response_row(msg, model, batch=False, seconds=time.monotonic() - started)
        except anthropic.APIError as exc:
            error = f"{type(exc).__name__}: {str(exc)[:300]}"
        append_jsonl(RAW, {
            "time": _utc_now(), "custom_id": f"calib_{config}_s{sample}_{row['post_id']}", "batch_id": None,
            "config": config, "requested_model": model, "effort": effort, "variant": "calib", "rep": sample,
            "post_id": row["post_id"], "prompt_sha1": prompt["sha1"], "error": error, **(resp or {}),
        })
        await finish(row, config, "calib", sample, prompt["sha1"], resp, "direct", error)

    async def go() -> None:
        fr._SLOTS = asyncio.Semaphore(IN_FLIGHT)
        await asyncio.gather(*(one(*job) for job in jobs))

    asyncio.run(go())
    report_calibration(rows)


def openrouter_reference(rows: list[dict]) -> dict[int, list[int]]:
    """Reasoning tokens of Claude Sonnet 5 through OpenRouter on these posts, with today's prompt
    (V2 on binaries), from the earlier replay."""
    wanted = {r["post_id"]: ("V2" if r["type"] == "binary" else "base") for r in rows}
    out: dict[int, list[int]] = {}
    for a in fr.recorded():
        if a["alias"] == "sonnet5" and a["status"] == "ok" and wanted.get(a["post_id"]) == a["variant"]:
            for u in a.get("usage", []):
                if "claude" in (u.get("model") or ""):
                    out.setdefault(a["post_id"], []).append(u.get("reasoning_tokens") or 0)
    return out


def report_calibration(rows: list[dict]) -> None:
    ref = openrouter_reference(rows)
    raws = [r for r in read_jsonl(RAW) if r.get("variant") == "calib" and r.get("usage")]
    configs = sorted({r["config"] for r in raws})
    print("\nThinking tokens per answer (each cell lists the samples):")
    header = "| configuration | " + " | ".join(f"{r['post_id']} {r['type'][:8]}" for r in rows) + " | median | output median | seconds median | $ per answer |"
    print(header)
    print("|---" * (len(rows) + 5) + "|")
    print("| sonnet5 OpenRouter high (earlier replay) | " + " | ".join(
        ", ".join(str(t) for t in ref.get(r["post_id"], [])) for r in rows
    ) + f" | {statistics.median([t for r in rows for t in ref.get(r['post_id'], [])] or [0]):.0f} | | | |")
    for config in configs:
        mine = [r for r in raws if r["config"] == config]
        cells = []
        for row in rows:
            cells.append(", ".join(str(r["usage"]["reasoning_tokens"]) for r in mine if r["post_id"] == row["post_id"]))
        thinking = [r["usage"]["reasoning_tokens"] or 0 for r in mine]
        outputs = [r["usage"]["completion_tokens"] for r in mine]
        secs = [r["seconds"] for r in mine if r.get("seconds") is not None]
        cost = statistics.mean(usage_cost(r["usage"]) for r in mine)
        stops = [r["stop_reason"] for r in mine if r["stop_reason"] != "end_turn"]
        print(
            f"| {config} | " + " | ".join(cells) + f" | {statistics.median(thinking):.0f} | {statistics.median(outputs):.0f}"
            f" | {statistics.median(secs) if secs else float('nan'):.0f} | {cost:.3f} |" + (f" stops: {stops}" if stops else "")
        )
    print(f"\nSpent so far: ${spent():.2f}")


# ---------------------------------------------------------------------------
# Commands: submit and collect batches
# ---------------------------------------------------------------------------


def cmd_submit(args) -> None:
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    reps = [int(r) for r in args.reps.split(",") if r.strip()]
    for c in configs:
        parse_config(c)
    rows = scored_rows()
    if args.posts:
        wanted = {int(p) for p in args.posts.split(",")}
        rows = [r for r in rows if r["post_id"] in wanted]
    prompts = load_prompts()
    done = {
        (a["post_id"], a["alias"], a["rep"]) for a in read_jsonl(ANSWERS)
        if a["variant"] == "base" and a["status"] == "ok"
    }
    batches = load_batches()
    pending = {
        (meta["post_id"], b["config"], b["rep"])
        for b in batches if not b.get("collected") for meta in b["requests"].values()
    }
    client = anthropic.Anthropic(max_retries=3)
    for config in configs:
        model, effort = parse_config(config)
        for rep in reps:
            todo = [r for r in rows if (r["post_id"], config, rep) not in done | pending]
            if not todo:
                print(f"{config} r{rep}: nothing to request")
                continue
            extra = sum(estimate(model, effort, prompts[r["post_id"]]["prompt"], batch=True) for r in todo)
            check_cap(extra, f"batch {config} r{rep} ({len(todo)} requests)")
            requests = [
                {"custom_id": f"{config}_r{rep}_{r['post_id']}", "params": request_params(config, prompts[r["post_id"]]["prompt"])}
                for r in todo
            ]
            batch = client.messages.batches.create(requests=requests)
            batches.append({
                "id": batch.id, "config": config, "rep": rep, "created": _utc_now(), "estimate": round(extra, 4),
                "requests": {
                    req["custom_id"]: {"post_id": r["post_id"], "sha1": prompts[r["post_id"]]["sha1"]}
                    for req, r in zip(requests, todo)
                },
                "collected": False,
            })
            save_batches(batches)
            print(f"{config} r{rep}: batch {batch.id} with {len(requests)} requests ({batch.processing_status})")


def cmd_collect(args) -> None:
    client = anthropic.Anthropic(max_retries=3)
    rows = {r["post_id"]: r for r in scored_rows()}
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
                print(
                    f"{_utc_now()} {b['config']} r{b['rep']} {b['id']}: {info.processing_status}, "
                    f"{counts.processing} processing, {counts.succeeded} succeeded, {counts.errored} errored"
                )
                continue
            model, effort = parse_config(b["config"])
            items = []
            for result in client.messages.batches.results(b["id"]):
                meta = b["requests"][result.custom_id]
                resp, error = None, None
                if result.result.type == "succeeded":
                    resp = response_row(result.result.message, model, batch=True)
                else:
                    detail = getattr(result.result, "error", None)
                    error = f"batch result {result.result.type}: {str(detail)[:300]}"
                append_jsonl(RAW, {
                    "time": _utc_now(), "custom_id": result.custom_id, "batch_id": b["id"], "config": b["config"],
                    "requested_model": model, "effort": effort, "variant": "base", "rep": b["rep"],
                    "post_id": meta["post_id"], "prompt_sha1": meta["sha1"], "error": error, **(resp or {}),
                })
                items.append((rows[meta["post_id"]], meta["sha1"], resp, error))

            async def parse_all(items=items, b=b) -> None:
                fr._SLOTS = asyncio.Semaphore(IN_FLIGHT)
                sem = asyncio.Semaphore(IN_FLIGHT)

                async def one(row, expected, resp, error):
                    async with sem:
                        await finish(row, b["config"], "base", b["rep"], expected, resp, "batch", error)

                await asyncio.gather(*(one(*item) for item in items))

            asyncio.run(parse_all())
            b["collected"] = True
            b["ended"] = info.ended_at.isoformat() if info.ended_at else None
            b["counts"] = {"succeeded": counts.succeeded, "errored": counts.errored,
                           "expired": counts.expired, "canceled": counts.canceled}
            save_batches(batches)
            print(f"{b['config']} r{b['rep']}: collected {len(items)} results; spent so far ${spent():.2f}")
        if not args.wait:
            break
        if not all(b.get("collected") for b in load_batches()):
            time.sleep(60)


def cmd_spend(_args) -> None:
    raws = [r for r in read_jsonl(RAW) if r.get("usage")]
    by: dict[str, float] = {}
    for r in raws:
        kind = {"calib": "calibration", "base": f"batch r{r['rep']}"}.get(r["variant"], r["variant"])
        key = f"{r['config']} {kind}"
        by[key] = by.get(key, 0.0) + usage_cost(r["usage"])
    parser = sum(usage_cost(u) for row in read_jsonl(ANSWERS) for u in row.get("usage", []) if is_parser(u))
    check = sum(usage_cost(u) for row in read_jsonl(PARSER_CHECK) for u in row.get("usage", []))
    for key, value in sorted(by.items()):
        print(f"  {key}: ${value:.3f}")
    print(f"  parser on replay answers: ${parser:.3f}")
    print(f"  parser check: ${check:.3f}")
    print(f"Total ${spent():.2f}; pending batches up to ${pending_estimate():.2f}")


# ---------------------------------------------------------------------------
# Scoring (offline)
# ---------------------------------------------------------------------------

REFERENCE = "sonnet5 OR high"
PRODUCTION = f"sol61 + {REFERENCE}"
# Claude Sonnet 5 asked again today through this script. On binaries the
# OpenRouter reference is the V2 run, the variant that was picked for
# production because it scored best on these same 29 questions, so it is
# biased upward; this one went through the same selection as the candidates.
FRESH = "s5-high direct"


def cmd_analyze(_args) -> None:
    logging.getLogger("forecasting_tools").setLevel(logging.ERROR)
    rows = {r["post_id"]: r for r in scored_rows()}
    questions = {pid: fr.question_object(r) for pid, r in rows.items()}
    old = [a for a in fr.recorded() if a["status"] == "ok" and a["post_id"] in rows]
    answers = read_jsonl(ANSWERS)
    new = [a for a in answers if a["status"] == "ok" and a["post_id"] in rows and a["variant"] == "base"]

    def latest(source: list[dict], alias: str, variant: str, rep: int) -> dict[int, object]:
        out = {}
        for a in source:
            if a["alias"] == alias and a["variant"] == variant and a["rep"] == rep:
                out[a["post_id"]] = a["prediction"]
        return out

    # name -> rep -> post -> member forecast
    members: dict[str, dict[int, dict[int, object]]] = {}
    for rep in (1, 2):
        v2, base = latest(old, "sonnet5", "V2", rep), latest(old, "sonnet5", "base", rep)
        members.setdefault(REFERENCE, {})[rep] = {
            pid: (v2 if r["type"] == "binary" else base)[pid]
            for pid, r in rows.items() if pid in (v2 if r["type"] == "binary" else base)
        }
        members.setdefault("sol61 OR high", {})[rep] = latest(old, "sol61", "base", rep)
        # The same Sonnet 5 answers with the pre-V2 binary prompt, to see how much of
        # V2's measured gain survives in a fresh run.
        members.setdefault("sonnet5 OR high pre-V2", {})[rep] = base
        # Their binaries were asked with the pre-V2 prompt: kept out.
        for alias in ("opus55", "sonnet55"):
            members.setdefault(f"{alias} OR high (no binaries)", {})[rep] = {
                pid: f for pid, f in latest(old, alias, "base", rep).items() if rows[pid]["type"] != "binary"
            }
    claude = [REFERENCE, "sonnet5 OR high pre-V2", "opus55 OR high (no binaries)", "sonnet55 OR high (no binaries)"]
    for config in sorted({a["alias"] for a in new}):
        name = f"{config} direct"
        claude.append(name)
        for rep in sorted({a["rep"] for a in new if a["alias"] == config}):
            members.setdefault(name, {})[rep] = latest(new, config, "base", rep)
    for name in list(claude):
        if name.endswith("(no binaries)"):
            continue
        pair = f"sol61 + {name}"
        for rep, mine in members[name].items():
            sol = members["sol61 OR high"].get(rep, {})
            members.setdefault(pair, {})[rep] = {
                pid: [f for f in (sol.get(pid), mine.get(pid)) if f is not None] for pid in set(sol) | set(mine)
            }

    scores: dict[str, dict[int, float]] = {}
    briers: dict[str, dict[int, float]] = {}

    async def score_all() -> None:
        for name, by_rep in members.items():
            for rep, by_post in by_rep.items():
                key = f"{name} r{rep}"
                for pid, forecast in by_post.items():
                    forecasts = forecast if name.startswith("sol61 + ") else [forecast]
                    if not forecasts:
                        continue
                    try:
                        s, b = await fr.score(forecasts, rows[pid], questions[pid])
                    except Exception as exc:  # noqa: BLE001
                        print(f"  scoring failed: {key} {pid}: {type(exc).__name__}: {str(exc)[:120]}")
                        continue
                    scores.setdefault(key, {})[pid] = s
                    if b is not None:
                        briers.setdefault(key, {})[pid] = b

    asyncio.run(score_all())

    # Per question, the mean of the two runs; this is what the tables compare.
    for name in members:
        for table in (scores, briers):
            if f"{name} r1" in table and f"{name} r2" in table:
                both = set(table[f"{name} r1"]) & set(table[f"{name} r2"])
                table[name] = {p: (table[f"{name} r1"][p] + table[f"{name} r2"][p]) / 2 for p in both}

    groups = {
        "all": set(rows),
        "binary": {p for p, r in rows.items() if r["type"] == "binary"},
        "numeric+discrete": {p for p, r in rows.items() if r["type"] in ("numeric", "discrete")},
        "multiple_choice": {p for p, r in rows.items() if r["type"] == "multiple_choice"},
    }
    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    def cell(a: str, b: str, posts: set[int], table: dict = scores, fmt: str = "+.2f") -> str:
        res = fr.paired(table.get(a, {}), table.get(b, {}), posts) if a != b else None
        if not res:
            return "-"
        return (f"{res['mean']:{fmt}} [{res['lo']:{fmt}}, {res['hi']:{fmt}}] {res['better']}/{res['worse']} (n={res['n']})")

    singles = [n for n in claude if n in scores]
    pairs = [f"sol61 + {n}" for n in claude if f"sol61 + {n}" in scores]

    def fresh_of(name: str) -> str:
        """The same-day Sonnet 5 counterpart: the fresh pair for a pair, the fresh model for a model."""
        if name.startswith("sol61 + "):
            return f"sol61 + {FRESH}"
        return FRESH if name != "sol61 OR high" else name

    emit("Scores: mean of 2 runs per question. Paired difference, mean [90% CI] better/worse.")
    emit(f"Last column: a pair against sol61 + {FRESH}, a single model against {FRESH}.")
    for group, posts in groups.items():
        emit(f"\n## {group} ({len(posts)} questions)\n")
        emit(f"| configuration | n | mean log score | Brier | vs {PRODUCTION} | vs {REFERENCE} | vs same-day Sonnet 5 ({FRESH}) |")
        emit("|---|---|---|---|---|---|---|")
        for name in pairs + singles + ["sol61 OR high"]:
            mine = {p: s for p, s in scores.get(name, {}).items() if p in posts}
            if not mine:
                continue
            b = [briers[name][p] for p in mine if p in briers.get(name, {})]
            emit(
                f"| {name} | {len(mine)} | {statistics.mean(mine.values()):+.2f} | "
                f"{statistics.mean(b) if b else float('nan'):.4f} | {cell(name, PRODUCTION, posts)} | {cell(name, REFERENCE, posts)}"
                f" | {cell(name, fresh_of(name), posts)} |"
            )

    emit(f"\n## Brier on binaries (negative = better)\n")
    emit(f"| configuration | vs {PRODUCTION} | vs same-day Sonnet 5 |")
    emit("|---|---|---|")
    for name in pairs + singles:
        if name in briers:
            emit(
                f"| {name} | {cell(name, PRODUCTION, groups['binary'], briers, '+.4f')} | "
                f"{cell(name, fresh_of(name), groups['binary'], briers, '+.4f')} |"
            )

    emit("\n## Repeat runs: second run against the first, same configuration\n")
    emit("| configuration | all | binary | numeric+discrete | mean absolute change, all |")
    emit("|---|---|---|---|---|")
    for name in pairs + singles + ["sol61 OR high"]:
        r1, r2 = f"{name} r1", f"{name} r2"
        if r1 not in scores or r2 not in scores:
            continue
        common = set(scores[r1]) & set(scores[r2])
        absolute = statistics.mean(abs(scores[r2][p] - scores[r1][p]) for p in common)
        emit(
            f"| {name} | {cell(r2, r1, groups['all'])} | {cell(r2, r1, groups['binary'])} | "
            f"{cell(r2, r1, groups['numeric+discrete'])} | {absolute:.1f} |"
        )

    # Largest per-question differences of each direct configuration against the reference.
    emit(f"\n## Questions that move each direct configuration most (2 runs; {FRESH} against {REFERENCE}, the others against {FRESH})\n")
    for name in singles:
        if not name.endswith("direct"):
            continue
        ref = REFERENCE if name == FRESH else FRESH
        if ref not in scores:
            continue
        common = set(scores[name]) & set(scores[ref])
        diffs = sorted(((scores[name][p] - scores[ref][p], p) for p in common), reverse=True)
        top = ", ".join(f"{p} {rows[p]['type'][:3]} {d:+.1f}" for d, p in diffs[:4])
        bottom = ", ".join(f"{p} {rows[p]['type'][:3]} {d:+.1f}" for d, p in diffs[-4:])
        emit(f"- {name}: best {top}; worst {bottom}")

    # Tokens, cost and failures of the direct runs.
    emit("\n## Direct runs: tokens, cost and failures\n")
    emit(
        "| configuration | route | answers ok | failed | input tokens (mean) | thinking tokens median [p25, p75] | thinking mean"
        " | output tokens median | stops at max_tokens | refusals | $/answer paid | $/answer at list price | parser $/answer | median s |"
    )
    emit("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    groups_cost: dict[tuple, list[dict]] = {}
    for a in answers:
        groups_cost.setdefault((a["alias"], a["route"]), []).append(a)
    for (config, route), items in sorted(groups_cost.items()):
        forecaster = [u for a in items for u in a.get("usage", []) if not is_parser(u)]
        thinking = [u.get("reasoning_tokens") or 0 for u in forecaster]
        outputs = [u.get("completion_tokens") or 0 for u in forecaster]
        inputs = [u.get("prompt_tokens") or 0 for u in forecaster]
        q = statistics.quantiles(thinking, n=4) if len(thinking) > 3 else [float("nan")] * 3
        paid = statistics.mean(usage_cost(u) for u in forecaster) if forecaster else float("nan")
        listed = statistics.mean(list_cost(u) for u in forecaster) if forecaster else float("nan")
        parser = statistics.mean(sum(usage_cost(u) for u in a.get("usage", []) if is_parser(u)) for a in items)
        secs = [a["seconds"] for a in items if a.get("seconds") is not None]
        emit(
            f"| {config} | {route} | {sum(a['status'] == 'ok' for a in items)} | {sum(a['status'] == 'failed' for a in items)}"
            f" | {statistics.mean(inputs) if inputs else 0:.0f} | {statistics.median(thinking) if thinking else 0:.0f}"
            f" [{q[0]:.0f}, {q[2]:.0f}] | {statistics.mean(thinking) if thinking else 0:.0f}"
            f" | {statistics.median(outputs) if outputs else 0:.0f} | {sum(a.get('stop_reason') == 'max_tokens' for a in items)}"
            f" | {sum(a.get('stop_reason') == 'refusal' for a in items)} | {paid:.4f} | {listed:.4f} | {parser:.4f}"
            f" | {statistics.median(secs) if secs else float('nan'):.0f} |"
        )
    # Thinking by question type, base runs only.
    emit("\n## Thinking tokens by question type (median), base runs\n")
    kinds = ("binary", "multiple_choice", "numeric")
    emit("| configuration | " + " | ".join(kinds) + " |")
    emit("|---|---|---|---|")
    ref_by_kind: dict[str, list[int]] = {}
    for a in fr.recorded():
        if a["alias"] == "sonnet5" and a["status"] == "ok" and a["variant"] == ("V2" if a["type"] == "binary" else "base"):
            kind = "numeric" if a["type"] == "discrete" else a["type"]
            ref_by_kind.setdefault(kind, []).extend(
                u.get("reasoning_tokens") or 0 for u in a["usage"] if "claude" in (u.get("model") or "")
            )
    emit(f"| {REFERENCE} | " + " | ".join(f"{statistics.median(ref_by_kind.get(k) or [0]):.0f}" for k in kinds) + " |")
    for config in sorted({a["alias"] for a in new}):
        by_kind: dict[str, list[int]] = {}
        for a in new:
            if a["alias"] == config:
                kind = "numeric" if a["type"] == "discrete" else a["type"]
                by_kind.setdefault(kind, []).extend(u.get("reasoning_tokens") or 0 for u in a["usage"] if not is_parser(u))
        emit(f"| {config} direct | " + " | ".join(f"{statistics.median(by_kind.get(k) or [0]):.0f}" for k in kinds) + " |")

    failures = [a for a in answers if a["status"] == "failed"]
    if failures:
        emit("\n## Failed answers\n")
        for a in failures:
            emit(f"- {a['post_id']} {a['type']} {a['alias']} {a['variant']} r{a['rep']}: {a['error']} {a.get('warnings') or ''}")
    emit(f"\nEstimated spend on the Anthropic key: ${spent():.2f}")

    with open(os.path.join(OUT, "analysis.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT, "scores.json"), "w", encoding="utf-8") as fh:
        json.dump({"log": scores, "brier": briers}, fh, indent=1)
    print(f"\nWrote {OUT}/analysis.md and {OUT}/scores.json")


def main_cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prompts", help="build and check every prompt (no LLM calls)")
    check = sub.add_parser("check-parser", help="re-parse recorded answers with the Anthropic parser")
    check.add_argument("--aliases", default="sonnet5,sol61")
    check.add_argument("--n", type=int, default=0, help="at most N answers (0 = all)")
    cal = sub.add_parser("calibrate", help="plain calls on 3 questions, thinking tokens per effort")
    cal.add_argument("--configs", required=True)
    cal.add_argument("--samples", type=int, default=1)
    submit = sub.add_parser("submit", help="send one batch per configuration and repeat")
    submit.add_argument("--configs", required=True)
    submit.add_argument("--reps", default="1")
    submit.add_argument("--posts", default="", help="comma-separated post ids")
    collect = sub.add_parser("collect", help="fetch finished batches and parse their answers")
    collect.add_argument("--wait", action="store_true", help="poll every 60 s until every batch is collected")
    sub.add_parser("analyze", help="score everything recorded (no LLM calls)")
    sub.add_parser("spend", help="estimated spend so far")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%H:%M:%S")
    logger.setLevel(logging.INFO)
    os.makedirs(OUT, exist_ok=True)
    {
        "prompts": cmd_prompts, "check-parser": cmd_check_parser, "calibrate": cmd_calibrate,
        "submit": cmd_submit, "collect": cmd_collect, "analyze": cmd_analyze, "spend": cmd_spend,
    }[args.command](args)


if __name__ == "__main__":
    main_cli()
