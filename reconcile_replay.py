"""
Tests a reconciler step for the two-member ensemble on MiniBench round 1:
instead of averaging GPT-6.1 Sol and Claude Sonnet 5, a third call reads both
answers, lists where they disagree on facts or on how to read the question,
settles each point by the quality of the evidence in the research, and gives
the final forecast. The question is whether that beats the production mean on
frozen replays.

Inputs, all recorded earlier, so no member is asked again:
- the 56 scored questions, the frozen research and the original forecast
  dates, from forecast_replay.py;
- GPT-6.1 Sol (OpenRouter, high), runs 1 and 2, from logs/replay_r1/;
- Claude Sonnet 5 (Anthropic API direct, high), runs 1 and 2, from
  logs/replay_claude/. These are today's fresh runs, not the V2 answers of
  logs/replay_r1/, which were picked as the best of four variants on these
  same binaries and are biased upward (reports/Replay Claude direto.md).
Run 1 of one member is paired with run 1 of the other, run 2 with run 2.

The reconciler is claude-opus-5-5 or claude-sonnet-5 at effort high with
adaptive thinking, through the Message Batches API (half the list price),
on the Anthropic key only: claude_direct_replay's guard refuses any LLM call
that is not anthropic/. Its answer is parsed by the bot's own parsers (the
claude-haiku-4-5 parser that reproduced the gpt-4o-mini parses exactly), and
scored like a single member with forecast_replay.score.

Variants, both scored from the same reconciler answers:
- A: the reconciled forecast on every question;
- B: the reconciled forecast only where the members disagree materially
  (binary 15 points or more; multiple choice total variation distance 0.2 or
  more; numeric medians more than 25% of the wider member's P10 to P90 range
  apart), the production mean elsewhere.

Outputs go to logs/replay_reconcile/ (git-ignored): prompts.jsonl, raw.jsonl
(every reconciler response, with token usage and stop reason), answers.jsonl
(parsed forecasts), batches.json, analysis.md, scores.json, and the bot's own
forecast log for these runs. Spending is estimated from token usage at
Anthropic list prices, halved for batches; a request that could take the
total past SPEND_CAP is refused.

Usage (the anthropic SDK is not a project dependency, hence --with):
    uv run --with anthropic python reconcile_replay.py inputs                 # prompts and triggers, no LLM calls
    uv run --with anthropic python reconcile_replay.py smoke --posts 1,2,3     # plain calls, run 1, both reconcilers
    uv run --with anthropic python reconcile_replay.py submit --configs o55-high,s5-high --reps 1,2
    uv run --with anthropic python reconcile_replay.py collect [--wait]
    uv run --with anthropic python reconcile_replay.py analyze
    uv run --with anthropic python reconcile_replay.py spend
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

# claude_direct_replay sets the Anthropic parser before main.py is imported
# and installs the guard that refuses any LLM call outside the Anthropic key.
import claude_direct_replay as cdr  # noqa: I001
import anthropic
import bot as bot_module
import forecast_replay as fr

logger = logging.getLogger("reconcile_replay")

OUT = os.path.join("logs", "replay_reconcile")
PROMPTS = os.path.join(OUT, "prompts.jsonl")
RAW = os.path.join(OUT, "raw.jsonl")
ANSWERS = os.path.join(OUT, "answers.jsonl")
BATCHES = os.path.join(OUT, "batches.json")

# The bot records every parsed member here instead of logs/forecasts.jsonl.
bot_module.FORECAST_LOG = os.path.join(OUT, "forecasts.jsonl")

SPEND_CAP = 60.0  # dollars, estimated from token usage
# Thinking plus answer; a stop at the cap is recorded. In the smoke test Claude
# Sonnet 5 at high thought up to 8,400 tokens, so batches, which have no HTTP
# timeout, get twice the 16,000 of plain calls.
MAX_TOKENS = 32000
PLAIN_MAX_TOKENS = 16000
IN_FLIGHT = 4  # plain calls or parses at once

CONFIGS = {"o55-high": ("claude-opus-5-5", "high"), "s5-high": ("claude-sonnet-5", "high")}
# Output tokens assumed per answer before any is measured, for the guard only.
OUTPUT_GUESS = {"claude-opus-5-5": 5000, "claude-sonnet-5": 6000}

SOL = "sol61"  # GPT-6.1 Sol in logs/replay_r1/answers.jsonl
SONNET = "s5-high"  # Claude Sonnet 5 in logs/replay_claude/answers.jsonl
OPUS_DIRECT = "o55-high"  # Claude Opus 5.5 as a forecaster, same file; a control in analyze
REPS = (1, 2)

# Material disagreement thresholds for variant B.
BINARY_GAP = 0.15
MC_TVD = 0.20
NUMERIC_SHIFT = 0.25


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


read_jsonl = cdr.read_jsonl
append_jsonl = cdr.append_jsonl


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Members and disagreement
# ---------------------------------------------------------------------------


def load_members(rows: dict[int, dict]) -> dict[int, dict[int, dict[str, dict]]]:
    """rep -> post -> {"sol": answer, "sonnet": answer}, the latest successful answer of each."""
    out: dict[int, dict[int, dict[str, dict]]] = {rep: {} for rep in REPS}
    for a in fr.recorded():
        if a["alias"] == SOL and a["variant"] == "base" and a["status"] == "ok" and a["rep"] in REPS and a["post_id"] in rows:
            out[a["rep"]].setdefault(a["post_id"], {})["sol"] = a
    for a in read_jsonl(cdr.ANSWERS):
        if (a["alias"] == SONNET and a["variant"] == "base" and a["route"] == "batch" and a["status"] == "ok"
                and a["rep"] in REPS and a["post_id"] in rows):
            out[a["rep"]].setdefault(a["post_id"], {})["sonnet"] = a
    return out


def _quantile(points: list, q: float) -> float:
    pts = sorted((float(h), float(v)) for h, v in points)
    return float(np.interp(q, [h for h, _ in pts], [v for _, v in pts]))


def disagreement(kind: str, f1, f2, options: list[str] | None = None) -> tuple[float, bool]:
    """Size of the disagreement between two member forecasts, and whether it is material."""
    if kind == "binary":
        gap = abs(f1 - f2)
        return gap, gap >= BINARY_GAP - 1e-9
    if kind == "multiple_choice":
        a, b = fr.clamp_options(f1, options), fr.clamp_options(f2, options)
        tvd = 0.5 * sum(abs(a[o] - b[o]) for o in options)
        return tvd, tvd >= MC_TVD - 1e-9
    m1, m2 = _quantile(f1, 0.5), _quantile(f2, 0.5)
    width = max(_quantile(f1, 0.9) - _quantile(f1, 0.1), _quantile(f2, 0.9) - _quantile(f2, 0.1))
    shift = abs(m1 - m2) / width if width > 0 else float("inf") if m1 != m2 else 0.0
    return shift, shift > NUMERIC_SHIFT


# ---------------------------------------------------------------------------
# The reconciler prompt
# ---------------------------------------------------------------------------


def _num(value: float) -> str:
    """A number as a forecaster would write it: no scientific notation, no float noise."""
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    text = f"{value:.6g}"
    if "e" in text:
        text = f"{value:.10f}".rstrip("0").rstrip(".")
    return text


def show_forecast(kind: str, prediction, options: list[str] | None) -> str:
    if kind == "binary":
        return f"Probability: {_num(round(100 * prediction, 2))}%"
    if kind == "multiple_choice":
        return "\n".join(f"{o}: {_num(round(100 * prediction.get(o, 0.0), 2))}%" for o in options)
    return "\n".join(f"Percentile {round(100 * h)}: {_num(v)}" for h, v in prediction)


def answer_format(kind: str, question, bot) -> str:
    """The bot's own answer instructions for this question type, so its parsers read the result."""
    if kind == "binary":
        return 'The last thing you write is your final answer as: "Probability: ZZ%", 0-100'
    if kind == "multiple_choice":
        return (
            f"The last thing you write is your final probabilities for the N options\n"
            f"in this order {question.options} as:\n"
            "Option_A: Probability_A\nOption_B: Probability_B\n...\nOption_N: Probability_N"
        )
    upper_msg, lower_msg = bot._bound_messages(question)
    return (
        f"{lower_msg}\n{upper_msg}\n\n"
        "Formatting Instructions:\n"
        "- Please notice the units requested and give your answer in these units\n"
        "  (e.g. whether you represent a number as 1,000,000 or 1 million).\n"
        "- Never use scientific notation.\n"
        "- Always start with a smaller number (more negative if negative) and\n"
        "  then increase from there. The value for percentile 10 should always\n"
        "  be less than the value for percentile 20, and so on.\n\n"
        "The last thing you write is your final answer as:\n"
        '"\n'
        "Percentile 10: XX (lowest number value)\n"
        "Percentile 20: XX\n"
        "Percentile 40: XX\n"
        "Percentile 60: XX\n"
        "Percentile 80: XX\n"
        "Percentile 90: XX (highest number value)\n"
        '"'
    )


BINARY_CONVENTIONS = (
    "Both forecasters were told two Metaculus conventions, which still apply: if the research does"
    " not positively show that the event has already occurred, assume it has not; and a question"
    ' phrased "will X happen before <date>" asks about the window between today and that date.'
)

INSTRUCTIONS = """\
How to reconcile:

1. List every point where A and B disagree on a fact (a number, a date, a status, whether an event \
happened, what a source says) or on how to read the question and its resolution criteria. Leave out \
differences of wording or emphasis that would not move the forecast. If they agree on every material \
point, say so.
2. For each point, decide which side the research notes support better. Judge the evidence, not the \
forecaster's confidence:
   - The source named in the resolution criteria, or another primary source (an official statement, \
dataset, filing, schedule or record), outranks a secondary report of it.
   - A dated figure outranks an undated one, and a more recent one outranks an older one. Check the date \
and the year of each document a claim rests on.
   - A claim backed by a passage in the research notes outranks a claim with no support there. A \
forecaster's assertion is not evidence by itself.
   - If neither side has support in the notes, or the notes conflict with no way to rank the sources, \
mark the point unresolved.
3. Build your forecast from the side that wins each point. Do not average A and B by default. Land \
between them only where a material point stays unresolved, or where both read the same evidence and \
differ only in judgment, and say which case applies. Going beyond the range of A and B needs evidence in \
the notes that both of them missed.
4. Keep the habits both forecasters were asked to follow: start from the status quo and the base rate \
for this kind of event, account for the time left until the outcome is known, and do not be more \
confident than the evidence allows. Forecasts are scored with a log score, so an extreme forecast on the \
wrong side costs far more than a cautious one.

Write the list of disagreements and your ruling on each, then a short justification of the final \
forecast."""


def first_is_sol(post_id: int, rep: int) -> bool:
    """Which member is shown as A, alternated by a hash so that position cannot favor one model."""
    return int(hashlib.sha1(f"{post_id}-{rep}".encode()).hexdigest(), 16) % 2 == 0


def reconciler_prompt(row: dict, question, pair: dict[str, dict], rep: int, bot) -> tuple[str, str]:
    """The prompt for one question and run, and which member is A ("sol" or "sonnet")."""
    kind = row["type"] if row["type"] != "discrete" else "numeric"
    options = getattr(question, "options", None)
    order = ("sol", "sonnet") if first_is_sol(row["post_id"], rep) else ("sonnet", "sol")
    today = datetime.fromisoformat(row["forecast_time"]).strftime("%Y-%m-%d")

    parts = [
        "You are the lead forecaster of a small team. Two forecasters on your team, A and B, answered the"
        " same forecasting question independently. Both had the same question, the same research notes"
        " and the same instructions, and both worked on the same day. Your job is to reconcile their two"
        " answers into one final forecast.",
        f"Today is {today}. The question, the research notes and both answers are as of this date.",
        f"<question>\n{question.question_text}\n</question>",
    ]
    if kind == "multiple_choice":
        parts.append(f"<options>\n{options}\n</options>")
    parts += [
        f"<background>\n{question.background_info}\n</background>",
        f"<resolution_criteria>\n{question.resolution_criteria}\n</resolution_criteria>",
        f"<fine_print>\n{question.fine_print}\n</fine_print>",
    ]
    if kind == "numeric":
        unit = question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"
        parts.append(f"<units>\n{unit}\n</units>")
    parts.append(f"<research_notes>\n{row['research']}\n</research_notes>")
    for label, member in zip(("a", "b"), order):
        answer = pair[member]
        parts.append(
            f"<forecaster_{label}>\n"
            f"Final forecast, as read from the answer:\n{show_forecast(kind, answer['prediction'], options)}\n\n"
            f"Full answer:\n{answer['raw'].strip()}\n"
            f"</forecaster_{label}>"
        )
    if kind == "binary":
        parts.append(BINARY_CONVENTIONS)
    parts += [INSTRUCTIONS, answer_format(kind, question, bot)]
    return "\n\n".join(parts), order[0]


def build_inputs() -> tuple[dict[int, dict], dict, list[dict]]:
    """Every prompt for both runs, with the disagreement between the members."""
    rows = {r["post_id"]: r for r in cdr.scored_rows()}
    members = load_members(rows)
    bot = cdr.replay_bot([])
    items = []
    for rep in REPS:
        for pid, row in rows.items():
            pair = members[rep].get(pid, {})
            if set(pair) != {"sol", "sonnet"}:
                logger.warning(f"{pid} r{rep}: missing member(s), have {sorted(pair)}")
                continue
            question = fr.question_object(row)
            kind = row["type"] if row["type"] != "discrete" else "numeric"
            size, material = disagreement(
                kind, pair["sol"]["prediction"], pair["sonnet"]["prediction"], getattr(question, "options", None)
            )
            prompt, first = reconciler_prompt(row, question, pair, rep, bot)
            items.append({
                "post_id": pid, "type": row["type"], "rep": rep, "first": first, "disagreement": size,
                "material": material, "sha1": sha1(prompt), "chars": len(prompt), "prompt": prompt,
            })
    return rows, members, items


# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------


def spent() -> float:
    """Every reconciler response (raw.jsonl) plus every parser call recorded."""
    total = sum(cdr.usage_cost(r["usage"]) for r in read_jsonl(RAW) if r.get("usage"))
    total += sum(cdr.usage_cost(u) for a in read_jsonl(ANSWERS) for u in a.get("parser_usage", []))
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


def estimate(model: str, prompt: str, batch: bool) -> float:
    """Upper estimate of one request plus its parse: 1 token per 3 characters of prompt, and
    1.5 times the mean output measured for this model (or a high guess before any)."""
    measured = [r["usage"]["completion_tokens"] for r in read_jsonl(RAW) if r.get("usage") and r.get("model") == model]
    out = 1.5 * statistics.mean(measured) if measured else OUTPUT_GUESS[model]
    pin, pout = cdr.PRICES[model]
    share = cdr.BATCH_SHARE if batch else 1.0
    parser = 2 * (4000 * 1.0 + 100 * 5.0) / 1e6  # two Haiku validation samples
    return share * (len(prompt) / 3 * pin + out * pout) / 1e6 + parser


def check_cap(extra: float, what: str) -> None:
    done, pending = spent(), pending_estimate()
    print(f"Spent so far ${done:.2f}, pending batches up to ${pending:.2f}, {what} up to ${extra:.2f} (cap ${SPEND_CAP:.0f})")
    if done + pending + extra > SPEND_CAP:
        raise SystemExit("Refused: this could take the estimated spend past the cap.")


# ---------------------------------------------------------------------------
# Calls and parsing
# ---------------------------------------------------------------------------


def request_params(config: str, prompt: str, max_tokens: int = MAX_TOKENS) -> dict:
    model, effort = CONFIGS[config]
    return {
        "model": model,
        "max_tokens": max_tokens,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": effort},
        "messages": [{"role": "user", "content": prompt}],
    }


def _last(pattern: str, text: str) -> re.Match | None:
    found = list(re.finditer(pattern, text, flags=re.I | re.M))
    return found[-1] if found else None


def format_check(kind: str, text: str, prediction, options: list[str] | None) -> str | None:
    """Compares the parser's reading with the final answer lines read by a regex; None when they agree."""
    try:
        if kind == "binary":
            m = _last(r"Probability:\s*\**\s*([0-9]+(?:\.[0-9]+)?)\s*%", text)
            if not m:
                return "no 'Probability: ZZ%' line"
            value = min(0.99, max(0.01, float(m.group(1)) / 100))
            return None if abs(value - prediction) < 0.005 else f"regex {value:.3f}, parser {prediction:.3f}"
        if kind == "multiple_choice":
            found = {}
            for k, o in enumerate(options):
                # The answer template labels options Option_A, Option_B, ... in question order.
                label = rf"(?:{re.escape(o)}|Option_{chr(65 + k)}(?:\s*\({re.escape(o)}\))?)"
                m = _last(label + r"\**\s*:\s*\**\s*([0-9]+(?:\.[0-9]+)?)\s*%?", text)
                if not m:
                    return f"option {o!r} not found"
                found[o] = float(m.group(1))
            total = sum(found.values()) or 1.0
            worst = max(abs(found[o] / total - prediction[o]) for o in options)
            return None if worst < 0.02 else f"largest option difference {worst:.3f}"
        lines = {}
        for p in (10, 20, 40, 60, 80, 90):
            m = _last(rf"Percentile\s*{p}\s*:\s*\**\s*\$?\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)", text)
            if not m:
                return f"no 'Percentile {p}' line"
            lines[p / 100] = float(m.group(1).replace(",", ""))
        worst = 0.0
        for h, v in prediction:
            ref = lines.get(round(h, 2))
            if ref is None:
                return f"parser gave percentile {h}"
            worst = max(worst, abs(v - ref) / max(abs(ref), 1e-9))
        return None if worst < 0.01 else f"largest relative difference {worst:.3f}"
    except Exception as exc:  # noqa: BLE001
        return f"check failed: {type(exc).__name__}: {exc}"


async def parse_answer(row: dict, item: dict, config: str, variant: str, resp: dict | None,
                       error: str | None, route: str) -> dict:
    """Runs the bot's own parse on a reconciler answer and records the result."""
    model, effort = CONFIGS[config]
    call: dict = {}
    fr._CALL.set(call)
    fr._TODAY.set(datetime.fromisoformat(row["forecast_time"]))
    status, prediction, check = "ok", None, None
    kind = row["type"] if row["type"] != "discrete" else "numeric"
    question = fr.question_object(row)
    try:
        if error:
            raise RuntimeError(error)
        if resp["stop_reason"] != "end_turn":
            raise RuntimeError(f"stop_reason {resp['stop_reason']}")
        if not resp["text"].strip():
            raise RuntimeError("empty answer")
        bot = cdr.replay_bot([cdr.CannedLlm(f"anthropic/{model}", resp["text"], None)])
        result = await fr.forecast_once(bot, question, row["research"])
        prediction = fr.serialize(result.prediction_value)
        check = format_check(kind, resp["text"], prediction, getattr(question, "options", None))
    except Exception as exc:  # noqa: BLE001
        status, error = "failed", f"{type(exc).__name__}: {str(exc)[:300]}"
    out = {
        "time": _utc_now(), "post_id": row["post_id"], "type": row["type"], "config": config, "model": model,
        "effort": effort, "variant": variant, "rep": item["rep"], "route": route, "first": item["first"],
        "material": item["material"], "disagreement": item["disagreement"], "prompt_sha1": item["sha1"],
        "status": status, "error": error, "prediction": prediction, "format_check": check,
        "stop_reason": resp.get("stop_reason") if resp else None, "seconds": resp.get("seconds") if resp else None,
        "reconciler_usage": resp.get("usage") if resp else None,
        "parser_usage": [u for u in call.get("usage", []) if cdr.is_parser(u)],
        "warnings": call.get("warnings"), "raw": resp.get("text") if resp else None,
    }
    append_jsonl(ANSWERS, out)
    shown = prediction if not isinstance(prediction, list) else f"P10..P90 {prediction[0][1]}..{prediction[-1][1]}"
    if isinstance(shown, dict):
        shown = {k[:12]: round(v, 2) for k, v in shown.items()}
    thinking = ((resp or {}).get("usage") or {}).get("reasoning_tokens")
    logger.info(
        f"{row['post_id']} {row['type'][:8]} {config} {variant} r{item['rep']}: {status} "
        f"{shown if status == 'ok' else error} (thinking {thinking}, {(resp or {}).get('seconds')} s)"
        + (f" FORMAT CHECK: {check}" if check else "")
    )
    return out


def raw_row(item: dict, config: str, variant: str, custom_id: str, batch_id: str | None,
            resp: dict | None, error: str | None) -> dict:
    model, effort = CONFIGS[config]
    return {
        "time": _utc_now(), "custom_id": custom_id, "batch_id": batch_id, "config": config, "model": model,
        "effort": effort, "variant": variant, "rep": item["rep"], "post_id": item["post_id"],
        "prompt_sha1": item["sha1"], "error": error, **(resp or {}),
    }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_inputs(_args) -> None:
    rows, _members, items = build_inputs()
    with open(PROMPTS, "w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {len(items)} prompts to {PROMPTS}.")
    print("\n| post | type | r1 disagreement | r2 disagreement |")
    print("|---|---|---|---|")
    by = {(i["post_id"], i["rep"]): i for i in items}
    for pid, row in rows.items():
        cells = []
        for rep in REPS:
            i = by.get((pid, rep))
            cells.append("-" if i is None else f"{i['disagreement']:.2f}{' *' if i['material'] else ''}")
        print(f"| {pid} | {row['type']} | {cells[0]} | {cells[1]} |")
    for rep in REPS:
        mine = [i for i in items if i["rep"] == rep]
        kinds: dict[str, list[int]] = {}
        for i in mine:
            kinds.setdefault(i["type"], [0, 0])
            kinds[i["type"]][0] += i["material"]
            kinds[i["type"]][1] += 1
        print(f"r{rep}: material disagreement on {sum(i['material'] for i in mine)} of {len(mine)}: "
              + ", ".join(f"{k} {a}/{n}" for k, (a, n) in sorted(kinds.items())))
    chars = [i["chars"] for i in items]
    print(f"Prompt characters: median {statistics.median(chars):.0f}, max {max(chars)}")
    full = sum(estimate(CONFIGS[c][0], i["prompt"], batch=True) for c in CONFIGS for i in items)
    print(f"Upper estimate of the full run (both reconcilers, both runs, batches): ${full:.2f}")


def _select(items: list[dict], posts: str, reps: list[int]) -> list[dict]:
    wanted = {int(p) for p in posts.split(",") if p.strip()} if posts else None
    return [i for i in items if i["rep"] in reps and (wanted is None or i["post_id"] in wanted)]


def cmd_smoke(args) -> None:
    """Plain calls on a few questions, run 1, every reconciler configuration."""
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    rows, _members, items = build_inputs()
    picked = _select(items, args.posts, [1])
    done = {(a["post_id"], a["config"], a["rep"]) for a in read_jsonl(ANSWERS) if a["variant"] == "smoke"}
    jobs = [(i, c) for c in configs for i in picked if (i["post_id"], c, i["rep"]) not in done]
    check_cap(sum(estimate(CONFIGS[c][0], i["prompt"], batch=False) for i, c in jobs), f"{len(jobs)} smoke calls")
    client = anthropic.AsyncAnthropic(max_retries=3, timeout=900)

    async def one(item: dict, config: str, sem: asyncio.Semaphore) -> None:
        model, _ = CONFIGS[config]
        resp, error = None, None
        async with sem:
            try:
                started = time.monotonic()
                msg = await client.messages.create(**request_params(config, item["prompt"], PLAIN_MAX_TOKENS))
                resp = cdr.response_row(msg, model, batch=False, seconds=time.monotonic() - started)
            except anthropic.APIError as exc:
                error = f"{type(exc).__name__}: {str(exc)[:300]}"
            append_jsonl(RAW, raw_row(item, config, "smoke", f"smoke_{config}_r1_{item['post_id']}", None, resp, error))
            await parse_answer(rows[item["post_id"]], item, config, "smoke", resp, error, "direct")

    async def go() -> None:
        fr._SLOTS = asyncio.Semaphore(IN_FLIGHT)
        sem = asyncio.Semaphore(IN_FLIGHT)
        await asyncio.gather(*(one(i, c, sem) for i, c in jobs))

    asyncio.run(go())
    print(f"Spent so far: ${spent():.2f}")


def cmd_submit(args) -> None:
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    reps = [int(r) for r in args.reps.split(",") if r.strip()]
    unknown = [c for c in configs if c not in CONFIGS]
    if unknown:
        raise SystemExit(f"unknown configuration(s) {unknown}; use {list(CONFIGS)}")
    _rows, _members, items = build_inputs()
    done = {(a["post_id"], a["config"], a["rep"]) for a in read_jsonl(ANSWERS) if a["variant"] == "main" and a["status"] == "ok"}
    batches = load_batches()
    pending = {
        (meta["post_id"], b["config"], b["rep"])
        for b in batches if not b.get("collected") for meta in b["requests"].values()
    }
    client = anthropic.Anthropic(max_retries=3)
    for config in configs:
        model, _ = CONFIGS[config]
        for rep in reps:
            todo = [i for i in _select(items, args.posts, [rep]) if (i["post_id"], config, rep) not in done | pending]
            if not todo:
                print(f"{config} r{rep}: nothing to request")
                continue
            extra = sum(estimate(model, i["prompt"], batch=True) for i in todo)
            check_cap(extra, f"batch {config} r{rep} ({len(todo)} requests)")
            requests = [
                {"custom_id": f"{config}_r{rep}_{i['post_id']}", "params": request_params(config, i["prompt"])}
                for i in todo
            ]
            batch = client.messages.batches.create(requests=requests)
            batches.append({
                "id": batch.id, "config": config, "rep": rep, "created": _utc_now(), "estimate": round(extra, 4),
                "requests": {req["custom_id"]: {"post_id": i["post_id"], "sha1": i["sha1"]} for req, i in zip(requests, todo)},
                "collected": False,
            })
            save_batches(batches)
            print(f"{config} r{rep}: batch {batch.id} with {len(requests)} requests ({batch.processing_status})")


def cmd_collect(args) -> None:
    client = anthropic.Anthropic(max_retries=3)
    rows, _members, items = build_inputs()
    by_key = {(i["post_id"], i["rep"]): i for i in items}
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
            model, _ = CONFIGS[b["config"]]
            results = []
            for result in client.messages.batches.results(b["id"]):
                meta = b["requests"][result.custom_id]
                item = by_key[(meta["post_id"], b["rep"])]
                if item["sha1"] != meta["sha1"]:
                    raise SystemExit(f"prompt of {result.custom_id} changed since submission")
                resp, error = None, None
                if result.result.type == "succeeded":
                    resp = cdr.response_row(result.result.message, model, batch=True)
                else:
                    error = f"batch result {result.result.type}: {str(getattr(result.result, 'error', None))[:300]}"
                append_jsonl(RAW, raw_row(item, b["config"], "main", result.custom_id, b["id"], resp, error))
                results.append((item, resp, error))

            async def parse_all(results=results, b=b) -> None:
                fr._SLOTS = asyncio.Semaphore(IN_FLIGHT)
                sem = asyncio.Semaphore(IN_FLIGHT)

                async def one(item, resp, error):
                    async with sem:
                        await parse_answer(rows[item["post_id"]], item, b["config"], "main", resp, error, "batch")

                await asyncio.gather(*(one(*r) for r in results))

            asyncio.run(parse_all())
            b["collected"] = True
            b["ended"] = info.ended_at.isoformat() if info.ended_at else None
            b["counts"] = {"succeeded": counts.succeeded, "errored": counts.errored,
                           "expired": counts.expired, "canceled": counts.canceled}
            save_batches(batches)
            print(f"{b['config']} r{b['rep']}: collected {len(results)} results; spent so far ${spent():.2f}")
        if not args.wait:
            break
        if not all(b.get("collected") for b in load_batches()):
            time.sleep(60)


def cmd_spend(_args) -> None:
    by: dict[str, float] = {}
    for r in read_jsonl(RAW):
        if r.get("usage"):
            key = f"{r['config']} {r['variant']} r{r['rep']}"
            by[key] = by.get(key, 0.0) + cdr.usage_cost(r["usage"])
    for key, value in sorted(by.items()):
        print(f"  {key}: ${value:.3f}")
    parser = sum(cdr.usage_cost(u) for a in read_jsonl(ANSWERS) for u in a.get("parser_usage", []))
    print(f"  parser: ${parser:.3f}")
    print(f"Total ${spent():.2f}; pending batches up to ${pending_estimate():.2f}")


# ---------------------------------------------------------------------------
# Scoring (offline)
# ---------------------------------------------------------------------------


def cmd_analyze(_args) -> None:
    logging.getLogger("forecasting_tools").setLevel(logging.ERROR)
    rows, members, items = build_inputs()
    questions = {pid: fr.question_object(r) for pid, r in rows.items()}
    trigger = {(i["post_id"], i["rep"]): i["material"] for i in items}
    first = {(i["post_id"], i["rep"]): i["first"] for i in items}
    answers = [a for a in read_jsonl(ANSWERS) if a["variant"] == "main"]
    latest: dict[tuple, dict] = {}
    for a in answers:
        latest[(a["config"], a["post_id"], a["rep"])] = a
    configs = sorted({c for c, _, _ in latest})
    # Control: Claude Opus 5.5 at high asked the forecasting question itself (the
    # bot's own prompt, Anthropic API direct, runs 1 and 2 of claude_direct_replay).
    # It separates what the reconciler adds from what Opus's own judgment adds.
    opus: dict[tuple[int, int], object] = {}
    for a in read_jsonl(cdr.ANSWERS):
        if a["alias"] == OPUS_DIRECT and a["variant"] == "base" and a["route"] == "batch" and a["status"] == "ok":
            opus[(a["post_id"], a["rep"])] = a["prediction"]

    scores: dict[str, dict[int, float]] = {}
    briers: dict[str, dict[int, float]] = {}
    fallbacks: dict[str, int] = {}

    async def put(name: str, pid: int, forecasts: list) -> None:
        s, b = await fr.score(forecasts, rows[pid], questions[pid])
        scores.setdefault(name, {})[pid] = s
        if b is not None:
            briers.setdefault(name, {})[pid] = b

    async def put_three(name: str, pid: int, forecasts: list) -> None:
        """Three members aggregated as production would: on binaries the bot's
        _trimmed_mean, which for three values is the median; elsewhere as in score()
        (mean per option, pointwise median of the CDFs)."""
        row = rows[pid]
        if row["type"] != "binary":
            await put(name, pid, forecasts)
            return
        p = bot_module._trimmed_mean([min(0.99, max(0.01, f)) for f in forecasts])
        scores.setdefault(name, {})[pid] = 100 * math.log(p if row["outcome"] == 1 else 1 - p)
        briers.setdefault(name, {})[pid] = (p - row["outcome"]) ** 2

    async def score_all() -> None:
        for rep in REPS:
            for pid in rows:
                pair = members[rep].get(pid, {})
                if set(pair) != {"sol", "sonnet"}:
                    continue
                sol, son = pair["sol"]["prediction"], pair["sonnet"]["prediction"]
                await put(f"mean r{rep}", pid, [sol, son])
                await put(f"sol61 r{rep}", pid, [sol])
                await put(f"sonnet5 r{rep}", pid, [son])
                # Upper bound for any picker: the member that turned out better, where B would reconcile.
                better = sol if scores[f"sol61 r{rep}"][pid] >= scores[f"sonnet5 r{rep}"][pid] else son
                await put(f"oracle pick r{rep}", pid, [better] if trigger[(pid, rep)] else [sol, son])
                if (pid, rep) in opus:
                    op = opus[(pid, rep)]
                    await put(f"opus direct r{rep}", pid, [op])
                    await put(f"sol61+opus direct r{rep}", pid, [sol, op])
                    await put_three(f"three members r{rep}", pid, [sol, son, op])
                for config in configs:
                    a = latest.get((config, pid, rep))
                    ok = a is not None and a["status"] == "ok"
                    if not ok:
                        # Production would publish the mean when the reconciler fails.
                        fallbacks[config] = fallbacks.get(config, 0) + 1
                    forecasts = [a["prediction"]] if ok else [sol, son]
                    await put(f"{config} A r{rep}", pid, forecasts)
                    await put(f"{config} B r{rep}", pid, forecasts if trigger[(pid, rep)] else [sol, son])

    asyncio.run(score_all())

    names = sorted({k.rsplit(" r", 1)[0] for k in scores})
    for name in names:
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
        res = fr.paired(table.get(a, {}), table.get(b, {}), posts)
        if not res:
            return "-"
        return f"{res['mean']:{fmt}} [{res['lo']:{fmt}}, {res['hi']:{fmt}}] {res['better']}/{res['worse']} (n={res['n']})"

    emit("Paired difference per question against the production mean of the same run(s); positive is better.")
    emit("Format: mean [90% bootstrap CI] questions better/worse (n). Two-run rows average each question over runs 1 and 2.")
    candidates = [f"{c} {v}" for c in configs for v in ("A", "B")] + [
        "sol61", "sonnet5", "oracle pick", "opus direct", "sol61+opus direct", "three members",
    ]
    for group, posts in groups.items():
        emit(f"\n## {group} ({len(posts)} questions)\n")
        emit("| configuration | mean log score (2 runs) | vs mean, 2 runs | vs mean, run 1 | vs mean, run 2 |")
        emit("|---|---|---|---|---|")
        mine = [scores["mean"][p] for p in posts if p in scores.get("mean", {})]
        emit(f"| production mean | {statistics.mean(mine):+.2f} | 0 | 0 | 0 |")
        for name in candidates:
            if name not in scores:
                continue
            vals = [scores[name][p] for p in posts if p in scores[name]]
            emit(
                f"| {name} | {statistics.mean(vals):+.2f} | {cell(name, 'mean', posts)} | "
                f"{cell(f'{name} r1', 'mean r1', posts)} | {cell(f'{name} r2', 'mean r2', posts)} |"
            )

    emit("\n## Brier on binaries, against the production mean (negative = better)\n")
    emit(f"- production mean: {statistics.mean(briers['mean'].values()):.4f}")
    for name in candidates:
        if name in briers:
            emit(f"- {name}: {cell(name, 'mean', groups['binary'], briers, '+.4f')}")

    emit("\n## Repeat noise: run 2 against run 1 of the same configuration\n")
    emit("| configuration | all | binary | numeric+discrete | mean absolute change, all |")
    emit("|---|---|---|---|---|")
    for name in ["mean"] + candidates:
        r1, r2 = f"{name} r1", f"{name} r2"
        if r1 in scores and r2 in scores:
            common = set(scores[r1]) & set(scores[r2])
            absolute = statistics.mean(abs(scores[r2][p] - scores[r1][p]) for p in common)
            emit(
                f"| {name} | {cell(r2, r1, groups['all'])} | {cell(r2, r1, groups['binary'])} | "
                f"{cell(r2, r1, groups['numeric+discrete'])} | {absolute:.1f} |"
            )

    emit("\n## How often variant B reconciles (material disagreement)\n")
    emit("| type | run 1 | run 2 | both runs | either run |")
    emit("|---|---|---|---|---|")
    for kind in ("binary", "multiple_choice", "numeric", "discrete", "all"):
        posts = [p for p, r in rows.items() if kind == "all" or r["type"] == kind]
        r1 = sum(trigger.get((p, 1), False) for p in posts)
        r2 = sum(trigger.get((p, 2), False) for p in posts)
        both = sum(trigger.get((p, 1), False) and trigger.get((p, 2), False) for p in posts)
        either = sum(trigger.get((p, 1), False) or trigger.get((p, 2), False) for p in posts)
        emit(f"| {kind} | {r1}/{len(posts)} | {r2}/{len(posts)} | {both} | {either} |")

    # The reconciled forecast on questions with and without material disagreement,
    # question-run pairs pooled (each pair counted once).
    emit("\n## Variant A split by disagreement (question-run pairs, not averaged over runs)\n")
    emit("| configuration | material disagreement | no material disagreement |")
    emit("|---|---|---|")
    for config in configs:
        cells = []
        for want in (True, False):
            diffs = []
            for rep in REPS:
                a, m = scores.get(f"{config} A r{rep}", {}), scores.get(f"mean r{rep}", {})
                diffs += [a[p] - m[p] for p in set(a) & set(m) if trigger[(p, rep)] == want]
            if diffs:
                arr = np.array(diffs)
                boot = np.random.default_rng(0).choice(arr, size=(20000, len(arr)), replace=True).mean(axis=1)
                cells.append(
                    f"{arr.mean():+.2f} [{np.percentile(boot, 5):+.2f}, {np.percentile(boot, 95):+.2f}] "
                    f"{int((arr > 1e-9).sum())}/{int((arr < -1e-9).sum())} (n={len(arr)})"
                )
            else:
                cells.append("-")
        emit(f"| {config} A | {cells[0]} | {cells[1]} |")

    # Where the reconciled binary forecast lands between the members, on material disagreements:
    # 0 is GPT-6.1 Sol's value, 1 is Claude Sonnet 5's.
    emit("\n## Binary questions with material disagreement: where the reconciled forecast lands\n")
    emit("Position 0 = GPT-6.1 Sol, 1 = Sonnet 5. 'Sided with the better member' counts forecasts on the side of the member that scored higher.")
    for config in configs:
        positions, right, wrong, shown_first = [], 0, 0, {"sol": [], "sonnet": []}
        for rep in REPS:
            for pid, row in rows.items():
                if row["type"] != "binary" or not trigger.get((pid, rep)):
                    continue
                a = latest.get((config, pid, rep))
                if not a or a["status"] != "ok":
                    continue
                ps, pn = members[rep][pid]["sol"]["prediction"], members[rep][pid]["sonnet"]["prediction"]
                t = (a["prediction"] - ps) / (pn - ps)
                positions.append(t)
                shown_first[first[(pid, rep)]].append(t)
                better_is_sonnet = scores[f"sonnet5 r{rep}"][pid] > scores[f"sol61 r{rep}"][pid]
                if t > 0.5:
                    right += better_is_sonnet
                    wrong += not better_is_sonnet
                elif t < 0.5:
                    right += not better_is_sonnet
                    wrong += better_is_sonnet
        if positions:
            arr = np.array(positions)
            emit(
                f"- {config}: n={len(arr)}, median position {np.median(arr):.2f}; nearer Sol (<0.25) {int((arr < 0.25).sum())},"
                f" middle {int(((arr >= 0.25) & (arr <= 0.75)).sum())}, nearer Sonnet (>0.75) {int((arr > 0.75).sum())},"
                f" outside the range {int(((arr < 0) | (arr > 1)).sum())}; sided with the better member {right}, with the worse {wrong};"
                f" median position when Sol is shown first {np.median(shown_first['sol']) if shown_first['sol'] else float('nan'):.2f},"
                f" when Sonnet is shown first {np.median(shown_first['sonnet']) if shown_first['sonnet'] else float('nan'):.2f}"
            )

    emit("\n## Numeric and discrete: reconciled spread and location\n")
    emit("Width is P10 to P90 relative to the mean width of the two members; 'inside' counts medians between the members' medians.")
    for label, source in [(c, {(p, r): a["prediction"] for (cc, p, r), a in latest.items() if cc == c and a["status"] == "ok"})
                          for c in configs] + [("opus direct", opus)]:
        ratios, inside, n = [], 0, 0
        for rep in REPS:
            for pid, row in rows.items():
                if row["type"] not in ("numeric", "discrete") or (pid, rep) not in source:
                    continue
                pair = members[rep][pid]
                f = source[(pid, rep)]
                widths = [_quantile(x, 0.9) - _quantile(x, 0.1) for x in (pair["sol"]["prediction"], pair["sonnet"]["prediction"])]
                if sum(widths) > 0:
                    ratios.append((_quantile(f, 0.9) - _quantile(f, 0.1)) / (sum(widths) / 2))
                medians = [_quantile(pair["sol"]["prediction"], 0.5), _quantile(pair["sonnet"]["prediction"], 0.5)]
                inside += min(medians) - 1e-9 <= _quantile(f, 0.5) <= max(medians) + 1e-9
                n += 1
        if ratios:
            q1, q2, q3 = np.percentile(ratios, [25, 50, 75])
            emit(f"- {label}: width ratio median {q2:.2f} [p25 {q1:.2f}, p75 {q3:.2f}]; median inside the members' {inside} of {n}")

    emit("\n## Questions that move variant A most (2 runs, against the mean)\n")
    for config in configs:
        name = f"{config} A"
        if name not in scores:
            continue
        diffs = sorted(((scores[name][p] - scores["mean"][p], p) for p in scores[name]), reverse=True)
        top = ", ".join(f"{p} {rows[p]['type'][:3]} {d:+.1f}" for d, p in diffs[:5])
        bottom = ", ".join(f"{p} {rows[p]['type'][:3]} {d:+.1f}" for d, p in diffs[-5:])
        emit(f"- {name}: best {top}; worst {bottom}")

    emit("\n## Reconciler runs: tokens, cost and failures\n")
    emit(
        "| configuration | answers ok | failed | fell back to the mean | input tokens (mean) | thinking median [p25, p75]"
        " | output median | $/answer paid (batch) | $/answer at list price | parser $/answer | format check mismatches |"
    )
    emit("|---|---|---|---|---|---|---|---|---|---|---|")
    for config in configs:
        mine = [a for (c, _, _), a in latest.items() if c == config]
        usage = [a["reconciler_usage"] for a in mine if a.get("reconciler_usage")]
        thinking = [u.get("reasoning_tokens") or 0 for u in usage]
        q = statistics.quantiles(thinking, n=4) if len(thinking) > 3 else [float("nan")] * 3
        emit(
            f"| {config} | {sum(a['status'] == 'ok' for a in mine)} | {sum(a['status'] != 'ok' for a in mine)} | {fallbacks.get(config, 0)}"
            f" | {statistics.mean(u['prompt_tokens'] for u in usage):.0f} | {statistics.median(thinking):.0f} [{q[0]:.0f}, {q[2]:.0f}]"
            f" | {statistics.median(u['completion_tokens'] for u in usage):.0f} | {statistics.mean(cdr.usage_cost(u) for u in usage):.4f}"
            f" | {statistics.mean(cdr.list_cost(u) for u in usage):.4f}"
            f" | {statistics.mean(sum(cdr.usage_cost(u) for u in a.get('parser_usage', [])) for a in mine):.4f}"
            f" | {sum(bool(a.get('format_check')) for a in mine)} |"
        )
    problems = [a for a in latest.values() if a["status"] != "ok" or a.get("format_check")]
    if problems:
        emit("\n## Failures and format check mismatches\n")
        for a in problems:
            emit(f"- {a['post_id']} {a['type']} {a['config']} r{a['rep']}: {a['status']} {a.get('error') or ''} {a.get('format_check') or ''}")
    emit(f"\nEstimated spend on the Anthropic key: ${spent():.2f}")

    with open(os.path.join(OUT, "analysis.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT, "scores.json"), "w", encoding="utf-8") as fh:
        json.dump({"log": scores, "brier": briers, "trigger": {f"{p} r{r}": v for (p, r), v in trigger.items()}}, fh, indent=1)
    print(f"\nWrote {OUT}/analysis.md and {OUT}/scores.json")


def main_cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inputs", help="build every reconciler prompt and show the disagreements (no LLM calls)")
    smoke = sub.add_parser("smoke", help="plain calls on a few questions, run 1")
    smoke.add_argument("--posts", required=True, help="comma-separated post ids")
    smoke.add_argument("--configs", default=",".join(CONFIGS))
    submit = sub.add_parser("submit", help="send one batch per configuration and run")
    submit.add_argument("--configs", default=",".join(CONFIGS))
    submit.add_argument("--reps", default="1,2")
    submit.add_argument("--posts", default="", help="comma-separated post ids (default: all)")
    collect = sub.add_parser("collect", help="fetch finished batches and parse their answers")
    collect.add_argument("--wait", action="store_true", help="poll every 60 s until every batch is collected")
    sub.add_parser("analyze", help="score everything recorded (no LLM calls)")
    sub.add_parser("spend", help="estimated spend so far")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%H:%M:%S")
    logger.setLevel(logging.INFO)
    # The library warns on every renormalized option list and untracked model; nothing to act on here.
    logging.getLogger("forecasting_tools").setLevel(logging.ERROR)
    os.makedirs(OUT, exist_ok=True)
    {
        "inputs": cmd_inputs, "smoke": cmd_smoke, "submit": cmd_submit, "collect": cmd_collect,
        "analyze": cmd_analyze, "spend": cmd_spend,
    }[args.command](args)


if __name__ == "__main__":
    main_cli()
