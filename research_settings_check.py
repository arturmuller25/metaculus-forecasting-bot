"""
Mechanism check for the research step: what reasoning effort changes in the
text each provider returns, and what it costs. Publishes nothing.

It does not measure accuracy: telling two settings apart by 0.01 of Brier
would take about 2000 questions. It answers a cheaper question that still
decides the setting: does more reasoning buy richer research (more distinct
sources, more dated facts), and at what price?

Uses the production calls (GeneralLlm, the prompt captured from run_research,
the same timeout) on open questions from MiniBench and the bot testing area.

Cost per call is computed from tokens with litellm's price table (including
the long-context rate). Per-search fees (about $0.01 each) are not in the
token counts; they only show in the key's real spend, printed at the end.

Usage: uv run python research_settings_check.py   (about $5 with all five settings)
"""

import asyncio
import contextvars
import io
import json
import re
import sys
import time

import dotenv
import litellm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
dotenv.load_dotenv(".env")

import forecasting_tools.ai_models.general_llm as gl  # noqa: E402
from forecasting_tools import GeneralLlm, MetaculusClient  # noqa: E402

from main import RESEARCH_MODEL, build_bot, openrouter_usage  # noqa: E402

N_QUESTIONS = 3
BUDGET = 4.00  # USD in tokens; above it, the remaining "high" calls are skipped
CLAUDE = "openrouter/anthropic/claude-sonnet-5:online"
CONFIGS = [  # label, model, reasoning effort (None = parameter not sent)
    ("claude-none", CLAUDE, None),
    ("claude-low", CLAUDE, "low"),
    ("gpt-low", RESEARCH_MODEL, "low"),
    ("gpt-medium", RESEARCH_MODEL, "medium"),
    ("gpt-high", RESEARCH_MODEL, "high"),
]

# Captures litellm responses to read the tokens and cost of each call.
_captured = contextvars.ContextVar("captured", default=None)
_original = gl.acompletion


async def _spy(*a, **k):
    r = await _original(*a, **k)
    captured = _captured.get()
    if captured is not None:
        captured.append(r)
    return r


gl.acompletion = _spy
spent = {"tokens": 0.0}


def richness(t: str) -> dict:
    urls = re.findall(r"https?://[^\s)\]>\"']+", t)
    domains = {re.sub(r"^www\.", "", u.split("/")[2]) for u in urls if len(u.split("/")) > 2}
    dates = re.findall(
        r"\b202[4-6]-\d\d-\d\d\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? 202[5-6]\b"
        r"|\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* 202[5-6]\b",
        t,
    )
    markets = bool(re.search(r"polymarket|kalshi|manifold", t, re.I))
    return {"chars": len(t), "urls": len(urls), "domains": len(domains), "dates": len(dates), "markets": markets}


def token_cost(r) -> float:
    try:
        return float(litellm.completion_cost(completion_response=r) or 0)
    except Exception:
        return 0.0


async def run_config(label, model, effort, prompt):
    kw = dict(model=model, temperature=0.1, timeout=180, allowed_tries=2)
    if effort:
        kw["reasoning_effort"] = effort
    captured = []
    _captured.set(captured)
    t0 = time.time()
    try:
        text = await GeneralLlm(**kw).invoke(prompt)
        error = None
    except Exception as e:  # record it and move on
        text, error = "", f"{type(e).__name__}: {str(e)[:120]}"
    elapsed = time.time() - t0
    usage = [r.usage for r in captured if getattr(r, "usage", None)]
    tokens_in = sum((u.prompt_tokens or 0) for u in usage)
    tokens_out = sum((u.completion_tokens or 0) for u in usage)
    reasoning_tokens = sum(
        (getattr(u.completion_tokens_details, "reasoning_tokens", 0) or 0)
        for u in usage
        if getattr(u, "completion_tokens_details", None)
    )
    cost = sum(token_cost(r) for r in captured)
    spent["tokens"] += cost
    return {"config": label, "seconds": round(elapsed), "tokens_in": tokens_in, "tokens_out": tokens_out,
            "reasoning_tokens": reasoning_tokens, "token_cost": round(cost, 4), "error": error,
            **richness(text), "text": text}


async def main():
    bot = build_bot(publish=False, samples=1)
    client = MetaculusClient()
    open_questions = client.get_all_open_questions_from_tournament("minibench")
    open_questions += client.get_all_open_questions_from_tournament("bot-testing-area")
    regular = [
        q for q in open_questions
        if not bot._meta_question_block(q).strip() and type(q).__name__ != "DateQuestion"
    ]
    by_type = {}  # varied question types, deterministic order
    for q in regular:
        by_type.setdefault(type(q).__name__, []).append(q)
    chosen = []
    while len(chosen) < N_QUESTIONS and any(by_type.values()):
        for t in list(by_type):
            if by_type[t] and len(chosen) < N_QUESTIONS:
                chosen.append(by_type[t].pop(0))
    print(f"{len(open_questions)} open, {len(regular)} not meta-questions. Chosen:")
    for q in chosen:
        print(f"  [{type(q).__name__}] {q.page_url} | {q.question_text[:90]}")

    # Captures the exact production prompt without calling any API.
    captured_prompts = []
    real_invoke = GeneralLlm.invoke

    async def grab(self, prompt, *a, **k):
        captured_prompts.append(prompt)
        return "stub"

    prompts = []
    for q in chosen:
        captured_prompts.clear()
        GeneralLlm.invoke = grab
        try:
            await bot.run_research(q)
        finally:
            GeneralLlm.invoke = real_invoke
        prompts.append(captured_prompts[0])

    before = openrouter_usage()

    async def run_question(q, prompt):
        rows = []
        for label, model, effort in CONFIGS:
            if label == "gpt-high" and spent["tokens"] > BUDGET:
                rows.append({"config": label, "error": "skipped: budget reached"})
                continue
            rows.append(await run_config(label, model, effort, prompt))
            print(f"  done {label:11s} {q.page_url}  (token spend so far ${spent['tokens']:.2f})", flush=True)
        return {"url": q.page_url, "type": type(q).__name__, "question": q.question_text, "rows": rows}

    results = await asyncio.gather(*(run_question(q, p) for q, p in zip(chosen, prompts)))
    await asyncio.sleep(20)  # let the key balance settle
    after = openrouter_usage()

    print(f"\n{'config':11s} {'sec':>4s} {'tok_in':>7s} {'tok_out':>7s} {'reason':>6s} {'USD':>6s} "
          f"{'chars':>6s} {'URLs':>4s} {'dom':>3s} {'dates':>5s} mkts")
    for r in results:
        print(f"--- [{r['type']}] {r['question'][:80]}")
        for row in r["rows"]:
            if row.get("error") and "chars" not in row:
                print(f"{row['config']:11s} {row['error']}")
                continue
            print(f"{row['config']:11s} {row['seconds']:4d} {row['tokens_in']:7d} {row['tokens_out']:7d} {row['reasoning_tokens']:6d} "
                  f"{row['token_cost']:6.3f} {row['chars']:6d} {row['urls']:4d} {row['domains']:3d} {row['dates']:5d} "
                  f"{'yes' if row['markets'] else '-'}{'  ERROR ' + row['error'] if row.get('error') else ''}")
    print(f"\nToken spend: ${spent['tokens']:.2f}")
    print(f"Real spend on the key (tokens + search fees; includes the live bot if it ran meanwhile): "
          f"${after - before:.2f}")
    io.open("logs/research_settings_check.json", "w", encoding="utf-8").write(json.dumps(results, ensure_ascii=False, indent=1))
    print("details (with the texts) in logs/research_settings_check.json")


asyncio.run(main())
