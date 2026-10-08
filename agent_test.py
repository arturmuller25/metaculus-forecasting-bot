"""
Dry run of the agentic forecaster (agent_forecaster.py) through the bot's own
forecast path, on a few live questions. Nothing is published, and only the
Anthropic key is spent (ANTHROPIC_API_KEY): the research step, which uses
OpenRouter, is replaced by a short briefing built by code from the resolution
sources (resolution_fetch.py), the parser is Claude Haiku 4.5 on the
Anthropic API, and any other model call is refused.

    uv run python agent_test.py check                      # offline checks, no API call
    uv run python agent_test.py list                       # MiniBench and Fall, open or recently closed
    uv run python agent_test.py run --posts 45951,45960    # forecast them with the agent
    uv run python agent_test.py run --posts ... --effort medium --uses 12 --limit 300

Writes logs/agent_test/: forecasts.jsonl (the bot's per-model record, in
place of logs/forecasts.jsonl), agent_calls.jsonl (one line per agent call)
and results.jsonl (one line per question, with the full answer).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

OUT = os.path.join("logs", "agent_test")
os.makedirs(OUT, exist_ok=True)
os.environ["AGENT_CALL_LOG"] = os.path.join(OUT, "agent_calls.jsonl")
# main.py builds the bot's LLMs from these when it is imported.
os.environ["PARSER_MODEL"] = "anthropic/claude-haiku-4-5"
os.environ["SHADOW_MODELS"] = ""

import bot as bot_module  # noqa: E402
import main  # noqa: E402  (loads .env)
from forecasting_tools import (  # noqa: E402
    BinaryQuestion,
    GeneralLlm,
    MetaculusApi,
    MetaculusClient,
    MonetaryCostManager,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.ai_models import general_llm  # noqa: E402

from agent_forecaster import AgentForecaster  # noqa: E402
from resolution_fetch import fetch_resolution_sources, urls_in  # noqa: E402

logger = logging.getLogger("agent_test")
bot_module.FORECAST_LOG = os.path.join(OUT, "forecasts.jsonl")
RESULTS = os.path.join(OUT, "results.jsonl")
SPEND_CAP = 15.0  # dollars, the agent's estimate plus the parser's

# Guard: any model call outside the Anthropic API is refused.
_acompletion = general_llm.acompletion


async def _anthropic_only(*args, **kwargs):
    model = str(kwargs.get("model") or (args[0] if args else ""))
    if not model.startswith("anthropic/"):
        raise RuntimeError(f"refused a call to {model}: this test only spends the Anthropic key")
    return await _acompletion(*args, **kwargs)


general_llm.acompletion = _anthropic_only


def _spent() -> float:
    if not os.path.exists(RESULTS):
        return 0.0
    with open(RESULTS, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return sum((r.get("agent_cost_usd") or 0) + (r.get("parser_cost_usd") or 0) for r in rows)


async def cmd_list(days: int) -> None:
    """Questions open now or closed in the last `days` days, in MiniBench and the Fall tournament."""
    from datetime import timedelta

    from forecasting_tools.helpers.metaculus_client import ApiFilter

    client = MetaculusClient()
    since = datetime.now(timezone.utc) - timedelta(days=days)
    for label, tid in (("minibench", client.CURRENT_MINIBENCH_ID), ("fall", client.CURRENT_AI_COMPETITION_ID)):
        found = await client.get_questions_matching_filter(
            ApiFilter(allowed_tournaments=[tid], allowed_statuses=["open", "closed"], close_time_gt=since),
            num_questions=60,
            error_if_question_target_missed=False,
        )
        print(f"\n{label} ({tid}): {len(found)} open or closed since {since:%Y-%m-%d}")
        for q in sorted(found, key=lambda q: q.close_time or since):
            n_urls = len(urls_in(q.resolution_criteria, q.fine_print))
            close = q.close_time.strftime("%m-%d %H:%M") if q.close_time else "-"
            state = getattr(q.state, "value", q.state)
            print(f"  {q.id_of_post:>6}  {type(q).__name__[:8]:8}  {state:6}  close {close}  urls {n_urls}  {q.question_text[:85]}")


def cmd_check() -> None:
    """Offline checks of the answer-block detection, no API call."""
    from agent_forecaster import _options, finish_answer

    binary = 'The last thing you write is your final answer as: "Probability: ZZ%", 0-100'
    numeric = "Percentile 10: XX (lowest number value)\nPercentile 20: XX\n"
    mc = "The options are: ['Yes, a lot', 'No', 'Maybe 2']\n...\nOption_A: Probability_A\n"
    pcts = "Percentile 10: 1,000\nPercentile 20: 2\nPercentile 40: 3\nPercentile 60: 4\nPercentile 80: 5\n"
    cases = [
        (finish_answer("x\n**Probability: 35%**\nThanks!", binary), "x\n**Probability: 35%**"),
        (finish_answer("no answer", binary), None),
        (finish_answer("x\n" + pcts + "Percentile 90: 6.5 (highest)\nbye", numeric), "x\n" + pcts + "Percentile 90: 6.5 (highest)"),
        (finish_answer("Percentile 10: 1\nPercentile 90: 2", numeric), None),
        (finish_answer("Yes, a lot: 20%\nNo: 50%\nMaybe 2: 30%", mc), "Yes, a lot: 20%\nNo: 50%\nMaybe 2: 30%"),
        (finish_answer("Yes, a lot: 20%\nNo: 50%", mc), None),
        (finish_answer("x\n\nOption_A: 0.68\nOption_B: 0.29\n- **Option_C**: 3%\n\nDone.", mc), "x\n\nOption_A: 0.68\nOption_B: 0.29\n- **Option_C**: 3%"),
        (_options(mc), ["Yes, a lot", "No", "Maybe 2"]),
    ]
    for i, (got, want) in enumerate(cases, 1):
        assert got == want, f"case {i}: {got!r} != {want!r}"
    print(f"{len(cases)} offline checks passed")


def _host(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def _briefing(question) -> str:
    """A short research briefing made by code, without any model: the resolution sources downloaded."""
    fetched = fetch_resolution_sources(question)
    if not fetched.strip():
        return "No research briefing is available for this question; the resolution criteria name no URL."
    return f"## Source: Resolution source download\n{fetched.strip()}"


async def forecast_one(bot, agent: AgentForecaster, post_id: int) -> dict:
    question = MetaculusApi.get_question_by_post_id(post_id)
    research = await asyncio.to_thread(_briefing, question)
    criteria_urls = urls_in(question.resolution_criteria, question.fine_print)
    row: dict = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "post_id": post_id,
        "url": question.page_url,
        "type": type(question).__name__,
        "question": question.question_text,
        "criteria_urls": criteria_urls,
        "briefing_chars": len(research),
    }
    start = time.monotonic()
    n_calls = len(agent.calls)
    with MonetaryCostManager(hard_limit=5.0) as cost:
        try:
            if isinstance(question, BinaryQuestion):
                result = await bot._run_forecast_on_binary(question, research)
            elif isinstance(question, MultipleChoiceQuestion):
                result = await bot._run_forecast_on_multiple_choice(question, research)
            elif isinstance(question, NumericQuestion):
                result = await bot._run_forecast_on_numeric(question, research)
            else:
                raise RuntimeError(f"unsupported type {type(question).__name__}")
            row["parsed"] = True
            row["forecast"] = bot_module._serialize(result.prediction_value)
            row["answer"] = result.reasoning
        except Exception as exc:
            row["parsed"] = False
            row["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
        row["parser_cost_usd"] = round(cost.current_usage, 4)
    row["seconds_total"] = round(time.monotonic() - start, 1)

    stats = agent.calls[-1] if len(agent.calls) > n_calls else {}
    fetched_ok = [f["url"] for f in stats.get("fetches", []) if f.get("ok")]
    criteria_hosts = {_host(u) for u in criteria_urls}
    answer = row.get("answer") or ""
    first_line = answer.strip().splitlines()[0] if answer.strip() else ""
    row.update(
        agent_outcome=stats.get("outcome"),
        agent_seconds=stats.get("seconds"),
        agent_cost_usd=stats.get("cost_usd"),
        tokens={k: stats.get(k) for k in ("input_tokens", "cache_write_tokens", "cache_read_tokens", "output_tokens")},
        searches=stats.get("web_search_requests"),
        fetches=stats.get("web_fetch_requests"),
        fetches_read=len(fetched_ok),
        code_runs=stats.get("code_runs"),
        allowed_urls=stats.get("allowed_urls"),
        queries=stats.get("queries"),
        fetched=stats.get("fetches"),
        repaired=stats.get("repaired"),
        stop_reasons=stats.get("stop_reasons"),
        # Opened the source: fetched a page on a host the criteria name, or a
        # subdomain of it. Without a URL in the criteria, the agent's own
        # statement is all there is.
        opened_criteria_host=any(
            h == c or h.endswith("." + c) for h in {_host(u) for u in fetched_ok} for c in criteria_hosts
        ) if criteria_hosts else None,
        agent_says=first_line[:300],
    )
    with open(RESULTS, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


async def cmd_run(posts: list[int], effort: str, uses: int, limit: int, model: str) -> None:
    bot = main.build_bot(publish=False, samples=1)
    agent = AgentForecaster(model=model, effort=effort, max_tool_uses=uses, time_limit_s=limit)
    # The agent is the only ensemble member; no shadows; Anthropic parser.
    bot._ensemble = [agent]
    bot._shadows = []
    haiku = GeneralLlm(model="anthropic/claude-haiku-4-5", temperature=0.0, timeout=60, allowed_tries=2)
    bot.set_llm(haiku, "parser")
    bot.set_llm(haiku, "summarizer")

    for post_id in posts:
        if _spent() >= SPEND_CAP - 1.5:
            print(f"Stopping: ${_spent():.2f} spent, cap ${SPEND_CAP:.2f}")
            break
        row = await forecast_one(bot, agent, post_id)
        print(
            f"\n=== {post_id} {row['type']}: parsed={row['parsed']} forecast={row.get('forecast')}\n"
            f"    agent {row['agent_outcome']} {row['agent_seconds']}s, searches {row['searches']}, "
            f"fetches {row['fetches']} ({row['fetches_read']} read), code {row['code_runs']}, "
            f"allowed {len(row['allowed_urls'] or [])}, "
            f"cost ${row['agent_cost_usd']} + parser ${row['parser_cost_usd']}\n"
            f"    opened criteria host: {row['opened_criteria_host']}; agent says: {row['agent_says'][:160]}"
            + (f"\n    error: {row['error']}" if row.get("error") else "")
        )
    print(f"\nTotal estimated spend in {RESULTS}: ${_spent():.2f}")


def main_cli() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    lst = sub.add_parser("list")
    lst.add_argument("--days", type=int, default=7, help="also list questions closed in the last N days")
    r = sub.add_parser("run")
    r.add_argument("--posts", required=True, help="comma-separated post ids")
    r.add_argument("--model", default="claude-sonnet-5")
    r.add_argument("--effort", default="high")
    r.add_argument("--uses", type=int, default=12)
    r.add_argument("--limit", type=int, default=300)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpx2", "LiteLLM", "litellm"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.cmd == "check":
        cmd_check()
    elif args.cmd == "list":
        asyncio.run(cmd_list(args.days))
    else:
        posts = [int(x) for x in args.posts.split(",") if x.strip()]
        asyncio.run(cmd_run(posts, args.effort, args.uses, args.limit, args.model))


if __name__ == "__main__":
    sys.exit(main_cli())
