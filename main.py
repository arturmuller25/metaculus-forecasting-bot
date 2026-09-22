"""
Command-line entry point.

    uv run python main.py --mode test                  # bot testing area, dry run
    uv run python main.py --mode tournament            # seasonal tournament + MiniBench, dry run
    uv run python main.py --mode tournament --publish  # submit forecasts

Without --publish the bot runs the full pipeline and saves reports to logs/,
but submits nothing.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

import dotenv

# The Windows console defaults to cp1252, and question texts contain
# characters such as "≥" that make the logger crash mid-run. Force UTF-8.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

dotenv.load_dotenv()

from forecasting_tools import GeneralLlm, MetaculusClient, MonetaryCostManager

from bot import ForecasterBot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Two setups, chosen by the keys present in .env:
#
# 1. OpenRouter (how tournament credits are delivered): openrouter/ prefix.
#    The :online suffix turns on web search inside OpenRouter.
#
# 2. Metaculus token only: metaculus/ prefix. The library routes these calls
#    to the Metaculus LLM proxy, authenticated with METACULUS_TOKEN. Useful
#    for testing the pipeline.

_HAS_OPENROUTER = bool(os.getenv("OPENROUTER_API_KEY"))

if _HAS_OPENROUTER:
    # GPT-5.x as the final forecaster is the strongest signal repeated across
    # the last two tournament seasons (r=+0.42 in Metaculus's analysis).
    _FORECAST = "openrouter/openai/gpt-5.4"
    _RESEARCH = "openrouter/openai/gpt-5.4:online"
    _PARSER = "openrouter/openai/gpt-4o-mini"
    # No Google model: this key has no usable Gemini quota (checked
    # 2026-09-18: zero quota for gemini-3.1-pro, 20 requests/min for
    # gemini-3.8-flash, which also returns only reasoning tokens, and the
    # Vertex route is blocked by the key).
    #
    # Ensemble members have no :online suffix. Source diversity comes from the
    # research providers; with a web search per member, one question cost
    # $4.38 (measured 2026-09-22) and the members repeated the same engines.
    _ENSEMBLE = [
        "openrouter/openai/gpt-5.4",
        "openrouter/anthropic/claude-sonnet-4.6",
    ]
else:
    _FORECAST = "metaculus/claude-sonnet-4-5"
    _RESEARCH = "metaculus/gpt-4o-search-preview"
    _PARSER = "metaculus/gpt-4o-mini"
    # The Metaculus proxy only routes Anthropic and OpenAI models.
    _ENSEMBLE = [
        "metaculus/claude-sonnet-4-5",
        "metaculus/gpt-4o",
    ]

FORECAST_MODEL = os.getenv("FORECAST_MODEL", _FORECAST)
RESEARCH_MODEL = os.getenv("RESEARCH_MODEL", _RESEARCH)
PARSER_MODEL = os.getenv("PARSER_MODEL", _PARSER)

# Cost cap per run, in USD; the run aborts when it is exceeded. Web search
# from :online models is not tracked by the library.
MAX_COST_PER_RUN = float(os.getenv("MAX_COST_PER_RUN", "5.00"))

# Reasoning effort for the forecasting models. "high" is the best-supported
# finding in Metaculus's bot analyses: it beat "low" in 8 of 8 pairs
# (p=0.004). The parser takes no reasoning parameter.
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "high").strip()
# Research uses low effort. Measured 2026-09-22 with research_settings_check.py
# on 3 questions: low, medium and high cost $0.30, $0.49 and $0.59 per call in
# tokens, and higher effort did not return richer research.
RESEARCH_REASONING = os.getenv("RESEARCH_REASONING", "low").strip()


def _thinker(model: str, temperature: float, timeout: int, effort: str | None = None) -> GeneralLlm:
    """GeneralLlm with a reasoning effort, for reasoning models."""
    effort = REASONING_EFFORT if effort is None else effort
    kwargs = dict(model=model, temperature=temperature, timeout=timeout, allowed_tries=2)
    if effort in ("low", "medium", "high"):
        kwargs["reasoning_effort"] = effort
    return GeneralLlm(**kwargs)

TOURNAMENT_URLS = {
    "tournament": "https://www.metaculus.com/tournament/fall-futureeval-2026/",
    "minibench": "https://www.metaculus.com/aib/minibench",
    "cup": "https://www.metaculus.com/tournament/metaculus-cup/",
    "market_pulse": "https://www.metaculus.com/tournament/market-pulse-26q4/",
    "test": "https://www.metaculus.com/tournament/bot-testing-area/",
}


def build_bot(publish: bool, samples: int) -> ForecasterBot:
    llms = {
        "default": _thinker(FORECAST_MODEL, 0.3, 120),
        "researcher": _thinker(RESEARCH_MODEL, 0.1, 180, effort=RESEARCH_REASONING),
        "parser": GeneralLlm(model=PARSER_MODEL, temperature=0.0, timeout=60, allowed_tries=2),
        "summarizer": GeneralLlm(model=PARSER_MODEL, temperature=0.0, timeout=60),
    }
    # ENSEMBLE=0 falls back to the single default model; ENSEMBLE_MODELS
    # overrides the member list.
    _override = os.getenv("ENSEMBLE_MODELS", "").strip()
    _members = [m.strip() for m in _override.split(",") if m.strip()] if _override else _ENSEMBLE
    ensemble = (
        []
        if os.getenv("ENSEMBLE", "1") == "0"
        else [_thinker(m, 0.3, 120) for m in _members]
    )

    # Shadow models: forecast every question on the same research, recorded to
    # logs/forecasts.jsonl but never published. SHADOW_MODELS="model[@effort],..."
    # e.g. "openrouter/openai/gpt-5.4@medium" to test a cheaper forecaster.
    shadows = []
    for spec in (s.strip() for s in os.getenv("SHADOW_MODELS", "").split(",")):
        if spec:
            model, _, effort = spec.partition("@")
            shadows.append((spec, _thinker(model.strip(), 0.3, 120, effort=effort.strip() or None)))

    return ForecasterBot(
        ensemble=ensemble,
        shadows=shadows,
        # One research report per question, `samples` forecasts on it.
        research_reports_per_question=1,
        predictions_per_research_report=samples,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish,
        folder_to_save_reports_to="logs/",
        # Metaculus rule for bot-only tournaments: "Bot makers should only
        # submit one forecast per question in these bot-only tournaments."
        # Dry runs forecast everything.
        skip_previously_forecasted_questions=publish,
        extra_metadata_in_explanation=True,
        llms=llms,
    )


def check_env(publish: bool) -> None:
    # METACULUS_TOKEN is the only hard requirement: without a provider key the
    # bot uses the Metaculus proxy, which authenticates with the same token.
    if not os.getenv("METACULUS_TOKEN"):
        print("METACULUS_TOKEN is missing.", file=sys.stderr)
        print("Copy .env.example to .env and fill it in. See README.md.", file=sys.stderr)
        sys.exit(1)

    if not _HAS_OPENROUTER:
        print("No OPENROUTER_API_KEY: using the Metaculus LLM proxy.")
        print("Fine for testing the pipeline; the tournament setup uses OpenRouter.\n")

    if publish:
        print("PUBLISH MODE: forecasts WILL be submitted to Metaculus.\n")
    else:
        print("Dry run: nothing will be published. Use --publish to submit.\n")


def openrouter_usage() -> float | None:
    """
    Total spent on the OpenRouter key, in USD, read from OpenRouter itself.
    The library's cost tracking misses :online search and hidden reasoning
    tokens; the /key endpoint shows what is actually charged.
    """
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        return None
    import json
    import urllib.request

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/key",
        headers={"Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.load(resp).get("data", {})
    except Exception as exc:
        logger.warning(f"Could not read the OpenRouter balance: {exc}")
        return None

    # On BYOK keys (provider keys plugged into OpenRouter) spend shows in
    # byok_usage while usage stays at zero. Limit minus remaining is what
    # counts against the cap; without a limit, add the two fields.
    limit = data.get("limit")
    remaining = data.get("limit_remaining")
    if limit is not None and remaining is not None:
        return float(limit) - float(remaining)
    return float(data.get("usage") or 0.0) + float(data.get("byok_usage") or 0.0)


def _pick(questions: list, limit: int) -> list:
    """Picks up to `limit` questions, alternating question types, so a small test covers every code path."""
    by_type: dict[str, list] = {}
    for q in questions:
        by_type.setdefault(type(q).__name__, []).append(q)
    picked: list = []
    while len(picked) < limit and any(by_type.values()):
        for pending in by_type.values():
            if pending and len(picked) < limit:
                picked.append(pending.pop(0))
    return picked


async def run(mode: str, publish: bool, samples: int, limit: int | None) -> list:
    bot = build_bot(publish, samples)
    client = MetaculusClient()

    targets = {
        "tournament": [client.CURRENT_AI_COMPETITION_ID, client.CURRENT_MINIBENCH_ID],
        "minibench": [client.CURRENT_MINIBENCH_ID],
        "cup": [client.CURRENT_METACULUS_CUP_ID],
        "market_pulse": [client.CURRENT_MARKET_PULSE_ID],
        "test": ["bot-testing-area"],
    }
    if mode not in targets:
        raise ValueError(f"unknown mode: {mode}")
    if mode in ("cup", "test"):
        bot.skip_previously_forecasted_questions = False

    before = openrouter_usage()

    # The parameter is hard_limit, not max_cost (the library README is out of
    # date). It raises when the cap is exceeded.
    with MonetaryCostManager(hard_limit=MAX_COST_PER_RUN) as cost:
        if limit is None:
            reports = []
            for tid in targets[mode]:
                reports += await bot.forecast_on_tournament(tid, return_exceptions=True)
        else:
            open_questions = []
            for tid in targets[mode]:
                open_questions += client.get_all_open_questions_from_tournament(tid)
            chosen = _pick(open_questions, limit)
            kinds = ", ".join(type(q).__name__.replace("Question", "") for q in chosen)
            print(f"Limited to {len(chosen)} of {len(open_questions)} open questions: {kinds}\n")
            reports = await bot.forecast_questions(chosen, return_exceptions=True)

        tracked = cost.current_usage

    after = openrouter_usage()
    print(f"\nCost tracked by the library : ${tracked:.4f}")
    if before is not None and after is not None:
        real_cost = after - before
        n = max(1, sum(1 for r in reports if not isinstance(r, BaseException)))
        print(f"Real cost on OpenRouter     : ${real_cost:.4f}  (${real_cost / n:.4f} per question)")
        print(f"Total spent on the key      : ${after:.4f}")

    # log_report_summary raises RuntimeError with the full traceback when
    # everything fails; diagnose() explains the common cases more readably.
    try:
        bot.log_report_summary(reports)
    except RuntimeError:
        pass
    return reports


def diagnose(reports: list) -> None:
    """Explains the most common failures in one sentence plus a next step."""
    errors = [r for r in reports if isinstance(r, BaseException)]
    if not errors:
        return

    blob = " ".join(str(e) for e in errors)

    if "allowance" in blob:
        model = "the requested model"
        import re

        m = re.search(r"allowance for model <([^>]+)>", blob)
        if m:
            model = m.group(1)
        print(
            f"\nDIAGNOSIS: your account has no allowance for {model} on the Metaculus proxy."
            "\nThe token is valid and the questions were read; only LLM credit is missing."
            "\n\nTwo ways out:"
            "\n  1. Request the tournament credits: https://forms.gle/aQdYMq9Pisrf1v7d8"
            "\n     They arrive as an OpenRouter key. Put it in OPENROUTER_API_KEY in .env."
            "\n  2. Use your own OpenAI, Anthropic or OpenRouter key in .env."
        )
    elif "401" in blob or "Permission" in blob or "authenticat" in blob.lower():
        print(
            "\nDIAGNOSIS: METACULUS_TOKEN was rejected."
            "\nCreate a new one in Settings > My Forecasting Bots > Reveal API Key."
        )
    elif "429" in blob or "rate" in blob.lower():
        print(
            "\nDIAGNOSIS: rate limit reached."
            "\nLower _max_concurrent_questions in bot.py or wait a few minutes."
        )
    else:
        print("\nDIAGNOSIS: unrecognized failure. First full error:\n")
        print(f"  {type(errors[0]).__name__}: {str(errors[0])[:600]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Metaculus forecasting bot")
    parser.add_argument(
        "--mode",
        choices=["test", "tournament", "minibench", "cup", "market_pulse"],
        default="test",
        help="where to forecast (default: test, the bot testing area)",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="submit forecasts to Metaculus (without it, dry run)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help=(
            "how many times to run the forecast step per question (default: 1). "
            "Each sample queries every ensemble model."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="forecast at most N questions, alternating question types. Use it to test cheaply.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    check_env(args.publish)

    print(f"Mode      : {args.mode}")
    print(f"Model     : {FORECAST_MODEL}")
    print(f"Research  : {RESEARCH_MODEL}")
    print(f"Samples   : {args.samples} per question")
    print(f"Tournament: {TOURNAMENT_URLS.get(args.mode, '-')}")
    print()

    reports = asyncio.run(run(args.mode, args.publish, args.samples, args.limit))

    errors = [r for r in reports if isinstance(r, BaseException)]
    ok = len(reports) - len(errors)
    print(f"\nQuestions forecast: {ok}. Failures: {len(errors)}.")
    if ok:
        print("Reports saved in logs/")
    diagnose(reports)


if __name__ == "__main__":
    main()
