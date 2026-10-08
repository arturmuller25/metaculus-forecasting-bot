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
    # GPT-5.6 Sol and Claude Sonnet 5 replaced gpt-5.4 and claude-sonnet-4.6
    # on 2026-09-22: newer, and cheaper per token. GPT-6.1 Sol replaced
    # GPT-5.6 Sol as forecaster on 2026-10-07: in a replay of MiniBench round
    # 1 (forecast_replay.py, two runs) it tied on binaries and did better on
    # numerics (+2.8 points per question, 90% CI +0.05 to +5.7) at the same
    # price. GPT-5.6 Sol still does the research, and runs as a shadow
    # forecaster (SHADOW_MODELS) to catch a regression. Claude Opus 5.5 cost
    # 60% more without a clear gain, and Claude Sonnet 5.5 returned no
    # reasoning tokens through OpenRouter, so Sonnet 5 stays.
    _FORECAST = "openrouter/openai/gpt-6.1-sol"
    _RESEARCH = "openrouter/openai/gpt-5.6-sol:online"
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
        "openrouter/openai/gpt-6.1-sol",
        "openrouter/anthropic/claude-sonnet-5",
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
# on 3 questions (gpt-5.4): low, medium and high cost $0.30, $0.49 and $0.59
# per call in tokens, and higher effort did not return richer research.
RESEARCH_REASONING = os.getenv("RESEARCH_REASONING", "low").strip()


def _thinker(model: str, temperature: float, timeout: int, effort: str | None = None) -> GeneralLlm:
    """GeneralLlm with a reasoning effort, for reasoning models."""
    effort = REASONING_EFFORT if effort is None else effort
    kwargs = dict(model=model, temperature=temperature, timeout=timeout, allowed_tries=2)
    if effort in ("low", "medium", "high"):
        kwargs["reasoning_effort"] = effort
    return GeneralLlm(**kwargs)

class _DirectFirst:
    """
    A Claude ensemble member called through the Anthropic API first
    (ANTHROPIC_API_KEY, the bot maker's own API credit), and through
    OpenRouter when the direct call fails or that credit runs out. Both routes
    reason the same at the same effort (measured with claude_direct_replay.py
    on 56 questions). It keeps the OpenRouter name, so records and comments
    stay comparable over time.
    """

    def __init__(self, openrouter_model: str, temperature: float, timeout: int) -> None:
        self.model = openrouter_model
        provider, _, name = openrouter_model.removeprefix("openrouter/").partition("/")
        # OpenRouter writes versions with a dot (claude-opus-5.5), the
        # Anthropic API with a hyphen (claude-opus-5-5).
        self._direct = _thinker(f"{provider}/{name.replace('.', '-')}", temperature, timeout)
        self._backup = _thinker(openrouter_model, temperature, timeout)

    async def invoke(self, prompt: str) -> str:
        try:
            return await self._direct.invoke(prompt)
        except Exception as exc:
            # ANTHROPIC_FALLBACK=0 (the Claude-only runs) keeps every call off
            # the OpenRouter key: the member fails instead.
            if os.getenv("ANTHROPIC_FALLBACK", "").strip() == "0":
                raise
            logger.warning(f"Direct Anthropic call for {self.model} failed ({type(exc).__name__}); using OpenRouter")
            return await self._backup.invoke(prompt)


def _member(model: str):
    """An ensemble member: Claude models go direct first when an Anthropic key is set (DIRECT_ANTHROPIC=0 turns it off)."""
    direct = bool(os.getenv("ANTHROPIC_API_KEY", "").strip()) and os.getenv("DIRECT_ANTHROPIC", "").strip() != "0"
    if direct and model.startswith("openrouter/anthropic/"):
        return _DirectFirst(model, 0.3, 120)
    return _thinker(model, 0.3, 120)


TOURNAMENT_URLS = {
    "tournament": "https://www.metaculus.com/tournament/fall-futureeval-2026/",
    "minibench": "https://www.metaculus.com/aib/minibench",
    "cup": "https://www.metaculus.com/tournament/metaculus-cup/",
    "market_pulse": "https://www.metaculus.com/tournament/market-pulse-26q4/",
    "animal_futures": "https://www.metaculus.com/notebooks/43978/",
    "test": "https://www.metaculus.com/tournament/bot-testing-area/",
}


def build_bot(publish: bool, samples: int) -> ForecasterBot:
    researcher = _thinker(RESEARCH_MODEL, 0.1, 180, effort=RESEARCH_REASONING)
    if RESEARCH_MODEL.startswith("anthropic/"):
        # Claude through the Anthropic API searches the web with its own tool.
        researcher = GeneralLlm(
            model=RESEARCH_MODEL, temperature=0.1, timeout=180, allowed_tries=2,
            reasoning_effort=RESEARCH_REASONING, web_search_options={"search_context_size": "medium"},
        )
    llms = {
        "default": _thinker(FORECAST_MODEL, 0.3, 120),
        "researcher": researcher,
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
        else [_member(m) for m in _members]
    )

    # Shadow models: forecast every question on the same research, recorded to
    # logs/forecasts.jsonl but never published. SHADOW_MODELS="model[@effort],..."
    # e.g. "openrouter/openai/gpt-5.4@medium" to test a cheaper forecaster, or
    # "anthropic/claude-opus-5-5" for a shadow paid by the Anthropic key.
    shadows = []
    for spec in (s.strip() for s in os.getenv("SHADOW_MODELS", "").split(",")):
        if not spec:
            continue
        if spec.startswith("agent/"):
            # Agentic forecaster on the Anthropic API (agent_forecaster.py):
            # searches, reads pages and runs code before forecasting. It takes
            # minutes, so it runs after the run's forecasts are published.
            from agent_forecaster import AgentForecaster

            agent = AgentForecaster.from_spec(spec)
            agent.deferred = True
            shadows.append((spec, agent))
            continue
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


def _key_info(key: str | None) -> dict | None:
    """The /key record of an OpenRouter key, or None when it cannot be read."""
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
            return json.load(resp).get("data", {})
    except Exception as exc:
        logger.warning(f"Could not read the OpenRouter balance: {exc}")
        return None


def use_fallback_key_if_low() -> None:
    """
    Switches to OPENROUTER_FALLBACK_KEY, a key the bot maker funds, when the
    tournament key has less than FALLBACK_BELOW dollars left (default 3,
    about four questions), so questions are not missed while a top-up is
    pending. Checked at the start of every run; once the tournament key is
    topped up, the next run goes back to it. Without the variable nothing
    changes.
    """
    fallback = (os.getenv("OPENROUTER_FALLBACK_KEY") or "").strip()
    if not fallback:
        return
    floor = float((os.getenv("FALLBACK_BELOW") or "").strip() or 3)
    data = _key_info(os.getenv("OPENROUTER_API_KEY"))
    remaining = (data or {}).get("limit_remaining")
    if remaining is not None and float(remaining) < floor:
        os.environ["OPENROUTER_API_KEY"] = fallback
        print(f"Tournament key has ${float(remaining):.2f} left (below ${floor:.2f}): using the fallback key.\n")


def openrouter_usage() -> float | None:
    """
    Total spent on the OpenRouter key, in USD, read from OpenRouter itself.
    The library's cost tracking misses :online search and hidden reasoning
    tokens; the /key endpoint shows what is actually charged.
    """
    data = _key_info(os.getenv("OPENROUTER_API_KEY"))
    if data is None:
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


# Tournaments where the bot competes with humans and may update its
# forecasts. Market Pulse scores the forecast standing at each question's
# close; Animal Futures averages scores over time. Both reward a fresh
# forecast near the close and a first forecast early.
REFRESHED_MODES = ("market_pulse", "animal_futures")
REFRESH_DAYS = float(os.getenv("REFRESH_DAYS", "") or 14)


def _due(question) -> bool:
    """New question, a forecast older than REFRESH_DAYS, or within 2 days of closing with a forecast older than 12 hours."""
    from datetime import datetime, timezone

    try:
        last = question.timestamp_of_my_last_forecast
    except ValueError:
        last = None
    if last is None:
        return True
    now = datetime.now(timezone.utc)
    age_hours = (now - last).total_seconds() / 3600
    close = question.close_time
    if close is not None and (close - now).total_seconds() <= 2 * 86400:
        return age_hours > 12
    return age_hours > REFRESH_DAYS * 24


# Everything through the Anthropic API (the bot maker's own credit) and free
# sources, with no OpenRouter call: for tournaments outside FutureEval, which
# the tournament-funded key is not meant for.
CLAUDE_ONLY = {
    "FORECAST_MODEL": "anthropic/claude-sonnet-5",
    "RESEARCH_MODEL": "anthropic/claude-sonnet-5",
    "PARSER_MODEL": "anthropic/claude-haiku-4-5",
    "ENSEMBLE_MODELS": "openrouter/anthropic/claude-sonnet-5,openrouter/anthropic/claude-opus-5.5",
    "RESEARCH_PROVIDERS": "asknews",
    "SHADOW_MODELS": "",
    "DIRECT_ANTHROPIC": "1",
    "ANTHROPIC_FALLBACK": "0",
}


async def run(mode: str, publish: bool, samples: int, limit: int | None) -> list:
    bot = build_bot(publish, samples)
    client = MetaculusClient()

    targets = {
        "tournament": [client.CURRENT_AI_COMPETITION_ID, client.CURRENT_MINIBENCH_ID],
        "minibench": [client.CURRENT_MINIBENCH_ID],
        "cup": [client.CURRENT_METACULUS_CUP_ID],
        "market_pulse": [client.CURRENT_MARKET_PULSE_ID],
        "animal_futures": [33016],
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
        if mode in REFRESHED_MODES and limit is None:
            # Tournaments open to humans allow updates: forecast new questions
            # and refresh old forecasts (see _due).
            open_questions = []
            for tid in targets[mode]:
                open_questions += client.get_all_open_questions_from_tournament(tid)
            due = [q for q in open_questions if _due(q)]
            # Soonest close first, and a cap per run, so a first pass over a
            # whole tournament spreads over several iterations instead of
            # holding up the next FutureEval check.
            due.sort(key=lambda q: q.close_time.timestamp() if q.close_time else float("inf"))
            cap = int((os.getenv("REFRESH_MAX_PER_RUN") or "").strip() or 8)
            print(f"{len(due)} of {len(open_questions)} open questions due for a forecast; doing up to {cap}\n")
            due = due[:cap]
            bot.skip_previously_forecasted_questions = False
            reports = await bot.forecast_questions(due, return_exceptions=True) if due else []
        elif limit is None:
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

    # Slow shadows (the agent) run now, after every forecast is published.
    await bot.run_deferred_shadows()

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
    if bot.model_failures:
        failures = ", ".join(f"{name}: {n}" for name, n in sorted(bot.model_failures.items()))
        print(f"Failed or unparsable model answers this run: {failures}")
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
    elif "402" in blob or "credits" in blob.lower() or "key limit" in blob.lower():
        print(
            "\nDIAGNOSIS: the OpenRouter key is out of credit."
            "\nAsk Metaculus for a top-up, or set OPENROUTER_FALLBACK_KEY to a funded key of your own."
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
        choices=["test", "tournament", "minibench", "cup", "market_pulse", "animal_futures"],
        default="test",
        help="where to forecast (default: test, the bot testing area)",
    )
    parser.add_argument(
        "--claude-only",
        action="store_true",
        help="use only the Anthropic API (ANTHROPIC_API_KEY) and free sources, never the OpenRouter key",
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
    if args.claude_only:
        if not os.getenv("ANTHROPIC_API_KEY", "").strip():
            print("--claude-only needs ANTHROPIC_API_KEY.", file=sys.stderr)
            sys.exit(1)
        global FORECAST_MODEL, RESEARCH_MODEL, PARSER_MODEL
        os.environ.update(CLAUDE_ONLY)
        FORECAST_MODEL = CLAUDE_ONLY["FORECAST_MODEL"]
        RESEARCH_MODEL = CLAUDE_ONLY["RESEARCH_MODEL"]
        PARSER_MODEL = CLAUDE_ONLY["PARSER_MODEL"]
        print("Claude only: Anthropic API and free sources, no OpenRouter calls.\n")
    else:
        use_fallback_key_if_low()

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
