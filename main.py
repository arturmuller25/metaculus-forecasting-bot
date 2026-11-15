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
import time

# When this process started. The Claude Code forecasters end their calls
# RUN_DEADLINE_S seconds after it (see _claude_code).
_STARTED = time.time()

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
    # forecaster (SHADOW_MODELS) to catch a regression. Claude Sonnet 5.5
    # replaced Sonnet 5 on 2026-10-19: in a replay of MiniBench round 2 with
    # Claude on the Anthropic API (where it reasons; through OpenRouter it
    # returned no reasoning tokens in round 1), the pair with Sonnet 5.5 beat
    # the pair with Sonnet 5 by 3.1 points per question (90% CI +0.6 to +5.7,
    # 57 questions, two runs), at the same price; Sonnet 5 was the weakest
    # model alone, and the weaker member live in round 2.
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
        "openrouter/anthropic/claude-sonnet-5.5",
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
# Model that writes the evidence packet when EVIDENCE_PACKET=1.
PACKET_MODEL = "openrouter/openai/gpt-5.6-sol"


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


# Seconds after the start of this process by which every Claude Code call has
# ended (RUN_DEADLINE_S overrides it; 0 means no deadline). The workflow
# kills the main call of each iteration at 1500 seconds (`timeout 1500` in
# forecast.yml): 1320 leaves 180 for the closing lines, so the kill finds no
# CLI process running.
RUN_DEADLINE_S = 1320


def _claude_code(specs: list[str], what: str) -> list:
    """
    A forecaster on the Claude Code CLI for each spec ("claudecode/<model>@<effort>"
    reads the forecast prompt, "claudecode-agent/<model>@<effort>" also
    searches and reads pages): cc_forecaster.py, paid by the bot maker's
    Claude subscription instead of an API key. The module is imported only
    when a spec asks for it. Without the CLI or its credential, or on any
    error here, the list is empty and one log line says why: these
    forecasters must never stop a run.
    """
    try:
        import cc_forecaster

        ok, reason = cc_forecaster.available()
        if ok:
            seconds = float((os.getenv("RUN_DEADLINE_S") or "").strip() or RUN_DEADLINE_S)
            cc_forecaster.set_deadline(_STARTED + seconds if seconds > 0 else None)
            return [cc_forecaster.ClaudeCodeForecaster.from_spec(spec) for spec in specs]
    except Exception as exc:  # noqa: BLE001
        reason = f"{type(exc).__name__}: {exc}"
    logger.warning(f"Claude Code {what} skipped: {reason}")
    return []


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
    # AGENT_MEMBER=1 makes the agentic forecaster (agent_forecaster.py) a
    # member, run with the others before publishing and pooled like them,
    # instead of a deferred shadow. The rule set on 2026-10-22 (at least +1.0
    # points per question over the pair, 90% interval above zero) passed on
    # 2026-11-14: +1.99 (+0.73 to +3.28) over 171 questions, 113 replayed as
    # of each forecast time (timetravel_agent.py, +0.86) and 58 live as a
    # shadow (live_score.py, +4.19, the agent answering within 16 minutes of
    # the published forecast and over two hours before each question closed). With
    # ENSEMBLE_MODELS set to Claude Sonnet 5.5 alone the pool is Sonnet and the
    # agent, +3.80 (+1.93 to +5.66) over the pair on the same questions:
    # GPT-6.1 Sol alone scored 3.6 points per question below the pair in both
    # sets. That choice was made after seeing the live data and then held in
    # the replay, so RESERVE_MODEL keeps GPT-6.1 Sol answering every question:
    # recorded as a shadow, and pooled only when fewer than two members answer
    # (the agent needs the Anthropic credit and can time out).
    agent_member = None
    if ensemble and os.getenv("AGENT_MEMBER", "").strip() == "1":
        agent_spec = os.getenv("AGENT_MEMBER_SPEC", "").strip() or "agent/claude-sonnet-5-5@xhigh"
        if agent_spec.startswith("claudecode-agent/"):
            # The agent's seat on the Claude Code CLI and the Claude
            # subscription instead of the Anthropic credit. Without the CLI
            # or its credential there is no agent member and the reserve
            # covers the seat.
            agent_member = next(iter(_claude_code([agent_spec], "agent member")), None)
        elif os.getenv("ANTHROPIC_API_KEY", "").strip():
            from agent_forecaster import AgentForecaster

            agent_member = AgentForecaster.from_spec(agent_spec)
            agent_member.record_name = agent_spec
        if agent_member is not None:
            ensemble.append(agent_member)

    reserve = None
    reserve_model = os.getenv("RESERVE_MODEL", "").strip()
    if ensemble and reserve_model and reserve_model not in _members:
        reserve = (reserve_model, _member(reserve_model))

    # Shadow models: forecast every question on the same research, recorded to
    # logs/forecasts.jsonl but never published. SHADOW_MODELS="model[@effort],..."
    # e.g. "openrouter/openai/gpt-5.4@medium" to test a cheaper forecaster, or
    # "anthropic/claude-opus-5-5" for a shadow paid by the Anthropic key.
    shadows = []
    cc_specs = []
    for spec in (s.strip() for s in os.getenv("SHADOW_MODELS", "").split(",")):
        if not spec:
            continue
        if spec.startswith(("claudecode/", "claudecode-agent/")):
            cc_specs.append(spec)
            continue
        if spec.startswith("agent/"):
            # Agentic forecaster on the Anthropic API (agent_forecaster.py):
            # searches, reads pages and runs code before forecasting. It takes
            # minutes, so it runs after the run's forecasts are published.
            # Already a member with AGENT_MEMBER=1: not run twice.
            if agent_member is not None and not agent_member.model.startswith("claudecode-agent/"):
                continue
            from agent_forecaster import AgentForecaster

            agent = AgentForecaster.from_spec(spec)
            agent.deferred = True
            shadows.append((spec, agent))
            continue
        model, _, effort = spec.partition("@")
        shadows.append((spec, _thinker(model.strip(), 0.3, 120, effort=effort.strip() or None)))
    # Shadows on the Claude Code CLI run last, after the deferred ones
    # (bot.run_claude_code_shadows), under the name the forecaster gives
    # itself. One that is already the agent member is not run twice.
    cc_shadows = [
        (forecaster.model, forecaster)
        for forecaster in (_claude_code(cc_specs, "shadows") if cc_specs else [])
        if agent_member is None or forecaster.model != agent_member.model
    ]

    # Evidence packet (evidence_packet.py): one call per question at low
    # effort that sorts and dates the research. Off since 2026-10-21;
    # EVIDENCE_PACKET=1 turns it on. On MiniBench round 1 replays it added
    # 2.0 points per question with Claude Sonnet 5 as the second member
    # (90% interval +0.8 to +3.2), but with the current pair (GPT-6.1 Sol and
    # Sonnet 5.5) it lost 0.68 over rounds 1 and 2 (112 questions, 90%
    # interval -1.47 to +0.03; round 2 alone -1.22, -2.35 to -0.29), mostly
    # through Sonnet 5.5 on binaries.
    packet_llm = None
    if os.getenv("EVIDENCE_PACKET", "").strip() == "1":
        packet_llm = _thinker(os.getenv("PACKET_MODEL", "").strip() or PACKET_MODEL, 0.1, 120, effort="low")

    # Single-series market questions (market_query.py): a low-effort call
    # names the series, then code computes the baseline. MARKET_QUERY=0 turns
    # it off.
    market_llm = None
    if os.getenv("MARKET_QUERY", "1").strip() != "0":
        market_llm = _thinker(os.getenv("MARKET_QUERY_MODEL", "").strip() or PACKET_MODEL, 0.0, 120, effort="low")

    # Deferred panel experiments on the Anthropic API, recorded as shadows
    # after the run's forecasts are published: GAP_FILL=1 checks the research
    # against primary sources (gap_fill.py) and has the members answer again
    # on its findings; SUPERVISOR=1 reconciles every analysis (supervisor.py).
    # PANEL_SHARE (0 to 1, default 1) samples the questions they run on.
    gap_agent = supervisor_agent = None
    if os.getenv("ANTHROPIC_API_KEY", "").strip():
        if os.getenv("GAP_FILL", "").strip() == "1":
            import gap_fill

            gap_agent = gap_fill.make_agent()
        if os.getenv("SUPERVISOR", "").strip() == "1":
            import supervisor

            supervisor_agent = supervisor.make_agent()
    elif os.getenv("GAP_FILL", "").strip() == "1" or os.getenv("SUPERVISOR", "").strip() == "1":
        logger.warning("GAP_FILL and SUPERVISOR need ANTHROPIC_API_KEY; both stay off")

    return ForecasterBot(
        ensemble=ensemble,
        shadows=shadows,
        cc_shadows=cc_shadows,
        reserve=reserve,
        packet_llm=packet_llm,
        market_llm=market_llm,
        gap_agent=gap_agent,
        supervisor=supervisor_agent,
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
# forecasts. Market Pulse scores only the forecast standing at each
# question's close; Animal Futures averages scores over time.
REFRESHED_MODES = ("market_pulse", "animal_futures")
SPOT_SCORED_MODES = ("market_pulse",)
REFRESH_DAYS = float(os.getenv("REFRESH_DAYS", "") or 14)
REFRESH_FINAL_HOURS = float(os.getenv("REFRESH_FINAL_HOURS", "") or 12)


def _due(question, spot: bool = False) -> bool:
    """
    Whether a question should be forecast now. A new question always is.
    Where only the forecast at the close counts (spot=True): once more in the
    last REFRESH_FINAL_HOURS hours, and nothing in between, since a forecast
    that is replaced before the close scores nothing (until 2026-11-14 each
    Market Pulse question was forecast about five times, on a credit that ran
    out). Elsewhere: a forecast older than REFRESH_DAYS, or older than 12
    hours within 2 days of the close.
    """
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
    to_close = (close - now).total_seconds() / 3600 if close is not None else None
    if spot:
        return to_close is not None and to_close <= REFRESH_FINAL_HOURS and age_hours > REFRESH_FINAL_HOURS
    if to_close is not None and to_close <= 48:
        return age_hours > 12
    return age_hours > REFRESH_DAYS * 24


# Listings asked for when a tournament is listed by schedule: the one the
# library itself uses, to tell whether a post came in it, and one with no
# status filter (None), so a post the API files under any other status still
# comes.
SCHEDULE_STATUSES = ("open", None)


def _api_time(value):
    """An API date (ISO string or epoch seconds) as an aware datetime; None when absent."""
    from datetime import datetime, timezone

    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, timezone.utc)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _open_at(question_json: dict, now, check_resolution: bool = True) -> bool:
    """
    Whether the schedule of a question (its API JSON) says it takes forecasts
    at `now`: open_time <= now < scheduled_close_time, not closed early and
    not resolved. The `status` field is ignored on purpose.
    """
    opens = _api_time(question_json.get("open_time"))
    closes = _api_time(question_json.get("scheduled_close_time"))
    if opens is None or closes is None or not opens <= now < closes:
        return False
    for key in ("actual_close_time", "actual_resolve_time"):
        moment = _api_time(question_json.get(key))
        if moment is not None and moment <= now:
            return False
    # actual_resolve_time above is the check that counts: on 2026-11-14 the
    # API sent `resolution` as null on all 17 resolved Market Pulse questions
    # this account had not forecast.
    return not check_resolution or question_json.get("resolution") in (None, "")


def _open_by_time(posts: list, now=None, check_resolution: bool = True) -> list:
    """
    Question objects, built by the library's own constructors, for every
    question and group subquestion in `posts` (post JSON from the API) that
    is open at `now` by its schedule, whatever its `status` field says. Each
    one is built on its own: the library builds a whole group at once and
    drops it when any subquestion fails to parse, open or not. What cannot
    be parsed is skipped with a warning. `posts` and `now` are parameters so
    this runs offline in a test.
    """
    from datetime import datetime, timezone

    from forecasting_tools import DataOrganizer

    now = now or datetime.now(timezone.utc)
    found: dict = {}
    for post in posts:
        post_id = post.get("id") if isinstance(post, dict) else None
        try:
            if "group_of_questions" in post:
                group = post["group_of_questions"]
                raw = group["questions"]
            elif "question" in post:
                group, raw = None, [post["question"]]
            else:
                continue  # notebook or conditional
            group_ids = [q["id"] for q in raw]
        except Exception as exc:
            logger.warning(f"Listing by schedule: post {post_id} skipped: {type(exc).__name__}: {exc}")
            continue
        for question_json in raw:
            try:
                if not _open_at(question_json, now, check_resolution):
                    continue
                if group is None:
                    question = DataOrganizer.get_question_from_post_json(post)
                else:
                    alone = {**post, "group_of_questions": {**group, "questions": [question_json]}}
                    question = MetaculusClient._unpack_group_question(alone)[0]
                    question.question_ids_of_group = list(group_ids)
                found.setdefault(question.id_of_question, question)
            except Exception as exc:
                logger.warning(
                    f"Listing by schedule: question {question_json.get('id')} of post {post_id} skipped: "
                    f"{type(exc).__name__}: {exc}"
                )
    return list(found.values())


def _fetch_posts(client: MetaculusClient, tournament_id, statuses=SCHEDULE_STATUSES) -> list:
    """
    The posts of a tournament as API JSON, one request per listing (a status,
    or None for no status filter) and per page of 100 (500 posts per listing
    at most), in the library's own order. with_cp goes along, as in the
    library's listing: on 2026-11-14 every question then carried
    my_forecasts, which _due reads, and a listing without it carried none.
    Each post gets a "_listed_as" key naming the listings that returned it,
    for the warnings in _add_open_by_time. A listing that fails is skipped
    with a warning.
    """
    import time

    import requests

    posts: dict = {}
    for status in statuses:
        label = status or "no filter"
        try:
            for page in range(5):
                time.sleep(3)  # the forecasts go through the same API: stay far from its rate limit
                params = {
                    "tournaments": [tournament_id],
                    "limit": 100,
                    "offset": page * 100,
                    "order_by": "-published_time",
                    "with_cp": "true",
                    "include_conditional_cps": "true",
                }
                if status:
                    params["statuses"] = [status]
                response = requests.get(
                    f"{client.base_url}/posts/",
                    params=params,
                    headers={"Authorization": f"Token {client.token}", "Accept-Language": "en"},
                    timeout=client.timeout,
                )
                response.raise_for_status()
                results = response.json().get("results") or []
                for post in results:
                    posts.setdefault(post["id"], post).setdefault("_listed_as", []).append(label)
                if len(results) < 100:
                    break
        except Exception as exc:
            logger.warning(f"Listing by schedule: listing '{label}' of tournament {tournament_id} not read: {type(exc).__name__}: {exc}")
    return list(posts.values())


def _add_open_by_time(listed: list, client: MetaculusClient, tournament_id, now=None, posts=None) -> list:
    """
    The library's list of open questions plus the questions that are open by
    their schedule and missing from it. The library asks for posts with
    status open, builds every subquestion of each group and keeps the ones
    whose own `status` is "open": a post the API files under another status,
    a subquestion whose status lags its schedule and a sibling that fails to
    parse all end with the question absent and nothing in the log. In
    October 2026 the biweekly Market Pulse subquestions stayed out of that
    list through two whole windows.

    A missing question the API itself calls open is added. One it calls
    anything else is only reported, unless OPEN_BY_TIME_HOURS is set: then
    it is also added during its first hours. The default is 0: on the
    leaderboard of 2026-11-14 no participant had more scored questions than
    this bot, so the October subquestions probably never took forecasts, and
    a forecast the API refuses never shows, so _due calls the question new
    and every run pays for it again. The warnings say what the API reported
    for each question, the evidence that was missing in October. Any failure
    here leaves the library's list as it is. OPEN_BY_TIME=0 turns this off.
    """
    from datetime import datetime, timezone

    if (os.getenv("OPEN_BY_TIME") or "1").strip() == "0":
        return listed
    try:
        now = now or datetime.now(timezone.utc)
        hours = float((os.getenv("OPEN_BY_TIME_HOURS") or "").strip() or 0)
        if posts is None:
            posts = _fetch_posts(client, tournament_id)
        known = {q.id_of_question for q in listed}
        extra = [q for q in _open_by_time(posts, now) if q.id_of_question not in known]

        def describe(questions: list) -> str:
            return "; ".join(
                f"question {q.id_of_question} (post {q.id_of_post}, {q.group_question_option or 'single question'}): "
                f"question status '{getattr(q.state, 'value', q.state)}', post status '{q.api_json.get('status')}', "
                f"post listed as {q.api_json.get('_listed_as', 'not recorded')}, open_time {q.open_time}"
                for q in questions
            )

        # _due takes a question with no forecast of ours as new. Without the
        # my_forecasts block that would be every run, at full cost each time.
        blind = [q for q in extra if not isinstance(q.api_json["question"].get("my_forecasts"), dict)]
        left_out = {q.id_of_question for q in blind}
        late = [
            q
            for q in extra
            if q.id_of_question not in left_out
            and getattr(q.state, "value", q.state) != "open"
            and (now - q.open_time).total_seconds() > hours * 3600
        ]
        left_out |= {q.id_of_question for q in late}
        extra = [q for q in extra if q.id_of_question not in left_out]
        if blind:
            logger.warning(
                f"Tournament {tournament_id}: {len(blind)} question(s) open by schedule were left out, "
                f"the API sent no my_forecasts for them: {describe(blind)}"
            )
        if late:
            logger.warning(
                f"Tournament {tournament_id}: {len(late)} question(s) open by schedule were left out, the API does "
                f"not call them open (they are tried only in their first {hours:g} hours, OPEN_BY_TIME_HOURS): {describe(late)}"
            )
        if extra:
            logger.warning(
                f"Tournament {tournament_id}: {len(extra)} question(s) open by schedule were missing from "
                f"the library's open list and were added: {describe(extra)}"
            )
        return listed + extra
    except Exception as exc:
        logger.warning(
            f"Listing by schedule failed for tournament {tournament_id}; using the library's list only: "
            f"{type(exc).__name__}: {exc}"
        )
        return listed


# Everything through the Anthropic API (the bot maker's own credit) and free
# sources, with no OpenRouter call and no AskNews: for tournaments outside
# FutureEval, which the tournament-funded key and the sponsored AskNews plan
# are not meant for (Metaculus asked to keep its key on FutureEval and
# MiniBench, 2026-10-12). "none" matches no provider, so research is Claude's
# web search plus the resolution source download.
CLAUDE_ONLY = {
    "FORECAST_MODEL": "anthropic/claude-sonnet-5",
    "RESEARCH_MODEL": "anthropic/claude-sonnet-5",
    "PARSER_MODEL": "anthropic/claude-haiku-4-5",
    "ENSEMBLE_MODELS": "openrouter/anthropic/claude-sonnet-5.5,openrouter/anthropic/claude-opus-5.5",
    "RESEARCH_PROVIDERS": "none",
    "SHADOW_MODELS": "",
    # The agent as a member was measured on FutureEval and MiniBench only, and
    # the reserve would call the OpenRouter key.
    "AGENT_MEMBER": "0",
    "RESERVE_MODEL": "",
    "DIRECT_ANTHROPIC": "1",
    "ANTHROPIC_FALLBACK": "0",
    # The packet and the market-question reader were tested with GPT-5.6 Sol only.
    "EVIDENCE_PACKET": "0",
    "MARKET_QUERY": "0",
    "GAP_FILL": "0",
    "SUPERVISOR": "0",
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
                listed = client.get_all_open_questions_from_tournament(tid)
                open_questions += _add_open_by_time(listed, client, tid)
            due = [q for q in open_questions if _due(q, spot=mode in SPOT_SCORED_MODES)]
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
    # Then the Claude Code shadows, which end before the run's deadline.
    await bot.run_claude_code_shadows()

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
        global FORECAST_MODEL, RESEARCH_MODEL, PARSER_MODEL, MAX_COST_PER_RUN
        os.environ.update(CLAUDE_ONLY)
        FORECAST_MODEL = CLAUDE_ONLY["FORECAST_MODEL"]
        RESEARCH_MODEL = CLAUDE_ONLY["RESEARCH_MODEL"]
        PARSER_MODEL = CLAUDE_ONLY["PARSER_MODEL"]
        # The $5 cap protects the tournament key; eight Claude-only questions
        # cost about that much, and the Anthropic credit has its own limit.
        MAX_COST_PER_RUN = max(MAX_COST_PER_RUN, 15.0)
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
    # The Anthropic SDK used by the agentic forecaster leaves an idle anyio
    # worker thread (not a daemon) whose stop callback never runs, so the
    # interpreter waits for it forever after main() returns. On 2026-10-09
    # two GitHub runs hung for hours after their last forecast and a question
    # was missed. Everything is written by now: flush and exit explicitly.
    # os._exit skips the exit handlers, so a Claude Code CLI process still
    # alive is stopped here.
    if sys.modules.get("cc_forecaster") is not None:
        try:
            sys.modules["cc_forecaster"].shutdown()
        except Exception:  # noqa: BLE001 - nothing may stop the exit below
            pass
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
