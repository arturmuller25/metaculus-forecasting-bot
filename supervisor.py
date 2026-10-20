"""
Supervisor: one Claude call with web tools that reads the panel's
independent analyses (the ensemble members, the agentic forecaster with its
own research, and the members again with the gap-fill findings when those
ran), settles the factual disagreements against primary sources, and gives
the final forecast in the bot's answer format.

Design from edisonymy-bot (github.com/edisonymy/forecast-scaffold, a
consistent MiniBench top-15 bot): a supervisor reconciles three independent
research runs, sorts each disagreement into FACTUAL or JUDGMENT, settles the
factual ones and is told not to average. A tool-free reconciler that read
the two members' answers on the same research (reconcile_replay.py, MiniBench
round 1, Opus 5.5 at high effort) scored +1.6 points per question over the
mean, 90% interval -0.6 to +3.6: +4.0 on numeric questions (+0.6 to +7.1),
nothing on binaries. Two things differ here and are the point of the test:
the analyses rest on different research, and factual points can be checked.

The bot records the supervisor as a shadow (supervisor/<model>) after the
run's forecasts are published, to be scored against the published forecast
once questions resolve. Anthropic API only.
"""

from __future__ import annotations

import os

from agent_forecaster import AgentForecaster

SYSTEM = """\
You are the supervisor of a forecasting panel. You have web_search, \
web_fetch, a Python code_execution sandbox (call web_search and web_fetch \
from inside your code when that helps you filter or compute) and allow_url. \
Today is {today} (UTC).

The user message has two parts: the forecasting task the panel received (the \
question, resolution criteria, fine print, a research briefing and the answer \
format), then the panel's independent analyses, each ending with its final \
forecast. Some analysts researched on their own and report facts the \
briefing lacks; those facts can be right or wrong.

Produce the panel's final forecast:
1. List the points where the analyses disagree in a way that moves the \
forecast. Mark each FACTUAL (what is true now: a value, a date, a status, an \
event that did or did not happen, what the resolution source shows or how its \
rule reads) or JUDGMENT (how much weight a fact deserves, which base rate \
applies, how fast things move).
2. Settle each FACTUAL point that matters with the tools, against the primary \
source: the resolution source, an official page, a filing, a data download. \
Say what you found, with URL and date, or that you could not settle it.
3. Decide. Where you settled a factual point, follow the analysis that had it \
right, even when it is the outlier. Where the analyses differ only in \
judgment, stay near the panel's pooled forecast unless one analysis gives a \
specific, checkable reason the others missed. Do not average by habit and do \
not split differences mechanically; do not move outside the range of the \
analyses without a fact you verified. If the analyses agree and you found no \
factual error, give the pooled forecast.

Budget: at most {searches} web searches and {fetches} page fetches, and about \
{minutes} minutes in total. Spend them on factual disagreements and on the \
single claim the forecast leans on most, nothing else. web_fetch can only \
open URLs that already appear in the conversation. To open a URL you built \
yourself, pass it to allow_url first, then fetch it.

Write your whole answer in one final message after your last tool call: the \
disagreements and how you settled each, in at most 250 words, then the answer \
block exactly in the format the task specifies. The answer block is the very \
last thing you write. Write nothing after it."""

# Each analysis is cut to this many characters (its start, where the agent
# reports what it read, and its end, where the reasoning concludes).
_HEAD, _TAIL = 3000, 6000


def make_agent() -> AgentForecaster:
    """
    The supervisor, set by SUPERVISOR_MODEL (default claude-sonnet-5-5),
    SUPERVISOR_EFFORT (high), SUPERVISOR_TOOLS (searches plus fetches, 6) and
    SUPERVISOR_TIME (seconds, 300).
    """
    return AgentForecaster(
        model=(os.getenv("SUPERVISOR_MODEL") or "").strip() or "claude-sonnet-5-5",
        effort=(os.getenv("SUPERVISOR_EFFORT") or "").strip() or "high",
        max_tool_uses=int((os.getenv("SUPERVISOR_TOOLS") or "").strip() or 6),
        time_limit_s=int((os.getenv("SUPERVISOR_TIME") or "").strip() or 300),
        system=SYSTEM,
        role="supervisor",
    )


def _cut(text: str) -> str:
    text = text.strip()
    if len(text) <= _HEAD + _TAIL:
        return text
    return text[:_HEAD] + "\n[...]\n" + text[-_TAIL:]


def message(task: str, analyses: list[tuple[str, str, str]], pooled: str) -> str:
    """
    The supervisor's user message. `analyses` holds (how the analyst worked,
    final forecast as a short string, full text); names are left out so no
    model brand sways the supervisor. `pooled` is the panel's pooled forecast
    in words.
    """
    parts = [task.rstrip(), "", "=====", "The panel's independent analyses", "====="]
    for i, (how, final, text) in enumerate(analyses):
        parts += ["", f"### Analyst {chr(ord('A') + i)} ({how}); final forecast: {final}", _cut(text)]
    parts += ["", "=====", f"Pooled forecast of the panel: {pooled}", "=====", "",
              "Now do your job as supervisor, and end with the answer block the task asks for."]
    return "\n".join(parts)
