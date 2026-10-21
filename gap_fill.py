"""
Gap-fill: a second research pass that reads the research briefing the way a
forecaster would, then checks it against primary sources with Claude's web
tools, and returns dated, sourced findings only.

Adapted from the agentic gap-fill of the No-Stream bot (MIT,
github.com/No-Stream/metaculus-bot, docs/agentic_gap_fill.md). Its audit of
the worst misses found that most were not gaps in coverage but the panel
leaning on a briefing claim that was stale, misread or invented; so the first
job here is to verify the two or three claims a forecast would lean on
hardest, then to read the resolution source itself, then to fill what is
missing. Findings state facts, never a view on how the question resolves, so
the forecasters' judgment stays theirs.

It runs on the Anthropic API only (agent_forecaster.AgentForecaster with its
own instructions and no answer block). The bot records it as a shadow
experiment first (bot.ForecasterBot.run_deferred_shadows): the members answer
again on the briefing plus these findings, and those answers are scored
against the members' own once questions resolve.
"""

from __future__ import annotations

import logging
import os
import re

from agent_forecaster import AgentForecaster

logger = logging.getLogger(__name__)

HEADER = "## Source: Verified findings (second pass)"

SYSTEM = """\
You are a research analyst supporting a forecasting panel. You do not \
forecast. You have web_search, web_fetch, a Python code_execution sandbox \
(call web_search and web_fetch from inside your code when that helps you \
filter or compute) and allow_url. Today is {today} (UTC).

The user message holds the exact forecasting task the panel will receive: \
the question, its resolution criteria and fine print, and the research \
briefing that assistants prepared. The panel will read your findings next to \
that briefing.

Step 1, privately: work out how you would forecast this question from the \
briefing, and from that pick your targets:
- VERIFY: the two or three claims in the briefing that a forecast would lean \
on hardest (a current value, a date, a status, a count, a rule). A stale, \
misread or invented claim here sinks every forecaster, so check each against \
its primary source.
- RESOLUTION: if the criteria name a source (a page, dataset, index, table or \
official body), the latest value or status it shows, with its date, read from \
the source itself and not from news about it.
- FILL: facts the forecast needs that the briefing lacks, or has only \
secondhand or stale.
- TIMING: the exact date of each event that could decide the question, and \
whether it falls inside the question's window.
- BASE RATE: if the forecast would rest on a reference class that is niche \
or uncertain, the real count and denominator. Skip common-knowledge rates.

Step 2: pursue the targets with the tools, primary sources first (the \
resolution source, official sites, filings, dockets, data downloads). When \
the answer is in data, compute it in code. Make at least 3 web searches and \
open at least 2 pages, the resolution source first, even when the briefing \
looks complete: it can be stale, wrong, or missing the newest development.

Budget: at most {searches} web searches and {fetches} page fetches, and about \
{minutes} minutes in total. If a page cannot be read, try one alternative (an \
official page, a text or API version, an archived copy), then move on. \
web_fetch can only open URLs that already appear in the conversation. To open \
a URL you built yourself (an API query, a CSV export), pass it to allow_url \
first, then fetch it. Pages that need JavaScript come back empty; prefer the \
source's API or data download.

Write your whole answer in one final message after your last tool call, in \
exactly this form and nothing else:

Corrections to the briefing:
- <what the briefing says>; the source says <what you found> (<source>, <URL>, <date of the value>)
Confirmed:
- <claim> (<source>, <URL>, <date>)
New facts:
- <fact> (<source>, <URL>, <date>)
Could not check:
- <target> (<why>)

Write "None." under a heading with nothing to report. Every line states a \
fact with its source, URL and date. Never give a view on how the question \
will resolve: no probabilities, and none of "likely", "unlikely", \
"probably", "suggests", "points to", "we expect". A fact you could not \
source goes under "Could not check". At most 400 words."""

PROMPT = """\
Below is the forecasting task the panel will receive. Check its research \
briefing as your instructions say. Do not answer the task.

=====
{task}
====="""

# Verdict words the findings must not contain; counted for the log only.
_VERDICT = re.compile(r"(?i)\b(likely|unlikely|probably|suggests?|points? to|we expect)\b")


def make_agent() -> AgentForecaster:
    """
    The gap-fill agent, set by GAP_FILL_MODEL (default claude-sonnet-5-5),
    GAP_FILL_EFFORT (xhigh), GAP_FILL_TOOLS (searches plus fetches, 12) and
    GAP_FILL_TIME (seconds, 240). At medium effort, with "spot-check and move
    on" in place of the research floor, it made 0 to 2 searches on five
    production prompts of 2026-10-21; at xhigh with the floor, 4 to 6
    searches and 3 to 7 pages in 53 to 122 seconds (one of five passed 300).
    """
    return AgentForecaster(
        model=(os.getenv("GAP_FILL_MODEL") or "").strip() or "claude-sonnet-5-5",
        effort=(os.getenv("GAP_FILL_EFFORT") or "").strip() or "xhigh",
        max_tool_uses=int((os.getenv("GAP_FILL_TOOLS") or "").strip() or 12),
        time_limit_s=int((os.getenv("GAP_FILL_TIME") or "").strip() or 240),
        system=SYSTEM,
        role="gapfill",
        answer_block=False,
    )


async def findings(agent: AgentForecaster, task: str) -> str:
    """
    The findings section for a forecast prompt `task`, headed like a research
    source, or "" when the agent reports nothing. Raises on failure.
    """
    text = (await agent.invoke(PROMPT.format(task=task))).strip()
    # Anything before the first heading is the model thinking aloud.
    start = text.find("Corrections to the briefing")
    if start > 0:
        text = text[start:]
    verdicts = _VERDICT.findall(text)
    if verdicts:
        logger.info(f"Gap-fill findings use verdict words: {', '.join(sorted(set(w.lower() for w in verdicts)))}")
    return f"{HEADER}\n{text}" if text else ""


def with_findings(prompt: str, section: str) -> str:
    """
    The forecast prompt with the findings placed at the end of the research,
    right before the prompt's own "Today is" line (the last one: a research
    briefing can quote the phrase too).
    """
    if not section:
        return prompt
    at = prompt.rfind("\nToday is ")
    if at < 0:
        raise ValueError("no 'Today is' line in the forecast prompt")
    return prompt[:at] + "\n\n" + section + "\n" + prompt[at:]
