"""
Agentic forecaster: one Claude conversation that researches with Anthropic
server tools before it forecasts, for use as a shadow model (recorded, never
published).

The ensemble members get a fixed research briefing and no tools. Metaculus's
own agent bots (metac-agent 919 and metac-azimuth 853 points on MiniBench
round 1, against our 208) read the resolution page and primary documents and
compute on data before forecasting. This module gives Claude the same reach:

- web_search (web_search_20260318): $10 per 1,000 searches plus tokens.
- web_fetch (web_fetch_20260318): no charge beyond tokens. It can only open
  URLs already in the conversation (the prompt, or earlier search and fetch
  results), and it does not run JavaScript.
- code_execution (code_execution_20260120): a Python sandbox without internet.
  Both web tools are called from inside it ("dynamic filtering"), so a CSV
  can be downloaded and summed in code. Free when the web tools are present.
- allow_url: a local tool that only echoes a URL back. web_fetch refuses URLs
  the model wrote itself (an API query, a CSV export link); a URL in a client
  tool result is fetchable, so this lets the agent query data APIs. Nothing is
  downloaded locally.

Interface expected by bot.ForecasterBot._ask_models for a shadow: a `.model`
string and `async invoke(prompt) -> str`. The prompt is the bot's full
forecast prompt; the returned text ends with the answer block that prompt
asks for, so the bot's own parsers read it. Any failure (time limit, API
error, refusal, no answer block) raises, and the bot records a shadow failure.

Only ANTHROPIC_API_KEY is used, never the OpenRouter key.
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Anthropic list prices in USD per million tokens: input, 5-minute cache
# write, cache read, output. Read from platform.claude.com/docs pricing on
# 2026-10-08. The web tools with dynamic filtering need Claude 4.6 or later,
# so Haiku 4.5 is not listed.
PRICES: dict[str, tuple[float, float, float, float]] = {
    "claude-sonnet-5": (2.0, 2.5, 0.20, 10.0),
    "claude-sonnet-5-5": (2.0, 2.5, 0.10, 10.0),
    "claude-opus-5-5": (4.0, 5.0, 0.20, 20.0),
    "claude-opus-5": (5.0, 6.25, 0.50, 25.0),
    "claude-fable-5-1": (10.0, 12.5, 0.25, 50.0),
}
WEB_SEARCH_PRICE = 0.01  # USD per search
EFFORTS = ("low", "medium", "high", "xhigh", "max")

# Thinking plus code plus the written answer, per response. Streaming is used,
# so a large cap does not risk HTTP timeouts.
MAX_TOKENS = 32000
# A long server-tool turn can pause (stop_reason "pause_turn") and is resumed
# by re-sending it. It never paused in the tests of 2026-10-08, even with 14
# code runs in one turn.
MAX_CONTINUATIONS = 4

# Whether search and fetch results consumed inside code come back in the
# response ("full") or are dropped ("excluded"). "full" keeps the fetched URLs
# in the call log; "excluded" saved no tokens in an A/B test on 2026-10-08.
RESPONSE_INCLUSION = "full"

# Rounds of the local allow_url tool per call, and URLs it may allow.
MAX_CLIENT_ROUNDS = 6
MAX_ALLOWED_URLS = 8

# One JSON object per call (tokens, tools, URLs, cost, answer), for review.
CALL_LOG = os.path.join("logs", "agent_calls.jsonl")

_ALLOW_URL_TOOL = {
    "name": "allow_url",
    "description": (
        "web_fetch only opens URLs already present in the conversation. Call this with a URL "
        "you constructed yourself (for example an API query, a CSV or JSON download link, or a "
        "data page with query parameters) to make it fetchable. Wait for its result, then call "
        "web_fetch on exactly that URL; a fetch issued in the same step can be refused. This tool "
        "does not fetch anything itself."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"url": {"type": "string", "description": "The full http(s) URL, at most 250 characters."}},
        "required": ["url"],
        "additionalProperties": False,
    },
    "strict": True,
}

_SYSTEM = """\
You are an expert forecaster working alone with web_search, web_fetch, a \
Python code_execution sandbox (call web_search and web_fetch from inside your \
code when that helps you filter or compute) and allow_url. Today is {today} (UTC).

The user message is a forecasting task written for a forecaster without tools. \
It contains the question, the resolution criteria and fine print, and a \
research briefing prepared earlier by assistants. Treat the briefing as a \
starting point and a list of leads, not as settled fact: it can be stale, \
wrong about the current value, or missing the newest development.

Before you forecast, use the tools to check what matters most, in this order:
1. The resolution source. Open the exact page, dataset, table or document the \
resolution criteria or fine print name: fetch the URL if one is given, \
otherwise search for it and then fetch it. Read the latest value or status \
that decides the question, with its date. If the question has a threshold, \
compute how far the current value is from it.
2. Anything newer than the question text and the briefing. Search for the \
last few days of news on the specific facts the question depends on \
(schedules, counts, values, "as of" dates), including postponements, \
cancellations, withdrawals and results already announced, in the local \
language too when relevant.
3. Primary documents for scheduled events: the court docket or calendar, the \
official agenda, the filing (for example on SEC EDGAR), the publisher's \
release calendar. Check that each document's date and year are the ones this \
question is about.
4. When the answer depends on data, compute it in code instead of estimating: \
download the CSV, JSON or table with web_fetch and sum, count or extrapolate \
(a 7-day sum, a year-to-date count, the share of days above a level, a base \
rate from past releases).

Budget: at most {searches} web searches and {fetches} page fetches, and about \
{minutes} minutes in total. Spend the first calls on the resolution source. \
Do not redo searches the briefing already answers unless a number that drives \
the forecast needs checking. If a page cannot be read, try one alternative \
(another official page, a text or API version, an archived copy), then move on \
and say what you could not read. web_fetch can only open URLs that already \
appear in the conversation. To open a URL you built yourself (an API query, a \
CSV export, a filtered data view), pass it to allow_url first, then fetch it. \
Pages that need JavaScript come back empty; prefer the source's API or data \
download.

When you have finished researching, write your whole answer in one final \
message, after your last tool call:
- First line: "Resolution source: <name and URL> (opened: yes)" or \
"(opened: no, <reason>)", then the decisive value you read there with its date.
- Then the facts you found that are newer than the question or the briefing, \
with dates and sources, or "No newer facts found."
- Then answer the task exactly as the user message asks: the items it asks you \
to write before answering, and finally the answer block in exactly the format \
it specifies.
The answer block is the very last thing you write. Write nothing after it."""

_REPAIR = (
    "Your reply did not end with the answer block in the required format. "
    "Without using any tool, write only the final answer block now, exactly in "
    "the format the task specified:\n\n{fmt}"
)

_FORMATS = {
    "binary": 'Probability: ZZ% (one number from 0 to 100)',
    "numeric": (
        "Percentile 10: XX\nPercentile 20: XX\nPercentile 40: XX\n"
        "Percentile 60: XX\nPercentile 80: XX\nPercentile 90: XX"
    ),
    "multiple_choice": "<option name>: <probability> for every option, one per line, in the order given",
}

_BINARY_RE = re.compile(r"Probability\s*:\s*\**\s*\d+(?:\.\d+)?\s*%", re.IGNORECASE)
_PCT_RE = re.compile(r"Percentile\s*(10|20|40|60|80|90)\s*:\s*\**\s*[-+$]?\s*[\d.,]+", re.IGNORECASE)
_MC_LINE_RE = re.compile(r"^\s*[-*]*\s*[^:\n]{1,200}:\s*\**\s*\d+(?:\.\d+)?\s*%?\s*\**\s*$")
_OPTIONS_RE = re.compile(r"The options are:\s*(\[.*?\])\s*$", re.MULTILINE)
_QUESTION_RE = re.compile(r"Your interview question is:\s*\n\s*(.+)")


def _kind(prompt: str) -> str | None:
    """Which answer block the bot's prompt asks for."""
    if "Probability: ZZ%" in prompt:
        return "binary"
    if "Percentile 10:" in prompt:
        return "numeric"
    if "Option_A: Probability_A" in prompt:
        return "multiple_choice"
    return None


def _options(prompt: str) -> list[str]:
    m = _OPTIONS_RE.search(prompt)
    if not m:
        return []
    try:
        value = ast.literal_eval(m.group(1))
    except (ValueError, SyntaxError):
        return []
    return [str(v) for v in value] if isinstance(value, list) else []


def _line_end(text: str, pos: int) -> int:
    nl = text.find("\n", pos)
    return len(text) if nl < 0 else nl


def finish_answer(text: str, prompt: str) -> str | None:
    """
    The text cut right after its answer block, or None when the block the
    prompt asks for is missing. A trailing remark after the block is dropped,
    so the text ends with exactly that block.
    """
    kind = _kind(prompt)
    text = text.rstrip()
    if kind == "binary":
        matches = list(_BINARY_RE.finditer(text))
        return text[: _line_end(text, matches[-1].end())].rstrip() if matches else None
    if kind == "numeric":
        matches = list(_PCT_RE.finditer(text))
        last90 = [m for m in matches if m.group(1) == "90"]
        if not last90:
            return None
        end = _line_end(text, last90[-1].end())
        block = text[max(0, end - 1500): end]
        found = {m.group(1) for m in _PCT_RE.finditer(block)}
        return text[:end].rstrip() if found >= {"10", "20", "40", "60", "80", "90"} else None
    if kind == "multiple_choice":
        # The last run of "<name>: <number>" lines must cover every option.
        lines = text.splitlines()
        ends = [i for i, line in enumerate(lines) if _MC_LINE_RE.match(line)]
        if not ends:
            return None
        end = start = ends[-1]
        while start > 0 and _MC_LINE_RE.match(lines[start - 1]):
            start -= 1
        if end - start + 1 < max(2, len(_options(prompt))):
            return None
        return "\n".join(lines[: end + 1]).rstrip()
    return text or None


def _to_param(block: Any) -> dict:
    """A response content block as a request content block, unchanged apart from empty fields."""
    data = block.model_dump(mode="json", exclude_none=True) if hasattr(block, "model_dump") else dict(block)
    data.pop("parsed_output", None)
    return data


def _final_text(blocks: list[dict]) -> str:
    """Text written after the last tool call; all the text if that part is empty."""
    last_tool = -1
    for i, b in enumerate(blocks):
        t = b.get("type", "")
        if t in ("server_tool_use", "tool_use") or t.endswith("_tool_result"):
            last_tool = i
    tail = "".join(b.get("text", "") for b in blocks[last_tool + 1:] if b.get("type") == "text").strip()
    if tail:
        return tail
    return "\n\n".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()


class AgentForecaster:
    """
    A shadow forecaster that researches with web search, web fetch and code
    execution before answering the bot's forecast prompt.

    max_tool_uses caps web searches plus web fetches (half each, searches
    rounded down), through each tool's max_uses. The API counts max_uses over
    the whole assistant turn, including the requests that follow an allow_url
    round (measured 2026-10-08: a second search after a client tool round got
    max_uses_exceeded), so the same values go on every request and the cap is
    exact. Code runs are not capped (they cost nothing extra and the time
    limit bounds them); allow_url is capped at MAX_ALLOWED_URLS. time_limit_s
    is a hard limit on the whole call, after which invoke raises TimeoutError.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-5",
        effort: str = "high",
        max_tool_uses: int = 12,
        time_limit_s: int = 300,
    ) -> None:
        model = model.removeprefix("anthropic/")
        if effort not in EFFORTS:
            raise ValueError(f"effort must be one of {EFFORTS}, not {effort!r}")
        if max_tool_uses < 2:
            raise ValueError("max_tool_uses must be at least 2 (one search and one fetch)")
        self.api_model = model
        self.effort = effort
        self.max_tool_uses = int(max_tool_uses)
        self.time_limit_s = int(time_limit_s)
        self.search_cap = self.max_tool_uses // 2
        self.fetch_cap = self.max_tool_uses - self.search_cap
        self.model = f"agent/{model}"
        # Stats of every call made through this instance, newest last.
        self.calls: list[dict] = []
        if model not in PRICES:
            logger.warning(f"No list price for {model}; cost will not be estimated")
        if not os.getenv("ANTHROPIC_API_KEY", "").strip():
            logger.warning(f"{self.model}: ANTHROPIC_API_KEY is not set; every call will fail")

    @classmethod
    def from_spec(cls, spec: str) -> "AgentForecaster":
        """
        Builds an agent from a SHADOW_MODELS entry "agent/<model>[@effort]",
        for example "agent/claude-sonnet-5@high". AGENT_MAX_TOOL_USES and
        AGENT_TIME_LIMIT override the defaults (12 tools, 300 seconds).
        """
        body = spec.strip().removeprefix("agent/")
        model, _, effort = body.partition("@")
        uses = int((os.getenv("AGENT_MAX_TOOL_USES") or "").strip() or 12)
        limit = int((os.getenv("AGENT_TIME_LIMIT") or "").strip() or 300)
        return cls(model=model.strip() or "claude-sonnet-5", effort=effort.strip() or "high",
                   max_tool_uses=uses, time_limit_s=limit)

    # ------------------------------------------------------------------

    def _tools(self) -> list[dict]:
        # Identical on every request of a call: the caps hold over the whole
        # turn, and an unchanged tools list keeps the prompt cache valid.
        return [
            {"type": "web_search_20260318", "name": "web_search", "max_uses": self.search_cap,
             "response_inclusion": RESPONSE_INCLUSION},
            # use_cache False: the docs warn that cached content can lag the
            # live page, and resolution sources change daily. In the test of
            # 2026-10-08 an ECDC bulletin came back a week behind the page the
            # briefing had read (the cause was not isolated).
            {"type": "web_fetch_20260318", "name": "web_fetch", "max_uses": self.fetch_cap, "use_cache": False,
             "response_inclusion": RESPONSE_INCLUSION},
            {"type": "code_execution_20260120", "name": "code_execution"},
            _ALLOW_URL_TOOL,
        ]

    @staticmethod
    def _allow_url(block: dict, s: dict) -> dict:
        """Result of one allow_url call: the URL echoed back, which makes it fetchable."""
        url = str((block.get("input") or {}).get("url", "")).strip()
        result: dict[str, Any] = {"type": "tool_result", "tool_use_id": block.get("id")}
        if block.get("name") != "allow_url":
            result.update(content=f"Unknown tool {block.get('name')!r}.", is_error=True)
        elif len(s["allowed_urls"]) >= MAX_ALLOWED_URLS:
            result.update(content=f"Limit of {MAX_ALLOWED_URLS} URLs reached.", is_error=True)
        elif not re.match(r"https?://[^\s]+$", url) or len(url) > 250:
            result.update(content="Give one http(s) URL of at most 250 characters, without spaces.", is_error=True)
        else:
            s["allowed_urls"].append(url)
            result["content"] = f"{url}\nThis URL can now be opened with web_fetch."
        return result

    def _cost(self, s: dict) -> float | None:
        price = PRICES.get(self.api_model)
        if price is None:
            return None
        p_in, p_write, p_read, p_out = price
        tokens = (
            s["input_tokens"] * p_in
            + s["cache_write_tokens"] * p_write
            + s["cache_read_tokens"] * p_read
            + s["output_tokens"] * p_out
        ) / 1e6
        return round(tokens + s["web_search_requests"] * WEB_SEARCH_PRICE, 4)

    @staticmethod
    def _add_usage(s: dict, usage: Any) -> None:
        if usage is None:
            return
        s["input_tokens"] += usage.input_tokens or 0
        s["output_tokens"] += usage.output_tokens or 0
        s["cache_write_tokens"] += getattr(usage, "cache_creation_input_tokens", 0) or 0
        s["cache_read_tokens"] += getattr(usage, "cache_read_input_tokens", 0) or 0
        stu = getattr(usage, "server_tool_use", None)
        if stu is not None:
            s["web_search_requests"] += getattr(stu, "web_search_requests", 0) or 0
            s["web_fetch_requests"] += getattr(stu, "web_fetch_requests", 0) or 0

    @staticmethod
    def _read_blocks(s: dict, blocks: list[dict]) -> None:
        """Search queries, fetched URLs and code runs, from the response blocks."""
        for b in blocks:
            t = b.get("type", "")
            if t == "server_tool_use":
                name, inp = b.get("name"), b.get("input") or {}
                if name == "web_search":
                    s["queries"].append(str(inp.get("query", ""))[:200])
                elif name == "web_fetch":
                    s["fetches"].append({"id": b.get("id"), "url": str(inp.get("url", ""))[:300], "ok": None})
                elif name in ("code_execution", "bash_code_execution", "text_editor_code_execution"):
                    s["code_runs"] += 1
            elif t == "web_fetch_tool_result":
                c = b.get("content") or {}
                for f in s["fetches"]:
                    if f.get("id") == b.get("tool_use_id"):
                        f["ok"] = c.get("type") == "web_fetch_result"
                        if c.get("error_code"):
                            f["error"] = c["error_code"]
                        break
            elif t == "web_search_tool_result":
                c = b.get("content")
                if isinstance(c, dict) and c.get("error_code"):
                    s["search_errors"].append(c["error_code"])

    def _log(self, s: dict) -> None:
        self.calls.append(s)
        cost = s.get("cost_usd")
        logger.info(
            f"{self.model} ({self.effort}): {s['outcome']} in {s['seconds']:.0f}s, "
            f"{s['requests']} request(s), tokens in {s['input_tokens']:,} "
            f"(+{s['cache_write_tokens']:,} cache write, {s['cache_read_tokens']:,} cache read) "
            f"out {s['output_tokens']:,}; searches {s['web_search_requests']}, "
            f"fetches {s['web_fetch_requests']} ({sum(1 for f in s['fetches'] if f.get('ok'))} read), "
            f"code runs {s['code_runs']}, URLs allowed {len(s['allowed_urls'])}; est. cost "
            + (f"${cost:.3f}" if cost is not None else "unknown")
            + (" (partial: lower bound)" if s.get("partial") else "")
        )
        path = os.getenv("AGENT_CALL_LOG") or CALL_LOG
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning(f"Could not write {path}: {exc}")

    # ------------------------------------------------------------------

    async def invoke(self, prompt: str) -> str:
        s: dict[str, Any] = {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": self.model,
            "effort": self.effort,
            "question": (m.group(1).strip()[:300] if (m := _QUESTION_RE.search(prompt)) else ""),
            "kind": _kind(prompt),
            "max_tool_uses": self.max_tool_uses,
            "time_limit_s": self.time_limit_s,
            "requests": 0,
            "stop_reasons": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_write_tokens": 0,
            "cache_read_tokens": 0,
            "web_search_requests": 0,
            "web_fetch_requests": 0,
            "code_runs": 0,
            "queries": [],
            "fetches": [],
            "search_errors": [],
            "allowed_urls": [],
            "repaired": False,
        }
        live: dict[str, Any] = {}
        start = time.monotonic()
        try:
            text = await asyncio.wait_for(self._run(prompt, s, live), timeout=self.time_limit_s)
        except BaseException as exc:
            # Tokens of the request in flight are billed but only partly
            # known: add what its stream had reported.
            stream = live.get("stream")
            if stream is not None:
                try:
                    self._add_usage(s, stream.current_message_snapshot.usage)
                    s["partial"] = True
                except Exception:
                    s["partial"] = True
            timed_out = isinstance(exc, (asyncio.TimeoutError, TimeoutError))
            s["outcome"] = "timeout" if timed_out else f"error: {type(exc).__name__}: {str(exc)[:300]}"
            s["seconds"] = round(time.monotonic() - start, 1)
            s["cost_usd"] = self._cost(s)
            self._log(s)
            if timed_out:
                raise TimeoutError(f"{self.model} passed its {self.time_limit_s}s limit") from None
            raise
        s["outcome"] = "ok"
        s["seconds"] = round(time.monotonic() - start, 1)
        s["cost_usd"] = self._cost(s)
        s["answer_chars"] = len(text)
        s["answer_tail"] = text[-600:]
        self._log(s)
        return text

    async def _request(self, client: Any, messages: list[dict], tools: list[dict],
                       s: dict, live: dict, no_tools: bool = False) -> tuple[Any, list[dict]]:
        kwargs: dict[str, Any] = dict(
            model=self.api_model,
            max_tokens=MAX_TOKENS,
            system=_SYSTEM.format(
                today=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                searches=self.search_cap,
                fetches=self.fetch_cap,
                minutes=max(1, round(self.time_limit_s / 60)),
            ),
            messages=messages,
            tools=tools,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            # Automatic prompt caching: a resumed or repaired turn re-reads
            # the prompt and the tool results from the cache.
            cache_control={"type": "ephemeral"},
        )
        if no_tools:
            kwargs["tool_choice"] = {"type": "none"}
        async with client.messages.stream(**kwargs) as stream:
            live["stream"] = stream
            msg = await stream.get_final_message()
        live.pop("stream", None)
        s["requests"] += 1
        s["stop_reasons"].append(msg.stop_reason)
        self._add_usage(s, msg.usage)
        blocks = [_to_param(b) for b in msg.content]
        self._read_blocks(s, blocks)
        if msg.stop_reason == "refusal":
            details = getattr(msg, "stop_details", None)
            raise RuntimeError(f"refusal: {getattr(details, 'category', None)}")
        return msg, blocks

    async def _run(self, prompt: str, s: dict, live: dict) -> str:
        import anthropic

        if not os.getenv("ANTHROPIC_API_KEY", "").strip():
            raise RuntimeError("ANTHROPIC_API_KEY is not set")

        messages: list[dict] = [{"role": "user", "content": prompt}]
        written: list[dict] = []  # every assistant block, in order
        pauses = rounds = 0
        tools = self._tools()
        async with anthropic.AsyncAnthropic(max_retries=1, timeout=float(self.time_limit_s)) as client:
            while True:
                msg, blocks = await self._request(client, messages, tools, s, live)
                written.extend(blocks)
                if messages[-1]["role"] == "assistant":
                    # A resumed turn continues the paused assistant message.
                    messages[-1]["content"].extend(blocks)
                else:
                    messages.append({"role": "assistant", "content": list(blocks)})
                if msg.stop_reason == "pause_turn":
                    pauses += 1
                    if pauses > MAX_CONTINUATIONS:
                        raise RuntimeError(f"still paused after {MAX_CONTINUATIONS} resumes")
                    continue
                calls = [b for b in blocks if b.get("type") == "tool_use"]
                if msg.stop_reason == "tool_use" and calls:
                    rounds += 1
                    if rounds > MAX_CLIENT_ROUNDS:
                        raise RuntimeError(f"more than {MAX_CLIENT_ROUNDS} rounds of allow_url")
                    # Only tool_result blocks: the API then runs any server
                    # tool call it deferred and Claude continues the turn.
                    messages.append({"role": "user", "content": [self._allow_url(b, s) for b in calls]})
                    continue
                break

            text = finish_answer(_final_text(messages[-1]["content"]), prompt)
            if text is None:
                # One short turn, tools off, to restate the answer block.
                s["repaired"] = True
                fmt = _FORMATS.get(_kind(prompt) or "", "the answer format the task specified")
                repair = messages + [{"role": "user", "content": _REPAIR.format(fmt=fmt)}]
                _, extra = await self._request(client, repair, tools, s, live, no_tools=True)
                block = "".join(b.get("text", "") for b in extra if b.get("type") == "text").strip()
                text = finish_answer(_final_text(written) + "\n\n" + block, prompt)
        if text is None:
            raise RuntimeError(f"no {_kind(prompt) or 'final'} answer block in the reply (stops: {s['stop_reasons']})")
        return text
