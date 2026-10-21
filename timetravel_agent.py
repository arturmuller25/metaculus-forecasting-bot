"""
Agentic forecaster that researches as of a past date, to replay the agent on
resolved questions without the answer leaking in.

Live web search cannot be replayed: on a question that has resolved, it finds
the outcome. This agent is agent_forecaster.AgentForecaster with the Anthropic
web tools replaced by local tools that only see the world before the as-of
time (the moment the bot forecast the question):

- news_search: AskNews historical search (the tournament's sponsored plan),
  limited to articles AskNews crawled before the as-of time, so the summary
  it returns was written from the article as it stood then. Each result
  carries that summary, so an article never archived can still be read.
  Off unless TIMETRAVEL_ASKNEWS=1: on 2026-10-21 one 57-question round
  (about 300 historical searches) used up the plan's monthly credits and
  its $5 wallet, which left production without AskNews until the renewal.
- headline_search: Google News search (RSS, free, no key) restricted with
  its before: operator to articles published up to the day before the as-of
  date, and filtered again on each article's date. The as-of day itself is
  left out because Google dates articles by day and "before:" let same-day
  articles through in a check on 2026-10-21. A national edition can be
  chosen for local coverage.
- fetch_page: the Internet Archive's latest capture of the URL made before
  the as-of time (CDX index), as text. Never the live page, so a resolution
  source shows what it showed that day, and a page never archived by then
  cannot be read. Google News links are first turned into the publisher's
  URL (two requests to news.google.com, the method its own pages use).
- price_history: daily closes from Yahoo Finance or FRED, up to the day
  before the as-of date (market_baseline.py's readers).
- code_execution (Anthropic server tool, no internet), to compute on data.

GDELT's article search was tried first and dropped: it answered "limit
requests to one every 5 seconds" for minutes on end even with the spacing.

Claude Sonnet 5.5, Opus 5.5 and Fable 5.1 have a reliable knowledge cutoff of
June 2026 (platform.claude.com models overview, read 2026-10-21), so
questions forecast from late September 2026 on are clean for memory as well.
Residual leakage: a misdated article in the search, or an archived capture
dated before the as-of time that the Internet Archive stamped wrongly.

Interface as AgentForecaster: `.model` and `async invoke(prompt) -> str`.
`as_of` is a callable returning the as-of datetime for the call in progress
(forecast_replay.py sets it per question).
"""

from __future__ import annotations

import asyncio
import contextvars
import html
import json
import os
import re
import urllib.parse
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable

import requests

import market_baseline
from agent_forecaster import AgentForecaster
from resolution_fetch import _focus, _summarize_csv, _summarize_json, _Text, keywords

SYSTEM = """\
You are an expert forecaster working alone. Today is {today} (UTC). You have \
five tools: news_search (news articles up to today, each with a summary of \
its content), headline_search (Google News headlines before today, with a \
national edition for local coverage), fetch_page (the copy of a web page \
archived by the Internet Archive up to today; a page that was not archived \
cannot be read), price_history (daily closes of a market or economic series \
up to yesterday) and code_execution (a Python sandbox without internet; paste \
data into your code to compute on it).

The user message is a forecasting task written for a forecaster without tools. \
It contains the question, the resolution criteria and fine print, and a \
research briefing prepared earlier by assistants. Treat the briefing as a \
starting point and a list of leads, not as settled fact: it can be stale, \
wrong about the current value, or missing the newest development.

Before you forecast, use the tools to check what matters most, in this order:
1. The resolution source. Open the exact page, dataset or document the \
resolution criteria or fine print name with fetch_page. Read the latest value \
or status that decides the question, with its date. If the question has a \
threshold, compute how far the current value is from it.
2. Anything newer than the question text and the briefing: search the news \
for the specific facts the question depends on (schedules, counts, values, \
"as of" dates), including postponements, cancellations and results already \
announced, and open the most relevant articles with fetch_page. Use the \
national edition and the local language when the coverage is local.
3. Primary documents for scheduled events: the official agenda, calendar or \
filing. Check that each document's date and year are the ones this question \
is about.
4. When the answer depends on market data, get the series with price_history \
and compute in code instead of estimating (recent volatility, the move needed \
to cross a threshold, a range over the window).

Research before you forecast even when the briefing looks complete. Make at \
least 3 searches and open at least 2 pages, the resolution source first. \
Budget: at most {searches} searches (news and headlines together) and \
{fetches} page fetches, and about \
{minutes} minutes in total. Search queries work best as a few specific words \
or a quoted phrase.

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

# Google News editions: ceid -> (hl, gl).
EDITIONS = {
    "US:en": ("en-US", "US"), "GB:en": ("en-GB", "GB"), "IN:en": ("en-IN", "IN"),
    "BR:pt-419": ("pt-BR", "BR"), "MX:es-419": ("es-419", "MX"), "AR:es-419": ("es-419", "AR"),
    "ES:es": ("es", "ES"), "FR:fr": ("fr", "FR"), "DE:de": ("de", "DE"), "IT:it": ("it", "IT"),
    "TR:tr": ("tr", "TR"), "JP:ja": ("ja", "JP"), "KR:ko": ("ko", "KR"), "RU:ru": ("ru", "RU"),
}

_ASKNEWS_TOOL = {
    "name": "news_search",
    "description": (
        "Searches news articles up to today in many languages (AskNews archive). Returns up to 8 "
        "articles with date, outlet, title, link and an English summary of the content. Use a few "
        "specific words."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Search words, at most 160 characters."}},
        "required": ["query"],
        "additionalProperties": False,
    },
}
_NEWS_TOOL = {
    "name": "headline_search",
    "description": (
        "Searches news headlines published before today (Google News). Returns up to 10 articles with "
        "date, outlet, title and link; open one with fetch_page. Use a few specific words or a quoted "
        "phrase. `edition` picks a national edition for local coverage (default US:en)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search words, at most 120 characters."},
            "edition": {"type": "string", "enum": sorted(EDITIONS)},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}
_FETCH_TOOL = {
    "name": "fetch_page",
    "description": (
        "Returns the text of a web page as archived by the Internet Archive up to today (the latest "
        "capture before now), with the capture date. Accepts the links news_search returns. Data files "
        "(CSV, JSON) come back summarized. Pages never archived cannot be read; try the site's main "
        "page or another source."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"url": {"type": "string", "description": "The full http(s) URL."}},
        "required": ["url"],
        "additionalProperties": False,
    },
}
_PRICE_TOOL = {
    "name": "price_history",
    "description": (
        "Daily closes of a series up to yesterday: a Yahoo Finance symbol (e.g. AAPL, ^VIX, CL=F, "
        "BTC-USD, EURUSD=X, XU100.IS) or a FRED series id (e.g. DGS10, BAMLH0A0HYM2). Returns the last "
        "`days` closes (default 60, at most 400) and the recent daily volatility."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "symbol": {"type": "string"},
            "source": {"type": "string", "enum": ["yahoo", "fred"]},
            "days": {"type": "integer"},
        },
        "required": ["symbol", "source"],
        "additionalProperties": False,
    },
}

# Stats of the last call made in the current context (one replay job), read
# by forecast_replay.py; an instance serves several jobs at once.
LAST_CALL: contextvars.ContextVar[dict | None] = contextvars.ContextVar("timetravel_last_call", default=None)

_BROWSER = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
# The Internet Archive answers a browser-like user agent from a script with a
# "suspected abusive bot traffic" page, and an identified client normally
# (checked 2026-10-21).
_ARCHIVE = {"User-Agent": "metaculus-forecasting-bot/1.0 (research replay; github.com/arturmuller25/metaculus-forecasting-bot)"}
_ARCHIVE_BLOCK = re.compile(r"suspected abusive bot traffic|Your request is being blocked", re.IGNORECASE)
MAX_PAGE_TEXT = 7000


def as_utc(moment: datetime) -> datetime:
    """A naive datetime is taken as UTC (the replay's forecast times)."""
    return moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment.astimezone(timezone.utc)


def _ts(moment: datetime) -> str:
    return as_utc(moment).strftime("%Y%m%d%H%M%S")


def news_search(query: str, as_of: datetime, edition: str = "US:en", limit: int = 10) -> str:
    """Google News articles published before the as-of date, as text lines (relevance order)."""
    as_of = as_utc(as_of)
    day = as_of.date()
    hl, gl = EDITIONS.get(edition, EDITIONS["US:en"])
    ceid = edition if edition in EDITIONS else "US:en"
    q = f"{query} after:{(day - timedelta(days=180)).isoformat()} before:{(day - timedelta(days=1)).isoformat()}"
    r = requests.get("https://news.google.com/rss/search", params={"q": q, "hl": hl, "gl": gl, "ceid": ceid},
                     headers=_BROWSER, timeout=30)
    if r.status_code >= 400:
        return f"Search error: HTTP {r.status_code}"
    midnight = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    rows = []
    for item in re.findall(r"<item>(.*?)</item>", r.text, re.S):
        title = re.search(r"<title>(.*?)</title>", item, re.S)
        link = re.search(r"<link>(.*?)</link>", item, re.S)
        date = re.search(r"<pubDate>(.*?)</pubDate>", item, re.S)
        source = re.search(r"<source url=\"([^\"]*)\">(.*?)</source>", item, re.S)
        if not (title and link and date):
            continue
        try:
            published = parsedate_to_datetime(date.group(1)).astimezone(timezone.utc)
        except (TypeError, ValueError):
            continue
        if published >= midnight:
            continue
        outlet = html.unescape(source.group(2)) if source else ""
        rows.append(f"- {published:%Y-%m-%d} | {outlet} | {html.unescape(title.group(1))[:160]} | {link.group(1).strip()}")
        if len(rows) >= limit:
            break
    return "\n".join(rows) if rows else "No articles found before today for this query."


_ASKNEWS: list = []  # the SDK client, built on first use


def asknews_search(query: str, as_of: datetime, limit: int = 8) -> str:
    """AskNews articles crawled (and published) before as_of, with their summaries."""
    import os

    from asknews_sdk import AskNewsSDK

    if not _ASKNEWS:
        _ASKNEWS.append(AskNewsSDK(
            client_id=os.getenv("ASKNEWS_CLIENT_ID"), client_secret=os.getenv("ASKNEWS_SECRET"),
            api_key=os.getenv("ASKNEWS_API_KEY"), scopes={"news"},
        ))
    as_of = as_utc(as_of)
    end = int(as_of.timestamp())
    response = _ASKNEWS[0].news.search_news(
        query=query, n_articles=limit, return_type="dicts", historical=True, method="kw",
        time_filter="crawl_date", start_timestamp=end - 120 * 86400, end_timestamp=end,
    )
    rows = []
    for item in response.as_dicts or []:
        d = item.model_dump() if hasattr(item, "model_dump") else dict(item)
        stamps = [as_utc(x) for x in (d.get("pub_date"), d.get("crawl_date")) if isinstance(x, datetime)]
        if any(x >= as_of for x in stamps):
            continue
        when = min(stamps).strftime("%Y-%m-%d") if stamps else "?"
        title = d.get("eng_title") or d.get("title") or ""
        rows.append(f"- {when} | {d.get('source_id', '')} | {title[:160]} | {d.get('article_url', '')}\n"
                    f"  {(d.get('summary') or '')[:600]}")
    return "\n".join(rows) if rows else "No articles found up to today for this query."


def google_news_target(url: str) -> str:
    """The publisher's URL behind a news.google.com article link (the decoding its own pages do)."""
    art_id = urllib.parse.urlparse(url).path.rstrip("/").split("/")[-1]
    page = requests.get(f"https://news.google.com/rss/articles/{art_id}", headers=_BROWSER, timeout=30).text
    sig, ts = re.search(r'data-n-a-sg="([^"]+)"', page), re.search(r'data-n-a-ts="([^"]+)"', page)
    if not (sig and ts):
        raise RuntimeError("could not resolve the Google News link")
    inner = ["garturlreq", [["X", "X", ["X", "X"], None, None, 1, 1, "US:en", None, 1, None, None, None, None,
                              None, 0, 1], "X", "X", 1, [1, 1, 1], 1, 1, None, 0, 0, None, 0],
             art_id, int(ts.group(1)), sig.group(1)]
    r = requests.post(
        "https://news.google.com/_/DotsSplashUi/data/batchexecute",
        data={"f.req": json.dumps([[["Fbv4je", json.dumps(inner), None, "generic"]]])},
        headers={**_BROWSER, "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"}, timeout=30,
    )
    m = re.search(r'\[\\"garturlres\\",\\"(.*?)\\"', r.text)
    if not m:
        raise RuntimeError("could not resolve the Google News link")
    return m.group(1).encode().decode("unicode_escape")


def wayback_fetch(url: str, as_of: datetime, words: list[str]) -> str:
    """Text of the latest Internet Archive capture of url made before as_of."""
    if "news.google.com/" in url:
        url = google_news_target(url)
    r = requests.get(
        "https://web.archive.org/cdx/search/cdx",
        params={"url": url, "to": _ts(as_of), "filter": "statuscode:200", "limit": "-1",
                "output": "json", "fl": "timestamp,original,mimetype"},
        headers=_ARCHIVE, timeout=40,
    )
    rows = r.json() if r.text.strip().startswith("[") else []
    if len(rows) < 2 or rows[-1][0] >= _ts(as_of):
        return f"No archived copy of {url} from before {as_utc(as_of):%Y-%m-%d}."
    stamp, original, mime = rows[-1]
    page = requests.get(f"https://web.archive.org/web/{stamp}id_/{original}", headers=_ARCHIVE, timeout=60)
    body = page.content[:25_000_000].decode(page.encoding or "utf-8", errors="replace")
    if page.status_code >= 400 or _ARCHIVE_BLOCK.search(body[:2000]):
        raise RuntimeError(f"the Internet Archive refused the capture (HTTP {page.status_code})")
    path = original.lower().split("?")[0]
    if "csv" in mime or path.endswith((".csv", ".tsv")):
        text = _summarize_csv(body, words, False)
    elif "json" in mime or path.endswith(".json"):
        try:
            text = _summarize_json(body)
        except ValueError:
            text = body[:MAX_PAGE_TEXT]
    else:
        parser = _Text()
        parser.feed(body)
        text = _focus(parser.text(), words)
    when = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]} {stamp[8:10]}:{stamp[10:12]} UTC"
    return f"Archived copy of {original} captured {when}:\n{text[:MAX_PAGE_TEXT]}"


def price_history(symbol: str, source: str, days: int, as_of: datetime) -> str:
    """Daily closes before the as-of date, with the recent daily volatility."""
    bars = market_baseline.yahoo_bars(symbol, 3) if source == "yahoo" else market_baseline.fred_series(symbol)
    bars = [b for b in bars if b.day < as_utc(as_of).date()]
    if not bars:
        return f"No data for {symbol} ({source}) before {as_utc(as_of):%Y-%m-%d}."
    tail = bars[-max(5, min(400, int(days or 60))):]
    closes = [b.close for b in bars[-61:]]
    if source == "yahoo":
        moves = [b / a - 1 for a, b in zip(closes, closes[1:]) if a > 0]
        unit = "relative daily move (RMS, last 60 sessions)"
    else:
        moves = [b - a for a, b in zip(closes, closes[1:])]
        unit = "absolute daily change (RMS, last 60 observations)"
    vol = (sum(m * m for m in moves) / len(moves)) ** 0.5 if moves else float("nan")
    lines = [f"{symbol} ({source}), {len(tail)} closes up to {tail[-1].day.isoformat()}; {unit}: {vol:.4g}",
             "date,close,high"]
    lines += [f"{b.day.isoformat()},{b.close:.6g},{b.high:.6g}" for b in tail]
    return "\n".join(lines)


_ASKNEWS_LOCK: list = []


def _asknews_lock() -> asyncio.Lock:
    """One AskNews request at a time across the replay's concurrent jobs (plan rate limit)."""
    if not _ASKNEWS_LOCK:
        _ASKNEWS_LOCK.append(asyncio.Lock())
    return _ASKNEWS_LOCK[0]


class TimeTravelAgent(AgentForecaster):
    """AgentForecaster that researches only with tools limited to before `as_of()`."""

    def __init__(
        self,
        as_of: Callable[[], datetime],
        model: str = "claude-sonnet-5-5",
        effort: str = "xhigh",
        max_searches: int = 6,
        max_fetches: int = 6,
        time_limit_s: int = 480,
    ) -> None:
        super().__init__(model=model, effort=effort, max_tool_uses=max_searches + max_fetches,
                         time_limit_s=time_limit_s, system=SYSTEM, role="timetravel")
        self.as_of = as_of
        self.search_cap, self.fetch_cap = max_searches, max_fetches
        self.max_client_rounds = 24

    def _tools(self) -> list[dict]:
        news = [_ASKNEWS_TOOL] if os.getenv("TIMETRAVEL_ASKNEWS", "").strip() == "1" else []
        return [{"type": "code_execution_20260120", "name": "code_execution"},
                *news, _NEWS_TOOL, _FETCH_TOOL, _PRICE_TOOL]

    def _today(self) -> str:
        return as_utc(self.as_of()).strftime("%Y-%m-%d")

    def _cost(self, s: dict) -> float | None:
        # The searches here are free; the parent prices web_search_requests
        # at the Anthropic web search rate.
        searches, s["web_search_requests"] = s["web_search_requests"], 0
        try:
            return super()._cost(s)
        finally:
            s["web_search_requests"] = searches

    def _log(self, s: dict) -> None:
        super()._log(s)
        LAST_CALL.set(s)

    async def _one(self, block: dict, s: dict, as_of: datetime, words: list[str]) -> dict:
        name, args = block.get("name"), block.get("input") or {}
        result: dict[str, Any] = {"type": "tool_result", "tool_use_id": block.get("id")}
        try:
            if name in ("news_search", "headline_search"):
                if s["web_search_requests"] >= self.search_cap:
                    raise RuntimeError(f"search limit of {self.search_cap} reached; write your answer")
                s["web_search_requests"] += 1
                query = str(args.get("query", ""))[:160]
                s["queries"].append(f"{name}: {query}")
                if name == "news_search":
                    async with _asknews_lock():
                        result["content"] = await asyncio.to_thread(asknews_search, query, as_of)
                else:
                    result["content"] = await asyncio.to_thread(
                        news_search, query, as_of, str(args.get("edition") or "US:en"))
            elif name == "fetch_page":
                if s["web_fetch_requests"] >= self.fetch_cap:
                    raise RuntimeError(f"page limit of {self.fetch_cap} reached; write your answer")
                s["web_fetch_requests"] += 1
                url = str(args.get("url", ""))[:600]
                text = await asyncio.to_thread(wayback_fetch, url, as_of, words)
                s["fetches"].append({"url": url[:300], "ok": text.startswith("Archived copy")})
                result["content"] = text
            elif name == "price_history":
                result["content"] = await asyncio.to_thread(
                    price_history, str(args.get("symbol", "")), str(args.get("source", "yahoo")),
                    int(args.get("days") or 60), as_of)
            else:
                raise RuntimeError(f"unknown tool {name!r}")
        except Exception as exc:  # noqa: BLE001 - the model reads the error and moves on
            result.update(content=f"Tool error: {type(exc).__name__}: {str(exc)[:300]}", is_error=True)
        return result

    async def _client_results(self, calls: list[dict], s: dict) -> list[dict]:
        as_of = as_utc(self.as_of())
        words = keywords(s.get("question", ""))
        return list(await asyncio.gather(*(self._one(b, s, as_of, words) for b in calls)))
