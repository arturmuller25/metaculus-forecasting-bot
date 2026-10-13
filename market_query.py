"""
Statistical baselines for questions on one market series: "what will X close
at on DATE", "will X close above K on DATE", "will X trade above K between A
and B". About 15% of MiniBench questions have had this shape (118 of 804 in
13 rounds).

A cheap LLM call reads the question and names the series (a Yahoo Finance
symbol or a FRED series id), the quantity, the dates, the threshold and the
unit conversion; everything after that is code: the series is downloaded and
the distribution comes from the same filtered historical simulation as
market_baseline.py. A result is dropped when the latest value does not fit the
question's own range or threshold, which is how a wrong symbol or unit shows.

The bot records the result as a shadow forecast ("market-baseline"), to be
scored on live questions with full research. On 96 past MiniBench questions
(market_query_backtest.py) Claude with only the latest prices tied the
baseline, and seeing the baseline did not help it, so by default it is
neither shown nor mixed in (MARKET_QUERY_BLOCK=1 shows it). The baseline
itself is calibrated: 81% of outcomes inside its P10 to P90 over 2,450
walk-forward dates of 11 series (market_calibration.py), 83% on the 63
numeric MiniBench questions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import numpy as np

import market_baseline as mb

logger = logging.getLogger(__name__)

# Only questions whose text suggests a market series are sent to the LLM.
HINT = re.compile(
    r"clos(e|es|ing)\b|settle|price of|stock|share price|\bETF\b|index|S&P|Nasdaq|Dow Jones|\bVIX\b|Bitcoin|\bBTC\b|"
    r"Ethereum|\bETH\b|crypto|gold|crude|Brent|\bWTI\b|yield|Treasury|exchange rate|futures|spread|trade above|trade below",
    re.I,
)

KINDS = ("value_on_date", "max_in_window", "min_in_window", "above_on_date", "below_on_date", "touch_above", "touch_below")
# Numeric questions ask for a value; binary questions for a threshold.
VALUE_KINDS = ("value_on_date", "max_in_window", "min_in_window")

PROMPT = """A forecasting question may be about the value of one market series. Read it and answer with one JSON object and nothing else.

Question: {title}

Resolution criteria: {criteria}

Fine print: {fine_print}

Today is {today}.

Fields:
- "is_market_series": true only if the question resolves to the price, level, yield or rate of ONE traded instrument or daily published financial series (a stock, ETF, index, future, currency pair, cryptocurrency, bond yield, credit spread), or to whether that value is above or below a threshold. False for counts, company results, events, comparisons of two series, and anything else.
- "source": "yahoo" or "fred".
- "symbol": the Yahoo Finance symbol (examples: "FXI", "NVDA", "BTC-USD", "ETH-USD", "BZ=F" for front-month Brent, "CL=F" for WTI, "GC=F" for gold, "^NDX", "^GSPC", "^VIX", "^TNX" for the 10-year Treasury yield in percent, "7974.T", "005930.KS", "EURUSD=X", "XU100.IS") or the FRED series id (examples: "DGS10", "DGS30", "BAMLH0A0HYM2").
- "name": short name of the series.
- "moves": "log" for prices, index levels and cryptocurrencies, "diff" for yields, rates and spreads quoted in percent or points.
- "quantity": "value_on_date" (what will the value be on a date), "max_in_window" or "min_in_window" (the highest or lowest value over a period), "above_on_date" (will it be above or at a threshold on a date), "below_on_date", "touch_above" (will it be above a threshold on ANY day or at any point in a window), "touch_below".
- "start_date": first day of the window as YYYY-MM-DD for the window quantities (touch, max, min), otherwise null.
- "end_date": the target date, or the last day of the window, as YYYY-MM-DD.
- "threshold": the threshold in the question's unit, or null.
- "reference_date": when the threshold is the series' own close on an earlier date ("higher than its close on 2026-04-17"), that date as YYYY-MM-DD and threshold null; otherwise null.
- "scale": the number that multiplies the source's value to give the question's unit (1 in most cases; 0.000001 if the question asks for millions of the quoted currency).
- "intraday": true if the question counts intraday highs or lows ("at any point", "trade above"), false if it counts closes or settlements."""


@dataclass
class Query:
    source: str
    symbol: str
    name: str
    how: str
    kind: str
    end: date
    start: date | None = None
    threshold: float | None = None
    scale: float = 1.0
    intraday: bool = False
    reference: date | None = None


@dataclass
class Result:
    query: Query
    as_of: date
    latest: float  # in the question's unit
    steps: int
    percentiles: dict[int, float]  # of the quantity, in the question's unit
    probability: float | None  # for the threshold quantities
    samples: int
    vol_note: str

    def render(self) -> str:
        q = self.query
        pts = ", ".join(f"P{p} {v:,.5g}" for p, v in self.percentiles.items())
        lines = [
            "Statistical baseline computed by code from market data (no news):",
            f"- Series: {q.name} ({q.symbol}, {q.source}). Latest close ({self.as_of.isoformat()}): {self.latest:,.5g}.",
            f"- {self.vol_note}",
        ]
        days = "calendar days" if _trades_weekends_symbol(q) else "trading days"
        if q.kind == "value_on_date":
            lines.append(f"- Value on {q.end.isoformat()}, {self.steps} {days} ahead, from {self.samples} historical"
                         f" windows rescaled to today's volatility: {pts}.")
        elif q.kind in ("max_in_window", "min_in_window"):
            which = "Highest" if q.kind == "max_in_window" else "Lowest"
            since = q.start.isoformat() if q.start else "now"
            lines.append(f"- {which} value from {since} to {q.end.isoformat()} ({self.samples} historical windows"
                         f" rescaled to today's volatility; values already observed in the window included): {pts}.")
        else:
            side = "above" if q.kind.endswith("above") or q.kind == "above_on_date" else "below"
            when = (f"on {q.end.isoformat()}" if q.kind in ("above_on_date", "below_on_date")
                    else f"on any day from {q.start.isoformat() if q.start else 'now'} to {q.end.isoformat()}")
            extreme = "the window's extreme" if q.kind.startswith("touch") else "the value on that date"
            lines.append(f"- Chance of being {side} {q.threshold:,.5g} {when}: {100 * (self.probability or 0):.0f}%"
                         f" ({self.samples} historical windows rescaled to today's volatility).")
            lines.append(f"- Distribution of {extreme}: {pts}.")
        lines.append(
            "This is what the market's own variability implies. Move away from it only for a scheduled event or"
            " news the price does not already reflect. If the series or the unit does not match the question,"
            " ignore this block."
        )
        return "\n".join(lines)


_WEEKEND: dict[str, bool] = {}


def _trades_weekends_symbol(query: Query) -> bool:
    return _WEEKEND.get(query.symbol, False)


def _bars(query: Query, as_of: date) -> list[mb.Bar]:
    bars = mb.yahoo_bars(query.symbol) if query.source == "yahoo" else mb.fred_series(query.symbol)
    if query.how == "log":
        bars = [b for b in bars if b.close > 0 and b.high > 0]
    bars = [b for b in bars if b.day <= as_of]
    _WEEKEND[query.symbol] = any(b.day.weekday() >= 5 for b in bars[-15:])
    return bars


def _steps(last: date, target: date, weekends: bool) -> int:
    if target <= last:
        return 0
    return (target - last).days if weekends else int(np.busday_count(last + timedelta(days=1), target + timedelta(days=1)))


def compute(query: Query, as_of: date) -> Result | None:
    bars = _bars(query, as_of)
    if len(bars) < 300:
        return None
    weekends = _WEEKEND[query.symbol]
    closes = np.array([b.close for b in bars])
    lows = closes  # daily lows are not downloaded; closes stand in for them
    highs = np.array([b.high for b in bars]) if query.intraday else closes
    last = bars[-1]
    log = query.how == "log"
    moves = np.diff(np.log(closes)) if log else np.diff(closes)
    vol = mb._ewma_vol(moves)
    unit = "" if log else "points"
    note = mb._vol_note(float(vol[-1]), vol[60:], query.how, unit)
    steps = _steps(last.day, query.end, weekends)
    if steps < 1:
        return None

    def scaled(change: np.ndarray, start_vol: np.ndarray) -> np.ndarray:
        """Historical changes rescaled to today's volatility, as values in the question's unit."""
        z = change / start_vol
        step = z * vol[-1]
        return query.scale * (last.close * np.exp(step) if log else last.close + step)

    idx = np.arange(60, len(closes) - steps)
    if len(idx) < 100:
        return None
    threshold = query.threshold
    if threshold is None and query.reference is not None:
        # "Higher than its close on <reference date>": that close, or the
        # latest one when the reference day has not closed yet.
        before = [b for b in bars if b.day <= query.reference]
        threshold = query.scale * (before[-1].close if before else last.close)
    if query.kind in ("value_on_date", "above_on_date", "below_on_date"):
        change = np.log(closes[idx + steps] / closes[idx]) if log else closes[idx + steps] - closes[idx]
        samples = scaled(change, vol[idx - 1])
        probability = None
        if query.kind != "value_on_date":
            if threshold is None:
                return None
            above = float(np.mean(samples >= threshold))
            probability = above if query.kind == "above_on_date" else 1 - above
    else:
        upward = query.kind in ("touch_above", "max_in_window")
        if query.kind.startswith("touch") and threshold is None:
            return None
        gap = _steps(last.day, query.start - timedelta(days=1), weekends) if query.start else 0
        gap = min(gap, steps - 1)
        series = highs if upward else lows
        extremes = []
        for i in idx:
            window = series[i + gap + 1: i + steps + 1]
            ext = window.max() if upward else window.min()
            extremes.append(math.log(ext / closes[i]) if log else ext - closes[i])
        samples = scaled(np.array(extremes), vol[idx - 1])
        # A window already under way: what has been observed counts too.
        seen = [b for b in bars if query.start and b.day >= query.start]
        observed = [query.scale * (b.high if query.intraday else b.close) for b in seen] if upward else [
            query.scale * b.close for b in seen]
        probability = None
        if query.kind == "max_in_window" and observed:
            samples = np.maximum(samples, max(observed))
        elif query.kind == "min_in_window" and observed:
            samples = np.minimum(samples, min(observed))
        elif query.kind == "touch_above":
            hit = any(v >= threshold for v in observed)
            probability = 1.0 if hit else float(np.mean(samples >= threshold))
        elif query.kind == "touch_below":
            hit = any(v <= threshold for v in observed)
            probability = 1.0 if hit else float(np.mean(samples <= threshold))
    if probability is not None:
        probability = min(0.99, max(0.01, probability))
    if threshold is not None and query.threshold is None:
        query.threshold = threshold
    return Result(query, last.day, query.scale * last.close, steps,
                  {p: float(np.percentile(samples, p)) for p in mb.PERCENTILES}, probability, len(samples), note)


def parse(text: str) -> Query | None:
    """The LLM's JSON answer as a Query, or None when it is not a usable market series."""
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        if not data.get("is_market_series"):
            return None
        kind = data.get("quantity")
        source = data.get("source")
        symbol = str(data.get("symbol") or "").strip()
        if kind not in KINDS or source not in ("yahoo", "fred") or not symbol:
            return None
        threshold = data.get("threshold")
        start = data.get("start_date")
        reference = data.get("reference_date")
        return Query(
            source=source, symbol=symbol, name=str(data.get("name") or symbol),
            how="diff" if data.get("moves") == "diff" else "log", kind=kind,
            end=date.fromisoformat(str(data["end_date"])[:10]),
            start=date.fromisoformat(str(start)[:10]) if start else None,
            threshold=float(threshold) if threshold is not None else None,
            scale=float(data.get("scale") or 1.0), intraday=bool(data.get("intraday")),
            reference=date.fromisoformat(str(reference)[:10]) if reference else None,
        )
    except (ValueError, KeyError, TypeError):
        return None


def plausible(result: Result, question) -> bool:
    """False when the series' latest value does not fit the question: a wrong symbol or unit."""
    q = result.query
    if q.kind == "value_on_date":
        low, high = getattr(question, "lower_bound", None), getattr(question, "upper_bound", None)
        if low is None or high is None:
            return True
        span = high - low
        return low - span <= result.latest <= high + span
    if not q.threshold or result.latest <= 0:
        return q.how == "diff"
    return 0.2 <= result.latest / q.threshold <= 5


def prompt_for(question) -> str:
    return PROMPT.format(
        title=question.question_text,
        criteria=(getattr(question, "resolution_criteria", "") or "").strip()[:3000],
        fine_print=(getattr(question, "fine_print", "") or "").strip()[:1500],
        today=datetime.now(timezone.utc).date().isoformat(),
    )


async def for_question(question, llm, as_of: date | None = None) -> Result | None:
    """The baseline for a single-series market question, or None."""
    text = " ".join([question.question_text or "", getattr(question, "resolution_criteria", "") or ""])
    if not HINT.search(text):
        return None
    query = parse(await llm.invoke(prompt_for(question)))
    if query is None:
        return None
    # A binary question needs a threshold quantity, a numeric one a value.
    if (query.kind in VALUE_KINDS) == (type(question).__name__ == "BinaryQuestion"):
        return None
    result = await asyncio.to_thread(compute, query, as_of or datetime.now(timezone.utc).date())
    if result is None or not plausible(result, question):
        logger.info(f"Market query dropped for {getattr(question, 'page_url', '')}: {query.symbol} {query.kind}")
        return None
    return result
