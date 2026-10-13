"""
Statistical baselines for questions about market series, computed in code.

For a question on a traded price, an index level, a yield, a credit spread,
the gap between two assets' returns or the maximum of the VIX over a window,
the market itself sets most of the answer: the current value and how much the
series usually moves in that many trading days. LLM forecasters guess that
width badly (a bot's stock forecasts were 1.51 times more extreme than the
community's, smingers-bot docs/analysis_market_close_questions.md), so this
module downloads the series and computes the distribution its own history
gives at today's volatility, and the bot shows it to the forecasters as one
research block.

Method: filtered historical simulation. Every historical move over the same
number of trading days is divided by the volatility at its start (an EWMA of
daily moves, lambda 0.94) and multiplied by today's volatility, so the
baseline keeps the series' own fat tails and today's volatility regime. Prices
and the VIX move in logs; yields and spreads move in points. The VIX maximum is
read from history directly: the highest intraday value over the window divided
by the close at the start, taken from days whose starting level was close to
today's.

Data: Yahoo Finance daily bars (chart API, no key) and FRED (fredgraph CSV,
with a reader fallback for networks where FRED times out). Nothing here calls
an LLM.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import numpy as np

logger = logging.getLogger(__name__)

PERCENTILES = (1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99)
EWMA_LAMBDA = 0.94
# Historical windows keep each series' own drift unless MARKET_DEMEAN=1. On
# the 96 resolved Market Pulse questions, removing the drift scored 0.9 points
# worse per question (90% interval -2.0 to +0.2), so it stays off.
DEMEAN = os.getenv("MARKET_DEMEAN", "0").strip() == "1"
# Yahoo answers 429 to a full browser string from a script and accepts this
# short one (checked 2026-10-12).
_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "*/*"}
_DASHES = "-" + chr(0x2013) + chr(0x2014)

# Names in question titles -> (source, symbol, how the series moves).
# "log" for prices and the VIX, "diff" for yields and spreads.
SERIES = {
    "nvidia": ("yahoo", "NVDA", "log"),
    "microsoft": ("yahoo", "MSFT", "log"),
    "apple": ("yahoo", "AAPL", "log"),
    "crude oil futures": ("yahoo", "CL=F", "log"),
    "gold futures": ("yahoo", "GC=F", "log"),
    "nasdaq-100 futures": ("yahoo", "NQ=F", "log"),
    "s&p 500 futures": ("yahoo", "ES=F", "log"),
    "vix": ("yahoo", "^VIX", "log"),
    "ust 10y yield": ("yahoo", "^TNX", "diff"),
    "high yield option-adjusted spread": ("fred", "BAMLH0A0HYM2", "diff"),
}


@dataclass
class Bar:
    day: date
    high: float
    close: float


@dataclass
class Spec:
    kind: str  # "relative_return", "max_intraday", "level_end"
    series: list[tuple[str, str, str]]
    start: date  # first day of the window (inclusive)
    end: date  # last day of the window
    label: str = ""
    names: list[str] = field(default_factory=list)


@dataclass
class Baseline:
    spec: Spec
    as_of: date
    current: dict[str, float]
    percentiles: dict[int, float]
    horizon_days: int
    gap_days: int
    samples: int
    vol_note: str
    unit: str

    def render(self) -> str:
        cur = ", ".join(f"{k} {v:,.4g}" for k, v in self.current.items())
        pts = ", ".join(f"P{p} {v:,.4g}" for p, v in self.percentiles.items())
        what = {
            "relative_return": "the difference of the two simple returns over the window, in percentage points",
            "max_intraday": "the highest intraday value over the window",
            "level_end": "the value at the end of the window",
        }[self.spec.kind]
        gap = f", starting {self.gap_days} trading days after the latest close" if self.gap_days else ""
        return (
            "Statistical baseline computed by code from market data (no news):\n"
            f"- Series: {', '.join(self.spec.names)}. Latest close ({self.as_of.isoformat()}): {cur}.\n"
            f"- Quantity: {what}; window {self.spec.start.isoformat()} to {self.spec.end.isoformat()},"
            f" {self.horizon_days} trading days{gap}.\n"
            f"- {self.vol_note}\n"
            f"- Distribution from {self.samples} historical windows rescaled to today's volatility,"
            f" in {self.unit}: {pts}.\n"
            "This is what the market's own variability implies. Move the center only for a scheduled event or"
            " news the price does not already reflect, and keep the width unless such an event makes the window"
            " unusually calm or volatile."
        )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

_CACHE: dict[tuple, list[Bar]] = {}


def _get(url: str, timeout: float = 30) -> bytes:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _yahoo_chart(symbol: str, years: int) -> dict:
    """The chart JSON, from a disk cache when MARKET_CACHE_DIR is set (backtests)."""
    cache_dir = os.getenv("MARKET_CACHE_DIR", "").strip()
    path = os.path.join(cache_dir, f"yahoo_{re.sub(r'[^A-Za-z0-9]', '_', symbol)}_{years}y.json") if cache_dir else ""
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    sym = urllib.parse.quote(symbol, safe="")
    last_error: Exception | None = None
    # Yahoo answers 429 to bursts; the second host and a pause usually pass.
    for attempt, host in enumerate(("query1", "query2", "query1", "query2")):
        try:
            raw = _get(f"https://{host}.finance.yahoo.com/v8/finance/chart/{sym}?range={years}y&interval=1d")
            data = json.loads(raw)
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(5 * attempt)
    else:
        raise RuntimeError(f"Yahoo chart failed for {symbol}: {last_error}")
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
    return data


def yahoo_bars(symbol: str, years: int = 10) -> list[Bar]:
    key = ("yahoo", symbol, years)
    if key in _CACHE:
        return _CACHE[key]
    data = _yahoo_chart(symbol, years)["chart"]["result"][0]
    quote = data["indicators"]["quote"][0]
    offset = data["meta"].get("gmtoffset", 0)
    bars = []
    for ts, high, close in zip(data["timestamp"], quote["high"], quote["close"]):
        if close is None or high is None:
            continue
        # Daily bars are stamped at the session open; shifted to the
        # exchange's zone, the date is the trading day.
        day = datetime.fromtimestamp(ts + offset, tz=timezone.utc).date()
        bars.append(Bar(day, float(high), float(close)))
    _CACHE[key] = bars
    return bars


def fred_series(series_id: str) -> list[Bar]:
    key = ("fred", series_id)
    if key in _CACHE:
        return _CACHE[key]
    cache_dir = os.getenv("MARKET_CACHE_DIR", "").strip()
    path = os.path.join(cache_dir, f"fred_{series_id}.json") if cache_dir else ""
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            bars = [Bar(date.fromisoformat(d), v, v) for d, v in json.load(fh)]
        _CACHE[key] = bars
        return bars
    rows: list[tuple[date, float]] = []
    try:
        text = _get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}", timeout=25).decode("utf-8")
        for rec in csv.reader(io.StringIO(text)):
            if len(rec) == 2 and re.match(r"\d{4}-\d\d-\d\d", rec[0]) and rec[1] not in ("", "."):
                rows.append((date.fromisoformat(rec[0]), float(rec[1])))
    except Exception as exc:  # noqa: BLE001 - fall back to the text table through a reader
        logger.info(f"FRED CSV failed for {series_id} ({type(exc).__name__}); trying the reader")
        text = _get(f"https://r.jina.ai/https://fred.stlouisfed.org/data/{series_id}.txt", timeout=60).decode("utf-8")
        for d, v in re.findall(r"\|\s*(\d{4}-\d\d-\d\d)\s*\|\s*([-\d.]+)\s*\|", text):
            rows.append((date.fromisoformat(d), float(v)))
    bars = [Bar(d, v, v) for d, v in sorted(rows)]
    if path and bars:
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump([[b.day.isoformat(), b.close] for b in bars], fh)
    _CACHE[key] = bars
    return bars


def bars_for(source: str, symbol: str) -> list[Bar]:
    bars = yahoo_bars(symbol) if source == "yahoo" else fred_series(symbol)
    # Prices move in logs, so a non-positive close (WTI on 2020-04-20) is
    # dropped; yields and spreads keep every value.
    if any(how == "log" and sym == symbol for _, sym, how in SERIES.values()):
        return [b for b in bars if b.close > 0 and b.high > 0]
    return bars


# ---------------------------------------------------------------------------
# Question parsing
# ---------------------------------------------------------------------------

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
_WINDOW = re.compile(r"\s*([A-Za-z]{3})[a-z]*\.? (\d{1,2})\s*[" + _DASHES + r"]\s*([A-Za-z]{3})[a-z]*\.? (\d{1,2})")


def _parse_window(label: str, year: int) -> tuple[date, date] | None:
    """'Jul 27 - Aug 7' -> (2026-07-27, 2026-08-07)."""
    m = _WINDOW.match(label)
    if not m:
        return None
    m1, d1, m2, d2 = _MONTHS.get(m.group(1).lower()), int(m.group(2)), _MONTHS.get(m.group(3).lower()), int(m.group(4))
    if not m1 or not m2:
        return None
    return date(year, m1, d1), date(year + (1 if m2 < m1 else 0), m2, d2)


def _find_series(text: str) -> list[tuple[str, tuple[str, str, str]]]:
    low = text.lower()
    found = [(name, SERIES[name], low.index(name)) for name in SERIES if name in low]
    return [(n, s) for n, s, _ in sorted(found, key=lambda x: x[2])]


def spec_for(title: str, label: str, year: int) -> Spec | None:
    """Recognizes the market-series question shapes used by Market Pulse."""
    window = _parse_window(label or "", year)
    found = _find_series(title)
    if not window or not found:
        return None
    start, end = window
    low = title.lower()
    if "exceed" in low and len(found) >= 2:
        return Spec("relative_return", [found[0][1], found[1][1]], start, end, label, [found[0][0], found[1][0]])
    if "maximum intraday" in low and found[0][1][1] == "^VIX":
        return Spec("max_intraday", [found[0][1]], start, end, label, [found[0][0]])
    if "ending value" in low:
        return Spec("level_end", [found[0][1]], start, end, label, [found[0][0]])
    return None


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def _trading_days(a: date, b: date) -> int:
    """Weekdays after a, up to and including b (US holidays ignored)."""
    if b <= a:
        return 0
    return int(np.busday_count(a + timedelta(days=1), b + timedelta(days=1)))


def _ewma_vol(moves: np.ndarray) -> np.ndarray:
    """Volatility known at the end of each day, from moves up to that day."""
    out = np.empty(len(moves))
    var = float(np.var(moves[:20]))
    for i, x in enumerate(moves):
        var = EWMA_LAMBDA * var + (1 - EWMA_LAMBDA) * x * x
        out[i] = math.sqrt(var)
    return out


def _upto(bars: list[Bar], as_of: date) -> list[Bar]:
    return [b for b in bars if b.day <= as_of]


def _quantiles(samples: np.ndarray) -> dict[int, float]:
    return {p: float(np.percentile(samples, p)) for p in PERCENTILES}


def _vol_note(cur: float, hist: np.ndarray, how: str, unit: str) -> str:
    med = float(np.median(hist))
    if how == "log":
        return (f"Daily volatility now {100 * cur:.2f}% (EWMA), against a median of {100 * med:.2f}% over the"
                f" history used ({100 * cur * math.sqrt(252):.0f}% vs {100 * med * math.sqrt(252):.0f}% a year).")
    return f"Daily volatility now {cur:.3f} {unit} (EWMA), against a median of {med:.3f} {unit} over the history used."


def compute(spec: Spec, as_of: date) -> Baseline | None:
    """Baseline as of the close of `as_of`; data after that day is never read."""
    # The window's own trading days: from the close before it starts to its
    # last close.
    window = _trading_days(spec.start - timedelta(days=1), spec.end)
    if spec.kind == "relative_return":
        # Only moves inside the window count, so the gap does not matter.
        (sa, ya, _), (sb, yb, _) = spec.series
        a, b = _upto(bars_for(sa, ya), as_of), _upto(bars_for(sb, yb), as_of)
        ca = {x.day: x.close for x in a}
        cb = {x.day: x.close for x in b}
        common = sorted(set(ca) & set(cb))
        if len(common) < 300 or window < 1:
            return None
        pa = np.array([ca[d] for d in common])
        pb = np.array([cb[d] for d in common])
        vol = _ewma_vol(np.diff(np.log(pa)) - np.diff(np.log(pb)))
        z = np.array([
            ((pa[i + window] / pa[i] - 1) - (pb[i + window] / pb[i] - 1)) / (vol[i - 1] * math.sqrt(window))
            for i in range(60, len(common) - window)
        ])
        if DEMEAN:
            # No drift: one asset's past outperformance is not a forecast.
            z = z - z.mean()
        samples = 100 * z * vol[-1] * math.sqrt(window)
        cur = {spec.names[0]: float(pa[-1]), spec.names[1]: float(pb[-1])}
        note = _vol_note(vol[-1], vol[60:], "log", "") + " Volatility of the daily return difference."
        gap = _trading_days(common[-1], spec.start - timedelta(days=1))
        return Baseline(spec, common[-1], cur, _quantiles(samples), window, gap, len(z), note, "percentage points")

    source, symbol, how = spec.series[0]
    bars = _upto(bars_for(source, symbol), as_of)
    if len(bars) < 300 or window < 1:
        return None
    closes = np.array([b.close for b in bars])
    highs = np.array([b.high for b in bars])
    last = bars[-1]
    # Trading days between the latest close and the window, counted from the
    # last session actually traded (a holiday leaves no bar).
    gap = _trading_days(last.day, spec.start - timedelta(days=1))
    unit = "points" if symbol == "^VIX" else ("percent" if how == "diff" else "price units")
    moves = np.diff(np.log(closes)) if how == "log" else np.diff(closes)
    vol = _ewma_vol(moves)

    if spec.kind == "max_intraday":
        # Days whose close was near today's level, so the VIX's mean reversion
        # and its level-dependent swings come from history itself.
        ratios: list[float] = []
        for lo, hi in ((0.8, 1.25), (0.67, 1.5), (0.5, 2.0)):
            ratios = [
                float(highs[i + gap + 1: i + gap + 1 + window].max() / closes[i])
                for i in range(len(closes) - gap - window - 1)
                if lo * last.close <= closes[i] <= hi * last.close
            ]
            if len(ratios) >= 300:
                break
        if len(ratios) < 100:
            return None
        samples = last.close * np.array(ratios)
        note = _vol_note(vol[-1], vol[60:], "log", unit) + f" Windows drawn from days that closed near {last.close:.2f}."
        return Baseline(spec, last.day, {spec.names[0]: last.close}, _quantiles(samples), window, gap, len(ratios), note, unit)

    # Level at the end of the window: the whole path from today counts.
    horizon = gap + window
    z = []
    for i in range(60, len(closes) - horizon):
        change = math.log(closes[i + horizon] / closes[i]) if how == "log" else closes[i + horizon] - closes[i]
        z.append(change / (vol[i - 1] * math.sqrt(horizon)))
    z = np.array(z)
    if DEMEAN:
        # No drift: a decade of rising prices or yields is not a forecast.
        z = z - z.mean()
    step = z * vol[-1] * math.sqrt(horizon)
    samples = last.close * np.exp(step) if how == "log" else last.close + step
    return Baseline(spec, last.day, {spec.names[0]: last.close}, _quantiles(samples), window, gap, len(z),
                    _vol_note(vol[-1], vol[60:], how, unit), unit)


def outcome(spec: Spec) -> float | None:
    """What the series did over the window, for backtests on resolved questions."""
    if spec.kind == "relative_return":
        out = []
        for source, symbol, _ in spec.series:
            bars = bars_for(source, symbol)
            before = [b for b in bars if b.day < spec.start]
            inside = [b for b in bars if spec.start <= b.day <= spec.end]
            if not before or not inside:
                return None
            out.append(inside[-1].close / before[-1].close - 1)
        return 100 * (out[0] - out[1])
    source, symbol, _ = spec.series[0]
    inside = [b for b in bars_for(source, symbol) if spec.start <= b.day <= spec.end]
    if not inside:
        return None
    if spec.kind == "max_intraday":
        return max(b.high for b in inside)
    return inside[-1].close


def baseline_for_question(question) -> Baseline | None:
    """The baseline for a bot question, or None when it is not a market series.

    Market Pulse subquestions carry their window in the title, for example
    "... in these biweekly periods of Q4 2026? (Oct 12 - Oct 23)". Data is
    cached in the process, so the research step and the forecast step share
    one download.
    """
    text = getattr(question, "question_text", "") or ""
    match = _WINDOW.search(text)
    if not match:
        return None
    close = getattr(question, "close_time", None) or datetime.now(timezone.utc)
    spec = spec_for(text, match.group(0).strip(), close.year)
    if spec is None:
        return None
    return compute(spec, datetime.now(timezone.utc).date())


def block_for_question(question) -> str:
    """The research block for a bot question, or "" when it is not a market series."""
    baseline = baseline_for_question(question)
    return baseline.render() if baseline else ""
