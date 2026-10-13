"""
Walk-forward calibration of market_query.py, with no LLM: on past dates of
each series, asks "what will the value be N steps ahead" and "will it be
above the level K" using only data up to that date, and compares with what
happened. A calibrated baseline puts 80% of outcomes between its P10 and P90
and its probabilities match the observed frequencies.

Usage:
    uv run python market_calibration.py
"""

from __future__ import annotations

import os
from datetime import timedelta

import numpy as np

os.environ.setdefault("MARKET_CACHE_DIR", os.path.join("logs", "market_backtest", "data"))

import market_baseline as mb  # noqa: E402
import market_query as mq  # noqa: E402

SERIES = [
    ("yahoo", "FXI", "log"), ("yahoo", "NVDA", "log"), ("yahoo", "DAL", "log"), ("yahoo", "^NDX", "log"),
    ("yahoo", "BZ=F", "log"), ("yahoo", "GC=F", "log"), ("yahoo", "BTC-USD", "log"), ("yahoo", "ETH-USD", "log"),
    ("yahoo", "005930.KS", "log"), ("yahoo", "XU100.IS", "log"), ("yahoo", "^TNX", "diff"),
]
STEPS = (5, 10)  # ahead: one and two weeks of trading days (calendar days for crypto)
EVERY = 7  # one simulated question every 7 bars
YEARS = 3


def main() -> None:
    rows = []
    for source, symbol, how in SERIES:
        try:
            bars = mb.yahoo_bars(symbol) if source == "yahoo" else mb.fred_series(symbol)
        except Exception as exc:  # noqa: BLE001
            print(f"{symbol}: no data ({type(exc).__name__})")
            continue
        if how == "log":
            bars = [b for b in bars if b.close > 0 and b.high > 0]
        first = bars[-1].day - timedelta(days=365 * YEARS)
        start = next(i for i, b in enumerate(bars) if b.day >= first)
        for steps in STEPS:
            inside80 = inside90 = n = 0
            brier, base_brier, probs, hits = [], [], [], []
            for i in range(start, len(bars) - steps, EVERY):
                target = bars[i + steps]
                # A threshold 2% above today's value, in the question's terms.
                level = bars[i].close * 1.02 if how == "log" else bars[i].close + 0.1
                value = mq.compute(mq.Query(source, symbol, symbol, how, "value_on_date", target.day), bars[i].day)
                above = mq.compute(mq.Query(source, symbol, symbol, how, "above_on_date", target.day, threshold=level), bars[i].day)
                if value is None or above is None:
                    continue
                n += 1
                inside80 += value.percentiles[10] <= target.close <= value.percentiles[90]
                inside90 += value.percentiles[5] <= target.close <= value.percentiles[95]
                hit = float(target.close >= level)
                probs.append(above.probability)
                hits.append(hit)
            if not n:
                continue
            probs_a, hits_a = np.array(probs), np.array(hits)
            brier = float(np.mean((probs_a - hits_a) ** 2))
            climate = float(np.mean((hits_a.mean() - hits_a) ** 2))
            rows.append((symbol, steps, n, inside80 / n, inside90 / n, float(probs_a.mean()), float(hits_a.mean()), brier, climate))
    print(f"{'series':10s} {'steps':>5s} {'n':>4s} {'P10-P90':>8s} {'P5-P95':>7s} {'mean p':>7s} {'freq':>6s} {'Brier':>7s} {'constant':>9s}")
    for r in rows:
        print(f"{r[0]:10s} {r[1]:5d} {r[2]:4d} {r[3]:8.2f} {r[4]:7.2f} {r[5]:7.2f} {r[6]:6.2f} {r[7]:7.3f} {r[8]:9.3f}")
    if rows:
        w = np.array([r[2] for r in rows], dtype=float)
        print(f"\nAll: P10-P90 coverage {np.average([r[3] for r in rows], weights=w):.3f} (0.80 if calibrated),"
              f" P5-P95 {np.average([r[4] for r in rows], weights=w):.3f} (0.90);"
              f" threshold Brier {np.average([r[7] for r in rows], weights=w):.3f}"
              f" vs {np.average([r[8] for r in rows], weights=w):.3f} for the constant base rate")


if __name__ == "__main__":
    main()
