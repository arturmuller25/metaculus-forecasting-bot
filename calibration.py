"""
Platt scaling for binary forecasts, applied after aggregation.

    p' = sigmoid(A * logit(p) + B)

A > 1 pushes forecasts toward the extremes (the bot is underconfident), A < 1
pulls them toward 50% (overconfident), and B != 0 corrects a systematic lean
toward Yes or No. The default, A=1 and B=0, changes nothing.

Fit the coefficients on resolved questions:

    uv run python calibration.py resolved.csv

A fit is recommended only when it beats the raw forecasts in leave-one-out
cross-validation. With two free parameters the in-sample score improves
almost always, even on pure noise.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass

import numpy as np

# Metaculus rejects 0% and 100%.
FLOOR = 0.01
CEIL = 0.99
MIN_QUESTIONS = 20

_EPS = 1e-6


def logit(p: float) -> float:
    p = min(max(p, _EPS), 1 - _EPS)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def apply_platt(p: float, a: float = 1.0, b: float = 0.0) -> float:
    """Applies the transform and clips to [FLOOR, CEIL]."""
    if a == 1.0 and b == 0.0:
        return min(max(p, FLOOR), CEIL)
    return min(max(sigmoid(a * logit(p) + b), FLOOR), CEIL)


# ----------------------------------------------------------------------------
# Fitting the coefficients on resolved questions
# ----------------------------------------------------------------------------


@dataclass
class FitResult:
    a: float
    b: float
    log_loss_before: float
    log_loss_after: float
    brier_before: float
    brier_after: float
    brier_held_out: float
    n: int

    def improved(self) -> bool:
        """True when the fit beats no calibration on held-out questions."""
        return self.brier_held_out < self.brier_before


def _log_loss(pairs: list[tuple[float, int]], a: float, b: float) -> float:
    total = 0.0
    for p, y in pairs:
        q = apply_platt(p, a, b)
        total -= math.log(q) if y == 1 else math.log(1 - q)
    return total / len(pairs)


def _brier(pairs: list[tuple[float, int]], a: float, b: float) -> float:
    return sum((apply_platt(p, a, b) - y) ** 2 for p, y in pairs) / len(pairs)


def _search(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Minimizes log loss over (A, B) by grid search with refinement, starting
    from no calibration. x holds the logits of the forecasts, y the outcomes.
    """

    def losses(a_grid: np.ndarray, b_grid: np.ndarray) -> np.ndarray:
        z = a_grid[:, None, None] * x[None, None, :] + b_grid[None, :, None]
        q = np.clip(1 / (1 + np.exp(-z)), FLOOR, CEIL)
        return -(y * np.log(q) + (1 - y) * np.log(1 - q)).mean(axis=2)

    best_a, best_b = 1.0, 0.0
    best_loss = losses(np.array([1.0]), np.array([0.0]))[0, 0]
    a_lo, a_hi, b_lo, b_hi = 0.3, 3.0, -1.5, 1.5
    for _ in range(4):
        a_grid, b_grid = np.linspace(a_lo, a_hi, 21), np.linspace(b_lo, b_hi, 21)
        grid = losses(a_grid, b_grid)
        i, j = np.unravel_index(np.argmin(grid), grid.shape)
        if grid[i, j] < best_loss:
            best_a, best_b, best_loss = float(a_grid[i]), float(b_grid[j]), grid[i, j]
        step_a, step_b = (a_hi - a_lo) / 20, (b_hi - b_lo) / 20
        a_lo, a_hi = best_a - step_a, best_a + step_a
        b_lo, b_hi = best_b - step_b, best_b + step_b
    return best_a, best_b


def fit_platt(pairs: list[tuple[float, int]]) -> FitResult:
    """
    Fits A and B on (predicted probability, outcome 0/1) pairs, and scores the
    fit on each question with coefficients fitted without that question.
    """
    if len(pairs) < MIN_QUESTIONS:
        raise ValueError(
            f"Too few resolved questions to calibrate: {len(pairs)} "
            f"(need at least {MIN_QUESTIONS})."
        )
    x = np.array([logit(p) for p, _ in pairs])
    y = np.array([float(o) for _, o in pairs])
    a, b = _search(x, y)
    held_out = []
    for i, (p, o) in enumerate(pairs):
        a_i, b_i = _search(np.delete(x, i), np.delete(y, i))
        held_out.append((apply_platt(p, a_i, b_i) - o) ** 2)
    return FitResult(
        a=round(a, 4),
        b=round(b, 4),
        log_loss_before=_log_loss(pairs, 1.0, 0.0),
        log_loss_after=_log_loss(pairs, a, b),
        brier_before=_brier(pairs, 1.0, 0.0),
        brier_after=_brier(pairs, a, b),
        brier_held_out=sum(held_out) / len(held_out),
        n=len(pairs),
    )


def load_pairs_from_csv(path: str) -> list[tuple[float, int]]:
    """Reads a CSV with columns `prediction` (0-1) and `outcome` (0 or 1)."""
    pairs: list[tuple[float, int]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            p = float(row["prediction"])
            y = int(row["outcome"])
            if y not in (0, 1):
                raise ValueError(f"outcome must be 0 or 1, got {y!r}")
            pairs.append((p, y))
    if not pairs:
        raise ValueError(f"No rows read from {path}")
    return pairs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fits Platt scaling coefficients on resolved questions.")
    parser.add_argument("csv", help="CSV with columns prediction (0-1) and outcome (0/1)")
    args = parser.parse_args()

    result = fit_platt(load_pairs_from_csv(args.csv))
    print(f"Resolved questions        : {result.n}")
    print(f"Brier, raw                : {result.brier_before:.4f}")
    print(f"Brier, calibrated         : {result.brier_after:.4f}  (in-sample)")
    print(f"Brier, calibrated         : {result.brier_held_out:.4f}  (held-out, the one that decides)")
    print(f"Log loss, raw / calibrated: {result.log_loss_before:.4f} / {result.log_loss_after:.4f}")
    print()
    if result.improved():
        print("Calibration helps on held-out questions. Set in .env (or as repository variables):")
        print(f"  CALIBRATION_A={result.a}")
        print(f"  CALIBRATION_B={result.b}")
        if result.a > 1.15:
            print("\nThe bot is underconfident: forecasts should move toward the extremes.")
        elif result.a < 0.85:
            print("\nThe bot is overconfident: forecasts should move toward 50%.")
        if abs(result.b) > 0.2:
            print(f"There is a systematic lean against {'Yes' if result.b > 0 else 'No'}.")
    else:
        print("Calibration does not beat the raw forecasts on held-out questions. Keep A=1 and B=0.")
