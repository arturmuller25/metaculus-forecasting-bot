"""
Alternative ways to turn declared percentiles into a Metaculus CDF.

forecasting-tools interpolates linearly between the declared percentiles
(P10 to P90 plus anchors at the bounds), which puts flat density between
each pair of points and a kink at every percentile. SmoothDistribution
builds the same anchors but interpolates with PCHIP, a monotone cubic
(everywhere, or only between the declared percentiles), and can stretch the
percentiles away from the median before interpolating.

Everything else follows the library: the same anchors at the bounds, the same
log scaling, the same standardization and validation. With method="linear"
and widen=1.0 it reproduces the library's CDF, which numeric_replay.py checks.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from forecasting_tools import NumericDistribution, Percentile
from forecasting_tools.data_models.numeric_report import NumericDefaults


def _edge_slope(h0: float, h1: float, m0: float, m1: float) -> float:
    """One-sided slope at an end point, limited so the curve keeps its shape."""
    d = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    if np.sign(d) != np.sign(m0):
        return 0.0
    if np.sign(m0) != np.sign(m1) and abs(d) > 3 * abs(m0):
        return 3 * m0
    return d


def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fritsch-Carlson slopes, as in scipy's PchipInterpolator."""
    h = np.diff(x)
    delta = np.diff(y) / h
    if len(x) == 2:
        return np.full(2, delta[0])
    slopes = np.zeros(len(x))
    w1 = 2 * h[1:] + h[:-1]
    w2 = h[1:] + 2 * h[:-1]
    same_sign = delta[:-1] * delta[1:] > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        harmonic = (w1 + w2) / (w1 / delta[:-1] + w2 / delta[1:])
    slopes[1:-1] = np.where(same_sign, harmonic, 0.0)
    slopes[0] = _edge_slope(h[0], h[1], delta[0], delta[1])
    slopes[-1] = _edge_slope(h[-1], h[-2], delta[-1], delta[-2])
    return slopes


def pchip(x: np.ndarray, y: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Monotone cubic interpolation of the points (x, y), evaluated at `at`."""
    slopes = _pchip_slopes(x, y)
    i = np.clip(np.searchsorted(x, at, side="right") - 1, 0, len(x) - 2)
    h = x[i + 1] - x[i]
    t = (at - x[i]) / h
    return (
        (1 + 2 * t) * (1 - t) ** 2 * y[i]
        + t * (1 - t) ** 2 * h * slopes[i]
        + t**2 * (3 - 2 * t) * y[i + 1]
        + t**2 * (t - 1) * h * slopes[i + 1]
    )


Method = Literal["pchip", "pchip-body", "linear"]


class SmoothDistribution(NumericDistribution):
    # "pchip" runs the cubic through the declared percentiles and the bound
    # anchors. With only P10 to P90 declared, its slope at an open bound can
    # fall to zero, leaving the buckets next to the bound with the minimum
    # mass. "pchip-body" uses the cubic between the declared percentiles
    # only and keeps the library's linear tails.
    method: Method = "pchip"
    # Stretch factor for the distance of each declared percentile from the
    # median, measured on the question's axis (log axis on log questions).
    widen: float = 1.0
    # Optional separate factor below the median. A model's low percentiles
    # often sit on a known floor (a count already confirmed), which
    # stretching would push below; widen_low=1 keeps them where they are.
    widen_low: float | None = None

    @classmethod
    def build(
        cls,
        percentiles: list[Percentile],
        question,
        method: Method = "pchip",
        widen: float = 1.0,
        widen_low: float | None = None,
    ) -> SmoothDistribution:
        base = NumericDistribution.from_question(percentiles, question)
        return cls(**base.model_dump(), method=method, widen=widen, widen_low=widen_low)

    def _stretched(self, declared: list[Percentile]) -> list[Percentile]:
        heights = np.array([p.percentile for p in declared])
        locations = np.array([self._nominal_location_to_cdf_location(p.value) for p in declared])
        median = float(np.interp(0.5, heights, locations))
        low = self.widen if self.widen_low is None else self.widen_low
        factor = np.where(locations < median, low, self.widen)
        stretched = median + factor * (locations - median)
        return [
            Percentile(percentile=float(h), value=self._cdf_location_to_nominal_location(float(loc)))
            for h, loc in zip(heights, stretched)
        ]

    def _anchor_points(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Declared percentiles plus the library's bound anchors, on the 0-1
        axis, and a mask of which points were declared.
        """
        declared = self.declared_percentiles
        if self.widen != 1.0 or self.widen_low is not None:
            declared = self._stretched(declared)
        anchors = self._add_explicit_upper_lower_bound_percentiles(declared)
        x = np.array([self._nominal_location_to_cdf_location(p.value) for p in anchors])
        y = np.array([p.percentile for p in anchors])
        # Points clamped onto a closed bound can coincide; the interpolant
        # needs strictly increasing positions.
        for k in range(1, len(x)):
            if x[k] <= x[k - 1]:
                x[k] = x[k - 1] + 1e-9
        is_declared = np.isin(np.round(y, 9), np.round([p.percentile for p in declared], 9))
        return x, y, is_declared

    def get_cdf(self) -> list[Percentile]:
        size = self.cdf_size or NumericDefaults.DEFAULT_CDF_SIZE
        # Same grid as the library, so CDFs from both can be aggregated together.
        locations = np.array([i / (size - 1) for i in range(size)])
        x, y, is_declared = self._anchor_points()
        heights = np.interp(locations, x, y)
        if self.method == "pchip":
            heights = pchip(x, y, locations)
        elif self.method == "pchip-body" and is_declared.sum() >= 2:
            xd, yd = x[is_declared], y[is_declared]
            body = (locations >= xd[0]) & (locations <= xd[-1])
            heights[body] = pchip(xd, yd, locations[body])
        heights = np.clip(heights, 0.0, 1.0).tolist()
        if self.standardize_cdf:
            heights = self._standardize_cdf(heights)
        percentiles = [
            Percentile(value=self._cdf_location_to_nominal_location(float(loc)), percentile=h)
            for loc, h in zip(locations, heights)
        ]
        # Same validation the library runs on its own CDFs.
        NumericDistribution.model_validate(
            NumericDistribution(
                declared_percentiles=percentiles,
                open_upper_bound=self.open_upper_bound,
                open_lower_bound=self.open_lower_bound,
                upper_bound=self.upper_bound,
                lower_bound=self.lower_bound,
                zero_point=self.zero_point,
                standardize_cdf=self.standardize_cdf,
            )
        )
        return percentiles
