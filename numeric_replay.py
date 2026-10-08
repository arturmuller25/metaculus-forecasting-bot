"""
Replays how the bot turns percentiles into a numeric CDF, on the numeric and
discrete questions it forecast. Uses only the Metaculus API and local files:
no LLM calls, no cost.

Each ensemble member's declared percentiles (P10 to P90) are rebuilt into
the published CDF in several ways (numeric_cdf.py) and aggregated exactly
like the live bot:

  linear         what the bot publishes today (the library's interpolation)
  pchip          monotone cubic through the percentiles and the bound anchors
  pchip body     monotone cubic between P10 and P90, linear tails
  pchip x1.15    pchip after stretching the percentiles 15% from the median
  linear x1.15   the same stretch with linear interpolation
  upper only     pchip stretching only the percentiles above the median, so
                 low percentiles sitting on a known floor stay put

Member percentiles come from forecasts.jsonl records when available (exact),
otherwise from the bot's comment. A question enters the comparison only if
the rebuilt linear CDF matches the CDF the bot actually published, which
proves the replay reproduces production for it.

On resolved questions each variant is scored against linear as
50 * ln(p_variant / p_linear), where p is the probability the CDF gives the
outcome's bucket: the change in the question's spot peer score with everyone
else's forecast fixed (numeric peer scores are halved, hence 50, not 100).

Usage:
    uv run python numeric_replay.py [--fetch-artifacts] [--all]

--all also checks open and closed questions: rebuild fidelity, and how far
each variant moves the published CDF.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import re
from types import SimpleNamespace

import numpy as np
from forecasting_tools import DataOrganizer, NumericDistribution, NumericReport, Percentile

import analyze_results as ar
from numeric_cdf import SmoothDistribution

VARIANTS = {
    "linear": ("linear", 1.0),
    "pchip": ("pchip", 1.0),
    "pchip body": ("pchip-body", 1.0),
    "pchip x1.15": ("pchip", 1.15),
    "linear x1.15": ("linear", 1.15),
    "pchip x1.15 upper only": ("pchip", 1.15, 1.0),
    "pchip x1.3 upper only": ("pchip", 1.3, 1.0),
}
# Largest gap between the rebuilt linear CDF and the published one for the
# question to count as reproduced. Comment values are rounded to 4 digits.
FIDELITY = 0.01
LEVELS = [10, 20, 40, 60, 80, 90]
# The bot has asked for 13 percentiles since 2026-10-09.
LEVELS_13 = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]


# ---------------------------------------------------------------------------
# Member percentiles
# ---------------------------------------------------------------------------


def _raw_percentiles(text: str) -> list[tuple[float, float]] | None:
    """The last complete percentile answer in a text (13 percentiles, or the older 6)."""
    found = re.findall(r"Percentile\s+(\d{1,2})\s*[:=]\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    for levels in (LEVELS_13, LEVELS):
        for start in range(len(found) - len(levels), -1, -1):
            block = found[start : start + len(levels)]
            if [int(p) for p, _ in block] == levels:
                return [(int(p) / 100, float(v.replace(",", ""))) for p, v in block]
    return None


def _close(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> bool:
    return len(a) == len(b) and all(
        pa == pb and abs(va - vb) <= 1e-3 * max(1.0, abs(vb)) for (pa, va), (pb, vb) in zip(a, b)
    )


def members_from_comment(text: str) -> tuple[dict[str, list[tuple[float, float]]], str]:
    """
    Per-model percentiles from the bot's comment. Ensemble comments have a
    "### <model> forecast: P10 x | ..." header per model, rounded for display,
    followed by that model's full answer; the exact values from the answer are
    used when they agree with the header. Older single-model comments only
    have the model's answer.
    """
    sections = re.split(r"^### ", text, flags=re.M)
    members: dict[str, list[tuple[float, float]]] = {}
    exact = True
    for section in sections[1:]:
        header, _, body = section.partition("\n")
        m = re.match(r"(\S+) forecast: (P\d+ .+)$", header.strip())
        if not m:
            continue
        shown = [
            (int(p) / 100, float(v.replace(",", "")))
            for p, v in re.findall(r"P(\d+) (-?[\d,]*\.?\d+(?:e[+-]?\d+)?)", m.group(2))
        ]
        raw = _raw_percentiles(body)
        if raw and _close(raw, shown):
            members[m.group(1)] = raw
        else:
            members[m.group(1)] = shown
            exact = False
    if members:
        return members, "comment" if exact else "comment (rounded)"
    raw = _raw_percentiles(text)
    return ({"single model": raw}, "comment (single model)") if raw else ({}, "none")


# ---------------------------------------------------------------------------
# CDFs and scoring
# ---------------------------------------------------------------------------


async def final_cdf(
    members: list[list[tuple[float, float]]], question, method: str, widen: float, widen_low: float | None = None
) -> np.ndarray:
    """The CDF the bot would publish if its members were built this way."""
    dists = [
        SmoothDistribution.build(
            [Percentile(percentile=p, value=v) for p, v in pts], question, method=method, widen=widen, widen_low=widen_low
        )
        for pts in members
    ]
    # Same steps as the live bot: the ensemble step in _run_forecast_on_numeric
    # (only with two or more members), the parent class's aggregation of the
    # single prediction per question, then get_cdf when publishing.
    combined = await NumericReport.aggregate_predictions(dists, question) if len(dists) > 1 else dists[0]
    top = await NumericReport.aggregate_predictions([combined], question)
    return np.array([p.percentile for p in top.get_cdf()])


def bucket_of(question, outcome, size: int) -> int:
    """Index into the PMF (0 = below the range, size = above it)."""
    inbound = size - 1
    if outcome == "below_lower_bound":
        return 0
    if outcome == "above_upper_bound":
        return inbound + 1
    # The library's own conversion from a value to the 0-1 axis (log-aware).
    axis = SimpleNamespace(
        lower_bound=question.lower_bound, upper_bound=question.upper_bound, zero_point=question.zero_point
    )
    location = NumericDistribution._nominal_location_to_cdf_location(axis, float(outcome))
    if location < 0:
        return 0
    if location > 1:
        return inbound + 1
    return min(inbound, int(math.floor(location * inbound + 1e-9)) + 1)


def pmf_at(cdf: np.ndarray, bucket: int) -> float:
    return float(np.diff(np.concatenate([[0.0], cdf, [1.0]]))[bucket])


def bootstrap(diffs: list[float], reps: int = 20000) -> tuple[float, float]:
    arr = np.array(diffs)
    means = np.random.default_rng(0).choice(arr, size=(reps, len(arr)), replace=True).mean(axis=1)
    return float(np.percentile(means, 5)), float(np.percentile(means, 95))


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fetch-artifacts", action="store_true", help="download forecast records from GitHub Actions first")
    parser.add_argument("--all", action="store_true", help="also check open and closed questions")
    args = parser.parse_args()

    if args.fetch_artifacts:
        ar.fetch_artifacts()
    records = ar.shadow_records()
    me = ar.get("/users/me/")["id"]
    statuses = ["open", "closed", "resolved"] if args.all else "resolved"
    posts = [p for p in ar.bot_posts(me, statuses) if (p.get("question") or {}).get("type") in ("numeric", "discrete")]
    print(f"Numeric and discrete questions the bot forecast ({'all statuses' if args.all else 'resolved'}): {len(posts)}")

    diffs: dict[str, list[float]] = {name: [] for name in VARIANTS if name != "linear"}
    member_scores: dict[str, list[float]] = {}
    skipped: list[str] = []
    for post in posts:
        pid = post["id"]
        post_json = ar.get(f"/posts/{pid}/")
        q_json = post_json["question"]
        question = DataOrganizer.get_question_from_post_json(post_json)
        published = np.array(((q_json.get("my_forecasts") or {}).get("latest") or {}).get("forecast_values") or [])

        members = {
            key.split(":", 1)[1]: [tuple(p) for p in row["forecast"]]
            for (post_id, key), row in records.items()
            if post_id == pid and key.startswith("member:")
        }
        source = "records"
        if not members:
            members, source = members_from_comment(ar.bot_comment_text(pid, me))
        if not members:
            skipped.append(f"{pid}: no member percentiles found")
            continue

        cdfs = {}
        for name, (method, widen, *low) in VARIANTS.items():
            try:
                cdfs[name] = asyncio.run(final_cdf(list(members.values()), question, method, widen, *low))
            except Exception as exc:  # noqa: BLE001
                skipped.append(f"{pid}: {name} failed: {type(exc).__name__}: {str(exc)[:100]}")
        if "linear" not in cdfs:
            continue
        gap = float(np.max(np.abs(cdfs["linear"] - published))) if len(published) == len(cdfs["linear"]) else float("nan")
        reproduced = gap <= FIDELITY
        moved = "  ".join(
            f"{name} {np.max(np.abs(cdf - cdfs['linear'])):.3f}" for name, cdf in cdfs.items() if name != "linear"
        )
        status = q_json.get("status") or post.get("status")
        print(f"  {pid} {question.__class__.__name__[:8]:8s} {status:8s} members {len(members)} from {source:22s} "
              f"rebuild gap {gap:.4f}{'' if reproduced else ' NOT REPRODUCED'} | max CDF shift: {moved}")

        outcome = ar.outcome_of(q_json)
        if outcome is None or not reproduced:
            if outcome is not None:
                skipped.append(f"{pid}: resolved but the rebuild does not match the published CDF (gap {gap:.3f})")
            continue
        size = len(cdfs["linear"])
        bucket = bucket_of(question, outcome, size)
        base = pmf_at(cdfs["linear"], bucket)
        for name, cdf in cdfs.items():
            if name != "linear":
                diffs[name].append(50 * math.log(pmf_at(cdf, bucket) / base))
        # Each member's own linear CDF against the published one: which model
        # did better on numerics.
        for model, pts in members.items():
            own = asyncio.run(final_cdf([pts], question, "linear", 1.0))
            member_scores.setdefault(model, []).append(50 * math.log(pmf_at(own, bucket) / base))

    if skipped:
        print("\nLeft out:")
        for line in skipped:
            print(f"  {line}")
    scored = len(next(iter(diffs.values())))
    print(f"\nResolved and reproduced: {scored}")
    if scored:
        print("Change in spot peer score per question against linear (positive = better):")
        print(f"  {'variant':14s} {'n':>4s} {'mean':>8s}  {'90% CI':>18s}  better/worse")
        for name, values in diffs.items():
            lo, hi = bootstrap(values) if len(values) >= 2 else (float("nan"), float("nan"))
            better = sum(v > 1e-9 for v in values)
            worse = sum(v < -1e-9 for v in values)
            print(f"  {name:14s} {len(values):4d} {np.mean(values):+8.2f}  [{lo:+7.2f}, {hi:+7.2f}]  {better}/{worse}")
        print("\nEach member alone against the published ensemble (linear, positive = member better):")
        for model, values in member_scores.items():
            print(f"  {model:44s} {len(values):4d} {np.mean(values):+8.2f}")
        print(
            "\nA public replay measured pchip at +2.39 per question (90% CI +0.21 to +4.59) on 97"
            " questions; an interval this wide needs about that many to settle."
        )


if __name__ == "__main__":
    main()
