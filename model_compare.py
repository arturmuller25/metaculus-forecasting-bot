"""
Compares forecasting models on a replayed round, with no new calls: every
model alone and every pair with GPT-6.1 Sol, each question averaged over the
repetitions available, against the production pair (GPT-6.1 Sol + Sonnet 5).

Usage:
    REPLAY_PROJECT=33129 REPLAY_OUT=logs/replay_r2 uv run python model_compare.py
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np

import forecast_replay as fr

SINGLES = ("sol61", "sonnet5d", "opus55d", "sonnet55d", "fable51d")
PARTNERS = ("sonnet5d", "opus55d", "sonnet55d", "fable51d")
REFERENCE = "sol61 + sonnet5d"


async def main_async() -> None:
    logging.getLogger("forecasting_tools").setLevel(logging.ERROR)
    rows = {r["post_id"]: r for r in fr.load_questions() if r["outcome"] is not None}
    questions = {pid: fr.question_object(r) for pid, r in rows.items()}
    answers: dict[tuple, object] = {}
    for a in fr.recorded():
        if a["status"] == "ok" and a["variant"] == "base" and a["post_id"] in rows:
            answers[(a["post_id"], a["alias"], a["rep"])] = a["prediction"]
    reps = sorted({r for (_, _, r) in answers})

    configs: dict[str, list[tuple[str, ...]]] = {s: [(s,)] for s in SINGLES}
    for p in PARTNERS:
        configs[f"sol61 + {p}"] = [("sol61", p)]

    async def scored(preds: list, row: dict, question) -> float | None:
        """As production publishes: a member whose CDF cannot be built is dropped."""
        try:
            s, _ = await fr.score(preds, row, question)
            return s
        except ValueError:
            usable = []
            for p in preds:
                try:
                    await fr.score([p], row, question)
                    usable.append(p)
                except ValueError:
                    unbuildable[0] += 1
            if not usable:
                return None
            s, _ = await fr.score(usable, row, question)
            return s

    unbuildable = [0]
    scores: dict[str, dict[int, float]] = {}
    for name, (members,) in ((n, c) for n, c in configs.items()):
        before = unbuildable[0]
        for pid, row in rows.items():
            vals = []
            for rep in reps:
                preds = [answers.get((pid, m, rep)) for m in members]
                if any(x is None for x in preds):
                    continue
                s = await scored(preds, row, questions[pid])
                if s is not None:
                    vals.append(s)
            if vals:
                scores.setdefault(name, {})[pid] = sum(vals) / len(vals)
        if unbuildable[0] > before:
            print(f"{name}: {unbuildable[0] - before} member forecasts could not be built (dropped, as in production)")

    kinds = {pid: r["type"] for pid, r in rows.items()}
    ref = scores.get(REFERENCE, {})
    for group, posts in (
        ("binary", {p for p, k in kinds.items() if k == "binary"}),
        ("numeric+discrete", {p for p, k in kinds.items() if k in ("numeric", "discrete")}),
        ("multiple_choice", {p for p, k in kinds.items() if k == "multiple_choice"}),
        ("all", set(rows)),
    ):
        print(f"\n== {group} ({len(posts)} questions); reference {REFERENCE}")
        for name, by_post in scores.items():
            common = set(by_post) & posts
            if not common:
                continue
            line = f"  {name:22s} n={len(common):2d} mean {np.mean([by_post[p] for p in common]):8.2f}"
            if name != REFERENCE and ref:
                d = fr.paired(by_post, ref, posts)
                if d:
                    line += (f"  vs reference {d['mean']:+.2f} (90% {d['lo']:+.2f} to {d['hi']:+.2f};"
                             f" better {d['better']}, worse {d['worse']})")
            print(line)
    print(f"\nRepetitions used: {reps}")


if __name__ == "__main__":
    asyncio.run(main_async())
