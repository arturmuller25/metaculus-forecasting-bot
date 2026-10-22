"""
Scores the offline panel experiments recorded by forecast_replay.py against
the production pair, question by question, with no model calls:

  agent3     pool of the pair and the time-travel agent (tt55x)      enters at +1.0/q, CI90 lower > 0
  sol+agent  GPT-6.1 Sol and the agent, the agent in Sonnet's seat   read only
  agent-son  the agent alone minus Sonnet 5.5 alone                   read only
  G          the pair answering with the gap-fill findings            enters at +1.0/q, CI90 lower > 0
  supG       supervisor over all analyses, gap-fill answers included  enters at +1.5/q, CI90 lower > 0
  sup        supervisor over the pair and the agent                   enters at +1.5/q, CI90 lower > 0
  pool5      pool of the pair, the pair with gap-fill, the agent      enters at +1.0/q, CI90 lower > 0
  C          the pair with the soft-clock rule (binaries)             enters at +0.5/q, CI90 lower > 0

Rules fixed before the results (2026-10-21 and 22). Binary members are pooled
in log-odds, numeric ones by the pointwise median of their CDFs, multiple
choice by the mean, as production does. The production pair is averaged over
its recorded runs; every candidate uses its run 1. Every table is shown with
and without multiple choice: in rounds 1 and 2 the pair's multiple-choice
answers predate the 2026-10-20 multiple-choice prompt the candidates used,
so on those questions a candidate's gain mixes in the prompt change.

Usage:
    uv run python panel_score.py --round 33125:logs/replay_r1:3,4:1,2 --round 33129:logs/replay_r2:1,2:1,2
Each --round is project:replay folder:GPT-6.1 Sol runs:Sonnet 5.5 runs of the
pair (round 1's Sol answers on the current prompt are runs 3 and 4).
Per-question differences go to logs/panel_scores/<project>.json.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import math
import os

import numpy as np

KEYS = ("agent3", "sol+agent", "agent-son", "G", "supG", "sup", "pool5", "C")
OUT = os.path.join("logs", "panel_scores")


def boot(d: list[float]) -> str:
    a = np.array(d)
    means = np.random.default_rng(0).choice(a, size=(20000, len(a)), replace=True).mean(axis=1)
    return (f"{a.mean():+.2f} [{np.percentile(means, 5):+.2f}, {np.percentile(means, 95):+.2f}]"
            f" better/worse {int((a > 1e-9).sum())}/{int((a < -1e-9).sum())} n={len(a)}")


def group_of(kind: str) -> str:
    return kind if kind in ("binary", "multiple_choice") else "numeric+discrete"


def score_round(project: str, folder: str, sol_runs: list[int], son_runs: list[int]) -> dict[str, dict[int, tuple[str, float]]]:
    os.environ["REPLAY_PROJECT"], os.environ["REPLAY_OUT"] = project, folder
    import bot
    import forecast_replay as fr

    importlib.reload(fr)
    rows = {r["post_id"]: r for r in fr.load_questions() if r["outcome"] is not None}
    latest = {}
    for a in fr.recorded():
        if a["status"] == "ok" and a["post_id"] in rows:
            latest[(a["post_id"], a["alias"], a["variant"], a["rep"])] = a["prediction"]

    async def score(forecasts, row, q) -> float:
        if row["type"] == "binary":
            p = bot._logodds_mean(forecasts)
            return 100 * math.log(p if row["outcome"] == 1 else 1 - p)
        s, _ = await fr.score(forecasts, row, q)
        return s

    diffs: dict[str, dict[int, tuple[str, float]]] = {k: {} for k in KEYS}

    async def go() -> None:
        for pid, row in rows.items():
            q = fr.question_object(row)
            pairs = [(latest.get((pid, "sol61", "base", a)), latest.get((pid, "sonnet55d", "base", b)))
                     for a, b in zip(sol_runs, son_runs)]
            pairs = [p for p in pairs if None not in p]
            if not pairs:
                continue
            base_scores = [await score(list(p), row, q) for p in pairs]
            base = sum(base_scores) / len(base_scores)
            agent = latest.get((pid, "tt55x", "base", 1))
            g = [latest.get((pid, "sol61", "G", 1)), latest.get((pid, "sonnet55d", "G", 1))]
            c = [latest.get((pid, "sol61", "C", 1)), latest.get((pid, "sonnet55d", "C", 1))]
            sup_g, sup = latest.get((pid, "ttsup", "G", 1)), latest.get((pid, "ttsup", "base", 1))
            cands = {
                "G": g if None not in g else None,
                "supG": [sup_g] if sup_g is not None else None,
                "sup": [sup] if sup is not None else None,
                "pool5": list(pairs[0]) + g + [agent] if None not in g and agent is not None else None,
                "C": c if None not in c else None,
            }
            kind = group_of(row["type"])
            for k, f in cands.items():
                if f:
                    diffs[k][pid] = (kind, await score(f, row, q) - base)
            if agent is not None:
                # Each pair run with the agent's run 1, then averaged.
                three, swap, alone = [], [], []
                for sol, son in pairs:
                    pair = await score([sol, son], row, q)
                    three.append(await score([sol, son, agent], row, q) - pair)
                    swap.append(await score([sol, agent], row, q) - pair)
                    alone.append(await score([agent], row, q) - await score([son], row, q))
                diffs["agent3"][pid] = (kind, sum(three) / len(three))
                diffs["sol+agent"][pid] = (kind, sum(swap) / len(swap))
                diffs["agent-son"][pid] = (kind, sum(alone) / len(alone))

    asyncio.run(go())
    return diffs


def report(title: str, diffs: dict[str, dict[int, tuple[str, float]]]) -> None:
    print(f"\n===== {title}")
    for k in KEYS:
        items = list(diffs[k].values())
        if not items:
            continue
        for g in ("all", "all but multiple choice", "binary", "numeric+discrete", "multiple_choice"):
            d = [v for kind, v in items if g == "all" or kind == g or (g == "all but multiple choice" and kind != "multiple_choice")]
            if d:
                print(f"{k:10s} {g:24s} {boot(d)}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--round", action="append", required=True,
                        help="project:replay folder:Sol runs:Sonnet runs, e.g. 33129:logs/replay_r2:1,2:1,2")
    args = parser.parse_args()
    logging.disable(logging.WARNING)
    os.makedirs(OUT, exist_ok=True)
    pooled: dict[str, dict] = {k: {} for k in KEYS}
    for spec in args.round:
        project, folder, sol, son = spec.split(":")
        diffs = score_round(project, folder, [int(x) for x in sol.split(",")], [int(x) for x in son.split(",")])
        with open(os.path.join(OUT, f"{project}.json"), "w", encoding="utf-8") as fh:
            json.dump({k: {str(p): v for p, v in by.items()} for k, by in diffs.items()}, fh)
        report(f"round {project}", diffs)
        for k in KEYS:
            pooled[k].update({(project, p): v for p, v in diffs[k].items()})
    if len(args.round) > 1:
        report("all rounds", pooled)


if __name__ == "__main__":
    main()
