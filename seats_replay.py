"""
Scores ensembles of different sizes from answers already recorded by
forecast_replay.py on MiniBench round 1, with no new LLM calls.

Every (question, model, repetition) answer in logs/replay_r1/answers.jsonl
read the same frozen research, so a k-seat ensemble is just k of those
answers aggregated the way the bot publishes (mean for binary and multiple
choice, pointwise median of the CDFs for numeric questions). Where a
configuration can be drawn in several ways (which repetition of each model),
the score of a question is the mean over every draw, so the comparison is not
decided by which repetition happened to be picked.

Usage:
    uv run python seats_replay.py            # base prompt, every question type
    uv run python seats_replay.py --binary-variant V2
    uv run python seats_replay.py --compare base,P --reps 3,4   # production pair, two variants
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import logging
import math
import statistics

import numpy as np

import forecast_replay as fr

# Aliases as recorded by forecast_replay.py.
PROD = ("sol61", "sonnet5")
FAMILIES5 = ("sol61", "sonnet5", "opus55", "sol56", "sonnet55")


def logit(p: float) -> float:
    p = min(0.99, max(0.01, p))
    return math.log(p / (1 - p))


def binary_score(probs: list[float], outcome: int, how: str) -> float:
    clipped = [min(0.99, max(0.01, p)) for p in probs]
    if how == "mean":
        p = sum(clipped) / len(clipped)
    elif how == "median":
        p = statistics.median(clipped)
    else:  # mean in log-odds
        z = sum(logit(p) for p in clipped) / len(clipped)
        p = 1 / (1 + math.exp(-z))
    return 100 * math.log(p if outcome == 1 else 1 - p)


def draws(by_alias: dict[str, dict[int, object]], aliases: tuple[str, ...], per_model: int) -> list[list]:
    """Every way to take `per_model` answers from each alias (repetitions 1 and 2)."""
    options = []
    for alias in aliases:
        reps = [by_alias[alias][r] for r in sorted(by_alias.get(alias, {}))]
        if len(reps) < per_model:
            return []
        options.append(list(itertools.combinations(reps, per_model)))
    return [[f for group in combo for f in group] for combo in itertools.product(*options)]


async def compare_variants(args, rows, questions, latest) -> None:
    """The production pair under two variants, each question averaged over the chosen repetitions."""
    first, second = [v.strip() for v in args.compare.split(",")]
    reps = {int(r) for r in args.reps.split(",") if r.strip()} or None
    scores: dict[str, dict[int, float]] = {first: {}, second: {}}
    for pid, row in rows.items():
        for variant in (first, second):
            vals = []
            for rep in sorted(reps or {r for (_, _, _, r) in latest}):
                pair = [latest.get((pid, alias, variant, rep)) for alias in PROD]
                if any(a is None for a in pair):
                    continue
                preds = [a["prediction"] for a in pair]
                if row["type"] == "binary":
                    vals.append(binary_score(preds, row["outcome"], "mean"))
                else:
                    s, _ = await fr.score(preds, row, questions[pid])
                    vals.append(s)
            if vals:
                scores[variant][pid] = sum(vals) / len(vals)
    kinds = {pid: r["type"] for pid, r in rows.items()}
    print(f"Production pair {'+'.join(PROD)}: {second} minus {first}, repetitions {sorted(reps) if reps else 'all'}\n")
    for group, posts in (
        ("binary", {p for p, k in kinds.items() if k == "binary"}),
        ("numeric+discrete", {p for p, k in kinds.items() if k in ("numeric", "discrete")}),
        ("multiple_choice", {p for p, k in kinds.items() if k == "multiple_choice"}),
        ("all", set(rows)),
    ):
        common = set(scores[first]) & set(scores[second]) & posts
        if not common:
            continue
        d = fr.paired(scores[second], scores[first], common)
        print(f"  {group:18s} n={d['n']:2d} {first} {np.mean([scores[first][p] for p in common]):8.2f}"
              f" {second} {np.mean([scores[second][p] for p in common]):8.2f}  diff {d['mean']:+.2f}"
              f" (90% {d['lo']:+.2f} to {d['hi']:+.2f}; better {d['better']}, worse {d['worse']})")


async def main_async(args) -> None:
    logging.getLogger("forecasting_tools").setLevel(logging.ERROR)
    rows = {r["post_id"]: r for r in fr.load_questions() if r["outcome"] is not None}
    questions = {pid: fr.question_object(r) for pid, r in rows.items()}
    latest: dict[tuple, dict] = {}
    for a in fr.recorded():
        if a["status"] == "ok" and a["post_id"] in rows:
            latest[(a["post_id"], a["alias"], a["variant"], a["rep"])] = a

    if args.compare:
        await compare_variants(args, rows, questions, latest)
        return

    configs = {
        "2 seats: sol61 + sonnet5 (production)": (PROD, 1),
        "4 seats: sol61 x2 + sonnet5 x2": (PROD, 2),
        "5 seats: five models": (FAMILIES5, 1),
        "10 seats: five models x2": (FAMILIES5, 2),
        "1 seat: sol61": (("sol61",), 1),
        "1 seat: sonnet5": (("sonnet5",), 1),
    }
    if args.binary_variant != "base":
        configs = {
            f"2 seats: sol56 + sonnet5 {args.binary_variant}": (("sol56", "sonnet5"), 1),
            f"4 seats: sol56 x2 + sonnet5 x2 {args.binary_variant}": (("sol56", "sonnet5"), 2),
        }

    scores: dict[str, dict[int, float]] = {}
    for pid, row in rows.items():
        kind = row["type"]
        variant = args.binary_variant if kind == "binary" else "base"
        if args.binary_variant != "base" and kind != "binary":
            continue
        by_alias: dict[str, dict[int, object]] = {}
        for (p, alias, v, rep), a in latest.items():
            if p == pid and v == variant:
                by_alias.setdefault(alias, {})[rep] = a["prediction"]
        for name, (aliases, per_model) in configs.items():
            combos = draws(by_alias, aliases, per_model)
            if not combos:
                continue
            if kind == "binary":
                for how in ("mean", "median", "logodds"):
                    vals = [binary_score(c, row["outcome"], how) for c in combos]
                    scores.setdefault(f"{name} [{how}]", {})[pid] = sum(vals) / len(vals)
            else:
                vals = []
                for c in combos:
                    s, _ = await fr.score(c, row, questions[pid])
                    vals.append(s)
                scores.setdefault(name, {})[pid] = sum(vals) / len(vals)

    # One table per question group. Binary questions have a row per
    # aggregation; "all" uses the bot's own binary aggregation, the mean.
    kinds = {pid: r["type"] for pid, r in rows.items()}
    reference = next(iter(configs))
    combined: dict[str, dict[int, float]] = {}
    for name in configs:
        merged = dict(scores.get(name, {}))
        merged.update(scores.get(f"{name} [mean]", {}))
        if merged:
            combined[name] = merged
    groups = {
        "binary": ({p for p, k in kinds.items() if k == "binary"}, f"{reference} [mean]", scores),
        "numeric+discrete": ({p for p, k in kinds.items() if k in ("numeric", "discrete")}, reference, scores),
        "multiple_choice": ({p for p, k in kinds.items() if k == "multiple_choice"}, reference, scores),
        "all (binary by mean)": (set(rows), reference, combined),
    }
    print(f"Questions scored: {len(rows)}; reference row: {reference}\n")
    for group, (posts, ref_name, table) in groups.items():
        ref = table.get(ref_name)
        if not ref or not posts & set(ref):
            continue
        print(f"== {group} ({len(posts & set(ref))} questions)")
        for name, by_post in table.items():
            common = set(by_post) & posts
            if not common:
                continue
            mean = float(np.mean([by_post[p] for p in common]))
            line = f"  {name:58s} n={len(common):2d} mean log score {mean:7.2f}"
            if name != ref_name:
                d = fr.paired(by_post, ref, posts)
                if d:
                    line += (f"  vs reference {d['mean']:+.2f} (90% {d['lo']:+.2f} to {d['hi']:+.2f};"
                             f" better {d['better']}, worse {d['worse']})")
            print(line)
        print()


def main_cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary-variant", default="base")
    parser.add_argument("--compare", default="", help="two variants to compare on the production pair, e.g. base,P")
    parser.add_argument("--reps", default="", help="repetitions to use with --compare, e.g. 3,4")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main_cli()
