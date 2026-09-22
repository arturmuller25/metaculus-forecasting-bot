"""
Post-cutoff eval: real forecasting skill or memorization?

The 2025 memorization probe could not tell memory from skill. BTF-3
(FutureSearch) questions dated 2026-04-29 to 2026-05-29 postdate the training
cutoffs of gpt-5.4 (Aug 2025) and sonnet-4.6 (Jan 2026), so their outcomes
cannot have been memorized. Each question ships with research frozen at that
date (no search leakage) and a FutureSearch SOTA forecast used as baseline.

Two conditions per model:
  BLIND     - question only, no research. Measures the model's own knowledge.
  RESEARCH  - with the frozen BTF-3 background. Measures the process, leak-free.

Interpretation (constant = always forecasting the base rate):
  - Post-cutoff blind Brier near the constant -> no skill without information;
    the 2025 edge (if any) was memory.
  - Post-cutoff blind Brier well below the constant -> real reasoning skill.
  - Compare each model with its own 2025 blind Brier (probe logs).

Setup (once):
  uv pip install pyarrow
  download https://huggingface.co/datasets/BTF-2/BTF-3/resolve/main/btf3_binary_questions_and_forecasts.parquet
  to data/btf3_binary.parquet (or set BTF3_PARQUET to where you saved it)

Usage: uv run python eval_post_cutoff.py [n_questions]
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import re
import statistics
import sys
from datetime import datetime

import dotenv
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
dotenv.load_dotenv(".env")

from forecasting_tools import GeneralLlm  # noqa: E402

PARQUET = os.getenv("BTF3_PARQUET", "data/btf3_binary.parquet")
MODELS = {
    "gpt-5.4": "openrouter/openai/gpt-5.4",
    "sonnet-4.6": "openrouter/anthropic/claude-sonnet-4.6",
    "fable-5.1": "openrouter/anthropic/claude-fable-5.1",
}
# 2025 blind Brier per model (from the memorization probe), for comparison
BLIND_2025 = {"gpt-5.4": 0.169, "sonnet-4.6": 0.123, "fable-5.1": 0.063}
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30


def load(n):
    rows = pq.read_table(PARQUET).to_pylist()
    out = []
    for r in rows:
        res = r.get("resolution")
        try:
            y = int(round(float(res)))
        except (TypeError, ValueError):
            continue
        if y not in (0, 1):
            continue
        sota = r.get("sota_forecast_probability")
        try:
            sota = float(sota) / 100 if sota is not None else None
        except ValueError:
            sota = None
        out.append(
            {
                "q": r["question"],
                "crit": r.get("resolution_criteria") or "",
                "bg": r.get("background") or "",
                "date": str(r.get("present_date"))[:10],
                "y": y,
                "sota": sota,
            }
        )
    # deterministic sample, evenly spread across the dataset
    step = max(1, len(out) // n)
    return out[::step][:n]


def blind_prompt(q):
    return f"""Today is {q['date']}. You have no web access. Forecast this question from your own knowledge and reasoning as of that date.

{q['q']}

Resolution criteria: {q['crit'][:600]}

Reason briefly, then end with exactly: "Probability: ZZ%" (0-100)."""


def research_prompt(q):
    return f"""Today is {q['date']}. Below is a frozen research briefing collected on that date. Use ONLY it, no web access.

QUESTION: {q['q']}
Resolution criteria: {q['crit'][:600]}

FROZEN RESEARCH (as of {q['date']}):
{q['bg'][:6000]}

Weigh the evidence, then end with exactly: "Probability: ZZ%" (0-100)."""


def parse(t):
    m = re.findall(r"Probability:\s*(\d+(?:\.\d+)?)\s*%", t)
    return min(0.99, max(0.01, float(m[-1]) / 100)) if m else None


def brier(pairs):
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs) if pairs else float("nan")


async def ask(llm, prompt):
    try:
        return parse(await llm.invoke(prompt))
    except Exception:
        return None


async def main():
    qs = load(N)
    base = sum(q["y"] for q in qs) / len(qs)
    print(f"{len(qs)} BTF-3 binary questions, dates {qs[0]['date']}..{qs[-1]['date']} (post-cutoff)")
    print(f"base rate of Yes: {base:.0%}\n")

    res = {n: {"blind": [], "research": []} for n in MODELS}
    sem = asyncio.Semaphore(4)

    async def run(name, model, q):
        blind = GeneralLlm(model=model, temperature=0.2, timeout=100)
        async with sem:
            pb = await ask(blind, blind_prompt(q))
            pr = await ask(blind, research_prompt(q))
        res[name]["blind"].append((q, pb))
        res[name]["research"].append((q, pr))

    t0 = datetime.now()
    await asyncio.gather(*(run(n, m, q) for n, m in MODELS.items() for q in qs))
    print(f"elapsed: {(datetime.now()-t0).seconds}s\n")

    ys = [q["y"] for q in qs]
    const = brier([(base, y) for y in ys])
    sota_pairs = [(q["sota"], q["y"]) for q in qs if q["sota"] is not None]
    print(f"{'':12s} {'Brier BLIND':>11s} {'Brier RSRCH':>13s} {'2025 blind':>10s} {'verdict':>28s}")
    for name in MODELS:
        pb = [(p, q["y"]) for q, p in res[name]["blind"] if p is not None]
        pr = [(p, q["y"]) for q, p in res[name]["research"] if p is not None]
        bb, br = brier(pb), brier(pr)
        v25 = BLIND_2025[name]
        # verdict
        if bb >= const - 0.02:
            verdict = "blind ~ constant: edge was info"
        elif bb < v25 - 0.03:
            verdict = "better blind than in 2025?!"
        else:
            verdict = "real skill when blind"
        print(f"{name:12s} {bb:>11.3f} {br:>13.3f} {v25:>10.3f}   {verdict:>28s}")
    print(f"{'constant':12s} {const:>11.3f}")
    print(f"{'SOTA BTF-3':12s} {'':>11s} {brier(sota_pairs):>13.3f}  (FutureSearch baseline)")

    dump = {
        n: {
            cond: [{"q": q["q"], "date": q["date"], "y": q["y"], "sota": q["sota"], "p": p} for q, p in res[n][cond]]
            for cond in ("blind", "research")
        }
        for n in MODELS
    }
    io.open("logs/eval_post_cutoff.json", "w", encoding="utf-8").write(json.dumps(dump, ensure_ascii=False, indent=1))
    print("\ndetails in logs/eval_post_cutoff.json")


asyncio.run(main())
