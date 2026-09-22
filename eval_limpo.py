"""
Diagnostico limpo: habilidade real ou memoria?

O problema: na sonda de 2025, um Brier baixo podia ser memoria ou talento.
Aqui separamos os dois, usando o BTF-3 (FutureSearch): perguntas com data de
referencia entre 29/04 e 29/05 de 2026, DEPOIS do corte de treino do gpt-5.4
(ago/2025) e do sonnet-4.6 (jan/2026). Nenhum dos dois pode ter decorado o
resultado. Cada pergunta traz pesquisa congelada na data (a prova contra
vazamento por busca) e um forecast SOTA da propria FutureSearch como baseline.

Duas condicoes por modelo:
  CEGO      - so a pergunta, sem pesquisa. Mede conhecimento puro do modelo.
  PESQUISA  - com o background congelado do BTF-3. Mede o processo, sem vazar.

O diagnostico:
  - Brier cego pos-corte perto da constante -> sem habilidade sem informacao;
    o edge de 2025 (se houve) era memoria.
  - Brier cego pos-corte bem abaixo da constante -> talento real de raciocinio.
  - Comparar cada modelo com o proprio Brier cego de 2025 (logs da sonda).

Preparo (uma vez):
  uv pip install pyarrow
  baixe https://huggingface.co/datasets/BTF-2/BTF-3/resolve/main/btf3_binary_questions_and_forecasts.parquet
  para data/btf3_binary.parquet (ou aponte BTF3_PARQUET para onde salvou)

Uso: uv run python eval_limpo.py [n_perguntas]
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
# Brier cego de 2025 (da sonda), para comparar
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
    # amostra determinística espalhada
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
    print(f"{len(qs)} binarias do BTF-3, datas {qs[0]['date']}..{qs[-1]['date']} (pos-corte)")
    print(f"taxa-base de Sim: {base:.0%}\n")

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
    print(f"tempo: {(datetime.now()-t0).seconds}s\n")

    ys = [q["y"] for q in qs]
    const = brier([(base, y) for y in ys])
    sota_pairs = [(q["sota"], q["y"]) for q in qs if q["sota"] is not None]
    print(f"{'':12s} {'Brier CEGO':>11s} {'Brier c/PESQ':>13s} {'2025 cego':>10s} {'veredito':>28s}")
    for name in MODELS:
        pb = [(p, q["y"]) for q, p in res[name]["blind"] if p is not None]
        pr = [(p, q["y"]) for q, p in res[name]["research"] if p is not None]
        bb, br = brier(pb), brier(pr)
        v25 = BLIND_2025[name]
        # veredito
        if bb >= const - 0.02:
            verd = "cego ~ constante: edge era info"
        elif bb < v25 - 0.03:
            verd = "melhor as cegas que em 2025?!"
        else:
            verd = "habilidade real as cegas"
        print(f"{name:12s} {bb:>11.3f} {br:>13.3f} {v25:>10.3f}   {verd:>28s}")
    print(f"{'constante':12s} {const:>11.3f}")
    print(f"{'SOTA BTF-3':12s} {'':>11s} {brier(sota_pairs):>13.3f}  (baseline da FutureSearch)")

    dump = {
        n: {
            cond: [{"q": q["q"], "date": q["date"], "y": q["y"], "sota": q["sota"], "p": p} for q, p in res[n][cond]]
            for cond in ("blind", "research")
        }
        for n in MODELS
    }
    io.open("logs/eval_limpo.json", "w", encoding="utf-8").write(json.dumps(dump, ensure_ascii=False, indent=1))
    print("\ndetalhe em logs/eval_limpo.json")


asyncio.run(main())
