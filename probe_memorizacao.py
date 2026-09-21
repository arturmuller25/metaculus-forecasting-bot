"""
Sonda de memorização: os modelos do bot decoraram as perguntas de 2025?

Motivo: o usuário quer usar perguntas já resolvidas para testar e calibrar
o bot. Isso só vale se o modelo NÃO souber o resultado por memória. Em vez
de assumir, medimos.

Duas sondas por modelo, sem busca web (modelos sem :online não têm ferramenta):

  A. RECALL  - pergunta direto se o modelo lembra da resolução.
  B. BLIND   - pede previsão "como se fosse a data de publicação".

Comparações:
  - Brier da previsão às cegas vs Brier da mediana dos profissionais
    (que tinham pesquisa mas não tinham o futuro) nas MESMAS perguntas.
  - Acurácia do recall explícito vs taxa-base.

Se um modelo de 2026 bate os profissionais às cegas em perguntas de
jan-abr/2025, é memória, e o conjunto não serve para calibrar.
Uso: uv run python probe_memorizacao.py [n_perguntas]
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime

import dotenv

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
dotenv.load_dotenv(".env")

from forecasting_tools import GeneralLlm  # noqa: E402

PRO_CSV = (
    r"C:\Users\Usuario\AppData\Local\Temp\claude\c--Users-Usuario-projetos-cripto-farm"
    r"\6045c2ce-e058-471a-8f3b-5c3e96264001\scratchpad\pro.csv"
)
MODELS = {
    "gpt-5.4": "openrouter/openai/gpt-5.4",
    "sonnet-4.6": "openrouter/anthropic/claude-sonnet-4.6",
    "fable-5.1": "openrouter/anthropic/claude-fable-5.1",
}
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30


def load_questions():
    """Perguntas binárias resolvidas + mediana dos profissionais (último palpite de cada)."""
    rows = list(csv.DictReader(io.open(PRO_CSV, encoding="utf-8")))
    by_q = defaultdict(list)
    for r in rows:
        if r["type"] != "binary" or r["resolution"] not in ("yes", "no"):
            continue
        by_q[r["question_id"]].append(r)
    out = []
    for qid, rs in by_q.items():
        last = {}
        for r in rs:  # último palpite de cada profissional
            f = r["forecaster"]
            if f not in last or r["created_at"] > last[f]["created_at"]:
                last[f] = r
        probs = [float(r["probability_yes"]) for r in last.values() if r["probability_yes"]]
        if len(probs) < 3:
            continue
        r0 = rs[0]
        out.append(
            {
                "id": qid,
                "title": r0["question_title"],
                "published": r0["published_at"][:10],
                "resolved": r0["actual_resolve_time"][:10],
                "y": 1 if r0["resolution"] == "yes" else 0,
                "pro_median": statistics.median(probs),
                "n_pros": len(probs),
            }
        )
    out.sort(key=lambda q: q["id"])
    return out


def recall_prompt(q):
    return f"""This is a Metaculus forecasting question that was published on {q['published']} and resolved on {q['resolved']}.

Question: {q['title']}

Do you actually remember, from your training data, how this specific question resolved? Be honest: distinguish genuine recall of the outcome from a guess based on reasoning.

Answer in exactly this format:
RECALL: yes or no
OUTCOME: yes, no, or unknown
CONFIDENCE: a number 0-100 that you truly remember the outcome"""


def blind_prompt(q):
    return f"""Pretend today is {q['published']}. You have no access to any information after that date and no web access.

Forecast this Metaculus question as a careful forecaster would have on that day:

{q['title']}

Reason briefly from what would have been known on {q['published']}, then end with exactly: "Probability: ZZ%" (0-100)."""


async def ask(llm, prompt, retries=2):
    for i in range(retries + 1):
        try:
            return await llm.invoke(prompt)
        except Exception as e:
            if i == retries:
                return f"ERRO: {type(e).__name__}"
            await asyncio.sleep(3)


def parse_recall(t):
    rec = re.search(r"RECALL:\s*(yes|no)", t, re.I)
    out = re.search(r"OUTCOME:\s*(yes|no|unknown)", t, re.I)
    conf = re.search(r"CONFIDENCE:\s*(\d+)", t)
    return (
        rec.group(1).lower() if rec else None,
        out.group(1).lower() if out else None,
        int(conf.group(1)) if conf else None,
    )


def parse_prob(t):
    m = re.findall(r"Probability:\s*(\d+(?:\.\d+)?)\s*%", t)
    if not m:
        return None
    return min(0.99, max(0.01, float(m[-1]) / 100))


def brier(pairs):
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs)


async def main():
    qs = load_questions()[:N]
    print(f"{len(qs)} perguntas binárias do Q1 2025 com mediana de profissionais")
    base = sum(q["y"] for q in qs) / len(qs)
    print(f"taxa-base de Sim nesta amostra: {base:.0%}\n")

    results = {name: {"recall": [], "blind": []} for name in MODELS}
    sem = asyncio.Semaphore(4)

    async def run_one(name, model, q):
        llm = GeneralLlm(model=model, temperature=0, timeout=120, allowed_tries=1)
        async with sem:
            a = await ask(llm, recall_prompt(q))
            b = await ask(llm, blind_prompt(q))
        results[name]["recall"].append((q, parse_recall(a), a))
        results[name]["blind"].append((q, parse_prob(b), b))

    tasks = [run_one(n, m, q) for n, m in MODELS.items() for q in qs]
    t0 = datetime.now()
    await asyncio.gather(*tasks)
    print(f"tempo: {(datetime.now()-t0).seconds}s\n")

    pro_pairs = [(q["pro_median"], q["y"]) for q in qs]
    print(f"{'modelo':12s} {'recall=sim':>10s} {'acerto recall':>13s} {'Brier cego':>10s} {'Brier pros':>10s} {'const 35%':>10s}")
    for name in MODELS:
        rec = results[name]["recall"]
        said_yes = [(q, o, c) for q, (r, o, c), _ in rec if r == "yes"]
        acc = (
            sum(1 for q, o, _ in said_yes if o == ("yes" if q["y"] else "no")) / len(said_yes)
            if said_yes
            else float("nan")
        )
        blind = [(p, q["y"]) for q, p, _ in results[name]["blind"] if p is not None]
        print(
            f"{name:12s} {len(said_yes):>4d}/{len(rec):<5d} {acc:>13.0%} "
            f"{brier(blind):>10.3f} {brier(pro_pairs):>10.3f} {brier([(0.35, q['y']) for q in qs]):>10.3f}"
        )

    # salva tudo para inspeção
    dump = {
        name: {
            "recall": [{"id": q["id"], "title": q["title"], "y": q["y"], "parsed": pr, "raw": raw[:600]} for q, pr, raw in results[name]["recall"]],
            "blind": [{"id": q["id"], "title": q["title"], "y": q["y"], "pro": q["pro_median"], "p": p, "raw": raw[-400:]} for q, p, raw in results[name]["blind"]],
        }
        for name in MODELS
    }
    io.open("logs/probe_memorizacao.json", "w", encoding="utf-8").write(json.dumps(dump, ensure_ascii=False, indent=1))
    print("\ndetalhe em logs/probe_memorizacao.json")

    # exemplos de recall explícito
    for name in MODELS:
        ex = [(q, o, c) for q, (r, o, c), _ in results[name]["recall"] if r == "yes"][:3]
        if ex:
            print(f"\n{name} diz que LEMBRA de, por exemplo:")
            for q, o, c in ex:
                ok = "certo" if o == ("yes" if q["y"] else "no") else "ERRADO"
                print(f"  [{ok}] conf {c}: {q['title'][:90]}")


asyncio.run(main())
