"""
Teste de MECANISMO da pesquisa: o que o esforco de raciocinio muda no texto
que os provedores devolvem, e quanto custa. Nao publica nada.

Nao mede acerto. Para ver 0,01 de Brier entre duas configuracoes seriam
~2000 perguntas (analise de poder sobre os dados do eval_limpo.py). Aqui a
pergunta e mais barata e ainda decide a configuracao: raciocinio a mais traz
pesquisa mais rica (mais fontes distintas, mais fatos datados)? Por quanto?

Mesmas chamadas da producao (GeneralLlm, prompt capturado do run_research,
mesmo timeout), em perguntas abertas da MiniBench e da area de testes.

Custo por chamada: tokens, pela tabela de precos do litellm (inclui a tarifa
de contexto longo). A taxa por busca (~US$ 0,01 cada) nao aparece nos tokens;
ela entra so no gasto real da chave, medido no fim.

Uso: uv run python teste_pesquisa.py   (cerca de US$ 5 com as 5 configuracoes)
"""

import asyncio
import contextvars
import io
import json
import re
import sys
import time

import dotenv
import litellm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
dotenv.load_dotenv(".env")

import forecasting_tools.ai_models.general_llm as gl  # noqa: E402
from forecasting_tools import GeneralLlm, MetaculusClient  # noqa: E402

from main import RESEARCH_MODEL, build_bot, openrouter_usage  # noqa: E402

N_PERGUNTAS = 3
TETO = 4.00  # US$ em tokens: acima disso, pula as chamadas "high" que faltarem
CLAUDE = "openrouter/anthropic/claude-sonnet-4.6:online"
CFGS = [  # rotulo, modelo, esforco (None = sem o parametro)
    ("claude-sem", CLAUDE, None),
    ("claude-low", CLAUDE, "low"),
    ("gpt-low", RESEARCH_MODEL, "low"),
    ("gpt-medium", RESEARCH_MODEL, "medium"),
    ("gpt-high", RESEARCH_MODEL, "high"),
]

# Espia as respostas do litellm para ler tokens e custo de cada chamada.
_caixa = contextvars.ContextVar("caixa", default=None)
_orig = gl.acompletion


async def _espia(*a, **k):
    r = await _orig(*a, **k)
    caixa = _caixa.get()
    if caixa is not None:
        caixa.append(r)
    return r


gl.acompletion = _espia
gasto = {"tokens": 0.0}


def riqueza(t: str) -> dict:
    urls = re.findall(r"https?://[^\s)\]>\"']+", t)
    doms = {re.sub(r"^www\.", "", u.split("/")[2]) for u in urls if len(u.split("/")) > 2}
    datas = re.findall(
        r"\b202[4-6]-\d\d-\d\d\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? 202[5-6]\b"
        r"|\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* 202[5-6]\b",
        t,
    )
    mercado = bool(re.search(r"polymarket|kalshi|manifold", t, re.I))
    return {"chars": len(t), "urls": len(urls), "dominios": len(doms), "datas": len(datas), "mercado": mercado}


def custo_tokens(r) -> float:
    try:
        return float(litellm.completion_cost(completion_response=r) or 0)
    except Exception:
        return 0.0


async def chamar(rot, modelo, esforco, prompt):
    kw = dict(model=modelo, temperature=0.1, timeout=180, allowed_tries=2)
    if esforco:
        kw["reasoning_effort"] = esforco
    caixa = []
    _caixa.set(caixa)
    t0 = time.time()
    try:
        texto = await GeneralLlm(**kw).invoke(prompt)
        erro = None
    except Exception as e:  # registra e segue
        texto, erro = "", f"{type(e).__name__}: {str(e)[:120]}"
    dt = time.time() - t0
    uso = [r.usage for r in caixa if getattr(r, "usage", None)]
    tin = sum((u.prompt_tokens or 0) for u in uso)
    tout = sum((u.completion_tokens or 0) for u in uso)
    trac = sum(
        (getattr(u.completion_tokens_details, "reasoning_tokens", 0) or 0)
        for u in uso
        if getattr(u, "completion_tokens_details", None)
    )
    custo = sum(custo_tokens(r) for r in caixa)
    gasto["tokens"] += custo
    return {"cfg": rot, "segundos": round(dt), "tokens_in": tin, "tokens_out": tout,
            "tokens_raciocinio": trac, "custo_tokens": round(custo, 4), "erro": erro,
            **riqueza(texto), "texto": texto}


async def main():
    bot = build_bot(publish=False, samples=1)
    c = MetaculusClient()
    abertas = c.get_all_open_questions_from_tournament("minibench")
    abertas += c.get_all_open_questions_from_tournament("bot-testing-area")
    normais = [
        q for q in abertas
        if not bot._meta_question_block(q).strip() and type(q).__name__ != "DateQuestion"
    ]
    por_tipo = {}  # tipos variados, deterministico
    for q in normais:
        por_tipo.setdefault(type(q).__name__, []).append(q)
    escolhidas = []
    while len(escolhidas) < N_PERGUNTAS and any(por_tipo.values()):
        for t in list(por_tipo):
            if por_tipo[t] and len(escolhidas) < N_PERGUNTAS:
                escolhidas.append(por_tipo[t].pop(0))
    print(f"{len(abertas)} abertas, {len(normais)} nao-meta. Escolhidas:")
    for q in escolhidas:
        print(f"  [{type(q).__name__}] {q.page_url} | {q.question_text[:90]}")

    # Captura o prompt EXATO da producao sem chamar API.
    capturados = []
    real = GeneralLlm.invoke

    async def pega(self, prompt, *a, **k):
        capturados.append(prompt)
        return "stub"

    prompts = []
    for q in escolhidas:
        capturados.clear()
        GeneralLlm.invoke = pega
        try:
            await bot.run_research(q)
        finally:
            GeneralLlm.invoke = real
        prompts.append(capturados[0])

    antes = openrouter_usage()

    async def por_pergunta(q, prompt):
        linhas = []
        for rot, modelo, esf in CFGS:
            if rot == "gpt-high" and gasto["tokens"] > TETO:
                linhas.append({"cfg": rot, "erro": "pulado: teto de gasto"})
                continue
            linhas.append(await chamar(rot, modelo, esf, prompt))
            print(f"  feito {rot:11s} {q.page_url}  (acumulado em tokens US$ {gasto['tokens']:.2f})", flush=True)
        return {"url": q.page_url, "tipo": type(q).__name__, "pergunta": q.question_text, "linhas": linhas}

    res = await asyncio.gather(*(por_pergunta(q, p) for q, p in zip(escolhidas, prompts)))
    await asyncio.sleep(20)  # deixa o saldo da chave assentar
    depois = openrouter_usage()

    print(f"\n{'cfg':11s} {'seg':>4s} {'tok_in':>7s} {'tok_out':>7s} {'racioc':>6s} {'US$':>6s} "
          f"{'chars':>6s} {'URLs':>4s} {'dom':>3s} {'datas':>5s} merc")
    for r in res:
        print(f"--- [{r['tipo']}] {r['pergunta'][:80]}")
        for l in r["linhas"]:
            if l.get("erro") and "chars" not in l:
                print(f"{l['cfg']:11s} {l['erro']}")
                continue
            print(f"{l['cfg']:11s} {l['segundos']:4d} {l['tokens_in']:7d} {l['tokens_out']:7d} {l['tokens_raciocinio']:6d} "
                  f"{l['custo_tokens']:6.3f} {l['chars']:6d} {l['urls']:4d} {l['dominios']:3d} {l['datas']:5d} "
                  f"{'sim' if l['mercado'] else '-'}{'  ERRO ' + l['erro'] if l.get('erro') else ''}")
    print(f"\nGasto em tokens: US$ {gasto['tokens']:.2f}")
    print(f"Gasto real na chave (tokens + taxas por busca; inclui o bot ao vivo se ele rodou junto): "
          f"US$ {depois - antes:.2f}")
    io.open("logs/teste_pesquisa.json", "w", encoding="utf-8").write(json.dumps(res, ensure_ascii=False, indent=1))
    print("detalhe (com os textos) em logs/teste_pesquisa.json")


asyncio.run(main())
