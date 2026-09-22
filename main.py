"""
Ponto de entrada do bot.

Uso tipico:
    uv run python main.py --mode test              # smoke test, nao publica
    uv run python main.py --mode tournament        # torneio principal + MiniBench
    uv run python main.py --mode tournament --publish

Sem --publish o bot calcula tudo e salva os relatorios em logs/, mas nao
envia nada para a Metaculus. Rode assim ate confiar na saida.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

import dotenv

# O console do Windows usa cp1252 por padrao. Os textos das perguntas da
# Metaculus vem com simbolos como "≥", "—" e acentos, e o logger quebra com
# UnicodeEncodeError no meio de uma execucao. Forca UTF-8 nas duas saidas.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

dotenv.load_dotenv()

from forecasting_tools import GeneralLlm, MetaculusClient, MonetaryCostManager

from bot import ForecasterBot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------

# O modelo base e a alavanca que mais move o placar. Troque aqui.
#
# Dois caminhos, escolhidos automaticamente pelas chaves que existem no .env:
#
# 1. COM OPENROUTER (quando os creditos do torneio chegarem)
#    Prefixo openrouter/. O sufixo :online liga busca web no proprio
#    OpenRouter, evitando contratar AskNews, Exa ou Perplexity so para pesquisar.
#
# 2. SO COM O TOKEN DA METACULUS (da para comecar hoje)
#    Prefixo metaculus/. A biblioteca redireciona para
#    llm-proxy.metaculus.com/proxy/anthropic quando o nome tem "claude" ou
#    "anthropic", e para .../proxy/openai/v1 no resto, autenticando com o
#    proprio METACULUS_TOKEN. Nao ha :online aqui, entao a pesquisa sai sem
#    busca web: serve para testar o encanamento, nao para competir.

_HAS_OPENROUTER = bool(os.getenv("OPENROUTER_API_KEY"))

if _HAS_OPENROUTER:
    # GPT-5.x como modelo de previsao final e o sinal mais forte que se repete
    # nas duas ultimas temporadas do torneio: r=+0.42 na tabela da Metaculus.
    # Claude Opus, na mesma tabela, ficou em r=-0.01. Isso nao quer dizer que
    # Claude seja ruim, quer dizer que quem usou GPT-5.x na decisao final
    # pontuou melhor. Por isso ele decide, e os outros entram no ensemble.
    _FORECAST = "openrouter/openai/gpt-5.4"
    _RESEARCH = "openrouter/openai/gpt-5.4:online"
    _PARSER = "openrouter/openai/gpt-4o-mini"
    # Sem Google no ensemble, e nao por escolha. Medido em 2026-09-18 com a
    # chave doada pela Metaculus:
    #   - gemini-3.1-pro pelo AI Studio: cota ZERO
    #     ("free_tier_input_token_count, limit: 0")
    #   - gemini-3.8-flash pelo AI Studio: free tier, 20 requisicoes/min
    #     ("free_tier_requests, limit: 20"), estoura com ensemble + parser
    #   - rota Vertex, que teria cota: bloqueada pela chave
    #     ("allowed-providers setting permits only: openai, anthropic,
    #     google-ai-studio")
    #   - e o Flash e modelo de raciocinio: devolve so os tokens de
    #     raciocinio, sem a linha "Probability", e o parser descarta.
    # Duas familias ja e heterogeneo. Se a Metaculus liberar cota do Google,
    # e so acrescentar aqui e testar de novo.
    #
    # Ensemble SEM :online, e a diversidade vem da PESQUISA, nao de cada
    # membro buscar de novo. Motivo medido em 2026-09-22: com os membros em
    # :online eram 4 buscas por pergunta (2 pesquisa + 2 ensemble) e o custo
    # real bateu US$ 4,38 numa unica pergunta, insustentavel. E as buscas
    # dos membros repetiam os mesmos dois motores que a pesquisa ja usa.
    # (No BTF-3 o gpt-5.4 foi de 0,159 as cegas para 0,189 com pesquisa, mas
    # isso e ruido: t=-0,94 em 30 perguntas. Nao conta como evidencia.)
    # Entao: a pesquisa (com varios provedores distintos) faz a busca uma
    # vez, e os dois modelos do ensemble raciocinam sobre esse briefing ja
    # diverso. Diversidade de familia no ensemble, de fonte na pesquisa.
    _ENSEMBLE = [
        "openrouter/openai/gpt-5.4",
        "openrouter/anthropic/claude-sonnet-4.6",
    ]
else:
    _FORECAST = "metaculus/claude-sonnet-4-5"
    _RESEARCH = "metaculus/gpt-4o-search-preview"
    _PARSER = "metaculus/gpt-4o-mini"
    # O proxy da Metaculus so roteia Anthropic e OpenAI, entao o ensemble
    # possivel aqui tem duas familias, nao tres.
    _ENSEMBLE = [
        "metaculus/claude-sonnet-4-5",
        "metaculus/gpt-4o",
    ]

FORECAST_MODEL = os.getenv("FORECAST_MODEL", _FORECAST)
RESEARCH_MODEL = os.getenv("RESEARCH_MODEL", _RESEARCH)
PARSER_MODEL = os.getenv("PARSER_MODEL", _PARSER)

# Teto de gasto por execucao, em dolares. O processo aborta ao estourar.
MAX_COST_PER_RUN = float(os.getenv("MAX_COST_PER_RUN", "5.00"))

# Esforco de raciocinio dos modelos que pensam (decisor, pesquisador, ensemble).
# "high" e o achado mais forte de todo o material: nos pares "high" contra
# "low" da Metaculus, o high venceu 8 de 8 vezes (p=0,004), um dos poucos
# resultados que sobrevive a correcao de comparacoes multiplas. Custa mais
# tokens de raciocinio, entao fica configuravel: REASONING_EFFORT=medium
# ou low no .env se o orcamento apertar. O parser (gpt-4o-mini) fica de fora,
# nao raciocina e nao aceita o parametro.
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "high").strip()
# A pesquisa usa raciocinio BAIXO de proposito. O esforco alto foi medido
# como valioso na PREVISAO final (8 de 8), nao na coleta de evidencia. E o
# teste de 2026-09-22 (teste_pesquisa.py, 3 perguntas, mesmo prompt, custo
# em tokens; as taxas por busca somaram ~US$ 0,13 por chamada em media, nao
# medidas uma a uma):
# baixo US$ 0,30 por chamada, medio 0,49, alto 0,59 (alto medido em uma).
# Medio e alto nao trouxeram pesquisa mais rica: em URLs, dominios e datas o
# baixo empatou ou venceu, fora 1 URL e 1 dominio a mais do alto numa
# pergunta. Raciocinio a mais aqui encurta o relatorio e cobra mais.
# Nao mede acerto (seriam ~2000 perguntas para ver 0,01 de Brier).
RESEARCH_REASONING = os.getenv("RESEARCH_REASONING", "low").strip()


def _thinker(model: str, temperature: float, timeout: int, effort: str | None = None) -> GeneralLlm:
    """GeneralLlm com esforco de raciocinio, para modelos que pensam."""
    effort = REASONING_EFFORT if effort is None else effort
    kwargs = dict(model=model, temperature=temperature, timeout=timeout, allowed_tries=2)
    if effort in ("low", "medium", "high"):
        kwargs["reasoning_effort"] = effort
    return GeneralLlm(**kwargs)

TOURNAMENT_URLS = {
    "tournament": "https://www.metaculus.com/tournament/fall-futureeval-2026/",
    "minibench": "https://www.metaculus.com/aib/minibench",
    "cup": "https://www.metaculus.com/tournament/metaculus-cup/",
    "market_pulse": "https://www.metaculus.com/tournament/market-pulse-26q4/",
    "test": "https://www.metaculus.com/tournament/bot-testing-area/",
}


def build_bot(publish: bool, samples: int) -> ForecasterBot:
    llms = {
        "default": _thinker(FORECAST_MODEL, 0.3, 120),
        "researcher": _thinker(RESEARCH_MODEL, 0.1, 180, effort=RESEARCH_REASONING),
        "parser": GeneralLlm(model=PARSER_MODEL, temperature=0.0, timeout=60, allowed_tries=2),
        "summarizer": GeneralLlm(model=PARSER_MODEL, temperature=0.0, timeout=60),
    }
    # O ensemble e a mudanca mais disputada do bot: um experimento controlado
    # em 202 perguntas deste torneio mostrou ganho com modelos DIFERENTES
    # (Brier 0,162 -> 0,153), mas a tabela de correlacao da Spring 2026 pontua
    # "agregar previsoes" em r=-0,19. Evidencia conflitante, entao fica um
    # interruptor: ENSEMBLE=0 no .env desliga e volta ao modelo unico.
    # Meca voce mesmo antes de acreditar em qualquer um dos dois estudos.
    # ENSEMBLE_MODELS="a,b,c" no .env sobrescreve a lista, para experimentos.
    _override = os.getenv("ENSEMBLE_MODELS", "").strip()
    _members = [m.strip() for m in _override.split(",") if m.strip()] if _override else _ENSEMBLE
    ensemble = (
        []
        if os.getenv("ENSEMBLE", "1") == "0"
        else [_thinker(m, 0.3, 120) for m in _members]
    )

    # Shadow models: forecast every question on the same research, recorded to
    # logs/forecasts.jsonl but never published. SHADOW_MODELS="model[@effort],..."
    # e.g. "openrouter/openai/gpt-5.4@medium" to test a cheaper forecaster.
    shadows = []
    for spec in (s.strip() for s in os.getenv("SHADOW_MODELS", "").split(",")):
        if spec:
            model, _, effort = spec.partition("@")
            shadows.append((spec, _thinker(model.strip(), 0.3, 120, effort=effort.strip() or None)))

    return ForecasterBot(
        ensemble=ensemble,
        shadows=shadows,
        # 1 pesquisa por pergunta, N previsoes sobre ela, agregadas.
        # Este e o formato do bot de referencia da propria Metaculus.
        research_reports_per_question=1,
        predictions_per_research_report=samples,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish,
        folder_to_save_reports_to="logs/",
        # NAO mexa nisso em modo publicacao. Regra da Metaculus para os
        # torneios so-de-bot, verbatim: "Bot makers should only submit one
        # forecast per question in these bot-only tournaments."
        # Em modo teste roda sempre, porque na area de testes reenviar e
        # justamente o que se quer.
        skip_previously_forecasted_questions=publish,
        extra_metadata_in_explanation=True,
        llms=llms,
    )


def check_env(publish: bool) -> None:
    # METACULUS_TOKEN e o unico realmente obrigatorio: sem chave de provedor
    # o bot cai no proxy da Metaculus, que usa esse mesmo token.
    if not os.getenv("METACULUS_TOKEN"):
        print("Faltando METACULUS_TOKEN.", file=sys.stderr)
        print("Copie .env.example para .env e preencha. Detalhes no README.md.", file=sys.stderr)
        sys.exit(1)

    if not _HAS_OPENROUTER:
        print("Sem OPENROUTER_API_KEY: usando o proxy de LLM da Metaculus.")
        print("A pesquisa sai sem busca web. Bom para testar, fraco para competir.\n")

    if publish:
        print("MODO PUBLICACAO: as previsoes VAO para a Metaculus.\n")
    else:
        print("Modo simulacao: nada sera publicado. Use --publish para valer.\n")


def openrouter_usage() -> float | None:
    """
    Gasto acumulado da chave da OpenRouter, em dolares, direto da fonte.

    Existe porque o MonetaryCostManager nao enxerga tudo: modelos com sufixo
    :online nao reportam custo para a biblioteca, e tokens de raciocinio
    escondidos tambem escapam. O endpoint /key da OpenRouter e o numero que
    vai de fato ser debitado dos seus creditos.
    """
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        return None
    import json
    import urllib.request

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/key",
        headers={"Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.load(resp).get("data", {})
    except Exception as exc:
        logger.warning(f"Nao consegui ler o saldo da OpenRouter: {exc}")
        return None

    # A chave doada pela Metaculus e BYOK: eles plugaram as proprias chaves
    # de OpenAI, Anthropic e Google na OpenRouter. O gasto aparece em
    # byok_usage, e o campo usage fica sempre em zero. Ler so usage daria
    # zero para sempre enquanto o saldo acaba.
    #
    # O numero mais confiavel e limite menos restante, porque e exatamente o
    # que conta contra o teto. Sem limite definido, soma os dois campos.
    limit = data.get("limit")
    remaining = data.get("limit_remaining")
    if limit is not None and remaining is not None:
        return float(limit) - float(remaining)
    return float(data.get("usage") or 0.0) + float(data.get("byok_usage") or 0.0)


def _pick(questions: list, limit: int) -> list:
    """
    Escolhe ate `limit` perguntas alternando entre os tipos.

    Num teste, cinco perguntas binarias provam cinco vezes a mesma coisa.
    Uma de cada tipo prova os tres caminhos de codigo do bot.
    """
    by_type: dict[str, list] = {}
    for q in questions:
        by_type.setdefault(type(q).__name__, []).append(q)
    picked: list = []
    while len(picked) < limit and any(by_type.values()):
        for fila in by_type.values():
            if fila and len(picked) < limit:
                picked.append(fila.pop(0))
    return picked


async def run(mode: str, publish: bool, samples: int, limit: int | None) -> list:
    bot = build_bot(publish, samples)
    client = MetaculusClient()

    targets = {
        "tournament": [client.CURRENT_AI_COMPETITION_ID, client.CURRENT_MINIBENCH_ID],
        "minibench": [client.CURRENT_MINIBENCH_ID],
        "cup": [client.CURRENT_METACULUS_CUP_ID],
        "market_pulse": [client.CURRENT_MARKET_PULSE_ID],
        "test": ["bot-testing-area"],
    }
    if mode not in targets:
        raise ValueError(f"modo desconhecido: {mode}")
    if mode in ("cup", "test"):
        bot.skip_previously_forecasted_questions = False

    antes = openrouter_usage()

    # Atencao: o parametro se chama hard_limit, nao max_cost (o README da
    # biblioteca esta desatualizado nesse ponto). Ao estourar, ele levanta erro.
    with MonetaryCostManager(hard_limit=MAX_COST_PER_RUN) as cost:
        if limit is None:
            reports = []
            for tid in targets[mode]:
                reports += await bot.forecast_on_tournament(tid, return_exceptions=True)
        else:
            abertas = []
            for tid in targets[mode]:
                abertas += client.get_all_open_questions_from_tournament(tid)
            escolhidas = _pick(abertas, limit)
            tipos = ", ".join(type(q).__name__.replace("Question", "") for q in escolhidas)
            print(f"Limitado a {len(escolhidas)} de {len(abertas)} perguntas abertas: {tipos}\n")
            reports = await bot.forecast_questions(escolhidas, return_exceptions=True)

        rastreado = cost.current_usage

    depois = openrouter_usage()
    print(f"\nCusto rastreado pela biblioteca : ${rastreado:.4f}")
    if antes is not None and depois is not None:
        real = depois - antes
        n = max(1, sum(1 for r in reports if not isinstance(r, BaseException)))
        print(f"Custo real debitado na OpenRouter: ${real:.4f}  (${real / n:.4f} por pergunta)")
        print(f"Gasto acumulado na chave         : ${depois:.4f}")

    # log_report_summary levanta RuntimeError com o traceback inteiro quando
    # tudo falha. Util para depurar, ilegivel para quem so quer saber o que
    # deu errado. O diagnostico abaixo cuida disso.
    try:
        bot.log_report_summary(reports)
    except RuntimeError:
        pass
    return reports


def diagnose(reports: list) -> None:
    """Traduz as falhas mais comuns em uma frase e um proximo passo."""
    errors = [r for r in reports if isinstance(r, BaseException)]
    if not errors:
        return

    blob = " ".join(str(e) for e in errors)

    if "allowance" in blob:
        model = "o modelo pedido"
        import re

        m = re.search(r"allowance for model <([^>]+)>", blob)
        if m:
            model = m.group(1)
        print(
            f"\nDIAGNOSTICO: sua conta nao tem cota liberada para {model} no proxy da Metaculus."
            "\nO token esta valido e as perguntas foram lidas, so falta credito de LLM."
            "\n\nDuas saidas:"
            "\n  1. Peca os creditos do torneio: https://forms.gle/aQdYMq9Pisrf1v7d8"
            "\n     Eles chegam como chave da OpenRouter. Cole em OPENROUTER_API_KEY no .env."
            "\n  2. Use chave propria da OpenAI, Anthropic ou OpenRouter no .env."
        )
    elif "401" in blob or "Permission" in blob or "authenticat" in blob.lower():
        print(
            "\nDIAGNOSTICO: o METACULUS_TOKEN foi recusado."
            "\nGere outro em Settings > My Forecasting Bots > Reveal API Key."
        )
    elif "429" in blob or "rate" in blob.lower():
        print(
            "\nDIAGNOSTICO: limite de requisicao atingido."
            "\nBaixe _max_concurrent_questions em bot.py ou espere alguns minutos."
        )
    else:
        print("\nDIAGNOSTICO: falha nao reconhecida. Primeiro erro completo:\n")
        print(f"  {type(errors[0]).__name__}: {str(errors[0])[:600]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bot de previsao para a Metaculus")
    parser.add_argument(
        "--mode",
        choices=["test", "tournament", "minibench", "cup", "market_pulse"],
        default="test",
        help="onde prever (padrao: test, a area de testes de bots)",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="envia as previsoes para a Metaculus (sem isso, so simula)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help=(
            "quantas vezes rodar o ensemble inteiro por pergunta (padrao: 1). "
            "Cada amostra ja consulta os 3 modelos, entao 3 amostras = 9 chamadas."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="preve no maximo N perguntas, alternando os tipos. Use para testar sem gastar o orcamento.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    check_env(args.publish)

    print(f"Modo      : {args.mode}")
    print(f"Modelo    : {FORECAST_MODEL}")
    print(f"Pesquisa  : {RESEARCH_MODEL}")
    print(f"Amostras  : {args.samples} por pergunta")
    print(f"Torneio   : {TOURNAMENT_URLS.get(args.mode, '-')}")
    print()

    reports = asyncio.run(run(args.mode, args.publish, args.samples, args.limit))

    errors = [r for r in reports if isinstance(r, BaseException)]
    ok = len(reports) - len(errors)
    print(f"\nPerguntas previstas: {ok}. Falhas: {len(errors)}.")
    if ok:
        print("Relatorios salvos em logs/")
    diagnose(reports)


if __name__ == "__main__":
    main()
