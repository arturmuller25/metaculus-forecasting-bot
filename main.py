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
    _ENSEMBLE = [
        "openrouter/openai/gpt-5.4",
        "openrouter/anthropic/claude-sonnet-4.6",
        "openrouter/google/gemini-3.1-pro-preview",
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

TOURNAMENT_URLS = {
    "tournament": "https://www.metaculus.com/tournament/fall-futureeval-2026/",
    "minibench": "https://www.metaculus.com/aib/minibench",
    "cup": "https://www.metaculus.com/tournament/metaculus-cup/",
    "market_pulse": "https://www.metaculus.com/tournament/market-pulse-26q4/",
    "test": "https://www.metaculus.com/tournament/bot-testing-area/",
}


def build_bot(publish: bool, samples: int) -> ForecasterBot:
    llms = {
        "default": GeneralLlm(model=FORECAST_MODEL, temperature=0.3, timeout=120, allowed_tries=2),
        "researcher": GeneralLlm(model=RESEARCH_MODEL, temperature=0.1, timeout=180, allowed_tries=2),
        "parser": GeneralLlm(model=PARSER_MODEL, temperature=0.0, timeout=60, allowed_tries=2),
        "summarizer": GeneralLlm(model=PARSER_MODEL, temperature=0.0, timeout=60),
    }
    # O ensemble e a mudanca mais disputada do bot: um experimento controlado
    # em 202 perguntas deste torneio mostrou ganho com modelos DIFERENTES
    # (Brier 0,162 -> 0,153), mas a tabela de correlacao da Spring 2026 pontua
    # "agregar previsoes" em r=-0,19. Evidencia conflitante, entao fica um
    # interruptor: ENSEMBLE=0 no .env desliga e volta ao modelo unico.
    # Meca voce mesmo antes de acreditar em qualquer um dos dois estudos.
    ensemble = (
        []
        if os.getenv("ENSEMBLE", "1") == "0"
        else [
            GeneralLlm(model=m, temperature=0.3, timeout=120, allowed_tries=2)
            for m in _ENSEMBLE
        ]
    )

    return ForecasterBot(
        ensemble=ensemble,
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


async def run(mode: str, publish: bool, samples: int) -> list:
    bot = build_bot(publish, samples)
    client = MetaculusClient()

    # Atencao: o parametro se chama hard_limit, nao max_cost (o README da
    # biblioteca esta desatualizado nesse ponto). Ao estourar, ele levanta erro.
    with MonetaryCostManager(hard_limit=MAX_COST_PER_RUN) as cost:
        if mode == "tournament":
            reports = await bot.forecast_on_tournament(
                client.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
            reports += await bot.forecast_on_tournament(
                client.CURRENT_MINIBENCH_ID, return_exceptions=True
            )
        elif mode == "minibench":
            reports = await bot.forecast_on_tournament(
                client.CURRENT_MINIBENCH_ID, return_exceptions=True
            )
        elif mode == "cup":
            bot.skip_previously_forecasted_questions = False
            reports = await bot.forecast_on_tournament(
                client.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )
        elif mode == "market_pulse":
            reports = await bot.forecast_on_tournament(
                client.CURRENT_MARKET_PULSE_ID, return_exceptions=True
            )
        elif mode == "test":
            bot.skip_previously_forecasted_questions = False
            reports = await bot.forecast_on_tournament(
                "bot-testing-area", return_exceptions=True
            )
        else:
            raise ValueError(f"modo desconhecido: {mode}")

        print(f"\nCusto desta execucao: ${cost.current_usage:.4f} de ${MAX_COST_PER_RUN:.2f}")

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
        default=5,
        help="previsoes independentes por pergunta, agregadas (padrao: 5)",
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

    reports = asyncio.run(run(args.mode, args.publish, args.samples))

    errors = [r for r in reports if isinstance(r, BaseException)]
    ok = len(reports) - len(errors)
    print(f"\nPerguntas previstas: {ok}. Falhas: {len(errors)}.")
    if ok:
        print("Relatorios salvos em logs/")
    diagnose(reports)


if __name__ == "__main__":
    main()
