"""
Bot de previsao para os torneios FutureEval da Metaculus.

Estrutura herdada de `ForecastBot` (biblioteca oficial forecasting-tools).
O fluxo da classe pai, por pergunta:
  1. roda `run_research` N vezes  (research_reports_per_question)
  2. roda o forecast M vezes por pesquisa  (predictions_per_research_report)
  3. agrega as N*M previsoes
  4. publica, se publish_reports_to_metaculus=True

Tres decisoes aqui sao baseadas nas analises publicas de desempenho dos
bots do torneio, nao em achismo:

  1. O modelo base pesa mais que o andaime. A analise Q2/2025 da Metaculus
     concluiu que trocar o modelo move mais o placar que sofisticar o
     scaffold. Por isso o modelo e uma constante unica no topo do arquivo,
     e nao ha cadeia de agentes elaborada.

  2. Agregar varias amostras independentes ganha pontos. Mantemos o padrao
     do template oficial: 5 previsoes por pesquisa, agregadas.

  3. Platt scaling sobre o numero final rende ~0,016 de Brier. Fica em
     `calibration.py`, desligado ate voce ter perguntas resolvidas.

O prompt de forecast segue o template oficial da Metaculus de perto, de
proposito: e o baseline que a propria casa usa e mede. Mude depois de ter
um placar, nao antes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from forecasting_tools import (
    BinaryPrediction,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    PredictedOption,
    PredictedOptionList,
    ReasonedPrediction,
    clean_indents,
    structure_output,
)

from calibration import apply_platt

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    """Le um float do ambiente tratando vazio e espacos como ausente."""
    raw = (os.getenv(name) or "").strip()
    return float(raw) if raw else default


def _trimmed_mean(valores: list[float]) -> float:
    """
    Consenso do ensemble descartando os extremos.

    E o metodo que pgodzinai descreveu depois de vencer o Q4 2024: ele rodava
    oito previsoes, jogava fora as duas mais extremas e tirava a media
    aritmetica das seis restantes. A media aparada protege contra um modelo
    que viaja sozinho, sem jogar fora a informacao da dispersao como a
    mediana pura faria.

    Com poucas amostras nao da para aparar sem ficar sem dados, entao:
      1 a 2 valores  -> media simples
      3 a 4 valores  -> mediana
      5 ou mais      -> descarta o menor e o maior, media do resto
    """
    vs = sorted(valores)
    n = len(vs)
    if n <= 2:
        return sum(vs) / n
    if n <= 4:
        meio = n // 2
        return vs[meio] if n % 2 else (vs[meio - 1] + vs[meio]) / 2
    miolo = vs[1:-1]
    return sum(miolo) / len(miolo)


# Every model's forecast, one JSON object per line, for scoring once questions
# resolve (analyze_results.py). The GitHub workflow uploads it as an artifact.
FORECAST_LOG = os.path.join("logs", "forecasts.jsonl")


def _serialize(prediction: Any) -> Any:
    if isinstance(prediction, float):
        return round(prediction, 4)
    if isinstance(prediction, PredictedOptionList):
        return {o.option_name: round(o.probability, 4) for o in prediction.predicted_options}
    if isinstance(prediction, NumericDistribution):
        return [[round(p.percentile, 4), p.value] for p in prediction.declared_percentiles]
    return str(prediction)


def _record(question: MetaculusQuestion, model: str, role: str, prediction: Any) -> None:
    row = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "post_id": question.id_of_post,
        "question_id": question.id_of_question,
        "url": question.page_url,
        "type": type(question).__name__,
        "model": model,
        "role": role,
        "forecast": _serialize(prediction),
    }
    try:
        os.makedirs(os.path.dirname(FORECAST_LOG), exist_ok=True)
        with open(FORECAST_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning(f"Could not write {FORECAST_LOG}: {exc}")


def _align_options(parsed: PredictedOptionList, options: list[str]) -> PredictedOptionList | None:
    """
    Maps a parsed answer onto the question's exact option names, in order, so
    answers from different models can be averaged. Falls back to position when
    the names do not match but the count does. Returns None when neither works.
    """
    by_name = {o.option_name.strip().lower(): o.probability for o in parsed.predicted_options}
    probs = [by_name.get(name.strip().lower()) for name in options]
    if any(p is None for p in probs):
        if len(parsed.predicted_options) != len(options):
            return None
        logger.warning("Option names did not match the question; using their order")
        probs = [o.probability for o in parsed.predicted_options]
    total = sum(probs)
    if total <= 0:
        return None
    return PredictedOptionList(
        predicted_options=[
            PredictedOption(option_name=name, probability=p / total)
            for name, p in zip(options, probs)
        ]
    )


def _fmt_num(value: float) -> str:
    return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.4g}"


def _quantile(dist: NumericDistribution, q: float) -> float:
    """Value at cumulative probability q, interpolated from the declared percentiles."""
    pts = sorted((p.percentile, p.value) for p in dist.declared_percentiles)
    for (p0, v0), (p1, v1) in zip(pts, pts[1:]):
        if p0 <= q <= p1:
            return v0 if p1 == p0 else v0 + (v1 - v0) * (q - p0) / (p1 - p0)
    return pts[0][1] if q < pts[0][0] else pts[-1][1]


class ForecasterBot(ForecastBot):
    # Uma pergunta por vez. Suba se seu provedor aguentar; o limite de
    # requisicao costuma ser o gargalo, nao a CPU.
    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    # Quantas vezes o parser tenta validar a extracao estruturada.
    _structure_output_validation_samples = 2

    def __init__(
        self,
        *args,
        ensemble: list[GeneralLlm] | None = None,
        shadows: list[tuple[str, GeneralLlm]] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # Shadow models answer every question alongside the ensemble but are
        # only recorded to FORECAST_LOG, never aggregated or published. They
        # let a candidate configuration be scored on live questions.
        self._shadows = list(shadows or [])
        if self._shadows:
            logger.info(f"Shadow models (recorded only): {', '.join(n for n, _ in self._shadows)}")

        # Coeficientes de calibracao vindos do .env. Identidade por padrao.
        # _env_float e nao float(os.getenv(...)): no GitHub Actions uma
        # variavel de repositorio nao definida chega como string VAZIA, nao
        # como ausente, e float("") derruba o bot antes da primeira pergunta.
        self._cal_a = _env_float("CALIBRATION_A", 1.0)
        self._cal_b = _env_float("CALIBRATION_B", 0.0)
        if (self._cal_a, self._cal_b) != (1.0, 0.0):
            logger.info(f"Calibracao ativa: A={self._cal_a}, B={self._cal_b}")

        # Ensemble heterogeneo: modelos de familias diferentes respondendo a
        # mesma pergunta. Lista vazia desliga e volta ao modelo unico.
        self._ensemble = list(ensemble or [])
        if self._ensemble:
            nomes = ", ".join(llm.model for llm in self._ensemble)
            logger.info(f"Ensemble com {len(self._ensemble)} modelos: {nomes}")

        # Provedores de pesquisa EXTRA (alem do primario). Auto-detecta pelas
        # chaves presentes, ou RESEARCH_PROVIDERS="asknews,exa" no .env forca.
        # "anthropic-search" e o segundo backend gratis (busca do Claude via
        # OpenRouter), diferente da busca do GPT do provedor primario.
        override = os.getenv("RESEARCH_PROVIDERS", "").strip()
        if override:
            wanted = [p.strip().lower() for p in override.split(",") if p.strip()]
        else:
            wanted = ["asknews", "exa", "perplexity", "anthropic-search"]
        # so fica o que tem credencial
        self._research_providers = [p for p in wanted if self._provider_ready(p)]
        if self._research_providers:
            logger.info(f"Provedores de pesquisa extra: {', '.join(self._research_providers)}")

    @staticmethod
    def _provider_ready(name: str) -> bool:
        if name == "asknews":
            # O AskNewsSearcher aceita OAuth (id + secret) ou chave de API.
            return bool(
                (os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"))
                or os.getenv("ASKNEWS_API_KEY")
            )
        if name == "exa":
            return bool(os.getenv("EXA_API_KEY"))
        if name == "perplexity":
            return bool(os.getenv("PERPLEXITY_API_KEY"))
        if name == "anthropic-search":
            return bool(os.getenv("OPENROUTER_API_KEY"))
        return False

    async def _provider(self, name: str, question: MetaculusQuestion, prompt: str) -> str:
        """Roda um provedor de pesquisa distinto e devolve o texto."""
        if name == "asknews":
            from forecasting_tools import AskNewsSearcher

            # A versao _async: get_formatted_news e sincrono e devolve str,
            # e "await" numa str quebra (o provedor falharia sempre, calado).
            return await AskNewsSearcher().get_formatted_news_async(question.question_text)
        if name == "exa":
            from forecasting_tools import SmartSearcher

            return await SmartSearcher(
                model="openrouter/openai/gpt-5.4", num_searches_to_run=2, num_sites_per_search=8
            ).invoke(prompt)
        if name == "perplexity":
            return await GeneralLlm(model="perplexity/sonar-pro", temperature=0.1).invoke(prompt)
        if name == "anthropic-search":
            # Raciocinio baixo: e busca, nao decisao. E medido em 2026-09-22:
            # sem o parametro, US$ 0,75 por chamada em tokens (media de 3); com
            # "low", US$ 0,14. Com raciocinio ligado o Claude puxa ~4x menos
            # conteudo de busca. Sem ele, uma chamada passou de 200 mil tokens
            # de entrada e caiu na tarifa de contexto longo (preco dobra).
            # Rende ~60% do texto por um quinto do preco.
            return await GeneralLlm(
                model="openrouter/anthropic/claude-sonnet-4.6:online",
                temperature=0.1, timeout=180, reasoning_effort="low",
            ).invoke(prompt)
        return ""

    # ------------------------------------------------------------------
    # PESQUISA
    # ------------------------------------------------------------------

    _META_RE = re.compile(
        r"community (?:prediction|forecast)[^?]{0,80}?(higher|lower|above|below|greater|less)"
        r"[^?]{0,40}?(\d+(?:\.\d+)?)\s*%",
        re.IGNORECASE,
    )

    def _meta_question_block(self, question: MetaculusQuestion) -> str:
        """
        Instrucao extra para as meta-perguntas do MiniBench.

        Muitas perguntas do MiniBench tem a forma "a previsao da comunidade
        vai estar acima de X% na data D para a pergunta Q?". Jeff Mohl (bot
        Delphi) perdeu suas TRES piores pontuacoes de uma rodada porque a
        busca web alucinou o valor atual da comunidade (leu 42% quando era
        35%, 75% quando era 90%) e o resto do raciocinio herdou o erro.

        A correcao ideal seria ler o numero direto da API. Testado em
        2026-09-19: o token de bot NAO enxerga a previsao da comunidade em
        perguntas que ele nao previu (0 de 6 perguntas abertas). Isso exige o
        "Bot Benchmarking Access Tier" da Metaculus. Enquanto ele nao chega,
        o melhor que da para fazer e obrigar a busca a ir na pagina exata e
        declarar a incerteza em vez de inventar.
        """
        text = question.question_text or ""
        m = self._META_RE.search(text)
        if not m:
            return ""
        direcao, limiar = m.group(1), m.group(2)
        return clean_indents(
            f"""
            (g) THIS IS A META-QUESTION about another Metaculus question's
                community prediction, with threshold {limiar}% ({direcao}).
                Your single most important job is the CURRENT value of that
                community prediction. Open the referenced Metaculus question
                page itself and read the number shown there. Report it as
                "Community prediction now: N% (seen on <date/time>)". Do NOT
                infer it from news articles or from memory. If you cannot open
                the page, write "Community prediction now: UNKNOWN" and say so.
                A wrong anchor here is worse than no anchor: the forecaster
                will reason about drift from whatever number you give.
            """
        )

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            prompt = clean_indents(
                f"""
                You are a research assistant to a superforecaster. You do NOT
                produce forecasts yourself. You produce the evidence the
                forecaster needs.

                Question:
                {question.question_text}

                Resolution criteria:
                {question.resolution_criteria}

                {question.fine_print}

                Today is {datetime.now().strftime("%Y-%m-%d")}.

                Write a concise but dense rundown covering:
                (a) The status quo: what is true right now, with dates and numbers.
                (b) The most recent relevant news, newest first, each with its date.
                (c) The base rate: how often has this kind of event happened in
                    comparable past cases? Name the reference class you used.
                (d) Scheduled events before resolution that could change the outcome.
                (e) What informed observers, experts or markets currently expect.
                (f) Prediction markets: search Polymarket, Kalshi, Manifold and
                    Metaculus for this exact question or its closest match. If
                    you find one, quote the current price, the market, and the
                    date you saw it. This is one input, not the answer.
                {self._meta_question_block(question)}
                State plainly where evidence is missing or contradictory. Never
                pad. If the question would resolve today on current information,
                say which way and why. Every number you report must come with
                its source and date; a number you cannot source is a guess, and
                you must label it as one.
                """
            )

            # Diversidade de PROVEDORES, nao so de consultas. O achado da
            # Metaculus: numero de fontes distintas r=+0,42, e "nenhum provedor
            # e bom por si". Entao a pesquisa compartilhada roda em paralelo o
            # provedor primario mais um segundo provedor DISTINTO, e concatena
            # os dois com rotulo. Cada bloco vira uma fonte independente que o
            # forecaster confronta. Os provedores extras degradam sozinhos: sem
            # chave, aquele bloco simplesmente nao aparece.
            async def primary() -> str:
                r = self.get_llm("researcher")
                if isinstance(r, GeneralLlm):
                    return await r.invoke(prompt)
                if not r or r in ("None", "no_research"):
                    return ""
                return await self.get_llm("researcher", "llm").invoke(prompt)

            blocos = await asyncio.gather(
                primary(),
                *(self._provider(name, question, prompt) for name in self._research_providers),
                return_exceptions=True,
            )
            partes = []
            rotulos = ["Primary web research"] + list(self._research_providers)
            for rotulo, b in zip(rotulos, blocos):
                if isinstance(b, Exception):
                    logger.warning(f"Provedor de pesquisa {rotulo} falhou: {type(b).__name__}: {b}")
                    continue
                if b and b.strip():
                    partes.append(f"## Source: {rotulo}\n{b.strip()}")
            research = "\n\n".join(partes)
            logger.info(f"Pesquisa para {question.page_url} ({len(partes)} fontes):\n{research}")
            return research

    # ------------------------------------------------------------------
    # ENSEMBLE
    # ------------------------------------------------------------------

    async def _ask_models(
        self,
        question: MetaculusQuestion,
        prompt: str,
        parse: Callable[[str], Awaitable[Any]],
    ) -> list[tuple[str, Any, str]]:
        """
        Sends the same prompt to every ensemble model (or to the default model
        when the ensemble is off) and parses each answer. Returns (model,
        prediction, reasoning) for the models that succeeded, so one failure
        does not sink the question. Shadow models run in parallel and are only
        recorded.

        The ensemble mixes model families on purpose: in a controlled test on
        202 tournament questions (Schneider and Schramm, 2025), different
        models improved Brier from 0.162 to 0.153, while repeating one model
        did not help.
        """
        members = [(llm.model, llm) for llm in self._ensemble]
        if not members:
            default = self.get_llm("default", "llm")
            members = [(default.model, default)]

        async def ask(name: str, llm: GeneralLlm, role: str) -> tuple[str, Any, str] | None:
            try:
                text = await llm.invoke(prompt)
                prediction = await parse(text)
            except Exception as exc:
                logger.warning(f"{name} ({role}) failed: {type(exc).__name__}: {exc}")
                return None
            if prediction is None:
                logger.warning(f"{name} ({role}): answer could not be parsed")
                return None
            _record(question, name, role, prediction)
            return name, prediction, text

        results = await asyncio.gather(
            *(ask(name, llm, "member") for name, llm in members),
            *(ask(name, llm, "shadow") for name, llm in self._shadows),
        )
        answers = [r for r in results[: len(members)] if r is not None]
        if not answers:
            raise RuntimeError("every forecasting model failed on this question")
        return answers

    # ------------------------------------------------------------------
    # BINARIA
    # ------------------------------------------------------------------

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}

            This question's outcome will be determined by the specific criteria
            below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) The base rate for this class of event, and the reference class
                you are using.
            (d) A brief description of a scenario that results in a No outcome.
            (e) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra
            weight on the status quo outcome since the world changes slowly most
            of the time. You also remember that most things that have never
            happened before do not happen in the next few months. Historically,
            forecasters like you have been overconfident, and only about 35% of
            Metaculus binary questions resolve Yes.

            Before you commit, check three things that sink otherwise good
            forecasts:
            - The principal actor may have a face-saving route to the outcome
              that you have not listed. A government can return a deportee by
              indicting him; a candidate can enter a race by resigning first; a
              company can skip its usual staged rollout. Name that route
              explicitly before dismissing the outcome.
            - If your research quotes a number as the current state, ask
              whether it is sourced and dated. A stale or misread anchor is the
              single most common cause of catastrophic misses.
            - If the question is about an index or a scale, check its floor and
              ceiling. An outcome outside the range is impossible, not merely
              unlikely.

            Two Metaculus resolution conventions that trip up forecasters:
            - If your research does not positively show that the event has
              already occurred, assume it has NOT occurred yet. Absence of
              evidence in the research is not evidence the event happened.
            - A question phrased "will X happen before <date>" is
              forward-looking: it asks about the window between now and that
              date, not about whether X ever happened in the past.

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )

        answers = await self._ask_models(question, prompt, self._parse_binary)
        if len(answers) == 1:
            _, value, text = answers[0]
            logger.info(f"Forecast {question.page_url}: {value}")
            return ReasonedPrediction(prediction_value=value, reasoning=text)

        values = [p for _, p, _ in answers]
        consensus = _trimmed_mean(values)
        details = "\n\n".join(f"### {m} forecast {p:.0%}\n{t}" for m, p, t in answers)
        summary = (
            f"Ensemble of {len(answers)} models: "
            + ", ".join(f"{p:.0%}" for p in values)
            + f" -> consensus {consensus:.1%}"
        )
        logger.info(f"{question.page_url}: {summary}")
        return ReasonedPrediction(prediction_value=consensus, reasoning=f"{summary}\n\n{details}")

    async def _parse_binary(self, reasoning: str) -> float:
        parsed: BinaryPrediction = await structure_output(
            reasoning,
            BinaryPrediction,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
        )
        valor = parsed.prediction_in_decimal

        # Alarme de 50% exato.
        #
        # Um participante da Spring 2026 perdeu a temporada inteira porque o
        # bot caiu num fallback silencioso e passou a enviar 0,5 em tudo. 50%
        # cravado quase nunca e uma conclusao de raciocinio, quase sempre e
        # parser falhando ou modelo se recusando a responder. Nao da para
        # corrigir automaticamente sem inventar numero, entao grita no log.
        if valor == 0.5:
            logger.warning(
                "PREVISAO EXATAMENTE 50%. Verifique se o parser leu o texto ou "
                "se o modelo se recusou a responder. Trecho do raciocinio: "
                f"{reasoning[:200]!r}"
            )

        # Metaculus nao aceita 0% nem 100%. A calibracao entra depois.
        return max(0.01, min(0.99, valor))

    # ------------------------------------------------------------------
    # MULTIPLA ESCOLHA
    # ------------------------------------------------------------------

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of a scenario that results in an unexpected outcome.

            You write your rationale remembering that (1) good forecasters put
            extra weight on the status quo outcome since the world changes slowly
            most of the time, and (2) good forecasters leave some moderate
            probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options
            in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )

        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}

            The text you are parsing may prepend these options with some variation
            of "Option" which you should remove if not part of the option names I
            just gave you.
            Additionally, you may sometimes need to parse a 0% probability. Please
            do not skip options with 0% but rather make it an entry in your final
            list with 0% probability.
            """
        )
        async def parse(text: str) -> PredictedOptionList | None:
            parsed: PredictedOptionList = await structure_output(
                text_to_structure=text,
                output_type=PredictedOptionList,
                model=self.get_llm("parser", "llm"),
                num_validation_samples=self._structure_output_validation_samples,
                additional_instructions=parsing_instructions,
            )
            return _align_options(parsed, question.options)

        answers = await self._ask_models(question, prompt, parse)
        if len(answers) == 1:
            _, predicted, text = answers[0]
            logger.info(f"Forecast {question.page_url}: {predicted}")
            return ReasonedPrediction(prediction_value=predicted, reasoning=text)

        # The library's own rule for combining samples: mean probability per option.
        combined = await super()._aggregate_predictions([p for _, p, _ in answers], question)

        def fmt(options: PredictedOptionList) -> str:
            return ", ".join(f"{o.option_name} {o.probability:.0%}" for o in options.predicted_options)

        details = "\n\n".join(f"### {m} forecast: {fmt(p)}\n{t}" for m, p, t in answers)
        summary = f"Ensemble of {len(answers)} models, averaged per option: {fmt(combined)}"
        logger.info(f"{question.page_url}: {summary}")
        return ReasonedPrediction(prediction_value=combined, reasoning=f"{summary}\n\n{details}")

    # ------------------------------------------------------------------
    # NUMERICA
    # ------------------------------------------------------------------

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_msg, lower_msg = self._bound_messages(question)

        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_msg}
            {upper_msg}

            Formatting Instructions:
            - Please notice the units requested and give your answer in these units
              (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and
              then increase from there. The value for percentile 10 should always
              be less than the value for percentile 20, and so on.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            You remind yourself that good forecasters are humble and set wide 90/10
            confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX (lowest number value)
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX (highest number value)
            "
            """
        )

        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a
            numeric question.
            - This text is trying to answer the numeric question: "{question.question_text}".
            - When parsing the text, please make sure to give the values (the ones
              assigned to percentiles) in terms of the correct units.
            - The units for the forecast are: {question.unit_of_measure}
            - Your work will be shown publicly with these units stated verbatim
              after the numbers you parse.
            - As an example, someone else guessed that the answer will be between
              {question.lower_bound} {question.unit_of_measure} and
              {question.upper_bound} {question.unit_of_measure}, so the numbers
              parsed from an answer like this would be verbatim
              "{question.lower_bound}" and "{question.upper_bound}".
            - If the answer doesn't give the answer in the correct units, you
              should parse it in the right units. For instance if the answer gives
              numbers as $500,000,000 and units are "B $" then you should parse the
              answer as 0.5 (since $500,000,000 is $0.5 billion).
            - If percentiles are not explicitly given (e.g. only a single value is
              given) please don't return a parsed output, but rather indicate that
              the answer is not explicitly given in the text.
            - Turn any values that are in scientific notation into regular numbers.
            """
        )
        async def parse(text: str) -> NumericDistribution:
            percentiles: list[Percentile] = await structure_output(
                text,
                list[Percentile],
                model=self.get_llm("parser", "llm"),
                additional_instructions=parsing_instructions,
                num_validation_samples=self._structure_output_validation_samples,
            )
            return NumericDistribution.from_question(percentiles, question)

        answers = await self._ask_models(question, prompt, parse)
        if len(answers) == 1:
            _, prediction, text = answers[0]
            logger.info(f"Forecast {question.page_url}: {prediction.declared_percentiles}")
            return ReasonedPrediction(prediction_value=prediction, reasoning=text)

        # The library's own rule for combining samples: the pointwise median of
        # the CDFs, which for two models is their average.
        combined = await super()._aggregate_predictions([p for _, p, _ in answers], question)

        def fmt(dist: NumericDistribution) -> str:
            return " | ".join(
                f"P{round(p.percentile * 100)} {_fmt_num(p.value)}" for p in dist.declared_percentiles
            )

        details = "\n\n".join(f"### {m} forecast: {fmt(p)}\n{t}" for m, p, t in answers)
        summary = f"Ensemble of {len(answers)} models, CDFs combined: " + " | ".join(
            f"P{round(q * 100)} {_fmt_num(_quantile(combined, q))}" for q in (0.1, 0.5, 0.9)
        )
        logger.info(f"{question.page_url}: {summary}")
        return ReasonedPrediction(prediction_value=combined, reasoning=f"{summary}\n\n{details}")

    def _bound_messages(self, question: NumericQuestion) -> tuple[str, str]:
        upper = (
            question.nominal_upper_bound
            if question.nominal_upper_bound is not None
            else question.upper_bound
        )
        lower = (
            question.nominal_lower_bound
            if question.nominal_lower_bound is not None
            else question.lower_bound
        )
        unit = question.unit_of_measure

        upper_msg = (
            f"The question creator thinks the number is likely not higher than {upper} {unit}."
            if question.open_upper_bound
            else f"The outcome can not be higher than {upper} {unit}."
        )
        lower_msg = (
            f"The question creator thinks the number is likely not lower than {lower} {unit}."
            if question.open_lower_bound
            else f"The outcome can not be lower than {lower} {unit}."
        )
        return upper_msg, lower_msg

    # ------------------------------------------------------------------
    # AGREGACAO + CALIBRACAO
    # ------------------------------------------------------------------

    async def _aggregate_predictions(self, predictions, question):
        """
        Agrega pela logica da classe pai e aplica Platt scaling em cima.

        A calibracao tem que vir DEPOIS da agregacao: corrigir cada amostra
        antes de juntar distorce a dispersao entre amostras, que e justamente
        o sinal que a agregacao usa.
        """
        aggregate = await super()._aggregate_predictions(predictions, question)

        if isinstance(aggregate, float) and (self._cal_a, self._cal_b) != (1.0, 0.0):
            calibrated = apply_platt(aggregate, self._cal_a, self._cal_b)
            logger.info(f"Calibracao: {aggregate:.4f} -> {calibrated:.4f}")
            return calibrated

        return aggregate
