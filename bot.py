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
import logging
import os
from datetime import datetime

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


class ForecasterBot(ForecastBot):
    # Uma pergunta por vez. Suba se seu provedor aguentar; o limite de
    # requisicao costuma ser o gargalo, nao a CPU.
    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    # Quantas vezes o parser tenta validar a extracao estruturada.
    _structure_output_validation_samples = 2

    def __init__(self, *args, ensemble: list[GeneralLlm] | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)

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

    # ------------------------------------------------------------------
    # PESQUISA
    # ------------------------------------------------------------------

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

                State plainly where evidence is missing or contradictory. Never
                pad. If the question would resolve today on current information,
                say which way and why.
                """
            )

            researcher = self.get_llm("researcher")
            if isinstance(researcher, GeneralLlm):
                research = await researcher.invoke(prompt)
            elif not researcher or researcher in ("None", "no_research"):
                research = ""
            else:
                research = await self.get_llm("researcher", "llm").invoke(prompt)

            logger.info(f"Pesquisa para {question.page_url}:\n{research}")
            return research

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
            happened before do not happen in the next few months.

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

        # Sem ensemble configurado: um modelo so, comportamento classico.
        if not self._ensemble:
            reasoning = await self.get_llm("default", "llm").invoke(prompt)
            parsed = await self._parse_binary(reasoning)
            logger.info(f"Previsao {question.page_url}: {parsed}")
            return ReasonedPrediction(prediction_value=parsed, reasoning=reasoning)

        # Com ensemble: pergunta a varios modelos DIFERENTES em paralelo.
        #
        # Por que modelos diferentes e nao varias amostras do mesmo: Schneider
        # e Schramm (2025) mediram os dois em 202 perguntas do proprio torneio.
        # Ensemble heterogeneo melhorou o Brier de 0,162 para 0,153 (p=0,014).
        # Tres instancias do MESMO modelo nao melhoraram nada (+0,007, p=0,12).
        # Repetir o mesmo modelo custa token e nao compra diversidade.
        async def uma(llm) -> tuple[str, float | None, str]:
            try:
                texto = await llm.invoke(prompt)
                return llm.model, await self._parse_binary(texto), texto
            except Exception as exc:  # um modelo fora nao derruba a pergunta
                logger.warning(f"{llm.model} falhou: {type(exc).__name__}: {exc}")
                return llm.model, None, ""

        respostas = await asyncio.gather(*(uma(llm) for llm in self._ensemble))
        validas = [(m, p, t) for m, p, t in respostas if p is not None]
        if not validas:
            raise RuntimeError("todos os modelos do ensemble falharam nesta pergunta")

        valores = [p for _, p, _ in validas]
        consenso = _trimmed_mean(valores)

        detalhe = "\n\n".join(
            f"### {m} previu {p:.0%}\n{t}" for m, p, t in validas
        )
        resumo = (
            f"Ensemble de {len(validas)} modelos: "
            + ", ".join(f"{p:.0%}" for p in valores)
            + f" -> consenso {consenso:.1%}"
        )
        logger.info(f"{question.page_url}: {resumo}")

        return ReasonedPrediction(
            prediction_value=consenso, reasoning=f"{resumo}\n\n{detalhe}"
        )

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

        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Raciocinio para {question.page_url}: {reasoning}")

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
        predicted: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
        )

        logger.info(f"Previsao {question.page_url}: {predicted}")
        return ReasonedPrediction(prediction_value=predicted, reasoning=reasoning)

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

        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Raciocinio para {question.page_url}: {reasoning}")

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
        percentiles: list[Percentile] = await structure_output(
            reasoning,
            list[Percentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )
        prediction = NumericDistribution.from_question(percentiles, question)

        logger.info(f"Previsao {question.page_url}: {prediction.declared_percentiles}")
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

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
