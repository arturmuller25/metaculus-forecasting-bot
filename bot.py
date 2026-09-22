"""
Forecasting bot for the Metaculus FutureEval tournaments.

Built on `ForecastBot` from forecasting-tools. For each question the parent
class runs `run_research`, then the forecast method for the question type,
aggregates, and publishes when publish_reports_to_metaculus is True.

This subclass adds research from several providers, a cross-family model
ensemble for every question type, per-model forecast records, and optional
Platt calibration. The forecast prompts follow the official Metaculus
template closely.
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
    """Reads a float from the environment, treating empty or blank values as unset."""
    raw = (os.getenv(name) or "").strip()
    return float(raw) if raw else default


def _trimmed_mean(values: list[float]) -> float:
    """
    Ensemble consensus that discards the extremes, protecting against one
    outlying model:
      1-2 values -> mean
      3-4 values -> median
      5 or more  -> drop the lowest and highest, mean of the rest
    """
    vs = sorted(values)
    n = len(vs)
    if n <= 2:
        return sum(vs) / n
    if n <= 4:
        mid = n // 2
        return vs[mid] if n % 2 else (vs[mid - 1] + vs[mid]) / 2
    middle = vs[1:-1]
    return sum(middle) / len(middle)


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
    # One question at a time. Rate limits, not CPU, are the bottleneck.
    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    # How many samples the parser uses to validate each structured extraction.
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

        # Platt calibration coefficients; identity by default. _env_float
        # because GitHub Actions passes unset repository variables as empty
        # strings, and float("") would crash before the first question.
        self._cal_a = _env_float("CALIBRATION_A", 1.0)
        self._cal_b = _env_float("CALIBRATION_B", 0.0)
        if (self._cal_a, self._cal_b) != (1.0, 0.0):
            logger.info(f"Calibration on: A={self._cal_a}, B={self._cal_b}")

        # Models from different families answering the same question. An
        # empty list falls back to the single default model.
        self._ensemble = list(ensemble or [])
        if self._ensemble:
            names = ", ".join(llm.model for llm in self._ensemble)
            logger.info(f"Ensemble of {len(self._ensemble)} models: {names}")

        # Research providers used in addition to the primary research model,
        # detected from the keys present or forced with RESEARCH_PROVIDERS.
        # "anthropic-search" is Claude's web search through OpenRouter, a
        # different engine from the primary GPT search.
        override = os.getenv("RESEARCH_PROVIDERS", "").strip()
        if override:
            wanted = [p.strip().lower() for p in override.split(",") if p.strip()]
        else:
            wanted = ["asknews", "exa", "perplexity", "anthropic-search"]
        # keep only providers that have credentials
        self._research_providers = [p for p in wanted if self._provider_ready(p)]
        if self._research_providers:
            logger.info(f"Extra research providers: {', '.join(self._research_providers)}")

    @staticmethod
    def _provider_ready(name: str) -> bool:
        if name == "asknews":
            # AskNewsSearcher accepts OAuth (client id + secret) or an API key.
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
        """Runs one additional research provider and returns its text."""
        if name == "asknews":
            from forecasting_tools import AskNewsSearcher

            # The _async variant: get_formatted_news is synchronous and returns
            # a str, so awaiting it fails (silently, since provider errors are
            # caught in run_research).
            return await AskNewsSearcher().get_formatted_news_async(question.question_text)
        if name == "exa":
            from forecasting_tools import SmartSearcher

            return await SmartSearcher(
                model="openrouter/openai/gpt-5.4", num_searches_to_run=2, num_sites_per_search=8
            ).invoke(prompt)
        if name == "perplexity":
            return await GeneralLlm(model="perplexity/sonar-pro", temperature=0.1).invoke(prompt)
        if name == "anthropic-search":
            # Low reasoning on purpose. Measured 2026-09-22: without the
            # parameter a call cost $0.75 in tokens (mean of 3) against $0.14
            # with "low", because reasoning makes Claude pull about 4x less
            # search content; one call went past 200k input tokens and was
            # billed at the long-context rate. "low" yields ~60% of the text
            # for a fifth of the price.
            return await GeneralLlm(
                model="openrouter/anthropic/claude-sonnet-4.6:online",
                temperature=0.1, timeout=180, reasoning_effort="low",
            ).invoke(prompt)
        return ""

    # ------------------------------------------------------------------
    # RESEARCH
    # ------------------------------------------------------------------

    _META_RE = re.compile(
        r"community (?:prediction|forecast)[^?]{0,80}?(higher|lower|above|below|greater|less)"
        r"[^?]{0,40}?(\d+(?:\.\d+)?)\s*%",
        re.IGNORECASE,
    )

    def _meta_question_block(self, question: MetaculusQuestion) -> str:
        """
        Extra research instruction for MiniBench meta-questions ("will the
        community prediction be above X% on date D for question Q?").

        Web search tends to hallucinate the current community value, and the
        whole forecast inherits that wrong anchor. Reading it from the API
        would be better, but a bot token cannot see the community prediction
        on questions it has not forecast (tested 2026-09-19), so the research
        is told to read the exact page or report the value as unknown.
        """
        text = question.question_text or ""
        m = self._META_RE.search(text)
        if not m:
            return ""
        direction, threshold = m.group(1), m.group(2)
        return clean_indents(
            f"""
            (g) THIS IS A META-QUESTION about another Metaculus question's
                community prediction, with threshold {threshold}% ({direction}).
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

            # The primary research model and every extra provider run in
            # parallel; each result becomes a labeled source block. A provider
            # that fails or has no key is simply left out.
            async def primary() -> str:
                r = self.get_llm("researcher")
                if isinstance(r, GeneralLlm):
                    return await r.invoke(prompt)
                if not r or r in ("None", "no_research"):
                    return ""
                return await self.get_llm("researcher", "llm").invoke(prompt)

            blocks = await asyncio.gather(
                primary(),
                *(self._provider(name, question, prompt) for name in self._research_providers),
                return_exceptions=True,
            )
            parts = []
            labels = ["Primary web research"] + list(self._research_providers)
            for label, b in zip(labels, blocks):
                if isinstance(b, Exception):
                    logger.warning(f"Research provider {label} failed: {type(b).__name__}: {b}")
                    continue
                if b and b.strip():
                    parts.append(f"## Source: {label}\n{b.strip()}")
            research = "\n\n".join(parts)
            logger.info(f"Research for {question.page_url} ({len(parts)} sources):\n{research}")
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
    # BINARY
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
        value = parsed.prediction_in_decimal

        # An exact 50% is almost never a reasoned conclusion; it usually means
        # the parser failed or the model refused to answer. It cannot be fixed
        # without inventing a number, so it is flagged loudly in the log.
        if value == 0.5:
            logger.warning(
                "FORECAST OF EXACTLY 50%. Check whether the parser read the text "
                "or the model refused to answer. Reasoning excerpt: "
                f"{reasoning[:200]!r}"
            )

        # Metaculus rejects 0% and 100%. Calibration is applied later.
        return max(0.01, min(0.99, value))

    # ------------------------------------------------------------------
    # MULTIPLE CHOICE
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
    # NUMERIC AND DISCRETE
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
    # AGGREGATION AND CALIBRATION
    # ------------------------------------------------------------------

    async def _aggregate_predictions(self, predictions, question):
        """
        Aggregates with the parent class logic, then applies Platt scaling to
        binary forecasts. Calibration comes after aggregation so it corrects
        the final number rather than distorting the spread between samples.
        """
        aggregate = await super()._aggregate_predictions(predictions, question)

        if isinstance(aggregate, float) and (self._cal_a, self._cal_b) != (1.0, 0.0):
            calibrated = apply_platt(aggregate, self._cal_a, self._cal_b)
            logger.info(f"Calibration: {aggregate:.4f} -> {calibrated:.4f}")
            return calibrated

        return aggregate
