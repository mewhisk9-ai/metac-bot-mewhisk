# main.py
# mewhisk — Conservative Forecasting Bot (OpenRouter Free Models)

import argparse
import asyncio
import logging
import os
from datetime import datetime

import numpy as np
from forecasting_tools import (
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    BinaryPrediction,
    PredictedOption,
    PredictedOptionList,
    ReasonedPrediction,
    clean_indents,
    structure_output,
)

from tavily import TavilyClient

# -----------------------------
# Environment & API Keys
# -----------------------------
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# -----------------------------
# Free Model IDs (OpenRouter)
# Slugs can change — verify at openrouter.ai/models
# -----------------------------
FREE_DEFAULT    = "openrouter/nvidia/nemotron-3-super-120b-a12b:free"
FREE_PARSER     = "openrouter/openai/gpt-oss-120b"
FREE_SUMMARIZER = "openrouter/openai/gpt-oss-20b"
FREE_RESEARCHER = "openrouter/free"  # auto-router picks best available free model

# Committee of 3 diverse free models for ensemble forecasting
FREE_COMMITTEE = [
    "openrouter/openai/gpt-oss-120b",
    "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
    "openrouter/openai/gpt-oss-20b",
]

# Research is considered "failed" if shorter than this
RESEARCH_MIN_CHARS = 150

# Timeouts
RESEARCH_TIMEOUT_S = float(os.getenv("RESEARCH_TIMEOUT_S", "45"))  # per research source
LLM_TIMEOUT_S      = float(os.getenv("LLM_TIMEOUT_S",      "90"))  # per committee model

# -----------------------------
# Logging setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mewhisk")


def _is_child_question(question: MetaculusQuestion) -> bool:
    """Returns True if this is a sub-question of a group — skip these."""
    try:
        if getattr(question, "group_id", None) is not None:
            return True
        if getattr(question, "parent_id", None) is not None:
            return True
        qtype = str(getattr(question, "question_type", "") or "").lower()
        if qtype in ("group", "conditional", "sub_question"):
            return True
    except Exception:
        pass
    return False


def _research_ok(research: str) -> bool:
    """Returns True if research has enough content to be useful."""
    return bool(research) and len(research.strip()) >= RESEARCH_MIN_CHARS


def _research_instruction(research: str) -> str:
    """Returns the research grounding instruction to inject into prompts."""
    if _research_ok(research):
        return (
            "You MUST ground your probability estimate in the research below. "
            "For each option/outcome, cite at least one specific fact from the research "
            "that supports or undermines it. If the research contradicts your prior, "
            "update toward the research."
        )
    return (
        "WARNING: Research is unavailable or returned no useful content. "
        "Your estimate should reflect higher uncertainty — widen your confidence "
        "interval and lean toward base rates rather than strong opinions."
    )


class mewhisk(ForecastBot):
    """
    Conservative forecasting bot using:
    - Research: Tavily + LLM researcher (OpenRouter free)
    - Models: Nemotron 120B, GPT-OSS 120B, GPT-OSS 20B (via OpenRouter)
    - Aggregation: Median across 3 forecasts
    - Research grounding: prompts require citing research; failed research widens priors
    - Child/parent: skips child questions automatically
    """

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def _llm_config_defaults(self) -> dict[str, str]:
        return {
            "default":    FREE_DEFAULT,
            "parser":     FREE_PARSER,
            "summarizer": FREE_SUMMARIZER,
            "researcher": FREE_RESEARCHER,
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

    # -----------------------------
    # Multi-Source Research
    # -----------------------------
    def call_tavily(self, query: str) -> str:
        if not self.tavily_client.api_key:
            return ""
        try:
            response = self.tavily_client.search(query=query, search_depth="advanced")
            return "\n".join([f"- {c['content']}" for c in response['results']])
        except Exception as e:
            return f"Tavily failed: {e}"

    async def run_research(self, question: MetaculusQuestion) -> str:
        if _is_child_question(question):
            logger.info("Skipping research for child question: %s", getattr(question, "question_text", "")[:80])
            return ""

        async def _safe(label: str, coro) -> str:
            try:
                result = await asyncio.wait_for(coro, timeout=RESEARCH_TIMEOUT_S)
                return result if isinstance(result, str) else str(result)
            except asyncio.TimeoutError:
                logger.warning("Research timeout (%ss): %s", RESEARCH_TIMEOUT_S, label)
                return f"({label} timed out)"
            except Exception as exc:
                logger.warning("Research error %s: %s", label, exc)
                return f"({label} error: {exc})"

        async with self._concurrency_limiter:
            loop = asyncio.get_running_loop()
            tavily_result = await _safe("tavily", loop.run_in_executor(None, self.call_tavily, question.question_text))
            llm_result    = await _safe("llm",    self.get_llm("researcher", "llm").invoke(question.question_text))

            raw_research = (
                f"--- SOURCE TAVILY ---\n{tavily_result}\n\n"
                f"--- SOURCE LLM ---\n{llm_result}\n\n"
            )
            if not _research_ok(raw_research):
                logger.warning("Research insufficient for Q: %s", getattr(question, "question_text", "")[:80])
            return raw_research

    # -----------------------------
    # Conservative Forecasting with Committee
    # -----------------------------
    async def _single_forecast(self, question, research: str, model_override: str = None):
        if model_override:
            self._llms["default"] = GeneralLlm(model=model_override)
            self._llms["parser"]  = GeneralLlm(model=FREE_PARSER)

        grounding = _research_instruction(research)
        research_block = research.strip() or "(no research available)"

        if isinstance(question, BinaryQuestion):
            prompt = clean_indents(f"""
            You are a professional forecaster known for conservative, well-calibrated predictions.

            Question: {question.question_text}
            Background: {question.background_info}
            Resolution criteria: {question.resolution_criteria}
            Fine print: {question.fine_print}
            Today: {datetime.now().strftime("%Y-%m-%d")}

            {grounding}

            Research:
            {research_block}

            Consider:
            (a) Time until resolution
            (b) Status quo (world changes slowly)
            (c) Base rates — if research is missing, stay close to them

            Be humble. Avoid overconfidence.
            End with: "Probability: ZZ%"
            """)
            reasoning = await self.get_llm("default", "llm").invoke(prompt)
            pred: BinaryPrediction = await structure_output(reasoning, BinaryPrediction, model=self.get_llm("parser", "llm"))
            result = max(0.01, min(0.99, pred.prediction_in_decimal))

        elif isinstance(question, MultipleChoiceQuestion):
            prompt = clean_indents(f"""
            You are a conservative forecaster.

            Question: {question.question_text}
            Options: {question.options}
            Background: {question.background_info}
            Resolution: {question.resolution_criteria}
            Today: {datetime.now().strftime("%Y-%m-%d")}

            {grounding}

            Research:
            {research_block}

            For each option, cite a specific fact from the research that supports your estimate.
            Assign non-zero probability to every option unless logically impossible.
            End with probabilities for each option in order.
            """)
            reasoning = await self.get_llm("default", "llm").invoke(prompt)
            result = await structure_output(
                reasoning, PredictedOptionList, model=self.get_llm("parser", "llm"),
                additional_instructions=f"Options must be exactly: {question.options}"
            )

        elif isinstance(question, NumericQuestion):
            lower_msg = f"Lower bound: {'open' if question.open_lower_bound else 'closed'} at {question.lower_bound or question.nominal_lower_bound}"
            upper_msg = f"Upper bound: {'open' if question.open_upper_bound else 'closed'} at {question.upper_bound or question.nominal_upper_bound}"
            prompt = clean_indents(f"""
            You are a conservative forecaster. Set wide 90/10 intervals.

            Question: {question.question_text}
            Units: {question.unit_of_measure or 'Infer from context'}
            Background: {question.background_info}
            Resolution: {question.resolution_criteria}
            {lower_msg}
            {upper_msg}
            Today: {datetime.now().strftime("%Y-%m-%d")}

            {grounding}

            Research:
            {research_block}

            Cite specific facts from the research when setting each percentile.
            Consider status quo, trends, expert views, and black swans.
            Provide percentiles: 10, 20, 40, 60, 80, 90.
            """)
            reasoning = await self.get_llm("default", "llm").invoke(prompt)
            percentile_list: list[Percentile] = await structure_output(reasoning, list[Percentile], model=self.get_llm("parser", "llm"))
            result = NumericDistribution.from_question(percentile_list, question)

        if model_override:
            self._llms["default"] = GeneralLlm(model=FREE_DEFAULT)
            self._llms["parser"]  = GeneralLlm(model=FREE_PARSER)

        return result, reasoning

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        if _is_child_question(question):
            logger.info("Skipping child binary question: %s", getattr(question, "question_text", "")[:80])
            return ReasonedPrediction(prediction_value=0.5, reasoning="Skipped: child question")

        forecasts, reasonings = [], []
        for model in FREE_COMMITTEE:
            try:
                pred, reason = await asyncio.wait_for(
                    self._single_forecast(question, research, model_override=model),
                    timeout=LLM_TIMEOUT_S
                )
                forecasts.append(pred)
                reasonings.append(reason)
            except asyncio.TimeoutError:
                logger.warning("Binary committee timeout: %s — skipping", model)
            except Exception as exc:
                logger.warning("Binary committee error: %s — %s", model, exc)

        if not forecasts:
            logger.error("All committee models failed/timed out — returning 0.5")
            return ReasonedPrediction(prediction_value=0.5, reasoning="All models failed")

        p_final = float(np.median(forecasts))

        # If research failed, blend toward 0.5 to reflect higher uncertainty
        if not _research_ok(research):
            logger.warning("Research failed — blending binary forecast toward 0.5")
            p_final = 0.6 * p_final + 0.4 * 0.5

        return ReasonedPrediction(prediction_value=p_final, reasoning=" | ".join(reasonings))

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str) -> ReasonedPrediction[PredictedOptionList]:
        if _is_child_question(question):
            logger.info("Skipping child MC question: %s", getattr(question, "question_text", "")[:80])
            n = len(question.options)
            return ReasonedPrediction(
                prediction_value=PredictedOptionList(predicted_options=[
                    PredictedOption(option_name=opt, probability=1.0/n)
                    for opt in question.options
                ]),
                reasoning="Skipped: child question"
            )

        forecasts, reasonings = [], []
        for model in FREE_COMMITTEE:
            try:
                pred, reason = await asyncio.wait_for(
                    self._single_forecast(question, research, model_override=model),
                    timeout=LLM_TIMEOUT_S
                )
                forecasts.append(pred)
                reasonings.append(reason)
            except asyncio.TimeoutError:
                logger.warning("MC committee timeout: %s — skipping", model)
            except Exception as exc:
                logger.warning("MC committee error: %s — %s", model, exc)

        if not forecasts:
            logger.error("All MC committee models failed/timed out — returning uniform")
            n0 = len(question.options)
            return ReasonedPrediction(
                prediction_value=PredictedOptionList(predicted_options=[
                    PredictedOption(option_name=opt, probability=1.0/n0)
                    for opt in question.options
                ]),
                reasoning="All models failed"
            )

        option_list = list(question.options)
        n = len(option_list)

        # Build per-model probability maps — PredictedOption is Pydantic, use attribute access
        per_model_maps = []
        for f in forecasts:
            mapped = {}
            for o in (getattr(f, "predicted_options", []) or []):
                name = getattr(o, "option_name", None)
                prob = getattr(o, "probability", 0.0)
                if name is not None:
                    mapped[str(name)] = max(0.0, float(prob))
            for opt in option_list:
                mapped.setdefault(opt, 0.0)
            s = sum(mapped.values()) or 1.0
            per_model_maps.append({k: v / s for k, v in mapped.items()})

        mat = np.array([[m[opt] for opt in option_list] for m in per_model_maps], dtype=float)
        median_probs = np.median(mat, axis=0)
        median_probs = np.maximum(median_probs, 1e-6)
        median_probs = median_probs / median_probs.sum()

        # If research failed, blend toward uniform to reflect higher uncertainty
        if not _research_ok(research):
            logger.warning("Research failed — blending MC forecast toward uniform")
            uniform = np.full(n, 1.0 / n)
            median_probs = 0.6 * median_probs + 0.4 * uniform
            median_probs = median_probs / median_probs.sum()

        median_forecast = PredictedOptionList(predicted_options=[
            PredictedOption(option_name=opt, probability=float(p))
            for opt, p in zip(option_list, median_probs)
        ])
        return ReasonedPrediction(prediction_value=median_forecast, reasoning=" | ".join(reasonings))

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        if _is_child_question(question):
            logger.info("Skipping child numeric question: %s", getattr(question, "question_text", "")[:80])
            return ReasonedPrediction(
                prediction_value=NumericDistribution.from_question(
                    [Percentile(percentile=p, value=0.0) for p in [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]],
                    question
                ),
                reasoning="Skipped: child question"
            )

        forecasts, reasonings = [], []
        for model in FREE_COMMITTEE:
            try:
                pred, reason = await asyncio.wait_for(
                    self._single_forecast(question, research, model_override=model),
                    timeout=LLM_TIMEOUT_S
                )
                forecasts.append(pred)
                reasonings.append(reason)
            except asyncio.TimeoutError:
                logger.warning("Numeric committee timeout: %s — skipping", model)
            except Exception as exc:
                logger.warning("Numeric committee error: %s — %s", model, exc)

        if not forecasts:
            logger.error("All numeric committee models failed/timed out")
            raise RuntimeError("All committee models timed out or failed for numeric question")

        target_percentiles = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
        aggregated = []
        for p in target_percentiles:
            values = []
            for f in forecasts:
                for item in f.declared_percentiles:
                    if abs(item.percentile - p) < 0.01:
                        values.append(item.value)
                        break
                else:
                    values.append(0.0)
            aggregated.append(Percentile(percentile=p, value=float(np.median(values))))

        return ReasonedPrediction(
            prediction_value=NumericDistribution.from_question(aggregated, question),
            reasoning=" | ".join(reasonings)
        )


# -----------------------------
# Entrypoint — Tournament Only
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run mewhisk.")
    parser.add_argument(
        "--tournament-ids",
        nargs="+",
        type=str,
        default=["33022", "market-pulse-26q2", MetaculusApi.CURRENT_MINIBENCH_ID],
    )
    args = parser.parse_args()

    bot = mewhisk(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=False,
    )

    try:
        all_reports = []
        for tid in args.tournament_ids:
            logger.info(f"Forecasting on tournament: {tid}")
            reports = asyncio.run(bot.forecast_on_tournament(tid, return_exceptions=True))
            all_reports.extend(reports)
        bot.log_report_summary(all_reports)
        logger.info("Run completed successfully.")
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
