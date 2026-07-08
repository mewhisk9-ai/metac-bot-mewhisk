# main.py
# mewhisk — Conservative Forecasting Bot (Vultr Serverless Inference)

import argparse
import asyncio
import logging
import os
from datetime import datetime

import numpy as np
from dotenv import load_dotenv
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

load_dotenv()

# -----------------------------
# Environment & API Keys
# -----------------------------
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
VULTR_API_KEY = (
    os.getenv("VULTR_SERVERLESS_INFERENCE_API_KEY")
    or os.getenv("VULTR_INFERENCE_API_KEY")
)
VULTR_BASE_URL = os.getenv(
    "VULTR_INFERENCE_BASE_URL", "https://api.vultrinference.com/v1"
)

# -----------------------------
# Vultr Serverless Inference models
# Catalog: https://api.vultrinference.com/v1/models (requires API key)
# Docs: https://docs.vultr.com/products/compute/serverless-inference/faq
# -----------------------------
VULTR_DEFAULT = os.getenv("VULTR_MODEL_DEFAULT", "llama-3.3-70b-instruct-fp8")
VULTR_PARSER = os.getenv("VULTR_MODEL_PARSER", "qwen2.5-32b-instruct")
VULTR_SUMMARIZER = os.getenv("VULTR_MODEL_SUMMARIZER", "mistral-nemo-instruct-2407")
VULTR_RESEARCHER = os.getenv("VULTR_MODEL_RESEARCHER", "llama-3.3-70b-instruct-fp8")

# Diverse committee across model families for ensemble forecasting
VULTR_COMMITTEE = [
    m.strip()
    for m in os.getenv(
        "VULTR_MODEL_COMMITTEE",
        ",".join(
            [
                "llama-3.3-70b-instruct-fp8",
                "qwen2.5-32b-instruct",
                "deepseek-r1-distill-llama-70b",
            ]
        ),
    ).split(",")
    if m.strip()
]

# Research is considered "failed" if shorter than this
RESEARCH_MIN_CHARS = 150

# Timeouts
RESEARCH_TIMEOUT_S = float(os.getenv("RESEARCH_TIMEOUT_S", "45"))
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "120"))

# -----------------------------
# Logging setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mewhisk")


def _require_vultr_key() -> str:
    if not VULTR_API_KEY:
        raise RuntimeError(
            "Missing Vultr Inference API key. Set "
            "VULTR_SERVERLESS_INFERENCE_API_KEY (or VULTR_INFERENCE_API_KEY) "
            "in your environment / GitHub Actions secrets."
        )
    return VULTR_API_KEY


def _vultr_llm(
    model_id: str,
    *,
    temperature: float | None = 0.2,
    timeout: float | None = None,
) -> GeneralLlm:
    """Build a GeneralLlm pointed at Vultr's OpenAI-compatible endpoint."""
    return GeneralLlm(
        model=f"openai/{model_id}",
        api_key=_require_vultr_key(),
        base_url=VULTR_BASE_URL,
        temperature=temperature,
        timeout=timeout or LLM_TIMEOUT_S,
        allowed_tries=2,
    )


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


def _extremize_binary(p: float, strength: float = 1.35) -> float:
    """
    Mildly push forecasts away from 0.5 after committee aggregation.
    Ensemble medians are often underconfident; this recovers some sharpness
    while keeping hard clamps at [0.01, 0.99].
    """
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    odds = (p / (1 - p)) ** strength
    return float(np.clip(odds / (1 + odds), 0.01, 0.99))


class mewhisk(ForecastBot):
    """
    Conservative forecasting bot using:
    - Research: Tavily + Vultr LLM researcher
    - Models: Vultr Serverless Inference (Llama / Qwen / DeepSeek committee)
    - Aggregation: Median across committee, then mild binary extremization
    - Research grounding: prompts require citing research; failed research widens priors
    - Child/parent: skips child questions automatically
    """

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def _llm_config_defaults(self) -> dict[str, GeneralLlm]:
        return {
            "default": _vultr_llm(VULTR_DEFAULT, temperature=0.25),
            "parser": _vultr_llm(VULTR_PARSER, temperature=0.0),
            "summarizer": _vultr_llm(VULTR_SUMMARIZER, temperature=0.1),
            "researcher": _vultr_llm(VULTR_RESEARCHER, temperature=0.2),
        }

    def __init__(self, *args, **kwargs):
        # Ensure Vultr key exists before ForecastBot builds default LLMs
        _require_vultr_key()
        if "llms" not in kwargs or kwargs["llms"] is None:
            kwargs["llms"] = self._llm_config_defaults()
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
            return "\n".join([f"- {c['content']}" for c in response["results"]])
        except Exception as e:
            return f"Tavily failed: {e}"

    async def run_research(self, question: MetaculusQuestion) -> str:
        if _is_child_question(question):
            logger.info(
                "Skipping research for child question: %s",
                getattr(question, "question_text", "")[:80],
            )
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

        research_prompt = clean_indents(
            f"""
            You are a research assistant for a superforecaster.
            Summarize the most decision-relevant facts for this question.
            Prefer base rates, recent developments, resolution criteria traps,
            and arguments for YES and NO (or each option).
            Do not produce a final probability.

            Question: {question.question_text}
            Background: {getattr(question, "background_info", None)}
            Resolution criteria: {getattr(question, "resolution_criteria", None)}
            Fine print: {getattr(question, "fine_print", None)}
            Today: {datetime.now().strftime("%Y-%m-%d")}
            """
        )

        async with self._concurrency_limiter:
            loop = asyncio.get_running_loop()
            tavily_result = await _safe(
                "tavily",
                loop.run_in_executor(None, self.call_tavily, question.question_text),
            )
            llm_result = await _safe(
                "llm",
                self.get_llm("researcher", "llm").invoke(research_prompt),
            )

            raw_research = (
                f"--- SOURCE TAVILY ---\n{tavily_result}\n\n"
                f"--- SOURCE LLM ---\n{llm_result}\n\n"
            )
            if not _research_ok(raw_research):
                logger.warning(
                    "Research insufficient for Q: %s",
                    getattr(question, "question_text", "")[:80],
                )
            return raw_research

    # -----------------------------
    # Conservative Forecasting with Committee
    # -----------------------------
    async def _single_forecast(
        self,
        question: MetaculusQuestion,
        research: str,
        model_id: str,
    ):
        forecaster = _vultr_llm(model_id, temperature=0.25)
        parser = _vultr_llm(VULTR_PARSER, temperature=0.0)

        grounding = _research_instruction(research)
        research_block = research.strip() or "(no research available)"

        if isinstance(question, BinaryQuestion):
            prompt = clean_indents(
                f"""
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
                """
            )
            reasoning = await forecaster.invoke(prompt)
            pred: BinaryPrediction = await structure_output(
                reasoning, BinaryPrediction, model=parser
            )
            result = max(0.01, min(0.99, pred.prediction_in_decimal))

        elif isinstance(question, MultipleChoiceQuestion):
            prompt = clean_indents(
                f"""
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
                """
            )
            reasoning = await forecaster.invoke(prompt)
            result = await structure_output(
                reasoning,
                PredictedOptionList,
                model=parser,
                additional_instructions=f"Options must be exactly: {question.options}",
            )

        elif isinstance(question, NumericQuestion):
            lower_msg = (
                f"Lower bound: {'open' if question.open_lower_bound else 'closed'} "
                f"at {question.lower_bound or question.nominal_lower_bound}"
            )
            upper_msg = (
                f"Upper bound: {'open' if question.open_upper_bound else 'closed'} "
                f"at {question.upper_bound or question.nominal_upper_bound}"
            )
            prompt = clean_indents(
                f"""
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
                """
            )
            reasoning = await forecaster.invoke(prompt)
            percentile_list: list[Percentile] = await structure_output(
                reasoning, list[Percentile], model=parser
            )
            result = NumericDistribution.from_question(percentile_list, question)

        else:
            raise TypeError(f"Unsupported question type: {type(question)}")

        return result, reasoning

    async def _committee_forecast(self, question: MetaculusQuestion, research: str):
        """Run the model committee in parallel; skip failures."""

        async def _one(model_id: str):
            try:
                pred, reason = await asyncio.wait_for(
                    self._single_forecast(question, research, model_id),
                    timeout=LLM_TIMEOUT_S,
                )
                return model_id, pred, reason
            except asyncio.TimeoutError:
                logger.warning("Committee timeout: %s — skipping", model_id)
                return model_id, None, None
            except Exception as exc:
                logger.warning("Committee error: %s — %s", model_id, exc)
                return model_id, None, None

        results = await asyncio.gather(*[_one(m) for m in VULTR_COMMITTEE])
        forecasts, reasonings = [], []
        for model_id, pred, reason in results:
            if pred is None:
                continue
            forecasts.append(pred)
            reasonings.append(f"[{model_id}] {reason}")
        return forecasts, reasonings

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        if _is_child_question(question):
            logger.info(
                "Skipping child binary question: %s",
                getattr(question, "question_text", "")[:80],
            )
            return ReasonedPrediction(
                prediction_value=0.5, reasoning="Skipped: child question"
            )

        forecasts, reasonings = await self._committee_forecast(question, research)

        if not forecasts:
            logger.error("All committee models failed/timed out — returning 0.5")
            return ReasonedPrediction(
                prediction_value=0.5, reasoning="All models failed"
            )

        p_final = float(np.median(forecasts))

        # If research failed, blend toward 0.5 to reflect higher uncertainty
        if not _research_ok(research):
            logger.warning("Research failed — blending binary forecast toward 0.5")
            p_final = 0.6 * p_final + 0.4 * 0.5
        else:
            p_final = _extremize_binary(p_final)

        return ReasonedPrediction(
            prediction_value=p_final, reasoning=" | ".join(reasonings)
        )

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        if _is_child_question(question):
            logger.info(
                "Skipping child MC question: %s",
                getattr(question, "question_text", "")[:80],
            )
            n = len(question.options)
            return ReasonedPrediction(
                prediction_value=PredictedOptionList(
                    predicted_options=[
                        PredictedOption(option_name=opt, probability=1.0 / n)
                        for opt in question.options
                    ]
                ),
                reasoning="Skipped: child question",
            )

        forecasts, reasonings = await self._committee_forecast(question, research)

        if not forecasts:
            logger.error("All MC committee models failed/timed out — returning uniform")
            n0 = len(question.options)
            return ReasonedPrediction(
                prediction_value=PredictedOptionList(
                    predicted_options=[
                        PredictedOption(option_name=opt, probability=1.0 / n0)
                        for opt in question.options
                    ]
                ),
                reasoning="All models failed",
            )

        option_list = list(question.options)
        n = len(option_list)

        per_model_maps = []
        for f in forecasts:
            mapped = {}
            for o in getattr(f, "predicted_options", []) or []:
                name = getattr(o, "option_name", None)
                prob = getattr(o, "probability", 0.0)
                if name is not None:
                    mapped[str(name)] = max(0.0, float(prob))
            for opt in option_list:
                mapped.setdefault(opt, 0.0)
            s = sum(mapped.values()) or 1.0
            per_model_maps.append({k: v / s for k, v in mapped.items()})

        mat = np.array(
            [[m[opt] for opt in option_list] for m in per_model_maps], dtype=float
        )
        median_probs = np.median(mat, axis=0)
        median_probs = np.maximum(median_probs, 1e-6)
        median_probs = median_probs / median_probs.sum()

        if not _research_ok(research):
            logger.warning("Research failed — blending MC forecast toward uniform")
            uniform = np.full(n, 1.0 / n)
            median_probs = 0.6 * median_probs + 0.4 * uniform
            median_probs = median_probs / median_probs.sum()

        median_forecast = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name=opt, probability=float(p))
                for opt, p in zip(option_list, median_probs)
            ]
        )
        return ReasonedPrediction(
            prediction_value=median_forecast, reasoning=" | ".join(reasonings)
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        if _is_child_question(question):
            logger.info(
                "Skipping child numeric question: %s",
                getattr(question, "question_text", "")[:80],
            )
            return ReasonedPrediction(
                prediction_value=NumericDistribution.from_question(
                    [
                        Percentile(percentile=p, value=0.0)
                        for p in [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
                    ],
                    question,
                ),
                reasoning="Skipped: child question",
            )

        forecasts, reasonings = await self._committee_forecast(question, research)

        if not forecasts:
            logger.error("All numeric committee models failed/timed out")
            raise RuntimeError(
                "All committee models timed out or failed for numeric question"
            )

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

        # Widen intervals when research failed
        if not _research_ok(research) and len(aggregated) >= 2:
            mid = aggregated[len(aggregated) // 2].value
            widened = []
            for item in aggregated:
                delta = item.value - mid
                widened.append(
                    Percentile(percentile=item.percentile, value=mid + 1.25 * delta)
                )
            aggregated = widened

        return ReasonedPrediction(
            prediction_value=NumericDistribution.from_question(aggregated, question),
            reasoning=" | ".join(reasonings),
        )


def _build_bot(*, publish: bool, skip_previous: bool) -> mewhisk:
    return mewhisk(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=publish,
        skip_previously_forecasted_questions=skip_previous,
    )


# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run mewhisk on Vultr Inference.")
    parser.add_argument(
        "--mode",
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
        help="Which question set to forecast on.",
    )
    parser.add_argument(
        "--tournament-ids",
        nargs="+",
        type=str,
        default=None,
        help="Override tournament IDs (tournament mode only).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not publish predictions to Metaculus.",
    )
    args = parser.parse_args()

    publish = not args.dry_run
    skip_previous = args.mode != "test_questions"

    if args.mode == "test_questions":
        tournament_ids = [MetaculusApi.CURRENT_MINIBENCH_ID]
        skip_previous = False
        # Do not publish test runs unless explicitly opted in
        publish = os.getenv("PUBLISH_TEST_QUESTIONS", "").lower() in {
            "1",
            "true",
            "yes",
        } and not args.dry_run
        logger.info("test_questions mode — publish=%s", publish)
    elif args.mode == "metaculus_cup":
        tournament_ids = [str(MetaculusApi.CURRENT_METACULUS_CUP_ID)]
    else:
        tournament_ids = args.tournament_ids or [
            str(MetaculusApi.CURRENT_AI_COMPETITION_ID),
            str(MetaculusApi.CURRENT_MARKET_PULSE_ID),
            str(MetaculusApi.CURRENT_MINIBENCH_ID),
        ]

    bot = _build_bot(publish=publish, skip_previous=skip_previous)

    try:
        all_reports = []
        for tid in tournament_ids:
            logger.info("Forecasting on tournament: %s", tid)
            reports = asyncio.run(
                bot.forecast_on_tournament(tid, return_exceptions=True)
            )
            all_reports.extend(reports)
        bot.log_report_summary(all_reports)
        logger.info("Run completed successfully.")
    except Exception as e:
        logger.error("Critical error: %s", e, exc_info=True)
        raise
