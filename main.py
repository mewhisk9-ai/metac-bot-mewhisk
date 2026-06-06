import argparse
import asyncio
import logging
import os
from datetime import datetime
from typing import Literal

import numpy as np
import requests
from duckduckgo_search import DDGS
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
    PredictedOptionList,
    ReasonedPrediction,
    clean_indents,
    structure_output,
)

# -----------------------------
# Logging setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ConservativeHybridBot")

# -----------------------------
# Free OpenRouter fallback models (no credits needed)
# -----------------------------
OPENROUTER_FREE_MODELS = [
    "openrouter/free",
    "openrouter/free",
    "openrouter/google/gemma-3-27b-it:free",
]


class ConservativeHybridBot(ForecastBot):
    """
    Conservative forecasting bot using:
    - Research: DuckDuckGo (web + news) + Perplexity (OpenRouter)
    - Models: gpt-5, gpt-4.1-mini, claude-sonnet-4 (OpenRouter)
    - Fallback: free OpenRouter models if primary models fail
    - Aggregation: Median across 3 forecasts
    - Compliance: structure_output + NumericDistribution.from_question()
    """

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def _llm_config_defaults(self) -> dict[str, str]:
        return {
            "default": "openrouter/openai/gpt-5.5",
            "parser": "openrouter/openai/gpt-4.1-mini",
            "summarizer": "openrouter/openai/gpt-5",
            "researcher": "openrouter/perplexity/sonar",
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ddgs = DDGS()

    # -----------------------------
    # Multi-Source Research (DuckDuckGo)
    # -----------------------------
    def call_ddg_web(self, query: str) -> str:
        """DuckDuckGo general web search — replaces Tavily."""
        try:
            results = self.ddgs.text(query, max_results=5)
            if not results:
                return ""
            return "\n".join(
                [f"- {r.get('title', '')}: {r.get('body', '')}" for r in results]
            )
        except Exception as e:
            return f"DDG web search failed: {e}"

    def call_ddg_news(self, query: str) -> str:
        """DuckDuckGo news search — replaces NewsAPI."""
        try:
            results = self.ddgs.news(query, max_results=5)
            if not results:
                return ""
            return "\n".join(
                [
                    f"- [{r.get('date', '')}] {r.get('title', '')}: {r.get('body', '')}"
                    for r in results
                ]
            )
        except Exception as e:
            return f"DDG news search failed: {e}"

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            loop = asyncio.get_running_loop()
            tasks = {
                "ddg_web": loop.run_in_executor(None, self.call_ddg_web, question.question_text),
                "ddg_news": loop.run_in_executor(None, self.call_ddg_news, question.question_text),
                "perplexity": self.get_llm("researcher", "llm").invoke(question.question_text),
            }
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            raw_research = ""
            for i, result in enumerate(results):
                raw_research += f"--- SOURCE {list(tasks.keys())[i].upper()} ---\n{result}\n\n"
            return raw_research

    # -----------------------------
    # LLM invocation with free fallback
    # -----------------------------
    async def _invoke_with_fallback(self, llm_key: str, prompt: str) -> str:
        """Try primary LLM; on failure, fall through to free OpenRouter models."""
        primary_llm = self.get_llm(llm_key, "llm")
        try:
            return await primary_llm.invoke(prompt)
        except Exception as e:
            logger.warning(f"Primary model ({llm_key}) failed: {e}. Trying free fallbacks.")
            for free_model in OPENROUTER_FREE_MODELS:
                try:
                    fallback_llm = GeneralLlm(model=free_model)
                    logger.info(f"Using free fallback: {free_model}")
                    return await fallback_llm.invoke(prompt)
                except Exception as fe:
                    logger.warning(f"Free fallback {free_model} also failed: {fe}")
            raise RuntimeError("All models (primary + free fallbacks) failed.")

    async def _structure_with_fallback(self, reasoning: str, output_type, **kwargs):
        """Try primary parser; on failure, fall through to free OpenRouter models."""
        primary_parser = self.get_llm("parser", "llm")
        try:
            return await structure_output(reasoning, output_type, model=primary_parser, **kwargs)
        except Exception as e:
            logger.warning(f"Primary parser failed: {e}. Trying free fallbacks.")
            for free_model in OPENROUTER_FREE_MODELS:
                try:
                    fallback_llm = GeneralLlm(model=free_model)
                    logger.info(f"Using free parser fallback: {free_model}")
                    return await structure_output(reasoning, output_type, model=fallback_llm, **kwargs)
                except Exception as fe:
                    logger.warning(f"Free parser fallback {free_model} also failed: {fe}")
            raise RuntimeError("All parsers (primary + free fallbacks) failed.")

    # -----------------------------
    # Conservative Forecasting with Committee
    # -----------------------------
    async def _single_forecast(self, question, research: str, model_override: str = None):
        original_default = None
        original_parser = None

        if model_override:
            original_default = self._llms.get("default")
            original_parser = self._llms.get("parser")
            self._llms["default"] = GeneralLlm(model=model_override)
            self._llms["parser"] = GeneralLlm(model="openrouter/openai/gpt-4.1-mini")

        try:
            if isinstance(question, BinaryQuestion):
                prompt = clean_indents(f"""
                You are a professional forecaster known for conservative, well-calibrated predictions.

                Question: {question.question_text}
                Background: {question.background_info}
                Resolution criteria: {question.resolution_criteria}
                Fine print: {question.fine_print}
                Research: {research}
                Today: {datetime.now().strftime("%Y-%m-%d")}

                Consider:
                (a) Time until resolution
                (b) Status quo (world changes slowly)
                (c) Base rates and community estimates (e.g., 30% for major population drops)

                Be humble. Avoid overconfidence.

                End with: "Probability: ZZ%"
                """)
                reasoning = await self._invoke_with_fallback("default", prompt)
                pred: BinaryPrediction = await self._structure_with_fallback(reasoning, BinaryPrediction)
                result = max(0.01, min(0.99, pred.prediction_in_decimal))

            elif isinstance(question, MultipleChoiceQuestion):
                prompt = clean_indents(f"""
                Conservative forecaster mode.

                Question: {question.question_text}
                Options: {question.options}
                Background: {question.background_info}
                Resolution: {question.resolution_criteria}
                Research: {research}
                Today: {datetime.now().strftime("%Y-%m-%d")}

                Assign probabilities. Do not assign 0% to any option unless logically impossible.

                End with probabilities for each option in order.
                """)
                reasoning = await self._invoke_with_fallback("default", prompt)
                result = await self._structure_with_fallback(
                    reasoning,
                    PredictedOptionList,
                    additional_instructions=f"Options must be exactly: {question.options}",
                )

            elif isinstance(question, NumericQuestion):
                lower_msg = f"Lower bound: {'open' if question.open_lower_bound else 'closed'} at {question.lower_bound or question.nominal_lower_bound}"
                upper_msg = f"Upper bound: {'open' if question.open_upper_bound else 'closed'} at {question.upper_bound or question.nominal_upper_bound}"
                prompt = clean_indents(f"""
                Conservative forecaster. Set wide 90/10 intervals.

                Question: {question.question_text}
                Units: {question.unit_of_measure or 'Infer from context'}
                Background: {question.background_info}
                Resolution: {question.resolution_criteria}
                {lower_msg}
                {upper_msg}
                Research: {research}
                Today: {datetime.now().strftime("%Y-%m-%d")}

                Consider status quo, trends, expert views, and black swans.

                Provide percentiles: 10, 20, 40, 60, 80, 90.
                """)
                reasoning = await self._invoke_with_fallback("default", prompt)
                percentile_list: list[Percentile] = await self._structure_with_fallback(
                    reasoning, list[Percentile]
                )
                result = NumericDistribution.from_question(percentile_list, question)

        finally:
            if model_override:
                if original_default is not None:
                    self._llms["default"] = original_default
                if original_parser is not None:
                    self._llms["parser"] = original_parser

        return result, reasoning

    # -----------------------------
    # Binary forecast committee
    # -----------------------------
    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        models = [
            "openrouter/openai/gpt-5.5",
            "openrouter/openai/o4-mini-deep-research",
            "openrouter/anthropic/claude-opus-4.1",
        ]
        forecasts, reasonings = [], []
        for model in models:
            pred, reason = await self._single_forecast(question, research, model_override=model)
            forecasts.append(pred)
            reasonings.append(reason)
        median_pred = float(np.median(forecasts))
        return ReasonedPrediction(
            prediction_value=median_pred, reasoning=" | ".join(reasonings)
        )

    # -----------------------------
    # Multiple-choice forecast committee
    # -----------------------------
    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        models = [
            "openrouter/openai/gpt-5.5",
            "openrouter/openai/o4-mini-deep-research",
            "openrouter/anthropic/claude-opus-4.1",
        ]
        forecasts, reasonings = [], []
        for model in models:
            pred, reason = await self._single_forecast(question, research, model_override=model)
            forecasts.append(pred)
            reasonings.append(reason)
        all_probs = np.array(
            [[opt["probability"] for opt in f.predicted_options] for f in forecasts]
        )
        median_probs = np.median(all_probs, axis=0)
        if median_probs.sum() > 0:
            median_probs = median_probs / median_probs.sum()
        else:
            median_probs = np.full_like(median_probs, 1.0 / len(median_probs))
        options = forecasts[0].predicted_options
        median_forecast = PredictedOptionList(
            [
                {"option": opt["option"], "probability": float(p)}
                for opt, p in zip(options, median_probs)
            ]
        )
        return ReasonedPrediction(
            prediction_value=median_forecast, reasoning=" | ".join(reasonings)
        )

    # -----------------------------
    # Numeric forecast committee
    # -----------------------------
    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        models = [
            "openrouter/openai/gpt-5.5",
            "openrouter/openai/o4-mini-deep-research",
            "openrouter/anthropic/claude-opus-4.1",
        ]
        forecasts, reasonings = [], []
        for model in models:
            pred, reason = await self._single_forecast(question, research, model_override=model)
            forecasts.append(pred)
            reasonings.append(reason)
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
        distribution = NumericDistribution.from_question(aggregated, question)
        return ReasonedPrediction(
            prediction_value=distribution, reasoning=" | ".join(reasonings)
        )


# -----------------------------
# Entrypoint — Tournament Only
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Conservative Hybrid Bot.")
    parser.add_argument(
        "--tournament-ids",
        nargs="+",
        type=str,
        default=["33022", "market-pulse-26q2", MetaculusApi.CURRENT_MINIBENCH_ID],
    )
    args = parser.parse_args()

    bot = ConservativeHybridBot(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
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
