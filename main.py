
import argparse
import asyncio
import logging
import os
from datetime import datetime
from typing import Literal

import numpy as np
import requests
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

from tavily import TavilyClient
from newsapi import NewsApiClient

# -----------------------------
# Environment & API Keys
# -----------------------------
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY")

# -----------------------------
# Free Model IDs (OpenRouter)
# Slugs can change — verify at openrouter.ai/models
# -----------------------------
FREE_DEFAULT    = "openrouter/deepseek/deepseek-r1:free"          # Strong reasoner — main forecasting
FREE_PARSER     = "openrouter/openai/gpt-oss-120b"  # Fast & reliable — structured output
FREE_SUMMARIZER = "openrouter/deepseek/deepseek-v3:free"           # Capable general model
FREE_RESEARCHER = "openrouter/openai/gpt-4o:free"                  # Good for research synthesis

# Committee of 3 diverse free models for ensemble forecasting
FREE_COMMITTEE = [
    "openrouter/deepseek/deepseek-r1:free",
    "openrouter/deepseek/deepseek-v3:free",
    "openrouter/free",
]

# -----------------------------
# Logging setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ConservativeHybridBot")


class ConservativeHybridBot(ForecastBot):
    """
    Conservative forecasting bot using:
    - Research: Tavily + NewsAPI + GPT-4o free (OpenRouter)
    - Models: DeepSeek R1, DeepSeek V3, Llama 3.3 70B (all free via OpenRouter)
    - Aggregation: Median across 3 forecasts
    - Compliance: structure_output + NumericDistribution.from_question()
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
        self.newsapi_client = NewsApiClient(api_key=NEWSAPI_API_KEY)

    # -----------------------------
    # Multi-Source Research
    # -----------------------------
    def call_tavily(self, query: str) -> str:
        if not self.tavily_client.api_key: return ""
        try:
            response = self.tavily_client.search(query=query, search_depth="advanced")
            return "\n".join([f"- {c['content']}" for c in response['results']])
        except Exception as e: return f"Tavily failed: {e}"

    def call_newsapi(self, query: str) -> str:
        if not self.newsapi_client.api_key: return ""
        try:
            articles = self.newsapi_client.get_everything(q=query, language='en', sort_by='relevancy', page_size=5)
            if not articles or not articles.get('articles'): return ""
            return "\n".join([f"- Title: {a['title']}\n  Snippet: {a.get('description', 'N/A')}" for a in articles['articles']])
        except Exception as e: return f"NewsAPI failed: {e}"

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            loop = asyncio.get_running_loop()
            tasks = {
                "tavily":  loop.run_in_executor(None, self.call_tavily, question.question_text),
                "newsapi": loop.run_in_executor(None, self.call_newsapi, question.question_text),
                "llm":     self.get_llm("researcher", "llm").invoke(question.question_text),
            }
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            raw_research = ""
            for i, result in enumerate(results):
                raw_research += f"--- SOURCE {list(tasks.keys())[i].upper()} ---\n{result}\n\n"
            return raw_research

    # -----------------------------
    # Conservative Forecasting with Committee
    # -----------------------------
    async def _single_forecast(self, question, research: str, model_override: str = None):
        if model_override:
            self._llms["default"] = GeneralLlm(model=model_override)
            self._llms["parser"]  = GeneralLlm(model=FREE_PARSER)

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
            reasoning = await self.get_llm("default", "llm").invoke(prompt)
            pred: BinaryPrediction = await structure_output(reasoning, BinaryPrediction, model=self.get_llm("parser", "llm"))
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
            reasoning = await self.get_llm("default", "llm").invoke(prompt)
            result = await structure_output(
                reasoning, PredictedOptionList, model=self.get_llm("parser", "llm"),
                additional_instructions=f"Options must be exactly: {question.options}"
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
            reasoning = await self.get_llm("default", "llm").invoke(prompt)
            percentile_list: list[Percentile] = await structure_output(reasoning, list[Percentile], model=self.get_llm("parser", "llm"))
            result = NumericDistribution.from_question(percentile_list, question)

        if model_override:
            # Restore defaults
            self._llms["default"] = GeneralLlm(model=FREE_DEFAULT)
            self._llms["parser"]  = GeneralLlm(model=FREE_PARSER)

        return result, reasoning

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        forecasts = []
        reasonings = []
        for model in FREE_COMMITTEE:
            pred, reason = await self._single_forecast(question, research, model_override=model)
            forecasts.append(pred)
            reasonings.append(reason)
        median_pred = float(np.median(forecasts))
        return ReasonedPrediction(prediction_value=median_pred, reasoning=" | ".join(reasonings))

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str) -> ReasonedPrediction[PredictedOptionList]:
        forecasts = []
        reasonings = []
        for model in FREE_COMMITTEE:
            pred, reason = await self._single_forecast(question, research, model_override=model)
            forecasts.append(pred)
            reasonings.append(reason)
        all_probs = np.array([[opt["probability"] for opt in f.predicted_options] for f in forecasts])
        median_probs = np.median(all_probs, axis=0)
        if median_probs.sum() > 0:
            median_probs = median_probs / median_probs.sum()
        else:
            median_probs = np.full_like(median_probs, 1.0 / len(median_probs))
        options = forecasts[0].predicted_options
        median_forecast = PredictedOptionList([
            {"option": opt["option"], "probability": float(p)} for opt, p in zip(options, median_probs)
        ])
        return ReasonedPrediction(prediction_value=median_forecast, reasoning=" | ".join(reasonings))

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        forecasts = []
        reasonings = []
        for model in FREE_COMMITTEE:
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
            median_val = float(np.median(values))
            aggregated.append(Percentile(percentile=p, value=median_val))
        distribution = NumericDistribution.from_question(aggregated, question)
        return ReasonedPrediction(prediction_value=distribution, reasoning=" | ".join(reasonings))


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
