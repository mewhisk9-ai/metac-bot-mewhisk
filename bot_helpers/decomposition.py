"""Question decomposition for complex multi-part forecasting questions."""

from __future__ import annotations

import logging
import re
from typing import Awaitable, Callable

from forecasting_tools import MetaculusQuestion, clean_indents

logger = logging.getLogger("mewhisk.decomposition")

COMPLEXITY_HINTS = re.compile(
    r"\b("
    r"and|or|both|either|whether|if .+ then|conditional|depends|"
    r"by \d{4}|before \d{4}|after \d{4}|at least|more than|less than|"
    r"which of|how many|combination|simultaneously"
    r")\b",
    re.IGNORECASE,
)


def looks_complex(question: MetaculusQuestion) -> bool:
    text = question.question_text or ""
    if len(text) > 220:
        return True
    hits = len(COMPLEXITY_HINTS.findall(text))
    if hits >= 2:
        return True
    # Multiple clauses / question marks often signal multi-part structure
    if text.count("?") >= 2 or text.count(";") >= 1:
        return True
    return False


async def maybe_decompose_question(
    question: MetaculusQuestion,
    invoke_llm: Callable[[str], Awaitable[str]],
    *,
    force: bool = False,
) -> str:
    """
    Return a markdown block of sub-questions + how they combine, or "".
    Uses the provided LLM invoker (already rate-limited by caller).
    """
    if not force and not looks_complex(question):
        return ""

    prompt = clean_indents(
        f"""
        You are helping a superforecaster decompose a complex question.

        Question: {question.question_text}
        Background: {getattr(question, "background_info", None)}
        Resolution criteria: {getattr(question, "resolution_criteria", None)}
        Fine print: {getattr(question, "fine_print", None)}

        Break this into 2–5 simpler sub-questions that would help forecast the
        original. For each sub-question:
        1) State the sub-question clearly
        2) Note why it matters
        3) Give a rough prior / base-rate hint (not a final forecast)

        Then explain briefly how answers to the sub-questions should combine
        into a forecast for the original question (AND/OR/weighted mix).
        Do not give a final probability for the original question.
        """
    )
    try:
        result = await invoke_llm(prompt)
        if not result or len(result.strip()) < 40:
            return ""
        logger.info("Decomposed complex question into sub-questions")
        return f"--- QUESTION DECOMPOSITION ---\n{result.strip()}\n"
    except Exception as exc:
        logger.warning("Decomposition failed: %s", exc)
        return ""
