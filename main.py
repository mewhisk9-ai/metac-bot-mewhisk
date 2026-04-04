"""
mewhisk — Metaculus forecasting bot
=====================================
Agent routing   : AgentRouter (https://agentrouter.org)
Primary model   : claude-sonnet-4-6       (fast, calibrated)
Checker model   : claude-opus-4-6         (deep adversarial)
Research        : perplexity/sonar-pro    (online, live web)  ← primary
                + openai/gpt-4o-search-preview (online, live web) ← secondary
                  Both run in parallel, concatenated into one research block.
                  Override via MEWHISK_RESEARCH_MODEL_1 / _MODEL_2 env vars.
Extremization   : ON — conservative (k_binary=1.15, k_mc=1.12)
Concurrency     : 4 questions in parallel, 6 LLM slots
Tournaments     : minibench + Spring AI Tournament only
Bot comments    : clean prose — no internal labels or model names published
"""

import argparse
import asyncio
import logging
import os
import textwrap
import re
import math
from datetime import datetime
from typing import List, Union, Dict, Any, Optional, Tuple

# ---------------------------------------------------------------
# AgentRouter env setup
# forecasting-tools uses litellm; pointing ANTHROPIC_API_BASE
# at AgentRouter makes every anthropic/* call route through it.
# All model calls (research + forecast) go through AgentRouter.
# ---------------------------------------------------------------
_AGENTROUTER_KEY  = os.getenv("AGENTROUTER_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
_AGENTROUTER_BASE = "https://agentrouter.org/v1"

if _AGENTROUTER_KEY:
    os.environ.setdefault("ANTHROPIC_API_KEY",  _AGENTROUTER_KEY)
    os.environ.setdefault("ANTHROPIC_API_BASE", _AGENTROUTER_BASE)

import requests as _requests

from forecasting_tools import (
    BinaryQuestion,
    ForecastBot,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    BinaryPrediction,
    PredictedOptionList,
    PredictedOption,
    ReasonedPrediction,
    clean_indents,
    structure_output,
)

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mewhisk")

# ============================================================
# Env / API key check
# ============================================================
AGENTROUTER_API_KEY = _AGENTROUTER_KEY
if not AGENTROUTER_API_KEY:
    logger.error("AGENTROUTER_API_KEY (or ANTHROPIC_API_KEY) not set.")

# ---------------------------------------------------------------
# Dual online research models — both web-search-capable,
# routed through AgentRouter. No separate API keys required.
# Override via env vars without touching code.
# ---------------------------------------------------------------
RESEARCH_MODEL_1 = os.getenv("MEWHISK_RESEARCH_MODEL_1", "perplexity/sonar-pro")
RESEARCH_MODEL_2 = os.getenv("MEWHISK_RESEARCH_MODEL_2", "openai/gpt-4o-search-preview")

# ============================================================
# Concurrency controls — module-level so they are shared across
# all bot instances and all async tasks in the process.
#
# _Q_SEM  : max questions being processed end-to-end at once
# _LLM_SEM: max simultaneous outbound LLM / research HTTP calls
# ============================================================
_Q_SEM   = asyncio.Semaphore(4)
_LLM_SEM = asyncio.Semaphore(6)

# ============================================================
# Helpers: stats + parsing
# ============================================================
def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def safe_median(lst: List[Union[float, int]]) -> float:
    """Returns 0.5 on empty input instead of raising."""
    vals = sorted(float(x) for x in lst if _is_num(x))
    if not vals:
        return 0.5
    n   = len(vals)
    mid = n // 2
    return (vals[mid - 1] + vals[mid]) / 2.0 if n % 2 == 0 else vals[mid]


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs: List[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def ci90(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return 0.0, 1.0
    m, s = mean(xs), stdev(xs)
    se   = s / math.sqrt(len(xs))
    return max(0.0, m - 1.645 * se), min(1.0, m + 1.645 * se)


def entropy(probs: Dict[str, float]) -> float:
    return -sum(p * math.log(p) for p in probs.values() if p > 0)


def safe_float(x: Any, default: Optional[float] = 0.0) -> Optional[float]:
    try:
        if x is None:
            return default
        s = str(x).strip().replace(",", "").replace("%", "")
        return float(s) if s else default
    except Exception:
        return default


def normalize_percentile(p: Any) -> float:
    perc = safe_float(p, default=0.5) or 0.5
    if perc > 1.0:
        perc /= 100.0
    return float(max(0.0, min(1.0, perc)))


def clamp01(p: float, lo: float = 0.01, hi: float = 0.99) -> float:
    return max(lo, min(hi, float(p)))


_PERCENT_RE = re.compile(r"(?i)\bprob(?:ability)?\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%")
_DEC_RE     = re.compile(r"(?i)\bdecimal\s*:\s*([0-9]*\.?[0-9]+)\b")


def extract_binary_prob_from_text(text: str) -> Optional[float]:
    if not text:
        return None
    m = _PERCENT_RE.search(text)
    if m:
        val = safe_float(m.group(1), default=None)
        return clamp01(val / 100.0, 0.0, 1.0) if val is not None else None
    m = _DEC_RE.search(text)
    if m:
        val = safe_float(m.group(1), default=None)
        return clamp01(val, 0.0, 1.0) if val is not None else None
    m2 = re.search(r"(?<!\d)([0-9]{1,3}(?:\.[0-9]+)?)\s*%", text)
    if m2:
        val = safe_float(m2.group(1), default=None)
        return clamp01(val / 100.0, 0.0, 1.0) if val is not None else None
    return None


def build_indexed_options(options: List[str]) -> List[str]:
    return [f"{i+1}) {opt}" for i, opt in enumerate(options)]


def extract_indexed_mc_probs(text: str, n_options: int) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for pat in (
        re.compile(r"(?i)\b(?:option\s*)?(\d{1,2})\s*[:\)\-]\s*([0-9]+(?:\.[0-9]+)?)\s*%"),
        re.compile(r"(?i)\b(\d{1,2})\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*%"),
    ):
        for m in pat.finditer(text):
            idx = int(m.group(1))
            if 1 <= idx <= n_options:
                pct = safe_float(m.group(2), default=None)
                if pct is not None:
                    out[idx] = pct / 100.0
    return out


def extract_numeric_percentiles(text: str, targets: List[float]) -> Dict[float, float]:
    out: Dict[float, float] = {}
    for pt in targets:
        pi = int(round(pt * 100))
        for pat in (
            re.compile(rf"(?i)\bpercentile\s*{pi}\s*:\s*([-+]?[0-9,]*\.?[0-9]+)"),
            re.compile(rf"(?i)\bp\s*{pi}\s*:\s*([-+]?[0-9,]*\.?[0-9]+)"),
            re.compile(rf"(?i)\bp{pi}\s*=\s*([-+]?[0-9,]*\.?[0-9]+)"),
        ):
            m = pat.search(text)
            if m:
                v = safe_float(m.group(1), default=None)
                if v is not None:
                    out[pt] = v
                    break
    return out


def build_query(question: MetaculusQuestion, max_chars: int = 450) -> str:
    """
    Build a compact search query from question text + resolution criteria
    (primary search signal) with background as fallback padding.
    All URLs are stripped first.
    """
    def _clean(s: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"http\S+", "", s or "")).strip()

    q  = _clean(question.question_text)
    rc = _clean(question.resolution_criteria)
    bg = _clean(question.background_info)

    for candidate in (
        f"{q} — {rc}",
        f"{q} — {textwrap.shorten(rc, width=max(10, max_chars - len(q) - 3), placeholder='…')}",
        f"{q} — {textwrap.shorten(bg, width=max(10, max_chars - len(q) - 3), placeholder='…')}",
        q,
    ):
        if candidate and len(candidate) <= max_chars:
            return candidate
    return textwrap.shorten(q, width=max_chars, placeholder="…")


# ============================================================
# EXTREMIZE — conservative / humble
# Binary : k=1.15  (gentle logit push toward conviction)
# MC     : k=1.12  (gentle power sharpening)
# Numeric: no extremization — preserve distributional calibration
# ============================================================
def _logit(p: float) -> float:
    p = clamp01(p, 1e-6, 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    z    = math.exp(-abs(x))
    base = 1.0 / (1.0 + z)
    return base if x >= 0 else 1.0 - base


def extremize_binary(p: float, k: float) -> float:
    if not _is_num(p) or not _is_num(k) or k <= 0 or abs(k - 1.0) < 1e-12:
        return float(p)
    return clamp01(_sigmoid(_logit(float(p)) * float(k)))


def extremize_mc(probs: Dict[str, float], k: float) -> Dict[str, float]:
    if not probs or not _is_num(k) or k <= 0 or abs(k - 1.0) < 1e-12:
        s = sum(max(0.0, float(v)) for v in probs.values())
        return (
            {a: max(0.0, float(v)) / s for a, v in probs.items()}
            if s > 0 else {a: 1.0 / len(probs) for a in probs}
        )
    powered = {a: max(0.0, float(v)) ** float(k) for a, v in probs.items()}
    s2      = sum(powered.values())
    return {a: v / s2 for a, v in powered.items()} if s2 > 0 else {a: 1.0 / len(probs) for a in probs}


# ============================================================
# Weighted blend — handles partial agent failure cleanly
# ============================================================
def _weighted_blend(pairs: List[Tuple[float, Optional[float]]]) -> float:
    """
    Blend (weight, value) pairs, skipping None values and
    renormalising remaining weights so partial failures degrade gracefully.
    Returns 0.5 only when every value is None.
    """
    valid   = [(w, float(v)) for w, v in pairs if v is not None and _is_num(v)]
    total_w = sum(w for w, _ in valid)
    if not valid or total_w == 0:
        return 0.5
    return sum(w * v / total_w for w, v in valid)


# ============================================================
# Isotonic regression for monotone percentile enforcement
# Pool-adjacent-violators algorithm — avoids tail distortion
# caused by the hard-clamp approach.
# ============================================================
def _isotonic_regression_increasing(vals: List[float]) -> List[float]:
    """
    PAV algorithm: returns a non-decreasing sequence of the same length.
    Each block carries equal weight (count = 1 per original element).
    """
    if not vals:
        return []
    # Represent each block as (average_value, list_of_original_indices)
    blocks: List[Tuple[float, List[int]]] = [(v, [i]) for i, v in enumerate(vals)]
    result = list(vals)

    i = 0
    while i < len(blocks) - 1:
        avg_i = blocks[i][0]
        avg_j = blocks[i + 1][0]
        if avg_i > avg_j:                        # monotonicity violation — merge
            merged_indices = blocks[i][1] + blocks[i + 1][1]
            merged_avg     = sum(result[j] for j in merged_indices) / len(merged_indices)
            for j in merged_indices:
                result[j] = merged_avg
            blocks = blocks[:i] + [(merged_avg, merged_indices)] + blocks[i + 2:]
            if i > 0:
                i -= 1                           # recheck left neighbour
        else:
            i += 1
    return result


def enforce_monotone(pts: List[Percentile]) -> List[Percentile]:
    """Apply isotonic regression to a sorted-by-percentile list of Percentile objects."""
    pts    = sorted(pts, key=lambda x: float(x.percentile))
    values = _isotonic_regression_increasing([float(p.value) for p in pts])
    for p, v in zip(pts, values):
        p.value = v
    return pts


# ============================================================
# Percentile interpolation — linear between bracketing points
# ============================================================
def interpolate_percentile(source: Dict[float, float], target: float) -> Optional[float]:
    """
    Linearly interpolate within the source dict; clamp to edge values
    outside the known range. Returns None only when source is empty.
    """
    if not source:
        return None
    keys = sorted(source.keys())
    if target <= keys[0]:
        return source[keys[0]]
    if target >= keys[-1]:
        return source[keys[-1]]
    for i in range(len(keys) - 1):
        lo, hi = keys[i], keys[i + 1]
        if lo <= target <= hi:
            frac = (target - lo) / (hi - lo) if hi != lo else 0.0
            return source[lo] + frac * (source[hi] - source[lo])
    return None


# ============================================================
# Dual online research via AgentRouter
# ============================================================
_RESEARCH_SYSTEM = (
    "You are a precise real-time news researcher. Today is {today}. "
    "Search the web and give a concise, factual digest relevant to the "
    "forecasting question below. Include exact dates, numbers, and source names. "
    "Max 400 words. Cite sources inline where possible."
)
_RESEARCH_USER = "Research for forecasting: {query}"


def _sync_online_research(model: str, query: str, label: str) -> str:
    """
    Single online research call through AgentRouter's
    OpenAI-compatible /v1/chat/completions endpoint.
    Only AGENTROUTER_API_KEY is required — no provider-specific keys.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        resp = _requests.post(
            f"{_AGENTROUTER_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {AGENTROUTER_API_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "model":    model,
                "messages": [
                    {"role": "system", "content": _RESEARCH_SYSTEM.format(today=today)},
                    {"role": "user",   "content": _RESEARCH_USER.format(query=query)},
                ],
                "max_tokens":  650,
                "temperature": 0.1,
            },
            timeout=45,
        )
        resp.raise_for_status()
        data    = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        # Perplexity-style inline citations when present
        citations = data.get("citations", [])
        if citations:
            content += "\nSources: " + " | ".join(str(c) for c in citations[:5])
        return content
    except Exception as e:
        logger.warning(f"Research model {label} ({model}) failed: {e}")
        return f"[{label} research unavailable: {e}]"


async def _run_research_pipeline(query: str, loop: asyncio.AbstractEventLoop) -> str:
    """
    Run both online research models in parallel under the shared LLM semaphore.
    Both results are concatenated into one research block for the forecaster.
    A failure in one source still leaves the other available.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    results: Dict[int, str] = {}

    async def fetch(slot: int, model: str, label: str) -> None:
        async with _LLM_SEM:
            results[slot] = await loop.run_in_executor(
                None, _sync_online_research, model, query, label
            )

    await asyncio.gather(
        fetch(1, RESEARCH_MODEL_1, "R1"),
        fetch(2, RESEARCH_MODEL_2, "R2"),
    )

    return (
        f"[Research as of {today}]\n\n"
        f"=== ONLINE SOURCE 1 ({RESEARCH_MODEL_1}) ===\n{results.get(1, '[unavailable]')}\n\n"
        f"=== ONLINE SOURCE 2 ({RESEARCH_MODEL_2}) ===\n{results.get(2, '[unavailable]')}"
    )


# ============================================================
# Bot
# ============================================================
class mewhisk(ForecastBot):
    """
    mewhisk — dual-model forecasting bot via AgentRouter.

    Research  : two online-capable models run in parallel
                (default: perplexity/sonar-pro + openai/gpt-4o-search-preview)
    Primary   : claude-sonnet-4-6
    Adversarial: claude-opus-4-6   (run concurrently with primary)

    All calls share a single AGENTROUTER_API_KEY.
    Extremization: ON, conservative — k_binary=1.15, k_mc=1.12.
    Numeric distributions: no extremization.
    Published comments: clean prose, no internal debug labels or model names.
    """

    _max_concurrent_questions            = 4
    _concurrency_limiter                 = _Q_SEM
    _structure_output_validation_samples = 3

    def _llm_config_defaults(self) -> Dict[str, str]:
        defaults = super()._llm_config_defaults()
        defaults.update({
            "forecaster_sonnet": "anthropic/claude-sonnet-4-6",
            "forecaster_opus":   "anthropic/claude-opus-4-6",
            "parser":            "anthropic/claude-sonnet-4-6",
        })
        return defaults

    def __init__(
        self,
        *args,
        extremize_enabled:  bool  = True,
        extremize_k_binary: float = 1.15,
        extremize_k_mc:     float = 1.12,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.extremize_enabled  = bool(extremize_enabled)
        self.extremize_k_binary = float(extremize_k_binary)
        self.extremize_k_mc     = float(extremize_k_mc)
        self._drop_counts:          Dict[str, int]            = {}
        self._drop_counts_by_model: Dict[str, Dict[str, int]] = {"sonnet": {}, "opus": {}}
        logger.info(
            f"mewhisk ready | AgentRouter={_AGENTROUTER_BASE} | "
            f"research=[{RESEARCH_MODEL_1}, {RESEARCH_MODEL_2}] | "
            f"forecast=[sonnet(primary), opus(adversarial)] | "
            f"extremize=ON k_bin={self.extremize_k_binary} k_mc={self.extremize_k_mc} | "
            f"concurrency=questions:{self._max_concurrent_questions} llm_slots:6"
        )

    def _inc_drop(self, tag: str, reason: str) -> None:
        self._drop_counts[reason] = self._drop_counts.get(reason, 0) + 1
        d = self._drop_counts_by_model.get(tag, {})
        d[reason] = d.get(reason, 0) + 1
        self._drop_counts_by_model[tag] = d

    # ----------------------------------------------------------
    # Research
    # ----------------------------------------------------------
    async def run_research(self, question: MetaculusQuestion) -> str:
        async with _Q_SEM:
            query = build_query(question)
            loop  = asyncio.get_running_loop()
            return await _run_research_pipeline(query, loop)

    # ----------------------------------------------------------
    # LLM helpers (all gated by _LLM_SEM)
    # ----------------------------------------------------------
    async def _invoke_llm(self, model_name: str, prompt: str) -> str:
        async with _LLM_SEM:
            return await self.get_llm(model_name, "llm").invoke(prompt)

    async def _invoke_with_format_retry(self, model_name: str, prompt: str, _: str) -> str:
        return await self._invoke_llm(model_name, prompt)

    # ----------------------------------------------------------
    # Parsers
    # ----------------------------------------------------------
    async def _parse_binary(self, raw: str, tag: str) -> Optional[float]:
        try:
            pred: BinaryPrediction = await structure_output(
                text_to_structure=raw,
                output_type=BinaryPrediction,
                model=self.get_llm("parser", "llm"),
                num_validation_samples=self._structure_output_validation_samples,
            )
            val = safe_float(getattr(pred, "prediction_in_decimal", None), default=None)
            if val is not None:
                return clamp01(float(val))
        except Exception:
            self._inc_drop(tag, "parse_error_binary_structured")
        val2 = extract_binary_prob_from_text(raw)
        if val2 is None:
            self._inc_drop(tag, "parse_error_binary_fallback")
        return clamp01(float(val2)) if val2 is not None else None

    async def _parse_mc(
        self, raw: str, question: MultipleChoiceQuestion, tag: str
    ) -> Optional[Dict[str, float]]:
        options = list(question.options)
        try:
            pred: PredictedOptionList = await structure_output(
                text_to_structure=raw,
                output_type=PredictedOptionList,
                model=self.get_llm("parser", "llm"),
                additional_instructions=f"Valid options: {options}.",
                num_validation_samples=self._structure_output_validation_samples,
            )
            pred_dict = {
                str(po.option_name).strip(): float(po.probability)
                for po in pred.predicted_options if _is_num(po.probability)
            }
            out: Dict[str, float] = {}
            for opt in options:
                if opt in pred_dict:
                    out[opt] = pred_dict[opt]
                else:
                    for k, v in pred_dict.items():
                        if k.casefold() == opt.casefold():
                            out[opt] = v
                            break
            if out:
                return out
        except Exception:
            self._inc_drop(tag, "parse_error_mc_structured")
        idx_probs = extract_indexed_mc_probs(raw, len(options))
        if not idx_probs:
            self._inc_drop(tag, "parse_error_mc_fallback")
            return None
        out2 = {
            options[i - 1]: float(idx_probs[i])
            for i in range(1, len(options) + 1) if i in idx_probs
        }
        if not out2:
            self._inc_drop(tag, "parse_error_mc_fallback_empty")
        return out2 or None

    async def _parse_numeric(
        self, raw: str, question: NumericQuestion, tag: str
    ) -> Optional[NumericDistribution]:
        targets = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
        try:
            percentile_list: List[Percentile] = await structure_output(
                text_to_structure=raw,
                output_type=list[Percentile],
                model=self.get_llm("parser", "llm"),
                num_validation_samples=self._structure_output_validation_samples,
            )
            pts = [
                Percentile(
                    value=float(v),
                    percentile=normalize_percentile(getattr(p, "percentile", 0.5)),
                )
                for p in percentile_list
                if (v := safe_float(getattr(p, "value", None), default=None)) is not None
            ]
            if pts:
                return NumericDistribution.from_question(enforce_monotone(pts), question)
        except Exception:
            self._inc_drop(tag, "parse_error_numeric_structured")
        extracted = extract_numeric_percentiles(raw, targets)
        if not extracted:
            self._inc_drop(tag, "parse_error_numeric_fallback")
            return None
        pts2 = [
            Percentile(percentile=pt, value=float(extracted[pt]))
            for pt in targets if pt in extracted
        ]
        return NumericDistribution.from_question(enforce_monotone(pts2), question)

    # ----------------------------------------------------------
    # Bounds helpers
    # ----------------------------------------------------------
    def _bounds_messages(self, q: NumericQuestion) -> Tuple[str, str]:
        low  = q.nominal_lower_bound if q.nominal_lower_bound is not None else q.lower_bound
        high = q.nominal_upper_bound if q.nominal_upper_bound is not None else q.upper_bound
        return (
            f"Cannot be lower than {low}." if not q.open_lower_bound else f"Unlikely below {low}.",
            f"Cannot be higher than {high}." if not q.open_upper_bound else f"Unlikely above {high}.",
        )

    def _numeric_midpoint(self, q: NumericQuestion) -> float:
        try:
            return (float(q.lower_bound or 0) + float(q.upper_bound or 100)) / 2.0
        except Exception:
            return 50.0

    # ----------------------------------------------------------
    # Prompts — no model names, no internal labels in output
    # ----------------------------------------------------------
    def _binary_prompt(self, q: BinaryQuestion, research: str, role: str) -> str:
        return clean_indents(f"""
            You are a calibrated forecaster. Maximise log score and Brier score.
            Principles:
            1) Apply outside-view base rates first.
            2) Steelman the opposite side before finalising.
            Role: {role} | Date: {datetime.now().strftime('%Y-%m-%d')}

            Question: {q.question_text}
            Background: {q.background_info}
            Resolution Criteria: {q.resolution_criteria}
            Fine Print: {q.fine_print}

            Research (current, treat as factual):
            {research}

            Write 5–8 reasoning bullets, then end EXACTLY with:
            Probability: ZZ%
            Decimal: 0.ZZ
        """)

    def _mc_prompt(self, q: MultipleChoiceQuestion, research: str, role: str) -> str:
        options = list(q.options)
        return clean_indents(f"""
            You are a calibrated forecaster. Maximise proper scoring rules.
            Principles:
            1) Apply outside-view base rates first.
            2) Steelman the leading alternative.
            Role: {role} | Date: {datetime.now().strftime('%Y-%m-%d')}

            Question: {q.question_text}
            Background: {q.background_info}
            Resolution Criteria: {q.resolution_criteria}
            Fine Print: {q.fine_print}

            Options:
            {chr(10).join(build_indexed_options(options))}

            Research (current, treat as factual):
            {research}

            Write 5–8 reasoning bullets, then end EXACTLY with {len(options)} lines:
            1: XX%  ...  {len(options)}: XX%   (must sum to 100%)
        """)

    def _numeric_prompt(self, q: NumericQuestion, research: str, role: str) -> str:
        low_msg, high_msg = self._bounds_messages(q)
        return clean_indents(f"""
            You are a calibrated forecaster. Maximise proper scoring rules.
            Principles:
            1) Apply outside-view base rates first.
            2) Steelman a much-lower or much-higher outcome.
            Role: {role} | Date: {datetime.now().strftime('%Y-%m-%d')}

            Question: {q.question_text}
            Background: {q.background_info}
            Resolution Criteria: {q.resolution_criteria}
            Fine Print: {q.fine_print}
            Units: {getattr(q, 'unit_of_measure', 'inferred')}
            Bounds: {low_msg} {high_msg}

            Research (current, treat as factual):
            {research}

            Write 5–8 reasoning bullets, then end EXACTLY with:
            Percentile 10: X
            Percentile 20: X
            Percentile 40: X
            Percentile 60: X
            Percentile 80: X
            Percentile 90: X
        """)

    # ----------------------------------------------------------
    # Per-role runners (with retry on parse failure)
    # ----------------------------------------------------------
    async def _run_binary_role(
        self, q: BinaryQuestion, research: str, tag: str, role: str
    ) -> ReasonedPrediction[float]:
        model = f"forecaster_{tag}"
        raw   = await self._invoke_with_format_retry(model, self._binary_prompt(q, research, role), "bin")
        val   = await self._parse_binary(raw, tag)
        if val is None:
            try:
                raw2 = await self._invoke_llm(model, "Output ONLY:\nProbability: ZZ%\nDecimal: 0.ZZ")
                val  = await self._parse_binary(raw2, tag)
                raw += "\n\n[RETRY]\n" + raw2
            except Exception:
                self._inc_drop(tag, "retry_failed_binary")
        if val is None:
            self._inc_drop(tag, "invalid_binary")
            val = 0.5
        return ReasonedPrediction(prediction_value=clamp01(val), reasoning=raw)

    async def _run_mc_role(
        self, q: MultipleChoiceQuestion, research: str, tag: str, role: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        model = f"forecaster_{tag}"
        raw   = await self._invoke_with_format_retry(model, self._mc_prompt(q, research, role), "mc")
        probs = await self._parse_mc(raw, q, tag)
        if probs is None:
            try:
                raw2  = await self._invoke_llm(model, "Output ONLY numbered % lines summing to 100%.")
                probs = await self._parse_mc(raw2, q, tag)
                raw  += "\n\n[RETRY]\n" + raw2
            except Exception:
                self._inc_drop(tag, "retry_failed_mc")
        if probs is None:
            self._inc_drop(tag, "invalid_mc")
            u     = 1.0 / max(1, len(q.options))
            probs = {opt: u for opt in q.options}
        return ReasonedPrediction(
            prediction_value=PredictedOptionList(
                predicted_options=[PredictedOption(option_name=o, probability=p) for o, p in probs.items()]
            ),
            reasoning=raw,
        )

    async def _run_numeric_role(
        self, q: NumericQuestion, research: str, tag: str, role: str
    ) -> ReasonedPrediction[NumericDistribution]:
        model = f"forecaster_{tag}"
        raw   = await self._invoke_with_format_retry(model, self._numeric_prompt(q, research, role), "num")
        dist  = await self._parse_numeric(raw, q, tag)
        if dist is None:
            try:
                raw2 = await self._invoke_llm(
                    model,
                    "Output ONLY:\nPercentile 10: X\nPercentile 20: X\nPercentile 40: X\n"
                    "Percentile 60: X\nPercentile 80: X\nPercentile 90: X",
                )
                dist  = await self._parse_numeric(raw2, q, tag)
                raw  += "\n\n[RETRY]\n" + raw2
            except Exception:
                self._inc_drop(tag, "retry_failed_numeric")
        if dist is None:
            self._inc_drop(tag, "invalid_numeric")
            dist = NumericDistribution.from_question(
                [Percentile(value=self._numeric_midpoint(q), percentile=0.5)], q
            )
        return ReasonedPrediction(prediction_value=dist, reasoning=raw)

    # Required single-agent overrides (used when _make_prediction is bypassed)
    async def _run_forecast_on_binary(self, q, research):
        return await self._run_binary_role(q, research, "sonnet", "PRIMARY")

    async def _run_forecast_on_multiple_choice(self, q, research):
        return await self._run_mc_role(q, research, "sonnet", "PRIMARY")

    async def _run_forecast_on_numeric(self, q, research):
        return await self._run_numeric_role(q, research, "sonnet", "PRIMARY")

    # ----------------------------------------------------------
    # Clean reasoning for public Metaculus comments.
    # Strips internal stat lines, agent labels, and separator lines.
    # ----------------------------------------------------------
    @staticmethod
    def _clean_for_publish(reasoning: str) -> str:
        output_lines: List[str] = []
        for line in reasoning.splitlines():
            stripped = line.strip()
            if stripped.startswith("[stats]"):
                continue
            if re.match(r"^\[AGENT_[A-Z]\]", stripped):
                continue
            if stripped == "---":
                continue
            output_lines.append(line)
        # Collapse runs of blank lines to a single blank
        cleaned  = "\n".join(output_lines)
        cleaned  = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    # ----------------------------------------------------------
    # Dual-agent prediction core
    # Sonnet + Opus run concurrently; results are blended with
    # weight-renormalisation so one failure degrades gracefully.
    # ----------------------------------------------------------
    async def _make_prediction(self, question: MetaculusQuestion, research: str):
        async with _Q_SEM:
            preds:      List[Tuple[str, Any]] = []
            reasonings: List[str]             = []
            w_s, w_o = 0.55, 0.45

            async def run_agent(tag: str, role: str) -> None:
                try:
                    if isinstance(question, BinaryQuestion):
                        p = await self._run_binary_role(question, research, tag, role)
                    elif isinstance(question, MultipleChoiceQuestion):
                        p = await self._run_mc_role(question, research, tag, role)
                    else:
                        p = await self._run_numeric_role(question, research, tag, role)
                    label = "A" if tag == "sonnet" else "B"
                    preds.append((tag, p.prediction_value))
                    reasonings.append(f"[AGENT_{label}]\n{p.reasoning}")
                except Exception as e:
                    logger.error(f"Agent {tag} failed: {e}")

            # Sonnet + Opus in parallel
            await asyncio.gather(
                run_agent("sonnet", "PRIMARY"),
                run_agent("opus",   "ADVERSARIAL_CHECKER"),
            )

            if not preds:
                raise RuntimeError("All forecast agents failed.")

            internal_reasoning = "\n\n---\n\n".join(reasonings)

            def get_val(tag: str) -> Any:
                for t, v in preds:
                    if t == tag:
                        return v
                return None

            s_pred = get_val("sonnet")
            o_pred = get_val("opus")

            # ---- BINARY ----
            if isinstance(question, BinaryQuestion):
                s_val = float(s_pred) if s_pred is not None and _is_num(s_pred) else None
                o_val = float(o_pred) if o_pred is not None and _is_num(o_pred) else None
                final = clamp01(_weighted_blend([(w_s, s_val), (w_o, o_val)]))
                pre   = final
                if self.extremize_enabled:
                    final = extremize_binary(final, self.extremize_k_binary)
                np_       = [v for v in [s_val, o_val] if v is not None] or [0.5]
                m, md, sd = mean(np_), safe_median(np_), stdev(np_)
                lo, hi    = ci90(np_)
                stats = (
                    f"[stats] n={len(np_)} mean={m:.3f} median={md:.3f} sd={sd:.3f} "
                    f"ci90=({lo:.3f},{hi:.3f}) "
                    f"extremize(k={self.extremize_k_binary:.2f}) {pre:.3f}→{final:.3f}"
                )
                return ReasonedPrediction(
                    prediction_value=final,
                    reasoning=self._clean_for_publish(stats + "\n\n" + internal_reasoning),
                )

            # ---- MC ----
            if isinstance(question, MultipleChoiceQuestion):
                options = list(question.options)

                def p2d(pol: Any) -> Dict[str, float]:
                    return (
                        {str(po.option_name).strip(): float(po.probability)
                         for po in pol.predicted_options if _is_num(po.probability)}
                        if isinstance(pol, PredictedOptionList) else {}
                    )

                s_d = p2d(s_pred) if s_pred is not None else {}
                o_d = p2d(o_pred) if o_pred is not None else {}
                blended: Dict[str, float] = {}
                for opt in options:
                    sv, ov       = s_d.get(opt), o_d.get(opt)
                    blended[opt] = _weighted_blend([(w_s, sv), (w_o, ov)]) or 1e-6
                total   = sum(blended.values())
                blended = (
                    {k: v / total for k, v in blended.items()}
                    if total > 0 else {o: 1.0 / len(options) for o in options}
                )
                if self.extremize_enabled:
                    blended = extremize_mc(blended, self.extremize_k_mc)
                stats = (
                    f"[stats] n_agents={len(preds)} entropy={entropy(blended):.3f} "
                    f"extremize_mc(k={self.extremize_k_mc:.2f})"
                )
                return ReasonedPrediction(
                    prediction_value=PredictedOptionList(
                        predicted_options=[
                            PredictedOption(option_name=o, probability=float(p))
                            for o, p in blended.items()
                        ]
                    ),
                    reasoning=self._clean_for_publish(stats + "\n\n" + internal_reasoning),
                )

            # ---- NUMERIC (no extremization — preserve distributional calibration) ----
            if isinstance(question, NumericQuestion):
                targets = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]

                def d2m(d: Any) -> Dict[float, float]:
                    if isinstance(d, NumericDistribution):
                        return {
                            normalize_percentile(getattr(item, "percentile", None)): float(v)
                            for item in d.declared_percentiles
                            if (v := safe_float(getattr(item, "value", None), default=None)) is not None
                        }
                    return {}

                s_m = d2m(s_pred) if s_pred is not None else {}
                o_m = d2m(o_pred) if o_pred is not None else {}
                pts: List[Percentile] = []
                for pt in targets:
                    sv = interpolate_percentile(s_m, pt) if s_m else None
                    ov = interpolate_percentile(o_m, pt) if o_m else None
                    v  = _weighted_blend([(w_s, sv), (w_o, ov)])
                    # _weighted_blend returns 0.5 when both are None — replace with domain midpoint
                    if sv is None and ov is None:
                        v = self._numeric_midpoint(question)
                    pts.append(Percentile(percentile=pt, value=float(v)))
                pts = enforce_monotone(pts)
                p10    = next((p.value for p in pts if abs(float(p.percentile) - 0.1) < 1e-9), None)
                p90    = next((p.value for p in pts if abs(float(p.percentile) - 0.9) < 1e-9), None)
                spread = (p90 - p10) if (p10 is not None and p90 is not None) else float("nan")
                stats  = f"[stats] n_agents={len(preds)} p10={p10} p90={p90} spread={spread:.3f}"
                return ReasonedPrediction(
                    prediction_value=NumericDistribution.from_question(pts, question),
                    reasoning=self._clean_for_publish(stats + "\n\n" + internal_reasoning),
                )

            # Fallback
            return ReasonedPrediction(
                prediction_value=preds[0][1],
                reasoning=self._clean_for_publish(internal_reasoning),
            )

    # ----------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------
    def log_internal_drop_stats(self) -> None:
        if self._drop_counts:
            logger.info(f"[drops] totals={self._drop_counts}")
            logger.info(f"[drops] by_agent={self._drop_counts_by_model}")


# ============================================================
# MAIN
# ============================================================
MINIBENCH_ID            = "minibench"
SPRING_AI_TOURNAMENT_ID = "32916"    # update to real ID if it changes

if __name__ == "__main__":
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("litellm").propagate = False

    parser = argparse.ArgumentParser(
        description=(
            "mewhisk: AgentRouter | dual online-model research | conservative extremize\n"
            "Research models: perplexity/sonar-pro + openai/gpt-4o-search-preview (parallel)\n"
            "Forecast models: claude-sonnet-4-6 (primary) + claude-opus-4-6 (adversarial)"
        )
    )
    parser.add_argument(
        "--tournament-ids", nargs="+", type=str,
        default=[MINIBENCH_ID, SPRING_AI_TOURNAMENT_ID],
        help="Tournament IDs to forecast (default: minibench + Spring AI Tournament)",
    )
    parser.add_argument("--no-extremize",       action="store_true", help="Disable extremization")
    parser.add_argument("--extremize-k-binary", type=float, default=1.15)
    parser.add_argument("--extremize-k-mc",     type=float, default=1.12)
    parser.add_argument(
        "--research-model-1", type=str, default=None,
        help=(
            "Primary online research model (env: MEWHISK_RESEARCH_MODEL_1). "
            "Must be web-search-capable and available on AgentRouter. "
            "Default: perplexity/sonar-pro"
        ),
    )
    parser.add_argument(
        "--research-model-2", type=str, default=None,
        help=(
            "Secondary online research model (env: MEWHISK_RESEARCH_MODEL_2). "
            "Must be web-search-capable and available on AgentRouter. "
            "Default: openai/gpt-4o-search-preview"
        ),
    )
    args = parser.parse_args()

    if args.research_model_1:
        os.environ["MEWHISK_RESEARCH_MODEL_1"] = args.research_model_1
        RESEARCH_MODEL_1 = args.research_model_1  # noqa: F811
    if args.research_model_2:
        os.environ["MEWHISK_RESEARCH_MODEL_2"] = args.research_model_2
        RESEARCH_MODEL_2 = args.research_model_2  # noqa: F811

    if not AGENTROUTER_API_KEY:
        logger.error("AGENTROUTER_API_KEY is required.")
        raise SystemExit(1)

    bot = mewhisk(
        research_reports_per_question=1,
        predictions_per_research_report=2,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
        extremize_enabled=not args.no_extremize,
        extremize_k_binary=args.extremize_k_binary,
        extremize_k_mc=args.extremize_k_mc,
    )

    async def run_all() -> List[Any]:
        all_reports: List[Any] = []
        for tid in args.tournament_ids:
            logger.info(f"mewhisk forecasting on tournament: {tid}")
            reports = await bot.forecast_on_tournament(tid, return_exceptions=True)
            all_reports.extend(reports)
        return all_reports

    try:
        reports = asyncio.run(run_all())
        bot.log_report_summary(reports)
        bot.log_internal_drop_stats()
        logger.info("mewhisk run complete.")
    except Exception as e:
        logger.error(f"Fatal: {e}")
        raise SystemExit(1)
