"""
mewhisk — Metaculus forecasting bot
=====================================
Agent routing   : AgentRouter (https://agentrouter.org)
Primary model   : claude-sonnet-4-6   (fast, calibrated)
Checker model   : claude-opus-4-6     (deep adversarial)
Research        : perplexity/sonar-pro via AgentRouter
                  (online model with live web — no extra API keys needed)
Extremization   : ON — conservative (k_binary=1.15, k_mc=1.12)
Tournaments     : minibench + Spring AI Tournament only
Bot comments    : never expose model names
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
# Research model — any online/web-search-capable model that
# AgentRouter supports.  Change to e.g. "openai/gpt-4o-search-preview"
# or "google/gemini-2.0-flash" if you prefer a different provider.
# ---------------------------------------------------------------
RESEARCH_MODEL = os.getenv("MEWHISK_RESEARCH_MODEL", "perplexity/sonar-pro")

# ============================================================
# Helpers: stats + parsing
# ============================================================
def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)

def median(lst: List[Union[float, int]]) -> float:
    vals = sorted(float(x) for x in lst if _is_num(x))
    if not vals:
        raise ValueError("median() called on empty list")
    n   = len(vals)
    mid = n // 2
    return (vals[mid - 1] + vals[mid]) / 2.0 if n % 2 == 0 else vals[mid]

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)

def stdev(xs: List[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

def ci90(xs: List[float]) -> Tuple[float, float]:
    m, s = mean(xs), stdev(xs)
    se   = s / math.sqrt(len(xs)) if xs else 0.0
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

def build_query(question: MetaculusQuestion, max_chars: int = 397) -> str:
    q  = re.sub(r"\s+", " ", re.sub(r"http\S+", "", question.question_text or "")).strip()
    bg = re.sub(r"\s+", " ", re.sub(r"http\S+", "", question.background_info or "")).strip()
    rc = re.sub(r"\s+", " ", re.sub(r"http\S+", "", question.resolution_criteria or "")).strip()
    # Prefer question + resolution criteria for search signal
    combined = f"{q} — {rc}" if rc else q
    if len(combined) <= max_chars:
        return combined
    if len(q) <= max_chars:
        space = max_chars - len(q) - 3
        if space > 10 and bg:
            return f"{q} — {textwrap.shorten(bg, width=space, placeholder='…')}"
        return q
    first = q.split(".")[0].strip()
    if len(first) > max_chars:
        return textwrap.shorten(first, width=max_chars, placeholder="…")
    remaining = max_chars - len(first) - 3
    if remaining > 10 and bg:
        combo = f"{first} — {textwrap.shorten(bg, width=remaining, placeholder='…')}"
        if len(combo) <= max_chars:
            return combo
    return textwrap.shorten(q, width=max_chars, placeholder="…")

# ============================================================
# EXTREMIZE — conservative / humble settings
# Binary: k=1.15  (gentle logit push toward conviction)
# MC:     k=1.12  (gentle power sharpening)
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
        return {a: max(0.0, float(v)) / s for a, v in probs.items()} if s > 0 else {a: 1.0 / len(probs) for a in probs}
    powered = {a: max(0.0, float(v)) ** float(k) for a, v in probs.items()}
    s2      = sum(powered.values())
    return {a: v / s2 for a, v in powered.items()} if s2 > 0 else {a: 1.0 / len(probs) for a in probs}

# ============================================================
# Weighted blend helper (handles partial agent failure cleanly)
# ============================================================
def _weighted_blend(pairs: List[Tuple[float, Optional[float]]]) -> float:
    """Blend (weight, value) pairs, ignoring None values, renormalising weights."""
    valid     = [(w, v) for w, v in pairs if v is not None and _is_num(v)]
    total_w   = sum(w for w, _ in valid)
    if not valid or total_w == 0:
        return 0.5
    return sum(w * float(v) / total_w for w, v in valid)

# ============================================================
# Research via AgentRouter online model
# ============================================================
def _sync_agentrouter_research(query: str) -> str:
    """
    Call an online/web-search-capable model through AgentRouter's
    OpenAI-compatible /v1/chat/completions endpoint.

    The model string is controlled by MEWHISK_RESEARCH_MODEL
    (default: perplexity/sonar-pro).  AgentRouter transparently
    routes this to the upstream provider with live web access —
    no separate API key is required beyond AGENTROUTER_API_KEY.
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
                "model": RESEARCH_MODEL,
                "messages": [
                    {
                        "role":    "system",
                        "content": (
                            f"You are a precise real-time news researcher. Today is {today}. "
                            "Search the web and give a concise, factual digest relevant to the "
                            "forecasting question. Include exact dates, numbers, and source names. "
                            "Max 400 words. Cite sources inline where possible."
                        ),
                    },
                    {
                        "role":    "user",
                        "content": f"Research for forecasting: {query}",
                    },
                ],
                "max_tokens":  600,
                "temperature": 0.1,
            },
            timeout=40,
        )
        resp.raise_for_status()
        data    = resp.json()
        content = data["choices"][0]["message"]["content"].strip()

        # Attach any citations the model returns (e.g. Perplexity-style)
        citations = data.get("citations", [])
        if citations:
            content += "\nSources: " + " | ".join(str(c) for c in citations[:5])

        return content

    except Exception as e:
        logger.error(f"AgentRouter research error ({RESEARCH_MODEL}): {e}")
        return f"[Research failed: {e}]"


async def _run_research_pipeline(query: str, loop: asyncio.AbstractEventLoop) -> str:
    """Single async wrapper around the AgentRouter online-model research call."""
    today = datetime.now().strftime("%Y-%m-%d")
    result = await loop.run_in_executor(None, _sync_agentrouter_research, query)
    return (
        f"[Research as of {today} — model: {RESEARCH_MODEL} via AgentRouter]\n\n"
        f"{result}"
    )

# ============================================================
# Bot
# ============================================================
class mewhisk(ForecastBot):
    """
    mewhisk — dual-model forecasting bot via AgentRouter.

    All LLM calls (research + forecast) route through AgentRouter:
      • Research  : online model (default perplexity/sonar-pro) — live web
      • Primary   : claude-sonnet-4-6
      • Adversarial checker: claude-opus-4-6

    Extremization: ON, conservative — k_binary=1.15, k_mc=1.12.
    Numeric distributions: no extremization.
    Bot comments: [AGENT_A] / [AGENT_B] — model names never shown.
    """

    _max_concurrent_questions          = 3   # parallelise across questions
    _concurrency_limiter               = asyncio.Semaphore(3)
    _structure_output_validation_samples = 3  # safer structured parse

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
            f"research_model={RESEARCH_MODEL} | "
            f"extremize=ON k_bin={self.extremize_k_binary} k_mc={self.extremize_k_mc}"
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
        async with self._concurrency_limiter:
            query  = build_query(question)
            loop   = asyncio.get_running_loop()
            result = await _run_research_pipeline(query, loop)
            return result

    # ----------------------------------------------------------
    # LLM helpers
    # ----------------------------------------------------------
    async def _invoke_llm(self, model_name: str, prompt: str) -> str:
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
        out2 = {options[i - 1]: float(idx_probs[i]) for i in range(1, len(options) + 1) if i in idx_probs}
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
            pts = []
            for p in percentile_list:
                v = safe_float(getattr(p, "value", None), default=None)
                if v is not None:
                    pts.append(
                        Percentile(
                            value=float(v),
                            percentile=normalize_percentile(getattr(p, "percentile", 0.5)),
                        )
                    )
            if pts:
                pts = self._enforce_monotone(pts)
                return NumericDistribution.from_question(pts, question)
        except Exception:
            self._inc_drop(tag, "parse_error_numeric_structured")
        extracted = extract_numeric_percentiles(raw, targets)
        if not extracted:
            self._inc_drop(tag, "parse_error_numeric_fallback")
            return None
        pts2 = sorted(
            [Percentile(percentile=pt, value=float(extracted[pt])) for pt in targets if pt in extracted],
            key=lambda x: float(x.percentile),
        )
        pts2 = self._enforce_monotone(pts2)
        return NumericDistribution.from_question(pts2, question)

    @staticmethod
    def _enforce_monotone(pts: List[Percentile]) -> List[Percentile]:
        """
        Fix monotonicity violations by averaging the two swapped neighbours
        rather than hard-clamping, which can distort tails.
        """
        pts = sorted(pts, key=lambda x: float(x.percentile))
        changed = True
        while changed:
            changed = False
            for i in range(1, len(pts)):
                if pts[i].value < pts[i - 1].value:
                    avg = (pts[i - 1].value + pts[i].value) / 2.0
                    pts[i - 1].value = avg
                    pts[i].value     = avg
                    changed          = True
        return pts

    @staticmethod
    def _interpolate_percentile(source: Dict[float, float], target: float) -> Optional[float]:
        """Linear interpolation between the two nearest bracketing percentile points."""
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

    def _bounds_messages(self, q: NumericQuestion) -> Tuple[str, str]:
        low  = q.nominal_lower_bound if q.nominal_lower_bound is not None else q.lower_bound
        high = q.nominal_upper_bound if q.nominal_upper_bound is not None else q.upper_bound
        return (
            f"Cannot be lower than {low}." if not q.open_lower_bound else f"Unlikely below {low}.",
            f"Cannot be higher than {high}." if not q.open_upper_bound else f"Unlikely above {high}.",
        )

    # ----------------------------------------------------------
    # Prompts — no model names ever appear in output
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
    # Per-role runners
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
            try:
                l, u = float(q.lower_bound or 0), float(q.upper_bound or 100)
            except Exception:
                l, u = 0.0, 100.0
            dist = NumericDistribution.from_question(
                [Percentile(value=(l + u) / 2, percentile=0.5)], q
            )
        return ReasonedPrediction(prediction_value=dist, reasoning=raw)

    # Required overrides (single-agent shortcuts — unused in dual-agent flow)
    async def _run_forecast_on_binary(self, q, research):
        return await self._run_binary_role(q, research, "sonnet", "PRIMARY")

    async def _run_forecast_on_multiple_choice(self, q, research):
        return await self._run_mc_role(q, research, "sonnet", "PRIMARY")

    async def _run_forecast_on_numeric(self, q, research):
        return await self._run_numeric_role(q, research, "sonnet", "PRIMARY")

    # ----------------------------------------------------------
    # Dual-agent prediction
    # ----------------------------------------------------------
    async def _make_prediction(self, question: MetaculusQuestion, research: str):
        async with self._concurrency_limiter:
            preds:     List[Any] = []
            reasonings: List[str] = []
            w_s, w_o = 0.55, 0.45

            # Run Sonnet (primary) and Opus (adversarial) in parallel
            async def run_agent(tag: str, role: str):
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

            await asyncio.gather(
                run_agent("sonnet", "PRIMARY"),
                run_agent("opus",   "ADVERSARIAL_CHECKER"),
            )

            if not preds:
                raise RuntimeError("All agents failed.")

            combined = "\n\n---\n\n".join(reasonings)

            def get_pred(tag: str) -> Any:
                for t, v in preds:
                    if t == tag:
                        return v
                return None

            s_pred = get_pred("sonnet")
            o_pred = get_pred("opus")

            # ---- BINARY ----
            if isinstance(question, BinaryQuestion):
                s_val = float(s_pred) if s_pred is not None and _is_num(s_pred) else None
                o_val = float(o_pred) if o_pred is not None and _is_num(o_pred) else None
                final = _weighted_blend([(w_s, s_val), (w_o, o_val)])
                final = clamp01(final)
                pre   = final
                if self.extremize_enabled:
                    final = extremize_binary(final, self.extremize_k_binary)
                np_   = [v for v in [s_val, o_val] if v is not None] or [0.5]
                m, md, sd = mean(np_), median(np_), stdev(np_)
                lo, hi    = ci90(np_)
                stats = (
                    f"[stats] n={len(np_)} mean={m:.3f} median={md:.3f} sd={sd:.3f} "
                    f"ci90=({lo:.3f},{hi:.3f}) extremize(k={self.extremize_k_binary:.2f}) "
                    f"{pre:.3f}→{final:.3f}"
                )
                return ReasonedPrediction(prediction_value=final, reasoning=stats + "\n\n" + combined)

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
                    sv, ov   = s_d.get(opt), o_d.get(opt)
                    blended[opt] = _weighted_blend([(w_s, sv), (w_o, ov)]) or 1e-6
                total   = sum(blended.values())
                blended = {k: v / total for k, v in blended.items()} if total > 0 else {o: 1.0 / len(options) for o in options}
                if self.extremize_enabled:
                    blended = extremize_mc(blended, self.extremize_k_mc)
                stats = (
                    f"[stats] n_agents={len(preds)} entropy={entropy(blended):.3f} "
                    f"extremize_mc(k={self.extremize_k_mc:.2f})"
                )
                return ReasonedPrediction(
                    prediction_value=PredictedOptionList(
                        predicted_options=[PredictedOption(option_name=o, probability=float(p)) for o, p in blended.items()]
                    ),
                    reasoning=stats + "\n\n" + combined,
                )

            # ---- NUMERIC (no extremization) ----
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
                    sv = self._interpolate_percentile(s_m, pt) if s_m else None
                    ov = self._interpolate_percentile(o_m, pt) if o_m else None
                    v  = _weighted_blend([(w_s, sv), (w_o, ov)])
                    if v is None or v == 0.5 and sv is None and ov is None:
                        try:
                            l, u = float(question.lower_bound or 0), float(question.upper_bound or 100)
                        except Exception:
                            l, u = 0.0, 100.0
                        v = (l + u) / 2.0
                    pts.append(Percentile(percentile=pt, value=float(v)))
                pts = self._enforce_monotone(pts)
                p10    = next((p.value for p in pts if abs(float(p.percentile) - 0.1) < 1e-9), None)
                p90    = next((p.value for p in pts if abs(float(p.percentile) - 0.9) < 1e-9), None)
                spread = (p90 - p10) if (p10 is not None and p90 is not None) else float("nan")
                stats  = f"[stats] n_agents={len(preds)} p10={p10} p90={p90} spread={spread:.3f}"
                return ReasonedPrediction(
                    prediction_value=NumericDistribution.from_question(pts, question),
                    reasoning=stats + "\n\n" + combined,
                )

            # Fallback — return first available prediction as-is
            return ReasonedPrediction(prediction_value=preds[0][1], reasoning=combined)

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
MINIBENCH_ID           = "minibench"
SPRING_AI_TOURNAMENT_ID = "32916"   # update to real tournament ID if it changes

if __name__ == "__main__":
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("litellm").propagate = False

    parser = argparse.ArgumentParser(
        description=(
            "mewhisk: AgentRouter | online-model research (perplexity/sonar-pro) | "
            "conservative extremize"
        )
    )
    parser.add_argument(
        "--tournament-ids", nargs="+", type=str,
        default=[MINIBENCH_ID, SPRING_AI_TOURNAMENT_ID],
        help="Tournament IDs to forecast (default: minibench + Spring AI Tournament)",
    )
    parser.add_argument("--no-extremize",         action="store_true", help="Disable extremization")
    parser.add_argument("--extremize-k-binary",   type=float, default=1.15)
    parser.add_argument("--extremize-k-mc",       type=float, default=1.12)
    parser.add_argument(
        "--research-model", type=str, default=None,
        help=(
            "Override MEWHISK_RESEARCH_MODEL env var. "
            "Any online-capable model on AgentRouter, e.g. "
            "'perplexity/sonar-pro', 'openai/gpt-4o-search-preview', "
            "'google/gemini-2.0-flash'."
        ),
    )
    args = parser.parse_args()

    if args.research_model:
        os.environ["MEWHISK_RESEARCH_MODEL"] = args.research_model
        RESEARCH_MODEL = args.research_model  # noqa: F811

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

    async def run_all():
        all_reports = []
        for tid in args.tournament_ids:
            logger.info(f"mewhisk forecasting on: {tid}")
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
