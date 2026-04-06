"""
mewhisk — Metaculus forecasting bot
=====================================
Agent routing   : AgentRouter (https://agentrouter.org) — research only
Primary model   : openrouter/qwen/qwen3.6-plus:free      (fast, calibrated)
Checker model   : openrouter/nvidia/nemotron-super-49b-v1:free  (adversarial)
Parser model    : openrouter/qwen/qwen3.6-plus:free
Research        : perplexity/sonar-pro    (online, live web)  ← primary
                + openai/gpt-4o-search-preview (online, live web) ← secondary
                  Both run in parallel via AgentRouter, concatenated into one block.
                  Override via MEWHISK_RESEARCH_MODEL_1 / _MODEL_2 env vars.
Extremization   : ON — conservative (k_binary=1.15, k_mc=1.12)
Concurrency     : 4 questions in parallel, 6 LLM slots
Tournaments     : minibench + Spring AI Tournament only
Bot comments    : clean prose — no internal labels or model names published

FIXES vs previous version:
  - Forecasters now call OpenRouter directly (separate OPENROUTER_API_KEY +
    base URL https://openrouter.ai/api/v1) to avoid the Aliyun WAF captcha
    wall that blocks AgentRouter's Anthropic endpoint from some regions.
  - Parser also uses OpenRouter so litellm never needs to resolve a bare
    "claude-*" model string against the overridden ANTHROPIC_API_BASE.
  - Research pipeline still uses AgentRouter (perplexity/sonar-pro +
    openai/gpt-4o-search-preview) via direct HTTP — unaffected by the WAF.
  - OPENAI_API_KEY stub set to suppress noisy openai.agents SDK warning.
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
# Suppress the "OPENAI_API_KEY is not set, skipping trace export"
# warning from the openai.agents tracing SDK.
# ---------------------------------------------------------------
os.environ.setdefault("OPENAI_API_KEY", "sk-dummy-suppress-warning")

# ---------------------------------------------------------------
# AgentRouter — used ONLY for research (perplexity/sonar-pro etc.)
# We do NOT set ANTHROPIC_API_BASE here anymore; that was causing
# litellm to route bare "claude-*" parser calls to AgentRouter,
# which gets geo-blocked by Aliyun WAF in some regions.
# ---------------------------------------------------------------
_AGENTROUTER_KEY  = os.getenv("AGENTROUTER_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
_AGENTROUTER_BASE = "https://agentrouter.org/v1"

# ---------------------------------------------------------------
# OpenRouter — used for ALL LLM forecast/parser calls.
# Requires OPENROUTER_API_KEY in environment.
# litellm routes "openrouter/*" models via this base URL
# when OPENROUTER_API_KEY is set.
# ---------------------------------------------------------------
_OPENROUTER_KEY  = os.getenv("OPENROUTER_API_KEY", "")
_OPENROUTER_BASE = "https://openrouter.ai/api/v1"

if _OPENROUTER_KEY:
    os.environ["OPENROUTER_API_KEY"] = _OPENROUTER_KEY

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
# Key checks
# ============================================================
if not _AGENTROUTER_KEY:
    logger.warning("AGENTROUTER_API_KEY (or ANTHROPIC_API_KEY) not set — research may fail.")
if not _OPENROUTER_KEY:
    logger.error("OPENROUTER_API_KEY not set — forecast calls will fail.")

# ---------------------------------------------------------------
# Forecaster / parser model strings — OpenRouter free tier.
# litellm recognises the "openrouter/" prefix and routes correctly
# when OPENROUTER_API_KEY is present.
# ---------------------------------------------------------------
FORECAST_MODEL_PRIMARY    = "openrouter/qwen/qwen3.6-plus:free"
FORECAST_MODEL_ADVERSARIAL = "openrouter/nvidia/nemotron-super-49b-v1:free"
PARSER_MODEL              = "openrouter/qwen/qwen3.6-plus:free"

# ---------------------------------------------------------------
# Dual online research models — routed through AgentRouter.
# Override via env vars without touching code.
# ---------------------------------------------------------------
RESEARCH_MODEL_1 = os.getenv("MEWHISK_RESEARCH_MODEL_1", "perplexity/sonar-pro")
RESEARCH_MODEL_2 = os.getenv("MEWHISK_RESEARCH_MODEL_2", "openai/gpt-4o-search-preview")

# ============================================================
# Concurrency controls
# ============================================================
_Q_SEM   = asyncio.Semaphore(4)
_LLM_SEM = asyncio.Semaphore(6)

# ============================================================
# Helpers: stats + parsing
# ============================================================
def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def safe_median(lst: List[Union[float, int]]) -> float:
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
# Weighted blend
# ============================================================
def _weighted_blend(pairs: List[Tuple[float, Optional[float]]]) -> float:
    valid   = [(w, float(v)) for w, v in pairs if v is not None and _is_num(v)]
    total_w = sum(w for w, _ in valid)
    if not valid or total_w == 0:
        return 0.5
    return sum(w * v / total_w for w, v in valid)


# ============================================================
# Isotonic regression (PAV) for monotone percentile enforcement
# ============================================================
def _isotonic_regression_increasing(vals: List[float]) -> List[float]:
    if not vals:
        return []
    blocks: List[Tuple[float, List[int]]] = [(v, [i]) for i, v in enumerate(vals)]
    result = list(vals)

    i = 0
    while i < len(blocks) - 1:
        avg_i = blocks[i][0]
        avg_j = blocks[i + 1][0]
        if avg_i > avg_j:
            merged_indices = blocks[i][1] + blocks[i + 1][1]
            merged_avg     = sum(result[j] for j in merged_indices) / len(merged_indices)
            for j in merged_indices:
                result[j] = merged_avg
            blocks = blocks[:i] + [(merged_avg, merged_indices)] + blocks[i + 2:]
            if i > 0:
                i -= 1
        else:
            i += 1
    return result


def enforce_monotone(pts: List[Percentile]) -> List[Percentile]:
    pts    = sorted(pts, key=lambda x: float(x.percentile))
    values = _isotonic_regression_increasing([float(p.value) for p in pts])
    for p, v in zip(pts, values):
        p.value = v
    return pts


# ============================================================
# Percentile interpolation
# ============================================================
def interpolate_percentile(source: Dict[float, float], target: float) -> Optional[float]:
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
# Dual online research via AgentRouter (direct HTTP — no litellm)
# ============================================================
_RESEARCH_SYSTEM = (
    "You are a precise real-time news researcher. Today is {today}. "
    "Search the web and give a concise, factual digest relevant to the "
    "forecasting question below. Include exact dates, numbers, and source names. "
    "Max 400 words. Cite sources inline where possible."
)
_RESEARCH_USER = "Research for forecasting: {query}"


def _sync_online_research(model: str, query: str, label: str) -> str:
    """Direct HTTP call to AgentRouter's chat/completions endpoint."""
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        resp = _requests.post(
            f"{_AGENTROUTER_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {_AGENTROUTER_KEY}",
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
        citations = data.get("citations", [])
        if citations:
            content += "\nSources: " + " | ".join(str(c) for c in citations[:5])
        return content
    except Exception as e:
        logger.warning(f"Research model {label} ({model}) failed: {e}")
        return f"[{label} research unavailable: {e}]"


async def _run_research_pipeline(query: str, loop: asyncio.AbstractEventLoop) -> str:
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
# Direct OpenRouter HTTP call — bypasses litellm entirely for
# forecaster/parser calls, avoiding any ANTHROPIC_API_BASE clash.
# ============================================================
def _sync_openrouter_call(model: str, prompt: str, max_tokens: int = 1200) -> str:
    """
    Call OpenRouter's chat completions endpoint directly.
    'model' should be the full OpenRouter model id WITHOUT the
    'openrouter/' litellm prefix, e.g. 'qwen/qwen3.6-plus:free'.
    """
    # Strip the litellm routing prefix if present
    or_model = model.removeprefix("openrouter/")
    try:
        resp = _requests.post(
            f"{_OPENROUTER_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {_OPENROUTER_KEY}",
                "Content-Type":  "application/json",
                "HTTP-Referer":  "https://github.com/mewhisk",
                "X-Title":       "mewhisk-forecasting-bot",
            },
            json={
                "model":    or_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens":  max_tokens,
                "temperature": 0.3,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise RuntimeError(f"OpenRouter call failed for {or_model}: {e}") from e


# ============================================================
# Bot
# ============================================================
class mewhisk(ForecastBot):
    """
    mewhisk — dual-model forecasting bot.

    Research    : AgentRouter → perplexity/sonar-pro + gpt-4o-search-preview (parallel)
    Primary     : OpenRouter  → qwen/qwen3.6-plus:free        (weight 0.55)
    Adversarial : OpenRouter  → nvidia/nemotron-super-49b-v1:free (weight 0.45)
    Parser      : OpenRouter  → qwen/qwen3.6-plus:free

    Extremization: ON, conservative — k_binary=1.15, k_mc=1.12.
    Numeric distributions: no extremization.
    Published comments: clean prose, no internal labels.
    """

    _max_concurrent_questions            = 4
    _concurrency_limiter                 = _Q_SEM
    _structure_output_validation_samples = 3

    def _llm_config_defaults(self) -> Dict[str, str]:
        # These are used by the parent class for its own internal calls.
        # We override _invoke_llm to bypass litellm for forecast/parser calls,
        # so these entries mainly satisfy the parent's init expectations.
        defaults = super()._llm_config_defaults()
        defaults.update({
            "forecaster_primary":    FORECAST_MODEL_PRIMARY,
            "forecaster_adversarial": FORECAST_MODEL_ADVERSARIAL,
            "parser":                PARSER_MODEL,
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
        self._drop_counts_by_model: Dict[str, Dict[str, int]] = {"primary": {}, "adversarial": {}}
        logger.info(
            f"mewhisk ready | "
            f"research=AgentRouter:[{RESEARCH_MODEL_1}, {RESEARCH_MODEL_2}] | "
            f"forecast=OpenRouter:[primary={FORECAST_MODEL_PRIMARY}, "
            f"adversarial={FORECAST_MODEL_ADVERSARIAL}] | "
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
    # LLM helpers — direct OpenRouter HTTP, gated by _LLM_SEM
    # ----------------------------------------------------------
    async def _invoke_openrouter(self, model: str, prompt: str) -> str:
        async with _LLM_SEM:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, _sync_openrouter_call, model, prompt
            )

    # Keep parent-compatible signatures
    async def _invoke_llm(self, model_name: str, prompt: str) -> str:
        # model_name here is a key like "forecaster_primary" or the full model string
        model = {
            "forecaster_primary":    FORECAST_MODEL_PRIMARY,
            "forecaster_adversarial": FORECAST_MODEL_ADVERSARIAL,
            "parser":                PARSER_MODEL,
        }.get(model_name, model_name)
        return await self._invoke_openrouter(model, prompt)

    async def _invoke_with_format_retry(self, model_name: str, prompt: str, _: str) -> str:
        return await self._invoke_llm(model_name, prompt)

    # ----------------------------------------------------------
    # Parser — uses OpenRouter directly (no litellm structure_output)
    # ----------------------------------------------------------
    async def _parse_binary(self, raw: str, tag: str) -> Optional[float]:
        # First try regex extraction from raw text
        val = extract_binary_prob_from_text(raw)
        if val is not None:
            return clamp01(float(val))

        # Fallback: ask parser model to extract
        try:
            extract_prompt = (
                f"Extract the probability from this forecasting text. "
                f"Reply with ONLY a single decimal between 0 and 1.\n\nText:\n{raw[:800]}"
            )
            result = await self._invoke_openrouter(PARSER_MODEL, extract_prompt)
            val2 = safe_float(result.strip().split()[0], default=None)
            if val2 is not None:
                return clamp01(float(val2))
        except Exception:
            self._inc_drop(tag, "parse_error_binary_fallback")

        self._inc_drop(tag, "parse_error_binary_total")
        return None

    async def _parse_mc(
        self, raw: str, question: MultipleChoiceQuestion, tag: str
    ) -> Optional[Dict[str, float]]:
        options = list(question.options)

        # Try regex extraction first
        idx_probs = extract_indexed_mc_probs(raw, len(options))
        if idx_probs:
            return {
                options[i - 1]: float(idx_probs[i])
                for i in range(1, len(options) + 1) if i in idx_probs
            }

        # Fallback: ask parser model
        try:
            opts_str = "\n".join(build_indexed_options(options))
            extract_prompt = (
                f"Extract probabilities for these options from the text. "
                f"Reply ONLY with lines like '1: XX%' summing to 100%.\n\n"
                f"Options:\n{opts_str}\n\nText:\n{raw[:800]}"
            )
            result = await self._invoke_openrouter(PARSER_MODEL, extract_prompt)
            idx_probs2 = extract_indexed_mc_probs(result, len(options))
            if idx_probs2:
                return {
                    options[i - 1]: float(idx_probs2[i])
                    for i in range(1, len(options) + 1) if i in idx_probs2
                }
        except Exception:
            pass

        self._inc_drop(tag, "parse_error_mc_total")
        return None

    async def _parse_numeric(
        self, raw: str, question: NumericQuestion, tag: str
    ) -> Optional[NumericDistribution]:
        targets = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]

        # Try regex extraction first
        extracted = extract_numeric_percentiles(raw, targets)
        if extracted:
            pts = [
                Percentile(percentile=pt, value=float(extracted[pt]))
                for pt in targets if pt in extracted
            ]
            if pts:
                return NumericDistribution.from_question(enforce_monotone(pts), question)

        # Fallback: ask parser model
        try:
            extract_prompt = (
                f"Extract percentile values from this text. "
                f"Reply ONLY with exactly these lines:\n"
                f"Percentile 10: X\nPercentile 20: X\nPercentile 40: X\n"
                f"Percentile 60: X\nPercentile 80: X\nPercentile 90: X\n\n"
                f"Text:\n{raw[:800]}"
            )
            result = await self._invoke_openrouter(PARSER_MODEL, extract_prompt)
            extracted2 = extract_numeric_percentiles(result, targets)
            if extracted2:
                pts2 = [
                    Percentile(percentile=pt, value=float(extracted2[pt]))
                    for pt in targets if pt in extracted2
                ]
                if pts2:
                    return NumericDistribution.from_question(enforce_monotone(pts2), question)
        except Exception:
            pass

        self._inc_drop(tag, "parse_error_numeric_total")
        return None

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
    # Prompts
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
        self, q: BinaryQuestion, research: str, tag: str, role: str, model: str
    ) -> ReasonedPrediction[float]:
        raw = await self._invoke_openrouter(model, self._binary_prompt(q, research, role))
        val = await self._parse_binary(raw, tag)
        if val is None:
            try:
                raw2 = await self._invoke_openrouter(
                    model, "Output ONLY:\nProbability: ZZ%\nDecimal: 0.ZZ"
                )
                val  = await self._parse_binary(raw2, tag)
                raw += "\n\n[RETRY]\n" + raw2
            except Exception:
                self._inc_drop(tag, "retry_failed_binary")
        if val is None:
            self._inc_drop(tag, "invalid_binary")
            val = 0.5
        return ReasonedPrediction(prediction_value=clamp01(val), reasoning=raw)

    async def _run_mc_role(
        self, q: MultipleChoiceQuestion, research: str, tag: str, role: str, model: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        raw   = await self._invoke_openrouter(model, self._mc_prompt(q, research, role))
        probs = await self._parse_mc(raw, q, tag)
        if probs is None:
            try:
                raw2  = await self._invoke_openrouter(
                    model, "Output ONLY numbered % lines summing to 100%."
                )
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
        self, q: NumericQuestion, research: str, tag: str, role: str, model: str
    ) -> ReasonedPrediction[NumericDistribution]:
        raw  = await self._invoke_openrouter(model, self._numeric_prompt(q, research, role))
        dist = await self._parse_numeric(raw, q, tag)
        if dist is None:
            try:
                raw2 = await self._invoke_openrouter(
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
        return await self._run_binary_role(q, research, "primary", "PRIMARY", FORECAST_MODEL_PRIMARY)

    async def _run_forecast_on_multiple_choice(self, q, research):
        return await self._run_mc_role(q, research, "primary", "PRIMARY", FORECAST_MODEL_PRIMARY)

    async def _run_forecast_on_numeric(self, q, research):
        return await self._run_numeric_role(q, research, "primary", "PRIMARY", FORECAST_MODEL_PRIMARY)

    # ----------------------------------------------------------
    # Clean reasoning for public Metaculus comments
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
        cleaned = "\n".join(output_lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    # ----------------------------------------------------------
    # Dual-agent prediction core
    # ----------------------------------------------------------
    async def _make_prediction(self, question: MetaculusQuestion, research: str):
        async with _Q_SEM:
            preds:      List[Tuple[str, Any]] = []
            reasonings: List[str]             = []
            w_p, w_a = 0.55, 0.45

            async def run_agent(tag: str, role: str, model: str) -> None:
                try:
                    if isinstance(question, BinaryQuestion):
                        p = await self._run_binary_role(question, research, tag, role, model)
                    elif isinstance(question, MultipleChoiceQuestion):
                        p = await self._run_mc_role(question, research, tag, role, model)
                    else:
                        p = await self._run_numeric_role(question, research, tag, role, model)
                    label = "A" if tag == "primary" else "B"
                    preds.append((tag, p.prediction_value))
                    reasonings.append(f"[AGENT_{label}]\n{p.reasoning}")
                except Exception as e:
                    logger.error(f"Agent {tag} failed: {e}")

            # Primary + Adversarial in parallel
            await asyncio.gather(
                run_agent("primary",    "PRIMARY",             FORECAST_MODEL_PRIMARY),
                run_agent("adversarial","ADVERSARIAL_CHECKER", FORECAST_MODEL_ADVERSARIAL),
            )

            if not preds:
                raise RuntimeError("All forecast agents failed.")

            internal_reasoning = "\n\n---\n\n".join(reasonings)

            def get_val(tag: str) -> Any:
                for t, v in preds:
                    if t == tag:
                        return v
                return None

            p_pred = get_val("primary")
            a_pred = get_val("adversarial")

            # ---- BINARY ----
            if isinstance(question, BinaryQuestion):
                p_val = float(p_pred) if p_pred is not None and _is_num(p_pred) else None
                a_val = float(a_pred) if a_pred is not None and _is_num(a_pred) else None
                final = clamp01(_weighted_blend([(w_p, p_val), (w_a, a_val)]))
                pre   = final
                if self.extremize_enabled:
                    final = extremize_binary(final, self.extremize_k_binary)
                np_       = [v for v in [p_val, a_val] if v is not None] or [0.5]
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

                p_d = p2d(p_pred) if p_pred is not None else {}
                a_d = p2d(a_pred) if a_pred is not None else {}
                blended: Dict[str, float] = {}
                for opt in options:
                    pv, av       = p_d.get(opt), a_d.get(opt)
                    blended[opt] = _weighted_blend([(w_p, pv), (w_a, av)]) or 1e-6
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

                p_m = d2m(p_pred) if p_pred is not None else {}
                a_m = d2m(a_pred) if a_pred is not None else {}
                pts: List[Percentile] = []
                for pt in targets:
                    pv = interpolate_percentile(p_m, pt) if p_m else None
                    av = interpolate_percentile(a_m, pt) if a_m else None
                    v  = _weighted_blend([(w_p, pv), (w_a, av)])
                    if pv is None and av is None:
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
SPRING_AI_TOURNAMENT_ID = "32916"

if __name__ == "__main__":
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("litellm").propagate = False

    parser = argparse.ArgumentParser(
        description=(
            "mewhisk: AgentRouter research | OpenRouter free-tier forecasters\n"
            "Research : perplexity/sonar-pro + openai/gpt-4o-search-preview (parallel, AgentRouter)\n"
            "Primary  : qwen/qwen3.6-plus:free (OpenRouter, weight=0.55)\n"
            "Adversarial: nvidia/nemotron-super-49b-v1:free (OpenRouter, weight=0.45)"
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
        help="Primary research model (env: MEWHISK_RESEARCH_MODEL_1). Default: perplexity/sonar-pro",
    )
    parser.add_argument(
        "--research-model-2", type=str, default=None,
        help="Secondary research model (env: MEWHISK_RESEARCH_MODEL_2). Default: openai/gpt-4o-search-preview",
    )
    args = parser.parse_args()

    if args.research_model_1:
        os.environ["MEWHISK_RESEARCH_MODEL_1"] = args.research_model_1
        RESEARCH_MODEL_1 = args.research_model_1  # noqa: F811
    if args.research_model_2:
        os.environ["MEWHISK_RESEARCH_MODEL_2"] = args.research_model_2
        RESEARCH_MODEL_2 = args.research_model_2  # noqa: F811

    if not _OPENROUTER_KEY:
        logger.error("OPENROUTER_API_KEY is required for forecast calls.")
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
