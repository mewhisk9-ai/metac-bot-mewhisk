"""
mewhisk — Metaculus forecasting bot
=====================================
All LLM calls (research + forecast + parse) go through OpenRouter via direct
HTTP, using the same pattern as the official metaculus forecasting-tools template.
No AgentRouter dependency.

🆕 UPDATED MODEL CONFIG (April 2026):
Primary forecaster  : meta-llama/llama-3.3-70b-instruct:free  (w=0.55) ✅ VERIFIED
Adversarial checker : google/gemma-3-12b-it:free              (w=0.45) ✅ VERIFIED
Research            : qwen/qwen3-next-80b-a3b-instruct:free + google/gemma-3-27b-it:free
Parser              : meta-llama/llama-3.3-70b-instruct:free

⚠️  MODEL ID FORMAT: Use bare IDs like "meta-llama/llama-3.3-70b-instruct:free"
    Do NOT include "openrouter/" prefix — API expects provider/model:tier format.

Extremization (per tournament):
  minibench  → 5-trigger aggressive system
  Spring AI  → conservative k_binary=1.15, k_mc=1.12

Env vars required:
  OPENROUTER_API_KEY   — all LLM traffic
  METACULUS_TOKEN      — posting forecasts (handled by forecasting-tools)
"""

import argparse
import asyncio
import logging
import os
import textwrap
import re
import math
import time
from datetime import datetime
from typing import List, Union, Dict, Any, Optional, Tuple

os.environ.setdefault("OPENAI_API_KEY", "sk-dummy-suppress-warning")
import requests as _requests

from forecasting_tools import (
    BinaryQuestion, ForecastBot, MetaculusQuestion, MultipleChoiceQuestion,
    NumericDistribution, NumericQuestion, Percentile, PredictedOptionList,
    PredictedOption, ReasonedPrediction, clean_indents,
)

# ============================================================
# Logging
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("mewhisk")

# ============================================================
# OpenRouter config
# ============================================================
_OPENROUTER_KEY  = os.getenv("OPENROUTER_API_KEY", "")
_OPENROUTER_BASE = "https://openrouter.ai/api/v1"

if not _OPENROUTER_KEY:
    logger.error("OPENROUTER_API_KEY not set — all LLM calls will fail.")

# ============================================================
# ✅ VERIFIED WORKING FREE MODEL IDs (April 2026)
# ⚠️  IMPORTANT: Use BARE model IDs (NO "openrouter/" prefix)
# ============================================================
FORECAST_MODEL_PRIMARY     = os.getenv(
    "MEWHISK_FORECAST_PRIMARY",
    "meta-llama/llama-3.3-70b-instruct:free",   # ✅ Strong 70B model, verified available
)
FORECAST_MODEL_ADVERSARIAL = os.getenv(
    "MEWHISK_FORECAST_ADVERSARIAL",
    "google/gemma-3-12b-it:free",               # ✅ Balanced, verified available
)
RESEARCH_MODEL_1 = os.getenv(
    "MEWHISK_RESEARCH_MODEL_1",
    "qwen/qwen3-next-80b-a3b-instruct:free",    # ✅ Long-context reasoning
)
RESEARCH_MODEL_2 = os.getenv(
    "MEWHISK_RESEARCH_MODEL_2",
    "google/gemma-3-27b-it:free",               # ✅ Multimodal research depth
)

# ============================================================
# 🔄 Fallback: Auto-router picks best available free model
# ============================================================
_AUTO_ROUTER = "openrouter/free"  # ✅ Always available, picks from working free models [[28]]

_FALLBACK_MODELS = {
    "meta-llama/llama-3.3-70b-instruct:free": _AUTO_ROUTER,
    "google/gemma-3-12b-it:free": _AUTO_ROUTER,
    "qwen/qwen3-next-80b-a3b-instruct:free": _AUTO_ROUTER,
    "google/gemma-3-27b-it:free": _AUTO_ROUTER,
}

# ============================================================
# Concurrency controls
# ============================================================
_Q_SEM   = asyncio.Semaphore(4)
_LLM_SEM = asyncio.Semaphore(6)

# ============================================================
# Helpers: stats + numeric utils
# ============================================================
def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)

def safe_median(lst: List[Union[float, int]]) -> float:
    vals = sorted(float(x) for x in lst if _is_num(x))
    if not vals: return 0.5
    n, mid = len(vals), len(vals) // 2
    return (vals[mid - 1] + vals[mid]) / 2.0 if n % 2 == 0 else vals[mid]

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0

def stdev(xs: List[float]) -> float:
    if len(xs) <= 1: return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

def ci90(xs: List[float]) -> Tuple[float, float]:
    if not xs: return 0.0, 1.0
    m, s = mean(xs), stdev(xs)
    se = s / math.sqrt(len(xs))
    return max(0.0, m - 1.645 * se), min(1.0, m + 1.645 * se)

def entropy(probs: Dict[str, float]) -> float:
    return -sum(p * math.log(p) for p in probs.values() if p > 0)

def safe_float(x: Any, default: Optional[float] = 0.0) -> Optional[float]:
    try:
        if x is None: return default
        s = str(x).strip().replace(",", "").replace("%", "")
        return float(s) if s else default
    except Exception:
        return default

def normalize_percentile(p: Any) -> float:
    perc = safe_float(p, default=0.5) or 0.5
    if perc > 1.0: perc /= 100.0
    return float(max(0.0, min(1.0, perc)))

def clamp01(p: float, lo: float = 0.01, hi: float = 0.99) -> float:
    return max(lo, min(hi, float(p)))

# ============================================================
# Model validation helper
# ============================================================
def _validate_openrouter_model(model_id: str) -> bool:
    if not model_id or "/" not in model_id:
        logger.error(f"Invalid model format: '{model_id}' — expected 'provider/model:tier'")
        return False
    if model_id.startswith("openrouter/"):
        logger.error(f"Invalid model format: '{model_id}' — remove 'openrouter/' prefix")
        return False
    return True

# ============================================================
# Parsers — text extraction
# ============================================================
_PERCENT_RE = re.compile(r"(?i)\bprob(?:ability)?\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%")
_DEC_RE     = re.compile(r"(?i)\bdecimal\s*:\s*([0-9]*\.?[0-9]+)\b")

def extract_binary_prob_from_text(text: str) -> Optional[float]:
    if not text: return None
    for pat, divisor in [(_PERCENT_RE, 100.0), (_DEC_RE, 1.0)]:
        m = pat.search(text)
        if m:
            v = safe_float(m.group(1), default=None)
            return clamp01(v / divisor, 0.0, 1.0) if v is not None else None
    m2 = re.search(r"(?<!\d)([0-9]{1,3}(?:\.[0-9]+)?)\s*%", text)
    if m2:
        v = safe_float(m2.group(1), default=None)
        return clamp01(v / 100.0, 0.0, 1.0) if v is not None else None
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
                if pct is not None: out[idx] = pct / 100.0
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
    q, rc, bg = _clean(question.question_text), _clean(question.resolution_criteria), _clean(question.background_info)
    for cand in (f"{q} — {rc}", f"{q} — {textwrap.shorten(rc, width=max(10, max_chars-len(q)-3), placeholder='…')}",
                 f"{q} — {textwrap.shorten(bg, width=max(10, max_chars-len(q)-3), placeholder='…')}", q):
        if cand and len(cand) <= max_chars: return cand
    return textwrap.shorten(q, width=max_chars, placeholder="…")

# ============================================================
# Extremization — standard logit-based
# ============================================================
def _logit(p: float) -> float:
    p = clamp01(p, 1e-6, 1.0 - 1e-6)
    return math.log(p / (1.0 - p))

def _sigmoid(x: float) -> float:
    z = math.exp(-abs(x))
    return 1.0 / (1.0 + z) if x >= 0 else 1.0 - 1.0 / (1.0 + z)

def extremize_binary(p: float, k: float) -> float:
    if not _is_num(p) or not _is_num(k) or k <= 0 or abs(k - 1.0) < 1e-12: return float(p)
    return clamp01(_sigmoid(_logit(float(p)) * float(k)))

def extremize_mc(probs: Dict[str, float], k: float) -> Dict[str, float]:
    if not probs or not _is_num(k) or k <= 0 or abs(k - 1.0) < 1e-12:
        s = sum(max(0.0, float(v)) for v in probs.values())
        return ({a: max(0.0, float(v)) / s for a, v in probs.items()} if s > 0 else {a: 1.0 / len(probs) for a in probs})
    powered = {a: max(0.0, float(v)) ** float(k) for a, v in probs.items()}
    s2 = sum(powered.values())
    return ({a: v / s2 for a, v in powered.items()} if s2 > 0 else {a: 1.0 / len(probs) for a in probs})

# ============================================================
# MINIBENCH EXTREMIZATION ENGINE — 5-trigger system
# ============================================================
MINIBENCH_K_BASE = 5.0; MINIBENCH_K_AGREE = 1.0; MINIBENCH_K_RESEARCH = 1.0; MINIBENCH_K_MAX = 7.0
MINIBENCH_K_MC = 3.5; MINIBENCH_GATE_LO = 0.40; MINIBENCH_GATE_HI = 0.60; MINIBENCH_GATE_AMP = 1.5
MINIBENCH_CONV_LO = 0.44; MINIBENCH_CONV_HI = 0.52; MINIBENCH_CONV_POS = 0.82; MINIBENCH_CONV_NEG = 0.18

_CONVICTION_RE = re.compile(r"(?i)\b(confirmed|officially|announced|signed|passed|enacted|launched|deployed|released|completed|achieved|won|elected|appointed|definitively|conclusively|clearly|undeniably|certainly|already\s+has|has\s+already|is\s+now|are\s+now|have\s+now|did\s+not|never\s+happened|no\s+evidence|ruled\s+out|impossible\s+by)\b")

def _research_is_strong(research: str) -> bool:
    return len(_CONVICTION_RE.findall(research or "")) >= 2

def _agents_agree(p_val: Optional[float], a_val: Optional[float]) -> bool:
    if p_val is None or a_val is None: return False
    return (p_val > 0.5 and a_val > 0.5) or (p_val < 0.5 and a_val < 0.5)

def minibench_extremize_binary(blend: float, p_val: Optional[float], a_val: Optional[float], research: str) -> Tuple[float, float, str]:
    agree = _agents_agree(p_val, a_val); strong = _research_is_strong(research); in_zone = MINIBENCH_CONV_LO <= blend <= MINIBENCH_CONV_HI
    if in_zone and agree and strong:
        pos = blend > 0.50; result = MINIBENCH_CONV_POS if pos else MINIBENCH_CONV_NEG
        return result, MINIBENCH_K_MAX, f"T5({'pos' if pos else 'neg'} {blend:.3f}→{result:.3f})"
    k = MINIBENCH_K_BASE; triggers = ["T1(base)"]
    if agree: k = min(k + MINIBENCH_K_AGREE, MINIBENCH_K_MAX); triggers.append("T2(agree)")
    if strong: k = min(k + MINIBENCH_K_RESEARCH, MINIBENCH_K_MAX); triggers.append("T3(research)")
    result = clamp01(_sigmoid(_logit(clamp01(blend, 1e-6, 1 - 1e-6)) * k))
    if MINIBENCH_GATE_LO <= result <= MINIBENCH_GATE_HI:
        result = clamp01(_sigmoid(_logit(clamp01(result, 1e-6, 1 - 1e-6)) * MINIBENCH_GATE_AMP))
        triggers.append("T4(gate)")
    return result, k, "+".join(triggers)

def _weighted_blend(pairs: List[Tuple[float, Optional[float]]]) -> float:
    valid = [(w, float(v)) for w, v in pairs if v is not None and _is_num(v)]
    total_w = sum(w for w, _ in valid)
    if not valid or total_w == 0: return 0.5
    return sum(w * v / total_w for w, v in valid)

def _isotonic(vals: List[float]) -> List[float]:
    if not vals: return []
    blocks = [(v, [i]) for i, v in enumerate(vals)]; result = list(vals); i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            merged = blocks[i][1] + blocks[i + 1][1]; avg = sum(result[j] for j in merged) / len(merged)
            for j in merged: result[j] = avg
            blocks = blocks[:i] + [(avg, merged)] + blocks[i + 2:]; i = max(0, i - 1)
        else: i += 1
    return result

def enforce_monotone(pts: List[Percentile]) -> List[Percentile]:
    pts = sorted(pts, key=lambda x: float(x.percentile)); values = _isotonic([float(p.value) for p in pts])
    for p, v in zip(pts, values): p.value = v
    return pts

def interpolate_percentile(source: Dict[float, float], target: float) -> Optional[float]:
    if not source: return None
    keys = sorted(source.keys())
    if target <= keys[0]: return source[keys[0]]
    if target >= keys[-1]: return source[keys[-1]]
    for i in range(len(keys) - 1):
        lo, hi = keys[i], keys[i + 1]
        if lo <= target <= hi:
            frac = (target - lo) / (hi - lo) if hi != lo else 0.0
            return source[lo] + frac * (source[hi] - source[lo])
    return None

def _numeric_midpoint(q: NumericQuestion) -> float:
    try: return (float(q.lower_bound or 0) + float(q.upper_bound or 100)) / 2.0
    except Exception: return 50.0

# ============================================================
# OpenRouter HTTP helper — with 404 handling + auto-router fallback
# ============================================================
def _sync_openrouter(model: str, messages: List[Dict[str, str]], max_tokens: int = 1200, temperature: float = 0.3, retries: int = 3, use_fallback: bool = True) -> str:
    headers = {"Authorization": f"Bearer {_OPENROUTER_KEY}", "Content-Type": "application/json", "HTTP-Referer": "https://github.com/mewhisk", "X-Title": "mewhisk-forecasting-bot"}
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    last_err: Exception = RuntimeError("no attempts"); current_model = model; tried_auto_router = False

    for attempt in range(retries):
        try:
            resp = _requests.post(f"{_OPENROUTER_BASE}/chat/completions", headers=headers, json=payload, timeout=90)
            if resp.status_code == 429:
                wait = 5 * (2 ** attempt)
                logger.warning(f"429 on {current_model} — sleeping {wait}s (attempt {attempt+1})")
                time.sleep(wait); last_err = RuntimeError(f"429 after {attempt+1} attempts"); continue
            if resp.status_code == 404:
                err_msg = resp.json().get("error", {}).get("message", "Not Found") if resp.content else "Not Found"
                logger.error(f"OpenRouter {current_model} error 404: {err_msg}")
                # Auto-router fallback for 404s
                if use_fallback and not tried_auto_router and current_model != _AUTO_ROUTER:
                    logger.info(f"Model '{current_model}' not found — falling back to {_AUTO_ROUTER}")
                    payload["model"] = _AUTO_ROUTER; current_model = _AUTO_ROUTER; tried_auto_router = True
                    continue
            if resp.status_code >= 400:
                try: err_json = resp.json(); logger.error(f"OpenRouter {current_model} error {resp.status_code}: {err_json}")
                except Exception: logger.error(f"OpenRouter {current_model} error {resp.status_code}: {resp.text[:300]}")
            resp.raise_for_status()
            return (resp.json()["choices"][0]["message"]["content"] or "").strip()
        except _requests.exceptions.Timeout:
            last_err = RuntimeError(f"Timeout on {current_model} attempt {attempt+1}"); logger.warning(str(last_err))
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                logger.warning(f"OpenRouter {current_model} attempt {attempt+1} error: {e}")
                if use_fallback and not tried_auto_router and current_model != _AUTO_ROUTER and model in _FALLBACK_MODELS:
                    fallback = _FALLBACK_MODELS[model]
                    logger.info(f"Retrying with fallback: {fallback}")
                    payload["model"] = fallback; current_model = fallback; tried_auto_router = (fallback == _AUTO_ROUTER)
    raise RuntimeError(f"OpenRouter failed for {model}: {last_err}") from last_err

async def _call(model: str, messages: List[Dict[str, str]], max_tokens: int = 1200, temperature: float = 0.3, use_fallback: bool = True) -> str:
    async with _LLM_SEM:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_openrouter, model, messages, max_tokens, temperature, 3, use_fallback)

async def _prompt(model: str, prompt: str, max_tokens: int = 1200) -> str:
    return await _call(model, [{"role": "user", "content": prompt}], max_tokens=max_tokens)

# ============================================================
# Research pipeline
# ============================================================
_RES_SYS = "You are a precise research assistant. Today is {today}. Reason carefully about the forecasting question using your knowledge. Give a concise factual digest: key recent facts, base rates, relevant trends. Max 350 words. Be specific with dates, numbers and source names where known."

async def _run_research(query: str) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    msgs = [{"role": "system", "content": _RES_SYS.format(today=today)}, {"role": "user", "content": f"Research for forecasting: {query}"}]
    results: Dict[int, str] = {}
    async def fetch(slot: int, model: str, label: str) -> None:
        try: results[slot] = await _call(model, msgs, max_tokens=500, temperature=0.1)
        except Exception as e: logger.warning(f"Research {label} ({model}) failed: {e}"); results[slot] = f"[{label} unavailable]"
    await asyncio.gather(fetch(1, RESEARCH_MODEL_1, "R1"), fetch(2, RESEARCH_MODEL_2, "R2"))
    return f"[Research as of {today}]\n\n=== SOURCE 1 ===\n{results.get(1, '[unavailable]')}\n\n=== SOURCE 2 ===\n{results.get(2, '[unavailable]')}"

# ============================================================
# Parse helpers
# ============================================================
async def _parse_bin(raw: str) -> Optional[float]:
    v = extract_binary_prob_from_text(raw)
    if v is not None: return clamp01(v)
    try:
        r = await _prompt(FORECAST_MODEL_PRIMARY, f"Extract the probability. Reply ONLY with a decimal 0–1.\n\n{raw[:500]}", max_tokens=10)
        vf = safe_float(r.strip().split()[0], default=None)
        if vf is not None: return clamp01(float(vf))
    except Exception: pass
    return None

async def _parse_mc_probs(raw: str, options: List[str]) -> Optional[Dict[str, float]]:
    ip = extract_indexed_mc_probs(raw, len(options))
    if ip: return {options[i-1]: float(ip[i]) for i in range(1, len(options)+1) if i in ip}
    try:
        opts = "\n".join(build_indexed_options(options))
        r = await _prompt(FORECAST_MODEL_PRIMARY, f"Extract probabilities for these options summing to 100%.\nReply ONLY with lines '1: XX%'.\n\nOptions:\n{opts}\n\nText:\n{raw[:500]}", max_tokens=60)
        ip2 = extract_indexed_mc_probs(r, len(options))
        if ip2: return {options[i-1]: float(ip2[i]) for i in range(1, len(options)+1) if i in ip2}
    except Exception: pass
    return None

async def _parse_num(raw: str, question: NumericQuestion) -> Optional[NumericDistribution]:
    targets = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]; extracted = extract_numeric_percentiles(raw, targets)
    if extracted:
        pts = [Percentile(percentile=pt, value=float(extracted[pt])) for pt in targets if pt in extracted]
        if pts: return NumericDistribution.from_question(enforce_monotone(pts), question)
    try:
        r = await _prompt(FORECAST_MODEL_PRIMARY, "Extract percentiles. Reply ONLY with:\nPercentile 10: X\nPercentile 20: X\nPercentile 40: X\nPercentile 60: X\nPercentile 80: X\nPercentile 90: X\n\nText:\n{raw[:500]}", max_tokens=60)
        ex2 = extract_numeric_percentiles(r, targets)
        if ex2:
            pts2 = [Percentile(percentile=pt, value=float(ex2[pt])) for pt in targets if pt in ex2]
            if pts2: return NumericDistribution.from_question(enforce_monotone(pts2), question)
    except Exception: pass
    return None

# ============================================================
# Bot class
# ============================================================
class mewhisk(ForecastBot):
    _max_concurrent_questions = 4; _concurrency_limiter = _Q_SEM; _structure_output_validation_samples = 1

    def __init__(self, *args, extremize_enabled: bool = True, extremize_k_binary: float = 1.15, extremize_k_mc: float = 1.12, **kwargs):
        super().__init__(*args, **kwargs)
        self.extremize_enabled = bool(extremize_enabled); self.extremize_k_binary = float(extremize_k_binary); self.extremize_k_mc = float(extremize_k_mc)
        self._active_tournament: str = ""; self._drop_counts: Dict[str, int] = {}
        for name, model in [("PRIMARY", FORECAST_MODEL_PRIMARY), ("ADVERSARIAL", FORECAST_MODEL_ADVERSARIAL), ("RESEARCH_1", RESEARCH_MODEL_1), ("RESEARCH_2", RESEARCH_MODEL_2)]:
            if not _validate_openrouter_model(model): logger.critical(f"Invalid {name} model config: '{model}' — bot will fail on API calls")
        logger.info(f"mewhisk ready | primary={FORECAST_MODEL_PRIMARY} | adversarial={FORECAST_MODEL_ADVERSARIAL} | minibench k≤{MINIBENCH_K_MAX} | other k_bin={self.extremize_k_binary}")

    def set_active_tournament(self, tid: str) -> None:
        self._active_tournament = str(tid).strip().lower(); logger.info(f"Active tournament: '{self._active_tournament}'")

    async def run_research(self, question: MetaculusQuestion) -> str:
        return await _run_research(build_query(question))

    def _binary_prompt(self, q: BinaryQuestion, research: str, role: str) -> str:
        return clean_indents(f"""You are a calibrated forecaster. Maximise log score and Brier score. 1) Apply outside-view base rates first. 2) Steelman the opposite side before finalising. Role: {role} | Date: {datetime.now().strftime('%Y-%m-%d')} Question: {q.question_text} Background: {q.background_info} Resolution Criteria: {q.resolution_criteria} Fine Print: {q.fine_print} Research: {research} Write 5-8 reasoning bullets, then end EXACTLY with: Probability: ZZ% Decimal: 0.ZZ""")

    def _mc_prompt(self, q: MultipleChoiceQuestion, research: str, role: str) -> str:
        opts = list(q.options)
        return clean_indents(f"""You are a calibrated forecaster. Maximise proper scoring rules. 1) Apply outside-view base rates first. 2) Steelman the leading alternative. Role: {role} | Date: {datetime.now().strftime('%Y-%m-%d')} Question: {q.question_text} Background: {q.background_info} Resolution Criteria: {q.resolution_criteria} Fine Print: {q.fine_print} Options: {chr(10).join(build_indexed_options(opts))} Research: {research} Write 5-8 reasoning bullets, then end EXACTLY with {len(opts)} lines: 1: XX% ... {len(opts)}: XX% (must sum to 100%)""")

    def _numeric_prompt(self, q: NumericQuestion, research: str, role: str) -> str:
        low = q.nominal_lower_bound if q.nominal_lower_bound is not None else q.lower_bound; high = q.nominal_upper_bound if q.nominal_upper_bound is not None else q.upper_bound
        lo_m = f"Cannot be lower than {low}." if not q.open_lower_bound else f"Unlikely below {low}."; hi_m = f"Cannot be higher than {high}." if not q.open_upper_bound else f"Unlikely above {high}."
        return clean_indents(f"""You are a calibrated forecaster. Maximise proper scoring rules. 1) Apply outside-view base rates first. 2) Steelman a much-lower or much-higher outcome. Role: {role} | Date: {datetime.now().strftime('%Y-%m-%d')} Question: {q.question_text} Background: {q.background_info} Resolution Criteria: {q.resolution_criteria} Fine Print: {q.fine_print} Units: {getattr(q, 'unit_of_measure', 'inferred')} Bounds: {lo_m} {hi_m} Research: {research} Write 5-8 reasoning bullets, then end EXACTLY with: Percentile 10: X Percentile 20: X Percentile 40: X Percentile 60: X Percentile 80: X Percentile 90: X""")

    async def _run_bin(self, q: BinaryQuestion, research: str, tag: str, role: str, model: str) -> ReasonedPrediction:
        raw = await _prompt(model, self._binary_prompt(q, research, role)); val = await _parse_bin(raw)
        if val is None:
            try: r2 = await _prompt(model, "Output ONLY:\nProbability: ZZ%\nDecimal: 0.ZZ", max_tokens=20); val = await _parse_bin(r2); raw += "\n[RETRY]\n" + r2
            except Exception: pass
        if val is None: self._drop_counts[f"{tag}_binary"] = self._drop_counts.get(f"{tag}_binary", 0) + 1; val = 0.5
        return ReasonedPrediction(prediction_value=clamp01(val), reasoning=raw)

    async def _run_mc(self, q: MultipleChoiceQuestion, research: str, tag: str, role: str, model: str) -> ReasonedPrediction:
        opts = list(q.options); raw = await _prompt(model, self._mc_prompt(q, research, role)); probs = await _parse_mc_probs(raw, opts)
        if probs is None:
            try: r2 = await _prompt(model, "Output ONLY numbered % lines summing to 100%.", max_tokens=50); probs = await _parse_mc_probs(r2, opts); raw += "\n[RETRY]\n" + r2
            except Exception: pass
        if probs is None: self._drop_counts[f"{tag}_mc"] = self._drop_counts.get(f"{tag}_mc", 0) + 1; u = 1.0 / max(1, len(opts)); probs = {o: u for o in opts}
        return ReasonedPrediction(prediction_value=PredictedOptionList(predicted_options=[PredictedOption(option_name=o, probability=p) for o, p in probs.items()]), reasoning=raw)

    async def _run_num(self, q: NumericQuestion, research: str, tag: str, role: str, model: str) -> ReasonedPrediction:
        raw = await _prompt(model, self._numeric_prompt(q, research, role)); dist = await _parse_num(raw, q)
        if dist is None:
            try: r2 = await _prompt(model, "Output ONLY:\nPercentile 10: X\nPercentile 20: X\nPercentile 40: X\nPercentile 60: X\nPercentile 80: X\nPercentile 90: X", max_tokens=60); dist = await _parse_num(r2, q); raw += "\n[RETRY]\n" + r2
            except Exception: pass
        if dist is None: self._drop_counts[f"{tag}_numeric"] = self._drop_counts.get(f"{tag}_numeric", 0) + 1; dist = NumericDistribution.from_question([Percentile(value=_numeric_midpoint(q), percentile=0.5)], q)
        return ReasonedPrediction(prediction_value=dist, reasoning=raw)

    async def _run_forecast_on_binary(self, q, research): return await self._run_bin(q, research, "primary", "PRIMARY", FORECAST_MODEL_PRIMARY)
    async def _run_forecast_on_multiple_choice(self, q, research): return await self._run_mc(q, research, "primary", "PRIMARY", FORECAST_MODEL_PRIMARY)
    async def _run_forecast_on_numeric(self, q, research): return await self._run_num(q, research, "primary", "PRIMARY", FORECAST_MODEL_PRIMARY)

    @staticmethod
    def _clean_for_publish(text: str) -> str:
        lines = [l for l in text.splitlines() if not l.strip().startswith("[stats]") and not re.match(r"^\[AGENT_[A-Z]\]", l.strip()) and l.strip() != "---"]
        return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()

    async def _make_prediction(self, question: MetaculusQuestion, research: str):
        async with _Q_SEM:
            preds: List[Tuple[str, Any]] = []; reasonings: List[str] = []; w_p, w_a = 0.55, 0.45; is_mb = self._active_tournament == MINIBENCH_ID.lower()
            async def run_agent(tag: str, role: str, model: str) -> None:
                try:
                    if isinstance(question, BinaryQuestion): p = await self._run_bin(question, research, tag, role, model)
                    elif isinstance(question, MultipleChoiceQuestion): p = await self._run_mc(question, research, tag, role, model)
                    else: p = await self._run_num(question, research, tag, role, model)
                    lbl = "A" if tag == "primary" else "B"; preds.append((tag, p.prediction_value)); reasonings.append(f"[AGENT_{lbl}]\n{p.reasoning}")
                except Exception as e: logger.error(f"Agent {tag} failed: {e}")
            await asyncio.gather(run_agent("primary", "PRIMARY", FORECAST_MODEL_PRIMARY), run_agent("adversarial", "ADVERSARIAL_CHECKER", FORECAST_MODEL_ADVERSARIAL))
            if not preds: raise RuntimeError("All forecast agents failed.")
            internal = "\n\n---\n\n".join(reasonings)
            def get_val(tag: str) -> Any:
                for t, v in preds:
                    if t == tag: return v
                return None
            p_pred, a_pred = get_val("primary"), get_val("adversarial")
            if isinstance(question, BinaryQuestion):
                p_val = float(p_pred) if p_pred is not None and _is_num(p_pred) else None; a_val = float(a_pred) if a_pred is not None and _is_num(a_pred) else None
                blend = clamp01(_weighted_blend([(w_p, p_val), (w_a, a_val)]))
                if self.extremize_enabled:
                    if is_mb: final, eff_k, trigs = minibench_extremize_binary(blend, p_val, a_val, research); ext = f"minibench({trigs} k_eff={eff_k:.1f}) {blend:.3f}→{final:.3f}"
                    else: final = extremize_binary(blend, self.extremize_k_binary); ext = f"extremize(k={self.extremize_k_binary:.2f}) {blend:.3f}→{final:.3f}"
                else: final, ext = blend, "extremize=OFF"
                np_ = [v for v in [p_val, a_val] if v is not None] or [0.5]; m, md, sd = mean(np_), safe_median(np_), stdev(np_); lo, hi = ci90(np_)
                stats = f"[stats] n={len(np_)} mean={m:.3f} median={md:.3f} sd={sd:.3f} ci90=({lo:.3f},{hi:.3f}) {ext}"
                return ReasonedPrediction(prediction_value=final, reasoning=self._clean_for_publish(stats + "\n\n" + internal))
            if isinstance(question, MultipleChoiceQuestion):
                opts = list(question.options)
                def p2d(pol): return ({str(po.option_name).strip(): float(po.probability) for po in pol.predicted_options if _is_num(po.probability)} if isinstance(pol, PredictedOptionList) else {})
                p_d, a_d = (p2d(p_pred) if p_pred is not None else {}), (p2d(a_pred) if a_pred is not None else {})
                blended = {opt: _weighted_blend([(w_p, p_d.get(opt)), (w_a, a_d.get(opt))]) or 1e-6 for opt in opts}; total = sum(blended.values())
                blended = ({k: v/total for k, v in blended.items()} if total > 0 else {o: 1.0/len(opts) for o in opts})
                if self.extremize_enabled:
                    k_mc = MINIBENCH_K_MC if is_mb else self.extremize_k_mc; blended = extremize_mc(blended, k_mc); mc_note = f"extremize_mc(k={k_mc:.2f}{'[mb]' if is_mb else ''})"
                else: mc_note = "extremize=OFF"
                stats = f"[stats] n={len(preds)} entropy={entropy(blended):.3f} {mc_note}"
                return ReasonedPrediction(prediction_value=PredictedOptionList(predicted_options=[PredictedOption(option_name=o, probability=float(p)) for o, p in blended.items()]), reasoning=self._clean_for_publish(stats + "\n\n" + internal))
            if isinstance(question, NumericQuestion):
                targets = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
                def d2m(d): return ({normalize_percentile(getattr(item, "percentile", None)): float(v) for item in d.declared_percentiles if (v := safe_float(getattr(item, "value", None), default=None)) is not None} if isinstance(d, NumericDistribution) else {})
                p_m, a_m = (d2m(p_pred) if p_pred is not None else {}), (d2m(a_pred) if a_pred is not None else {}); pts = []
                for pt in targets:
                    pv = interpolate_percentile(p_m, pt) if p_m else None; av = interpolate_percentile(a_m, pt) if a_m else None
                    v = _weighted_blend([(w_p, pv), (w_a, av)]); 
                    if pv is None and av is None: v = _numeric_midpoint(question)
                    pts.append(Percentile(percentile=pt, value=float(v)))
                pts = enforce_monotone(pts); p10 = next((p.value for p in pts if abs(float(p.percentile)-0.1) < 1e-9), None); p90 = next((p.value for p in pts if abs(float(p.percentile)-0.9) < 1e-9), None)
                spread = (p90-p10) if (p10 is not None and p90 is not None) else float("nan"); stats = f"[stats] n={len(preds)} p10={p10} p90={p90} spread={spread:.3f}"
                return ReasonedPrediction(prediction_value=NumericDistribution.from_question(pts, question), reasoning=self._clean_for_publish(stats + "\n\n" + internal))
            return ReasonedPrediction(prediction_value=preds[0][1], reasoning=self._clean_for_publish(internal))

    def log_internal_drop_stats(self) -> None:
        if self._drop_counts: logger.info(f"[drops] {self._drop_counts}")

# ============================================================
# MAIN
# ============================================================
MINIBENCH_ID = "minibench"; SPRING_AI_TOURNAMENT_ID = "32916"

if __name__ == "__main__":
    logging.getLogger("litellm").setLevel(logging.WARNING); logging.getLogger("litellm").propagate = False
    parser = argparse.ArgumentParser(description="mewhisk forecasting bot — all LLM via OpenRouter")
    parser.add_argument("--tournament-ids", nargs="+", default=[MINIBENCH_ID, SPRING_AI_TOURNAMENT_ID])
    parser.add_argument("--no-extremize", action="store_true")
    parser.add_argument("--extremize-k-binary", type=float, default=1.15)
    parser.add_argument("--extremize-k-mc", type=float, default=1.12)
    parser.add_argument("--forecast-primary", type=str, default=None, help="Override primary model (bare OpenRouter id, e.g. 'meta-llama/llama-3.3-70b-instruct:free')")
    parser.add_argument("--forecast-adversarial", type=str, default=None, help="Override adversarial model (bare OpenRouter id)")
    parser.add_argument("--research-model-1", type=str, default=None); parser.add_argument("--research-model-2", type=str, default=None)
    args = parser.parse_args()
    if args.forecast_primary: FORECAST_MODEL_PRIMARY = args.forecast_primary
    if args.forecast_adversarial: FORECAST_MODEL_ADVERSARIAL = args.forecast_adversarial
    if args.research_model_1: RESEARCH_MODEL_1 = args.research_model_1
    if args.research_model_2: RESEARCH_MODEL_2 = args.research_model_2
    if not _OPENROUTER_KEY: logger.error("OPENROUTER_API_KEY is required."); raise SystemExit(1)
    all_models = [FORECAST_MODEL_PRIMARY, FORECAST_MODEL_ADVERSARIAL, RESEARCH_MODEL_1, RESEARCH_MODEL_2]
    if not all(_validate_openrouter_model(m) for m in all_models): logger.critical("One or more model IDs are invalid — aborting to prevent runtime failures"); raise SystemExit(1)
    bot = mewhisk(research_reports_per_question=1, predictions_per_research_report=2, publish_reports_to_metaculus=True, skip_previously_forecasted_questions=True, extremize_enabled=not args.no_extremize, extremize_k_binary=args.extremize_k_binary, extremize_k_mc=args.extremize_k_mc)
    async def run_all() -> List[Any]:
        reports: List[Any] = []
        for tid in args.tournament_ids:
            bot.set_active_tournament(tid); logger.info(f"Forecasting tournament: {tid}")
            reports.extend(await bot.forecast_on_tournament(tid, return_exceptions=True))
        return reports
    try:
        reports = asyncio.run(run_all()); bot.log_report_summary(reports); bot.log_internal_drop_stats(); logger.info("mewhisk run complete.")
    except Exception as e: logger.error(f"Fatal: {e}"); raise SystemExit(1)
    