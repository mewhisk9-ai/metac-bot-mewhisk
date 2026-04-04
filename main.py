"""
mewhisk — Metaculus forecasting bot
=====================================
Agent routing   : AgentRouter (https://agentrouter.org)
Primary model   : claude-sonnet-4-6   (fast, calibrated)
Checker model   : claude-opus-4-6     (deep adversarial)
Research stack  : Perplexity sonar-pro (AI + live web, primary)
                  Tavily               (AI + live web, secondary)
                  Exa / Linkup         (raw snippets, fallback only when
                                        AI research looks insufficient)
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
# ---------------------------------------------------------------
_AGENTROUTER_KEY = os.getenv("AGENTROUTER_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
_AGENTROUTER_BASE = "https://agentrouter.org/v1"

# Patch env so litellm picks up AgentRouter automatically
if _AGENTROUTER_KEY:
    os.environ.setdefault("ANTHROPIC_API_KEY", _AGENTROUTER_KEY)
    os.environ.setdefault("ANTHROPIC_API_BASE", _AGENTROUTER_BASE)

# ---------------------------------------------------------------
# Optional raw-search SDKs (fallback only)
# ---------------------------------------------------------------
try:
    from exa_py import Exa
    EXA_AVAILABLE = True
except ImportError:
    EXA_AVAILABLE = False

try:
    from linkup import LinkupClient
    LINKUP_AVAILABLE = True
except ImportError:
    LINKUP_AVAILABLE = False

try:
    from asknews_sdk import AskNewsSDK
    ASKNEWS_SDK_AVAILABLE = True
except ImportError:
    ASKNEWS_SDK_AVAILABLE = False

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
# Env / API key checks
# ============================================================
AGENTROUTER_API_KEY = _AGENTROUTER_KEY
if not AGENTROUTER_API_KEY:
    logger.error("AGENTROUTER_API_KEY (or ANTHROPIC_API_KEY) not set.")

# Perplexity — primary AI search
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
PERPLEXITY_ENABLED = bool(PERPLEXITY_API_KEY)
if not PERPLEXITY_ENABLED:
    logger.warning("PERPLEXITY_API_KEY not set — primary AI search disabled.")

# Tavily — secondary AI search (already in your deps: tavily-python)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_ENABLED = bool(TAVILY_API_KEY)
if not TAVILY_ENABLED:
    logger.warning("TAVILY_API_KEY not set — secondary AI search disabled.")

if not PERPLEXITY_ENABLED and not TAVILY_ENABLED:
    logger.error("Neither Perplexity nor Tavily configured. At least one AI search provider required.")

# Exa / Linkup — raw fallback only
EXA_API_KEY = os.getenv("EXA_API_KEY")
EXA_ENABLED = EXA_AVAILABLE and bool(EXA_API_KEY)
LINKUP_API_KEY = os.getenv("LINKUP_API_KEY")
LINKUP_ENABLED = LINKUP_AVAILABLE and bool(LINKUP_API_KEY)

# AskNews — optional supplement
ASKNEWS_CLIENT_ID = os.getenv("ASKNEWS_CLIENT_ID")
ASKNEWS_CLIENT_SECRET = os.getenv("ASKNEWS_CLIENT_SECRET")
ASKNEWS_ENABLED = bool(ASKNEWS_CLIENT_ID and ASKNEWS_CLIENT_SECRET)

# ============================================================
# Helpers: stats + parsing
# ============================================================
def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)

def median(lst: List[Union[float, int]]) -> float:
    vals = sorted(float(x) for x in lst if _is_num(x))
    if not vals:
        raise ValueError("median() empty")
    n = len(vals)
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
    se = s / math.sqrt(len(xs)) if xs else 0.0
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
_DEC_RE = re.compile(r"(?i)\bdecimal\s*:\s*([0-9]*\.?[0-9]+)\b")

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
    q = re.sub(r"\s+", " ", re.sub(r"http\S+", "", question.question_text or "")).strip()
    bg = re.sub(r"\s+", " ", re.sub(r"http\S+", "", question.background_info or "")).strip()
    if len(q) <= max_chars:
        candidate = f"{q} — {bg}" if bg else q
        if len(candidate) <= max_chars:
            return candidate
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
    z = math.exp(-abs(x))
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
    s2 = sum(powered.values())
    return {a: v / s2 for a, v in powered.items()} if s2 > 0 else {a: 1.0 / len(probs) for a in probs}

# ============================================================
# Research quality check
# ============================================================
_LOW_CONFIDENCE_SIGNALS = [
    "i don't have", "i don't know", "no information", "not aware",
    "cannot find", "no recent", "no data", "unable to find",
    "insufficient", "limited information", "as of my knowledge",
    "my training data", "i'm not sure", "unclear",
]

def _research_looks_thin(text: str) -> bool:
    if not text or len(text.strip()) < 120:
        return True
    return sum(1 for sig in _LOW_CONFIDENCE_SIGNALS if sig in text.lower()) >= 2

# ============================================================
# AI Search #1 — Perplexity sonar-pro (PRIMARY, live web AI)
# ============================================================
def _sync_perplexity_search(query: str) -> str:
    if not PERPLEXITY_ENABLED:
        return "[Perplexity not configured]"
    try:
        resp = _requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "sonar-pro",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a precise real-time news researcher. "
                            "Give a concise, factual digest for the forecasting question. "
                            "Include dates, numbers, and source names. Max 350 words."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Latest news and context for forecasting: {query}",
                    },
                ],
                "max_tokens": 550,
                "temperature": 0.1,
                "search_recency_filter": "week",
                "return_citations": True,
            },
            timeout=25,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        citations = data.get("citations", [])
        if citations:
            content += "\nSources: " + " | ".join(citations[:4])
        return content
    except Exception as e:
        logger.error(f"Perplexity error: {e}")
        return f"[Perplexity failed: {e}]"

# ============================================================
# AI Search #2 — Tavily (SECONDARY, live web AI)
# tavily-python is already in your deps
# ============================================================
def _sync_tavily_search(query: str) -> str:
    if not TAVILY_ENABLED:
        return "[Tavily not configured]"
    try:
        resp = _requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "search_depth": "advanced",
                "include_answer": True,
                "include_raw_content": False,
                "max_results": 5,
                "topic": "news",
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("answer", "").strip()
        results = data.get("results", [])
        snippets = []
        for i, r in enumerate(results[:5]):
            title = r.get("title", "Untitled")
            content = textwrap.shorten(r.get("content", ""), width=220, placeholder="…")
            url = r.get("url", "")
            snippets.append(f"[T{i+1}] {title}: {content}" + (f" ({url})" if url else ""))
        body = (f"Summary: {answer}\n\n" if answer else "") + "\n".join(snippets)
        return body.strip() or "[Tavily: no results]"
    except Exception as e:
        logger.error(f"Tavily error: {e}")
        return f"[Tavily failed: {e}]"

# ============================================================
# Raw fallback: Exa (only activated when AI research is thin)
# ============================================================
def _sync_exa_search(query: str, max_results: int = 5) -> List[str]:
    if not EXA_ENABLED:
        return []
    try:
        client = Exa(api_key=EXA_API_KEY)
        results = client.search_and_contents(
            query, type="auto", num_results=max_results,
            highlights={"max_characters": 4000},
        )
        snippets = []
        for i, res in enumerate(results.results[:max_results]):
            title = getattr(res, "title", "Untitled") or "Untitled"
            url = getattr(res, "url", "") or ""
            hl = getattr(res, "highlights", None)
            content = (" ".join(hl) if hl and isinstance(hl, list) else (getattr(res, "text", "") or ""))[:500]
            snippets.append(
                f"[E{i+1}] {title}: {textwrap.shorten(content, width=240, placeholder='…')}"
                + (f" ({url})" if url else "")
            )
        return snippets
    except Exception as e:
        logger.error(f"Exa fallback error: {e}")
        return []

# ============================================================
# Raw fallback: Linkup (only activated when AI research is thin)
# ============================================================
def _sync_linkup_search(query: str, max_results: int = 5) -> List[str]:
    if not LINKUP_ENABLED:
        return []
    try:
        client = LinkupClient(api_key=LINKUP_API_KEY)
        response = client.search(query=query, depth="standard", output_type="searchResults")
        results = getattr(response, "results", []) or []
        snippets = []
        for i, res in enumerate(results[:max_results]):
            title = getattr(res, "name", "Untitled") or "Untitled"
            url = getattr(res, "url", "") or ""
            content = (getattr(res, "content", "") or "")[:500]
            snippets.append(
                f"[L{i+1}] {title}: {textwrap.shorten(content, width=240, placeholder='…')}"
                + (f" ({url})" if url else "")
            )
        return snippets
    except Exception as e:
        logger.error(f"Linkup fallback error: {e}")
        return []

# ============================================================
# AskNews (optional supplement)
# ============================================================
def _get_asknews_client() -> Any:
    if not ASKNEWS_ENABLED:
        return None
    if ASKNEWS_SDK_AVAILABLE:
        return AskNewsSDK(
            client_id=ASKNEWS_CLIENT_ID,
            client_secret=ASKNEWS_CLIENT_SECRET,
            scopes=["news"],
        )
    try:
        resp = _requests.post(
            "https://api.asknews.app/v1/oauth/token",
            data={
                "grant_type": "client_credentials",
                "client_id": ASKNEWS_CLIENT_ID,
                "client_secret": ASKNEWS_CLIENT_SECRET,
                "scope": "news",
            },
            timeout=10,
        )
        resp.raise_for_status()
        return {"token": resp.json()["access_token"]}
    except Exception as e:
        logger.error(f"AskNews auth failed: {e}")
        return None

def _sync_asknews_search(query: str, client: Any) -> List[str]:
    if client is None:
        return []
    try:
        if ASKNEWS_SDK_AVAILABLE:
            fn = getattr(client.news, "search_news", None) or getattr(client.news, "search_stories", None)
            if not fn:
                return []
            response = fn(query=query, n_articles=5, return_type="news",
                          method="kw", return_story_text=True)
            stories = getattr(response, "news", []) or []
        else:
            r = _requests.get(
                "https://api.asknews.app/v1/news",
                headers={"Authorization": f"Bearer {client['token']}"},
                params={"q": query, "n_articles": 5, "return_type": "news",
                        "return_story_text": "true"},
                timeout=15,
            )
            r.raise_for_status()
            stories = r.json().get("data", {}).get("news", [])
        snippets = []
        for i, story in enumerate(stories[:5]):
            title = (story.get("title") if isinstance(story, dict) else getattr(story, "title", "")) or "Untitled"
            text = ((story.get("text") if isinstance(story, dict) else getattr(story, "text", "")) or "")[:500]
            snippets.append(f"[A{i+1}] {title}: {textwrap.shorten(text, width=240, placeholder='…')}")
        return snippets
    except Exception as e:
        logger.error(f"AskNews search error: {e}")
        return []

# ============================================================
# Unified research runner
# ============================================================
async def _run_research_pipeline(query: str, loop: asyncio.AbstractEventLoop) -> str:
    """
    1. Run Perplexity + Tavily in parallel (both AI models with live web).
    2. If both results look thin → activate Exa + Linkup raw fallback.
    3. Return structured research block.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    pplx_result = "[Perplexity not configured]"
    tavily_result = "[Tavily not configured]"

    async def run_pplx():
        nonlocal pplx_result
        if PERPLEXITY_ENABLED:
            try:
                pplx_result = await loop.run_in_executor(None, _sync_perplexity_search, query)
            except Exception as e:
                pplx_result = f"[Perplexity error: {e}]"

    async def run_tavily():
        nonlocal tavily_result
        if TAVILY_ENABLED:
            try:
                tavily_result = await loop.run_in_executor(None, _sync_tavily_search, query)
            except Exception as e:
                tavily_result = f"[Tavily error: {e}]"

    await asyncio.gather(run_pplx(), run_tavily())

    # Only activate raw fallback when both AI searches look thin
    raw_block = ""
    if _research_looks_thin(pplx_result) and _research_looks_thin(tavily_result):
        if EXA_ENABLED or LINKUP_ENABLED:
            logger.info("AI research thin — activating Exa/Linkup raw fallback.")
            exa_snips: List[str] = []
            linkup_snips: List[str] = []

            async def run_exa():
                try:
                    exa_snips.extend(await loop.run_in_executor(None, _sync_exa_search, query))
                except Exception as e:
                    logger.error(f"Exa fallback error: {e}")

            async def run_linkup():
                try:
                    linkup_snips.extend(await loop.run_in_executor(None, _sync_linkup_search, query))
                except Exception as e:
                    logger.error(f"Linkup fallback error: {e}")

            await asyncio.gather(run_exa(), run_linkup())
            merged: List[str] = []
            for i in range(max(len(exa_snips), len(linkup_snips))):
                if i < len(exa_snips):
                    merged.append(exa_snips[i])
                if i < len(linkup_snips):
                    merged.append(linkup_snips[i])
            if merged:
                raw_block = "\n\n=== RAW WEB FALLBACK (Exa + Linkup) ===\n" + "\n".join(merged)

    return (
        f"[Research as of {today}]\n\n"
        f"=== AI SEARCH #1 — PRIMARY ===\n{pplx_result}\n\n"
        f"=== AI SEARCH #2 — SECONDARY ===\n{tavily_result}"
        + raw_block
    )

# ============================================================
# Bot
# ============================================================
class mewhisk(ForecastBot):
    """
    mewhisk — dual-model forecasting bot via AgentRouter.

    Claude model calls are routed through AgentRouter by setting
    ANTHROPIC_API_BASE=https://agentrouter.org/v1 at startup.
    litellm (used by forecasting-tools) picks this up automatically
    for all anthropic/* model strings.

    Extremization: ON, conservative — k_binary=1.15, k_mc=1.12.
    Numeric distributions: no extremization.
    Bot comments: [AGENT_A] / [AGENT_B] — model names never shown.
    """

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    _structure_output_validation_samples = 2

    def _llm_config_defaults(self) -> Dict[str, str]:
        defaults = super()._llm_config_defaults()
        defaults.update({
            # Both route through AgentRouter via ANTHROPIC_API_BASE
            "forecaster_sonnet": "anthropic/claude-sonnet-4-6",
            "forecaster_opus":   "anthropic/claude-opus-4-6",
            "parser":            "anthropic/claude-sonnet-4-6",
        })
        return defaults

    def __init__(
        self,
        *args,
        extremize_enabled: bool = True,
        extremize_k_binary: float = 1.15,
        extremize_k_mc: float = 1.12,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._asknews_client = _get_asknews_client()
        self.extremize_enabled = bool(extremize_enabled)
        self.extremize_k_binary = float(extremize_k_binary)
        self.extremize_k_mc = float(extremize_k_mc)
        self._drop_counts: Dict[str, int] = {}
        self._drop_counts_by_model: Dict[str, Dict[str, int]] = {"sonnet": {}, "opus": {}}
        logger.info(
            f"mewhisk ready | AgentRouter={_AGENTROUTER_BASE} | "
            f"Perplexity={PERPLEXITY_ENABLED} Tavily={TAVILY_ENABLED} "
            f"Exa={EXA_ENABLED}(fallback) Linkup={LINKUP_ENABLED}(fallback) | "
            f"extremize=ON k_bin={self.extremize_k_binary} k_mc={self.extremize_k_mc}"
        )

    def _inc_drop(self, tag: str, reason: str) -> None:
        self._drop_counts[reason] = self._drop_counts.get(reason, 0) + 1
        d = self._drop_counts_by_model.get(tag, {})
        d[reason] = d.get(reason, 0) + 1
        self._drop_counts_by_model[tag] = d

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            query = build_query(question)
            loop = asyncio.get_running_loop()
            research = await _run_research_pipeline(query, loop)
            if ASKNEWS_ENABLED and self._asknews_client is not None:
                try:
                    stories = await loop.run_in_executor(
                        None, _sync_asknews_search, query, self._asknews_client
                    )
                    if stories:
                        research += "\n\n=== ASKNEWS SUPPLEMENT ===\n" + "\n".join(stories)
                except Exception as e:
                    logger.error(f"AskNews supplement failed: {e}")
            return research

    async def _invoke_llm(self, model_name: str, prompt: str) -> str:
        return await self.get_llm(model_name, "llm").invoke(prompt)

    async def _invoke_with_format_retry(self, model_name: str, prompt: str, _: str) -> str:
        return await self._invoke_llm(model_name, prompt)

    async def _parse_binary(self, raw: str, tag: str) -> Optional[float]:
        try:
            pred: BinaryPrediction = await structure_output(
                text_to_structure=raw, output_type=BinaryPrediction,
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

    async def _parse_mc(self, raw: str, question: MultipleChoiceQuestion, tag: str) -> Optional[Dict[str, float]]:
        options = list(question.options)
        try:
            pred: PredictedOptionList = await structure_output(
                text_to_structure=raw, output_type=PredictedOptionList,
                model=self.get_llm("parser", "llm"),
                additional_instructions=f"Valid options: {options}.",
                num_validation_samples=self._structure_output_validation_samples,
            )
            pred_dict = {
                str(po.option_name).strip(): float(po.probability)
                for po in pred.predicted_options if _is_num(po.probability)
            }
            out = {}
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

    async def _parse_numeric(self, raw: str, question: NumericQuestion, tag: str) -> Optional[NumericDistribution]:
        targets = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
        try:
            percentile_list: List[Percentile] = await structure_output(
                text_to_structure=raw, output_type=list[Percentile],
                model=self.get_llm("parser", "llm"),
                num_validation_samples=self._structure_output_validation_samples,
            )
            pts = []
            for p in percentile_list:
                v = safe_float(getattr(p, "value", None), default=None)
                if v is not None:
                    pts.append(Percentile(value=float(v), percentile=normalize_percentile(getattr(p, "percentile", 0.5))))
            if pts:
                pts.sort(key=lambda x: float(x.percentile))
                for i in range(1, len(pts)):
                    if pts[i].value < pts[i - 1].value:
                        pts[i].value = pts[i - 1].value
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
        for i in range(1, len(pts2)):
            if pts2[i].value < pts2[i - 1].value:
                pts2[i].value = pts2[i - 1].value
        return NumericDistribution.from_question(pts2, question)

    def _bounds_messages(self, q: NumericQuestion) -> Tuple[str, str]:
        low = q.nominal_lower_bound if q.nominal_lower_bound is not None else q.lower_bound
        high = q.nominal_upper_bound if q.nominal_upper_bound is not None else q.upper_bound
        return (
            f"Cannot be lower than {low}." if not q.open_lower_bound else f"Unlikely below {low}.",
            f"Cannot be higher than {high}." if not q.open_upper_bound else f"Unlikely above {high}.",
        )

    # Prompts — no model names in output ever
    def _binary_prompt(self, q: BinaryQuestion, research: str, role: str) -> str:
        return clean_indents(f"""
            You are a calibrated forecaster. Maximize log score and Brier score.
            Principles:
            1) Apply outside-view base rates first.
            2) Steelman the opposite side before finalizing.
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
            You are a calibrated forecaster. Maximize proper scoring rules.
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
            You are a calibrated forecaster. Maximize proper scoring rules.
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

    async def _run_binary_role(self, q, research, tag, role) -> ReasonedPrediction[float]:
        model = f"forecaster_{tag}"
        raw = await self._invoke_with_format_retry(model, self._binary_prompt(q, research, role), "bin")
        val = await self._parse_binary(raw, tag)
        if val is None:
            try:
                raw2 = await self._invoke_llm(model, "Output ONLY:\nProbability: ZZ%\nDecimal: 0.ZZ")
                val = await self._parse_binary(raw2, tag)
                raw += "\n\n[RETRY]\n" + raw2
            except Exception:
                self._inc_drop(tag, "retry_failed_binary")
        if val is None:
            self._inc_drop(tag, "invalid_binary")
            val = 0.5
        return ReasonedPrediction(prediction_value=clamp01(val), reasoning=raw)

    async def _run_mc_role(self, q, research, tag, role) -> ReasonedPrediction[PredictedOptionList]:
        model = f"forecaster_{tag}"
        raw = await self._invoke_with_format_retry(model, self._mc_prompt(q, research, role), "mc")
        probs = await self._parse_mc(raw, q, tag)
        if probs is None:
            try:
                raw2 = await self._invoke_llm(model, "Output ONLY numbered % lines summing to 100%.")
                probs = await self._parse_mc(raw2, q, tag)
                raw += "\n\n[RETRY]\n" + raw2
            except Exception:
                self._inc_drop(tag, "retry_failed_mc")
        if probs is None:
            self._inc_drop(tag, "invalid_mc")
            u = 1.0 / max(1, len(q.options))
            probs = {opt: u for opt in q.options}
        return ReasonedPrediction(
            prediction_value=PredictedOptionList(
                predicted_options=[PredictedOption(option_name=o, probability=p) for o, p in probs.items()]
            ),
            reasoning=raw,
        )

    async def _run_numeric_role(self, q, research, tag, role) -> ReasonedPrediction[NumericDistribution]:
        model = f"forecaster_{tag}"
        raw = await self._invoke_with_format_retry(model, self._numeric_prompt(q, research, role), "num")
        dist = await self._parse_numeric(raw, q, tag)
        if dist is None:
            try:
                raw2 = await self._invoke_llm(
                    model,
                    "Output ONLY:\nPercentile 10: X\nPercentile 20: X\nPercentile 40: X\n"
                    "Percentile 60: X\nPercentile 80: X\nPercentile 90: X",
                )
                dist = await self._parse_numeric(raw2, q, tag)
                raw += "\n\n[RETRY]\n" + raw2
            except Exception:
                self._inc_drop(tag, "retry_failed_numeric")
        if dist is None:
            self._inc_drop(tag, "invalid_numeric")
            try:
                l, u = float(q.lower_bound or 0), float(q.upper_bound or 100)
            except Exception:
                l, u = 0.0, 100.0
            dist = NumericDistribution.from_question([Percentile(value=(l + u) / 2, percentile=0.5)], q)
        return ReasonedPrediction(prediction_value=dist, reasoning=raw)

    async def _run_forecast_on_binary(self, q, research):
        return await self._run_binary_role(q, research, "sonnet", "PRIMARY")

    async def _run_forecast_on_multiple_choice(self, q, research):
        return await self._run_mc_role(q, research, "sonnet", "PRIMARY")

    async def _run_forecast_on_numeric(self, q, research):
        return await self._run_numeric_role(q, research, "sonnet", "PRIMARY")

    async def _make_prediction(self, question: MetaculusQuestion, research: str):
        async with self._concurrency_limiter:
            preds: List[Any] = []
            reasonings: List[str] = []
            w_s, w_o = 0.55, 0.45

            for tag, role in (("sonnet", "PRIMARY"), ("opus", "ADVERSARIAL_CHECKER")):
                try:
                    if isinstance(question, BinaryQuestion):
                        p = await self._run_binary_role(question, research, tag, role)
                    elif isinstance(question, MultipleChoiceQuestion):
                        p = await self._run_mc_role(question, research, tag, role)
                    else:
                        p = await self._run_numeric_role(question, research, tag, role)
                    preds.append(p.prediction_value)
                    label = "A" if tag == "sonnet" else "B"
                    reasonings.append(f"[AGENT_{label}]\n{p.reasoning}")
                except Exception as e:
                    logger.error(f"Agent {tag} failed: {e}")

            if not preds:
                raise RuntimeError("All agents failed.")

            combined = "\n\n---\n\n".join(reasonings)

            # BINARY
            if isinstance(question, BinaryQuestion):
                s = float(preds[0]) if len(preds) >= 1 and _is_num(preds[0]) else None
                o = float(preds[1]) if len(preds) >= 2 and _is_num(preds[1]) else None
                np_ = [v for v in [s, o] if v is not None] or [0.5]
                final = clamp01(w_s * (s or 0.5) + w_o * (o or 0.5)) if s and o else clamp01(s or o or 0.5)
                pre = final
                if self.extremize_enabled:
                    final = extremize_binary(final, self.extremize_k_binary)
                m, md, sd = mean(np_), median(np_), stdev(np_)
                lo, hi = ci90(np_)
                stats = (
                    f"[stats] n={len(np_)} mean={m:.3f} median={md:.3f} sd={sd:.3f} "
                    f"ci90=({lo:.3f},{hi:.3f}) extremize(k={self.extremize_k_binary:.2f}) {pre:.3f}→{final:.3f}"
                )
                return ReasonedPrediction(prediction_value=final, reasoning=stats + "\n\n" + combined)

            # MC
            if isinstance(question, MultipleChoiceQuestion):
                options = list(question.options)

                def p2d(pol: Any) -> Dict[str, float]:
                    return ({str(po.option_name).strip(): float(po.probability)
                             for po in pol.predicted_options if _is_num(po.probability)}
                            if isinstance(pol, PredictedOptionList) else {})

                s_d = p2d(preds[0]) if len(preds) >= 1 else {}
                o_d = p2d(preds[1]) if len(preds) >= 2 else {}
                blended: Dict[str, float] = {}
                for opt in options:
                    sv, ov = s_d.get(opt), o_d.get(opt)
                    blended[opt] = (
                        w_s * float(sv) + w_o * float(ov) if sv is not None and ov is not None
                        else float(sv or ov or 1e-6)
                    )
                total = sum(blended.values())
                blended = {k: v / total for k, v in blended.items()} if total > 0 else {o: 1.0 / len(options) for o in options}
                if self.extremize_enabled:
                    blended = extremize_mc(blended, self.extremize_k_mc)
                stats = f"[stats] n_agents={len(preds)} entropy={entropy(blended):.3f} extremize_mc(k={self.extremize_k_mc:.2f})"
                return ReasonedPrediction(
                    prediction_value=PredictedOptionList(
                        predicted_options=[PredictedOption(option_name=o, probability=float(p)) for o, p in blended.items()]
                    ),
                    reasoning=stats + "\n\n" + combined,
                )

            # NUMERIC — no extremization
            if isinstance(question, NumericQuestion):
                targets = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]

                def d2m(d: Any) -> Dict[float, float]:
                    if isinstance(d, NumericDistribution):
                        return {normalize_percentile(getattr(item, "percentile", None)): float(v)
                                for item in d.declared_percentiles
                                if (v := safe_float(getattr(item, "value", None), default=None)) is not None}
                    return {}

                s_m = d2m(preds[0]) if len(preds) >= 1 else {}
                o_m = d2m(preds[1]) if len(preds) >= 2 else {}
                pts: List[Percentile] = []
                for pt in targets:
                    sv = min(s_m.items(), key=lambda kv: abs(kv[0] - pt))[1] if s_m else None
                    ov = min(o_m.items(), key=lambda kv: abs(kv[0] - pt))[1] if o_m else None
                    if sv is None and ov is None:
                        try:
                            l, u = float(question.lower_bound or 0), float(question.upper_bound or 100)
                        except Exception:
                            l, u = 0.0, 100.0
                        v = (l + u) / 2.0
                    elif sv is None:
                        v = float(ov)
                    elif ov is None:
                        v = float(sv)
                    else:
                        v = w_s * float(sv) + w_o * float(ov)
                    pts.append(Percentile(percentile=pt, value=float(v)))
                pts.sort(key=lambda x: float(x.percentile))
                for i in range(1, len(pts)):
                    if pts[i].value < pts[i - 1].value:
                        pts[i].value = pts[i - 1].value
                p10 = next((p.value for p in pts if abs(float(p.percentile) - 0.1) < 1e-9), None)
                p90 = next((p.value for p in pts if abs(float(p.percentile) - 0.9) < 1e-9), None)
                spread = (p90 - p10) if (p10 is not None and p90 is not None) else float("nan")
                stats = f"[stats] n_agents={len(preds)} p10={p10} p90={p90} spread={spread:.3f}"
                return ReasonedPrediction(
                    prediction_value=NumericDistribution.from_question(pts, question),
                    reasoning=stats + "\n\n" + combined,
                )

            return ReasonedPrediction(prediction_value=preds[0], reasoning=combined)

    def log_internal_drop_stats(self) -> None:
        if self._drop_counts:
            logger.info(f"[drops] totals={self._drop_counts}")
            logger.info(f"[drops] by_agent={self._drop_counts_by_model}")


# ============================================================
# MAIN
# ============================================================
MINIBENCH_ID = "minibench"
SPRING_AI_TOURNAMENT_ID = "32916"   # update if the real ID differs

if __name__ == "__main__":
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("litellm").propagate = False

    parser = argparse.ArgumentParser(
        description="mewhisk: AgentRouter | Perplexity+Tavily live AI search | conservative extremize"
    )
    parser.add_argument(
        "--tournament-ids", nargs="+", type=str,
        default=[MINIBENCH_ID, SPRING_AI_TOURNAMENT_ID],
        help="Tournament IDs to forecast (default: minibench + Spring AI Tournament)",
    )
    parser.add_argument("--no-extremize", action="store_true", help="Disable extremization")
    parser.add_argument("--extremize-k-binary", type=float, default=1.15)
    parser.add_argument("--extremize-k-mc", type=float, default=1.12)
    args = parser.parse_args()

    if not AGENTROUTER_API_KEY:
        logger.error("AGENTROUTER_API_KEY is required.")
        raise SystemExit(1)
    if not PERPLEXITY_ENABLED and not TAVILY_ENABLED:
        logger.error("At least one AI search provider required (PERPLEXITY_API_KEY or TAVILY_API_KEY).")
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
