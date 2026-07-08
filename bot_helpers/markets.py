"""Prediction-market / Metaforecast-style cross-checks (Polymarket + Manifold)."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

logger = logging.getLogger("mewhisk.markets")

_STOP = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "to",
    "for",
    "and",
    "or",
    "will",
    "be",
    "by",
    "at",
    "is",
    "are",
    "was",
    "were",
    "with",
    "from",
    "that",
    "this",
    "it",
    "as",
    "before",
    "after",
    "than",
    "more",
    "less",
    "how",
    "what",
    "when",
    "which",
    "who",
}


@dataclass
class MarketHit:
    platform: str
    title: str
    url: str
    probability: float | None
    volume: float | None = None

    def format_line(self) -> str:
        p = f"{100 * self.probability:.1f}%" if self.probability is not None else "n/a"
        vol = f", vol≈{self.volume:,.0f}" if self.volume else ""
        return f"- [{self.platform}] {self.title} → {p}{vol}\n  {self.url}"


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if len(w) > 2 and w not in _STOP}


def _overlap_score(query: str, title: str) -> float:
    q, t = _tokens(query), _tokens(title)
    if not q or not t:
        return 0.0
    return len(q & t) / len(q)


def _http_get_json(url: str, timeout: float = 12.0) -> Any:
    req = Request(url, headers={"User-Agent": "mewhisk-forecast-bot/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _search_manifold(query: str, limit: int = 5) -> list[MarketHit]:
    url = (
        "https://api.manifold.markets/v0/search-markets"
        f"?term={quote_plus(query[:120])}&limit={limit}&filter=open"
    )
    data = _http_get_json(url)
    hits: list[MarketHit] = []
    for row in data or []:
        title = row.get("question") or ""
        if _overlap_score(query, title) < 0.25:
            continue
        hits.append(
            MarketHit(
                platform="Manifold",
                title=title,
                url=row.get("url")
                or f"https://manifold.markets/{row.get('creatorUsername')}/{row.get('slug')}",
                probability=float(row["probability"])
                if row.get("probability") is not None
                else None,
                volume=float(row["volume"]) if row.get("volume") is not None else None,
            )
        )
    return hits


def _parse_poly_price(market: dict[str, Any]) -> float | None:
    prices = market.get("outcomePrices")
    outcomes = market.get("outcomes")
    try:
        if isinstance(prices, str):
            prices = json.loads(prices)
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        if not prices:
            return None
        if outcomes and "Yes" in outcomes:
            idx = outcomes.index("Yes")
            return float(prices[idx])
        return float(prices[0])
    except Exception:
        return None


def _search_polymarket(query: str, limit: int = 5) -> list[MarketHit]:
    url = (
        "https://gamma-api.polymarket.com/public-search"
        f"?q={quote_plus(query[:120])}&limit_per_type={limit}"
    )
    data = _http_get_json(url)
    hits: list[MarketHit] = []
    events = (data or {}).get("events") or []
    for event in events:
        markets = event.get("markets") or []
        # Prefer individual markets; fall back to event title
        candidates = markets or [event]
        for m in candidates[:3]:
            title = m.get("question") or m.get("title") or event.get("title") or ""
            if _overlap_score(query, title) < 0.2:
                continue
            slug = m.get("slug") or event.get("slug")
            hits.append(
                MarketHit(
                    platform="Polymarket",
                    title=title,
                    url=f"https://polymarket.com/event/{slug}" if slug else "https://polymarket.com",
                    probability=_parse_poly_price(m),
                    volume=float(m["volume"])
                    if m.get("volume") not in (None, "")
                    else (
                        float(event["volume"])
                        if event.get("volume") is not None
                        else None
                    ),
                )
            )
    return hits


def fetch_market_crosschecks(question_text: str, *, max_hits: int = 6) -> str:
    """
    Search Manifold + Polymarket for related markets and format a research block.
    Metaforecast GraphQL is currently unreliable, so we use live market APIs.
    """
    hits: list[MarketHit] = []
    for label, fn in (("Manifold", _search_manifold), ("Polymarket", _search_polymarket)):
        try:
            hits.extend(fn(question_text))
        except Exception as exc:
            logger.warning("%s search failed: %s", label, exc)

    # Deduplicate by title similarity / url
    seen: set[str] = set()
    unique: list[MarketHit] = []
    for h in sorted(
        hits,
        key=lambda x: (
            -_overlap_score(question_text, x.title),
            -(x.volume or 0),
        ),
    ):
        key = h.url.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(h)
        if len(unique) >= max_hits:
            break

    if not unique:
        return "--- MARKET CROSS-CHECKS ---\n(no closely related open markets found)\n"

    lines = ["--- MARKET CROSS-CHECKS ---"]
    lines.append(
        "Related prediction-market prices (treat as noisy priors; check resolution "
        "criteria carefully — they often differ from Metaculus):"
    )
    lines.extend(h.format_line() for h in unique)

    # Suggest a blended prior from binary-ish hits with prices
    priced = [h.probability for h in unique if h.probability is not None]
    if priced:
        avg = sum(priced) / len(priced)
        lines.append(
            f"\nRough market-implied average (unweighted): {100 * avg:.1f}% "
            f"across {len(priced)} priced markets."
        )
    return "\n".join(lines) + "\n"


def blend_binary_with_markets(p: float, market_block: str, weight: float = 0.2) -> float:
    """Mildly pull binary forecast toward market average if present."""
    m = re.search(r"Rough market-implied average[^:]*:\s*([0-9.]+)%", market_block or "")
    if not m:
        return p
    market_p = float(m.group(1)) / 100.0
    w = max(0.0, min(0.5, weight))
    return (1 - w) * p + w * market_p
