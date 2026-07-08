"""Recency-weighted research helpers for near-term questions."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from forecasting_tools import MetaculusQuestion

TimeRange = Literal["day", "week", "month", "year"]


def _as_aware(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def days_until_resolution(question: MetaculusQuestion) -> float | None:
    """Return days until scheduled resolution / close, if known."""
    now = datetime.now(timezone.utc)
    candidates = [
        _as_aware(getattr(question, "scheduled_resolution_time", None)),
        _as_aware(getattr(question, "close_time", None)),
    ]
    future = [t for t in candidates if t is not None and t > now]
    if not future:
        return None
    soonest = min(future)
    return max(0.0, (soonest - now).total_seconds() / 86400.0)


def recency_time_range(question: MetaculusQuestion) -> TimeRange | None:
    """
    Prefer recent sources for near-term questions.
    - <= 14 days: week
    - <= 60 days: month
    - <= 365 days: year
    - otherwise / unknown: None (no hard filter)
    """
    days = days_until_resolution(question)
    if days is None:
        return None
    if days <= 14:
        return "week"
    if days <= 60:
        return "month"
    if days <= 365:
        return "year"
    return None


def recency_instruction(question: MetaculusQuestion) -> str:
    days = days_until_resolution(question)
    tr = recency_time_range(question)
    if days is None:
        return (
            "Resolution timing is unclear — balance recent news with longer-run "
            "base rates."
        )
    if tr == "week":
        return (
            f"This question resolves in ~{days:.0f} days. Heavily weight the last "
            "7–14 days of developments; older base rates matter less unless they "
            "directly constrain near-term outcomes."
        )
    if tr == "month":
        return (
            f"This question resolves in ~{days:.0f} days. Prefer evidence from the "
            "last 30–60 days, then layer longer-run base rates."
        )
    if tr == "year":
        return (
            f"This question resolves in ~{days:.0f} days. Use recent trends, but "
            "do not ignore multi-year base rates."
        )
    return (
        f"This question is long-horizon (~{days:.0f} days). Emphasize structural "
        "base rates over day-to-day news."
    )
