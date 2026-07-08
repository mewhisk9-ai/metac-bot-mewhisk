"""Offline unit checks for bot_helpers (no API keys required)."""

from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from bot_helpers.calibration import CalibrationStore, apply_binary_calibration, extremize
from bot_helpers.decomposition import looks_complex
from bot_helpers.markets import blend_binary_with_markets, fetch_market_crosschecks
from bot_helpers.model_memory import ModelMemoryStore
from bot_helpers.rate_limit import VultrRateLimiter
from bot_helpers.rationale_rag import RationaleStore
from bot_helpers.recency import days_until_resolution, recency_time_range


def test_recency():
    soon = SimpleNamespace(
        scheduled_resolution_time=datetime.now(timezone.utc) + timedelta(days=5),
        close_time=None,
    )
    mid = SimpleNamespace(
        scheduled_resolution_time=datetime.now(timezone.utc) + timedelta(days=40),
        close_time=None,
    )
    far = SimpleNamespace(
        scheduled_resolution_time=datetime.now(timezone.utc) + timedelta(days=400),
        close_time=None,
    )
    assert recency_time_range(soon) == "week"
    assert recency_time_range(mid) == "month"
    assert recency_time_range(far) is None
    assert days_until_resolution(soon) is not None
    print("recency ok")


def test_decomposition_heuristic():
    simple = SimpleNamespace(question_text="Will Bitcoin hit $200k in 2026?")
    complex_q = SimpleNamespace(
        question_text=(
            "Will both the US and China announce AI safety regulations before 2027, "
            "and will at least one major lab pause training if either does?"
        )
    )
    assert not looks_complex(simple)
    assert looks_complex(complex_q)
    print("decomposition heuristic ok")


def test_calibration():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "cal.json"
        store = CalibrationStore(path)
        for p, y in [(0.2, 0), (0.3, 0), (0.7, 1), (0.8, 1)] * 6:
            store.add_resolution(p, y)
        store.refit()
        out = apply_binary_calibration(0.6, store)
        assert 0.01 <= out <= 0.99
        assert extremize(0.7) > 0.7
        print("calibration ok", store.as_dict(), "sample", round(out, 4))


def test_model_memory_weights():
    with tempfile.TemporaryDirectory() as td:
        mem = ModelMemoryStore(
            path=Path(td) / "forecasts.jsonl",
            weights_path=Path(td) / "weights.json",
        )
        for i in range(10):
            mem.record_forecast(
                question_id=i,
                question_text=f"q{i}",
                model_id="good",
                question_type="binary",
                prediction=0.8 if i % 2 == 0 else 0.2,
            )
            mem.record_forecast(
                question_id=i,
                question_text=f"q{i}",
                model_id="bad",
                question_type="binary",
                prediction=0.2 if i % 2 == 0 else 0.8,
            )
            mem.mark_resolution(i, i % 2 == 0)
        assert mem.weights["good"] > mem.weights["bad"]
        p = mem.weighted_median_binary(["good", "bad"], [0.9, 0.1])
        assert 0.01 <= p <= 0.99
        print("model memory ok", mem.weights, "median", p)


def test_rationale_rag():
    with tempfile.TemporaryDirectory() as td:
        store = RationaleStore(Path(td) / "rationales.jsonl")
        store.add(
            question_id=1,
            question_text="Will OpenAI release GPT-6 before July 2027?",
            prediction=0.35,
            reasoning="Product cadence and compute constraints suggest delay.",
            question_type="binary",
        )
        store.add(
            question_id=2,
            question_text="Will Anthropic release Claude 5 in 2026?",
            prediction=0.55,
            reasoning="Competitive pressure may accelerate releases.",
            question_type="binary",
        )
        ctx = store.format_context(
            "Will Google DeepMind release Gemini Ultra 2 in 2026?"
        )
        assert "SIMILAR PAST RATIONALES" in ctx or ctx == ""
        print("rationale rag ok", "chars", len(ctx))


def test_rate_limiter():
    async def _run():
        lim = VultrRateLimiter(requests_per_minute=120, burst=2)
        t0 = asyncio.get_event_loop().time()
        await lim.acquire()
        await lim.acquire()
        await lim.acquire()  # should wait for refill
        elapsed = asyncio.get_event_loop().time() - t0
        assert elapsed >= 0.2
        print("rate limiter ok", round(elapsed, 3))

    asyncio.run(_run())


def test_market_blend_and_live_search():
    block = (
        "--- MARKET CROSS-CHECKS ---\n"
        "Rough market-implied average (unweighted): 40.0% across 2 priced markets.\n"
    )
    blended = blend_binary_with_markets(0.6, block, weight=0.2)
    assert abs(blended - (0.8 * 0.6 + 0.2 * 0.4)) < 1e-9
    # Live search (best-effort; should not raise)
    try:
        text = fetch_market_crosschecks("Will AI achieve AGI before 2030?")
        assert "MARKET CROSS-CHECKS" in text
        print("markets ok", text.splitlines()[0], "lines", len(text.splitlines()))
    except Exception as exc:
        print("markets live search skipped:", exc)


if __name__ == "__main__":
    test_recency()
    test_decomposition_heuristic()
    test_calibration()
    test_model_memory_weights()
    test_rationale_rag()
    test_rate_limiter()
    test_market_blend_and_live_search()
    print("ALL HELPER TESTS PASSED")
