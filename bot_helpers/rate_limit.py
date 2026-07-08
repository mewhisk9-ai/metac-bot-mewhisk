"""Vultr-oriented rate limiting for LLM calls."""

from __future__ import annotations

import asyncio
import logging
import os
import time

logger = logging.getLogger("mewhisk.rate_limit")


class VultrRateLimiter:
    """
    Simple token-bucket limiter tuned for Vultr Serverless Inference quotas.

    Defaults: burst of 4 requests, refill ~0.5 req/s (≈30/min).
    Override with VULTR_RPM / VULTR_BURST.
    """

    def __init__(
        self,
        requests_per_minute: float | None = None,
        burst: float | None = None,
    ) -> None:
        rpm = float(
            requests_per_minute
            if requests_per_minute is not None
            else os.getenv("VULTR_RPM", "30")
        )
        self.capacity = float(
            burst if burst is not None else os.getenv("VULTR_BURST", "4")
        )
        self.refresh_rate = max(rpm, 0.1) / 60.0  # tokens per second
        self._tokens = self.capacity
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()
        self.allowed_tries = int(os.getenv("VULTR_ALLOWED_TRIES", "3"))

    async def acquire(self, cost: float = 1.0) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self._updated
                self._updated = now
                self._tokens = min(
                    self.capacity, self._tokens + elapsed * self.refresh_rate
                )
                if self._tokens >= cost:
                    self._tokens -= cost
                    return
                need = cost - self._tokens
                wait_s = need / self.refresh_rate
                logger.debug("Vultr rate limit: waiting %.2fs", wait_s)
                await asyncio.sleep(wait_s)


# Process-wide limiter shared by all Vultr LLM calls
VULTR_LIMITER = VultrRateLimiter()
