"""rate_limit — async concurrency + requests-per-minute throttling for LLM calls.

Calling spec:
    limiter = RateLimiter(max_concurrency=8, rpm=60.0)
    async with limiter.slot():
        response = await provider_call(...)

``slot()`` first waits for a requests-per-minute token (token bucket), then
acquires a concurrency slot (semaphore), releasing the slot on exit. The token
bucket starts full (burst = ``rpm`` tokens) and refills at ``rpm / 60`` tokens
per second. Single event loop only.

Side effects: ``asyncio.sleep`` while throttled.
Dependencies: none (asyncio synchronization only).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class _TokenBucket:
    """A single token bucket. Not thread-safe — drive it from one event loop."""

    __slots__ = ("_max_tokens", "_tokens", "_refill_rate", "_last_refill")

    def __init__(self, rpm: float) -> None:
        self._max_tokens = rpm
        self._tokens = rpm  # start full (burst capacity)
        self._refill_rate = rpm / 60.0  # tokens per second
        self._last_refill = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._max_tokens, self._tokens + elapsed * self._refill_rate)
        self._last_refill = now

    def try_acquire(self) -> bool:
        """Consume one token if available. Returns True on success."""
        self._refill()
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False

    def wait_time(self) -> float:
        """Seconds until the next token becomes available."""
        self._refill()
        if self._tokens >= 1.0:
            return 0.0
        return (1.0 - self._tokens) / self._refill_rate


class RateLimiter:
    """Caps in-flight requests (semaphore) and request rate (token bucket).

    The two limits are independent: the bucket smooths the request *rate* while
    the semaphore bounds *concurrent* in-flight requests. Acquire the rate token
    first so a burst of coroutines does not all pass the semaphore at once.
    """

    def __init__(self, max_concurrency: int = 8, rpm: float = 60.0) -> None:
        self._sem = asyncio.Semaphore(max_concurrency)
        self._bucket = _TokenBucket(rpm)
        self._lock = asyncio.Lock()

    async def _acquire_token(self) -> None:
        while True:
            async with self._lock:
                if self._bucket.try_acquire():
                    return
                wait = self._bucket.wait_time()
            await asyncio.sleep(wait)

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Wait for an RPM token, then hold a concurrency slot for the body."""
        await self._acquire_token()
        await self._sem.acquire()
        try:
            yield
        finally:
            self._sem.release()
