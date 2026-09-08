"""retry — async exponential-backoff retry for transient LLM failures.

Calling spec:
    policy = RetryPolicy(max_attempts=3, base_delay_s=1.0, max_delay_s=30.0)
    result = await retry_async(policy, make_call, label="planner.director")

``make_call`` is a zero-arg callable returning a *fresh* awaitable per attempt.
Only LlmTransientError (incl. LlmRateLimitError) is retried; every other
exception — including terminal LlmProviderError (auth, bad JSON) — propagates
immediately. The last transient error is re-raised after the final attempt.
Rate-limit ``retry_after`` is honored when larger than the computed backoff.

Side effects: ``asyncio.sleep`` between attempts.
Dependencies: ib.llm.service error types.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from ib.llm.service import LlmRateLimitError, LlmTransientError

logger = logging.getLogger("ib.llm.retry")

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential-backoff retry configuration with symmetric jitter."""

    max_attempts: int = 3
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0
    jitter_fraction: float = 0.25

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {self.max_attempts}")
        if self.base_delay_s < 0 or self.max_delay_s < 0:
            raise ValueError("retry delays must be >= 0")
        if not 0 <= self.jitter_fraction <= 1:
            raise ValueError(f"jitter_fraction must be 0-1, got {self.jitter_fraction}")

    def delay_for(self, attempt: int, retry_after: float | None) -> float:
        """Backoff seconds before the next attempt (0-indexed attempt number)."""
        delay = min(self.base_delay_s * (2**attempt), self.max_delay_s)
        if self.jitter_fraction > 0:
            jitter = delay * self.jitter_fraction * (2 * random.random() - 1)
            delay = max(0.0, delay + jitter)
        if retry_after is not None and retry_after > delay:
            delay = retry_after
        return delay


async def retry_async(
    policy: RetryPolicy,
    make_call: Callable[[], Awaitable[T]],
    *,
    label: str = "",
) -> T:
    """Run make_call with retries on transient errors. Pure (no shared state)."""
    prefix = f"[{label}] " if label else ""
    last_error: LlmTransientError | None = None
    for attempt in range(policy.max_attempts):
        try:
            return await make_call()
        except LlmTransientError as exc:
            last_error = exc
            logger.warning(
                "%sattempt %d/%d failed: %s", prefix, attempt + 1, policy.max_attempts, exc
            )
            if attempt < policy.max_attempts - 1:
                retry_after = getattr(exc, "retry_after", None) if isinstance(exc, LlmRateLimitError) else None
                await asyncio.sleep(policy.delay_for(attempt, retry_after))
    assert last_error is not None  # loop only exits here after a transient failure
    raise last_error
