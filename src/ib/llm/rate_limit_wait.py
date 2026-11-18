"""rate_limit_wait — how long a synchronous LLM call waits out a 429 before it retries; pure.

Calling spec:
    seconds = rate_limit_wait(exc, tries, waited, now=datetime.now(timezone.utc), rng=random)
        -> float, or None when the call should stop waiting (not a 429, or the budget is spent)

service.complete calls it on every LlmTransientError. A 429 (LlmRateLimitError) is waited out
without using one of the call's transient attempts, until RATE_LIMIT_BUDGET_S of waiting in total;
after that the 429 counts as an ordinary transient error. The wait before the next try is, first
found: the error's retry_after (Retry-After header), the time until "Limit resets at:
YYYY-MM-DD HH:MM:SS UTC" in the error text (a proxy's per-user limit), else
BACKOFF_BASE_S * 2**tries capped at BACKOFF_MAX_S. Jitter (up to JITTER_S added, or +-50% of the
backoff) spreads the workers that hit the same limit, so they do not all retry at its reset.
A wait that would overrun the budget is cut to what is left of it.
Side effects: none.
"""

from __future__ import annotations

import random
import re
from datetime import datetime, timezone
from typing import Any

from ib.llm.types import LlmRateLimitError

RATE_LIMIT_BUDGET_S = 300.0
BACKOFF_BASE_S = 5.0
BACKOFF_MAX_S = 60.0
JITTER_S = 5.0
_RESETS_AT = re.compile(r"Limit resets at:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*UTC")


def rate_limit_wait(
    exc: Exception, tries: int, waited: float, *, now: datetime | None = None, rng: Any = random
) -> float | None:
    """Return the seconds to wait before retrying a 429, or None (see the module spec)."""
    if not isinstance(exc, LlmRateLimitError) or waited >= RATE_LIMIT_BUDGET_S:
        return None
    reset = _RESETS_AT.search(str(exc))
    if exc.retry_after is not None:
        wait = exc.retry_after + rng.uniform(0, JITTER_S)
    elif reset:
        at = datetime.strptime(reset.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        wait = max(0.0, (at - (now or datetime.now(timezone.utc))).total_seconds()) + rng.uniform(1.0, JITTER_S)
    else:
        wait = min(BACKOFF_BASE_S * 2**tries, BACKOFF_MAX_S) * rng.uniform(0.5, 1.5)
    return min(wait, RATE_LIMIT_BUDGET_S - waited)
