"""llm — unified entrypoint for MSI-Bench LLM calls.

Calling spec:
    from ib.llm import LlmRequest, complete, complete_json, render_prompt   # sync
    from ib.llm import LlmService, LlmConfig, CostTracker                   # async

    # Sync (single calls, deterministic/test paths):
    text = complete(LlmRequest(provider="openai", model="gpt-4.1-mini", messages=[...])).text

    # Async (concurrent batch calls with rate limiting + cost tracking):
    service = LlmService(LlmConfig())
    payload = await service.acomplete_json(LlmRequest(...), module="planner.director")
    summary = service.cost.summary()
    await service.aclose()

Role-specific modules own validation; this layer owns template rendering, provider
dispatch, credential gating, rate limiting, retries, and cost tracking.

Side effects: live providers may call external model APIs.
"""

from ib.llm.config import LlmConfig
from ib.llm.cost import PRICING_TABLE, CostTracker, LlmUsage, price_lookup
from ib.llm.async_service import LlmService
from ib.llm.rate_limit import RateLimiter
from ib.llm.retry import RetryPolicy, retry_async
from ib.llm.service import (
    LlmProviderError,
    LlmRateLimitError,
    LlmRequest,
    LlmResponse,
    LlmTransientError,
    complete,
    complete_json,
    parse_json_object,
)
from ib.llm.templates import render_prompt

__all__ = [
    # request/response + sync dispatch
    "LlmProviderError",
    "LlmRateLimitError",
    "LlmTransientError",
    "LlmRequest",
    "LlmResponse",
    "complete",
    "complete_json",
    "parse_json_object",
    "render_prompt",
    # async service + infra
    "LlmService",
    "LlmConfig",
    "RateLimiter",
    "RetryPolicy",
    "retry_async",
    "CostTracker",
    "LlmUsage",
    "price_lookup",
    "PRICING_TABLE",
]
