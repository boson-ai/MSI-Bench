"""llm — unified entrypoint for MSI-Bench LLM calls.

Calling spec:
    from ib.llm import LlmRequest, complete, complete_json, render_prompt   # sync
    from ib.llm import LlmService, LlmConfig, UsageTracker                  # async
    from ib.llm import reaches_anthropic   # (provider, model) -> bool: served by Anthropic

    # Sync (single calls, deterministic/test paths):
    text = complete(LlmRequest(provider="openai", model="gpt-4.1-mini", messages=[...])).text

    # Async (concurrent batch calls with rate limiting + usage tracking):
    service = LlmService(LlmConfig())
    payload = await service.acomplete_json(LlmRequest(...), module="planner.director")
    summary = service.usage.summary()
    await service.aclose()

Role-specific modules own validation; this layer owns template rendering, provider
dispatch, credential gating, rate limiting, retries, and call/token counting.

Side effects: live providers may call external model APIs.
"""

from ib.llm.config import LlmConfig
from ib.llm.usage import LlmUsage, UsageTracker
from ib.llm.async_service import LlmService
from ib.llm.openrouter import reaches_anthropic
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
    "reaches_anthropic",
    # async service + infra
    "LlmService",
    "LlmConfig",
    "RateLimiter",
    "RetryPolicy",
    "retry_async",
    "UsageTracker",
    "LlmUsage",
]
