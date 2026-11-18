"""types — LLM request/response records, the provider protocol and error types; no logic.

Calling spec:
    LlmRequest(provider, model, messages, ...)   one call's inputs
    LlmResponse(...)                             one call's text + usage
    LlmProvider                                  protocol: complete(request) -> LlmResponse
    LlmProviderError > LlmTransientError > LlmRateLimitError   terminal / retryable / 429
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class LlmProviderError(ValueError):
    """Raised when a unified LLM provider is unavailable or returns invalid output.

    Treated as terminal by the async retry policy (no retry) unless it is one of
    the retryable subclasses below.
    """


class LlmBudgetError(LlmProviderError):
    """The provider refused because the key or account has no money left (ib.llm.budget). Terminal."""


class LlmTransientError(LlmProviderError):
    """Transient provider failure (5xx, connection drop, timeout). Retryable."""


class LlmRateLimitError(LlmTransientError):
    """Provider returned 429 / rate limit. Retryable; carries optional retry_after."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


@dataclass(frozen=True)
class LlmRequest:
    """One provider call after role-specific prompt templates have been rendered."""

    provider: str
    model: str
    messages: list[dict[str, Any]]
    response_format: dict[str, Any] | None = None
    seed: int | None = None
    max_tokens: int | None = None
    api_key: str | None = None
    base_url: str | None = None
    extra_body: dict[str, Any] | None = None
    client: Any | None = None
    timeout_s: float | None = None
    verbosity: str | None = None
    reasoning_effort: str | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: str | None = None


@dataclass(frozen=True)
class LlmResponse:
    """Provider response normalized to text plus optional call id and token usage."""

    text: str
    call_id: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    model: str | None = None
    cached_input_tokens: int = 0
    cache_write_input_tokens: int = 0


class LlmProvider(Protocol):
    """Adapter contract for provider-specific synchronous completions."""

    def complete(self, request: LlmRequest) -> LlmResponse: ...
