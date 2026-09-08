"""async_service — LlmService: the async entrypoint for live LLM calls.

Calling spec:
    service = LlmService(LlmConfig())
    text = (await service.acomplete(request, module="planner.director")).text
    payload = await service.acomplete_json(request, module="scoring.judge")
    summary = service.cost.summary()
    await service.aclose()

One LlmService is built per run and shared across all concurrent calls so the
rate limiter throttles globally and the cost tracker aggregates the whole run.
Pipeline: rate-limit slot -> retry(timeout(adapter)) -> record cost -> return.

``request.client`` overrides the shared client (test injection / a provider's own
client); otherwise OpenAI-compatible clients are built lazily per provider, API
key, and base URL. Gemini uses the Interactions REST adapter directly.

Side effects: HTTP calls via provider adapters, asyncio.sleep via retries.
Dependencies: ib.llm.{service,config,rate_limit,retry,cost}
"""

from __future__ import annotations

import asyncio
from typing import Any

from ib.llm.config import LlmConfig
from ib.llm.cost import CostTracker, LlmUsage
from ib.llm.rate_limit import RateLimiter
from ib.llm.retry import RetryPolicy, retry_async
from ib.llm.service import (
    AsyncGeminiLlmProvider,
    AsyncOpenAiLlmProvider,
    LlmProviderError,
    LlmRequest,
    LlmResponse,
    LlmTransientError,
    parse_json_object,
    _with_provider_defaults,
)


class LlmService:
    """Async facade: provider-routed LLM calls with rate limiting + cost tracking."""

    def __init__(self, config: LlmConfig | None = None, cost: CostTracker | None = None) -> None:
        self._config = config or LlmConfig()
        self._rate_limiter = RateLimiter(
            max_concurrency=self._config.max_concurrency, rpm=self._config.rpm
        )
        self._retry_policy = RetryPolicy(
            max_attempts=self._config.retry_max_attempts,
            base_delay_s=self._config.retry_base_delay_s,
            max_delay_s=self._config.retry_max_delay_s,
        )
        self.cost = cost or CostTracker(
            budget_usd=self._config.budget_usd,
            warn_at_percent=self._config.warn_at_percent,
            default_cost_per_1m=self._config.default_cost_per_1m_tokens,
        )
        self._openai_adapter = AsyncOpenAiLlmProvider()
        self._gemini_adapter = AsyncGeminiLlmProvider()
        self._clients: dict[tuple[str, str | None, str | None], Any] = {}

    def _client_for(self, request: LlmRequest) -> Any:
        if request.client is not None:
            return request.client
        if request.provider == "gemini":
            return None
        if request.base_url is not None:
            return None
        key = (request.provider, request.api_key, request.base_url)
        client = self._clients.get(key)
        if client is None:
            client = self._build_client(request)
            self._clients[key] = client
        return client

    def _build_client(self, request: LlmRequest) -> Any:
        import os

        resolved = request.api_key or os.getenv("OPENAI_API_KEY")
        if request.base_url is not None:
            resolved = resolved or "EMPTY"
        if not resolved:
            raise LlmProviderError(
                f"OPENAI_API_KEY is required for LLM provider {request.provider!r}"
            )
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise LlmProviderError("openai package is required for LLM provider 'openai'") from exc
        kwargs: dict[str, Any] = {"api_key": resolved}
        if request.base_url is not None:
            kwargs["base_url"] = request.base_url
        return AsyncOpenAI(**kwargs)

    def _adapter_for(self, request: LlmRequest) -> AsyncOpenAiLlmProvider | AsyncGeminiLlmProvider:
        if request.provider == "gemini":
            return self._gemini_adapter
        return self._openai_adapter

    async def acomplete(self, request: LlmRequest, *, module: str = "") -> LlmResponse:
        """Rate-limited, retried single completion; records cost on success."""
        request = _with_provider_defaults(request)
        client = self._client_for(request)
        adapter = self._adapter_for(request)

        async def _call() -> LlmResponse:
            try:
                return await asyncio.wait_for(
                    adapter.complete(request, client),
                    timeout=self._config.request_timeout_s,
                )
            except TimeoutError as exc:
                raise LlmTransientError(
                    f"request to {request.model!r} timed out after {self._config.request_timeout_s}s"
                ) from exc

        async with self._rate_limiter.slot():
            response = await retry_async(
                self._retry_policy, _call, label=module or request.model
            )

        await self.cost.record(
            LlmUsage(
                model=response.model or request.model,
                module=module,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )
        )
        return response

    async def acomplete_json(self, request: LlmRequest, *, module: str = "") -> dict:
        """acomplete + strict JSON-object parse. Parse failure is terminal."""
        response = await self.acomplete(request, module=module)
        return parse_json_object(response.text)

    async def aclose(self) -> None:
        """Close any AsyncOpenAI clients built by this service. Safe to repeat."""
        for client in self._clients.values():
            close = getattr(client, "close", None)
            if close is not None:
                try:
                    await close()
                except Exception:  # noqa: BLE001 — cleanup must not mask the real error
                    pass
        self._clients.clear()
