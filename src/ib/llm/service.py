"""service — provider dispatch and response parsing for LLM calls.

Calling spec:
    request = LlmRequest(provider="openai", model="gpt-4.1-mini", messages=[...])
    response = complete(request) -> LlmResponse
    payload = complete_json(request) -> dict

Deterministic logic: provider dispatch is a dict; JSON parsing is strict.
Side effects: live adapters may call external APIs and read provider API keys.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from typing import Any

from ib.llm.budget import budget_error, is_budget_refusal, refuse_if_tripped
from ib.llm.cache_keys import OPENAI_GPT55_DEFAULT_CACHE_KEY
from ib.llm.claude_direct import ANTHROPIC_PROVIDER, ClaudeLlmProvider, route_claude
from ib.llm.config import LlmConfig
from ib.llm.gemini import GeminiLlmProvider
from ib.llm.rate_limit_wait import rate_limit_wait
from ib.llm.openrouter import (
    OPENROUTER_API_KEY_ENV,
    OPENROUTER_BASE_URL,
    OPENROUTER_PROVIDER,
    openrouter_model,
    reaches_anthropic,
    with_cache_breakpoints,
)
from ib.llm.transport import (
    completion_from_stream,
    retry_after_http_headers,
    timeout_for_request,
    usage_value,
)
from ib.llm.types import (
    LlmProvider,
    LlmProviderError,
    LlmRateLimitError,
    LlmRequest,
    LlmResponse,
    LlmTransientError,
)

# Sync-path backoff: short, like the async RetryPolicy. The attempt count is the
# configured LlmConfig.retry_max_attempts, shared with the async service.
SYNC_RETRY_BASE_DELAY_S = 3.0
SYNC_RETRY_MAX_DELAY_S = 30.0

_sync_retry_max_attempts: int = LlmConfig().retry_max_attempts

# Hosted OpenAI-compatible endpoints: the base URL a request without one is sent to,
# and the env var holding the bearer key when the request carries none.
HOSTED_BASE_URLS: dict[str, str] = {
    OPENROUTER_PROVIDER: OPENROUTER_BASE_URL,
}
HOSTED_API_KEY_ENVS: dict[str, str] = {
    OPENROUTER_PROVIDER: OPENROUTER_API_KEY_ENV,
}
# Providers whose replies are streamed so the HTTP read timeout bounds only the gap between chunks.
STREAMED_PROVIDERS: frozenset[str] = frozenset()


def set_sync_retry_max_attempts(max_attempts: int) -> None:
    """Apply LlmConfig.retry_max_attempts (e.g. from --llm-retry-max-attempts) to complete()."""
    global _sync_retry_max_attempts
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
    _sync_retry_max_attempts = max_attempts


class OpenAiLlmProvider:
    """OpenAI chat-completions adapter used by all text LLM call sites."""

    def complete(self, request: LlmRequest) -> LlmResponse:
        refuse_if_tripped()
        if request.base_url is not None and request.client is None:
            return _complete_openai_compatible_http(request)
        client = request.client or self._client(request)
        kwargs = _request_kwargs(request)
        try:
            response = client.chat.completions.create(**kwargs)
        except LlmProviderError:
            raise
        except Exception as exc:  # noqa: BLE001 — re-raised classified below
            raise _classify_openai_error(exc) from exc
        return LlmResponse(
            text=response.choices[0].message.content or "",
            call_id=getattr(response, "id", None),
            input_tokens=usage_value(getattr(response, "usage", None), "prompt_tokens", "promptTokens"),
            output_tokens=usage_value(
                getattr(response, "usage", None),
                "completion_tokens",
                "completionTokens",
            ),
            model=getattr(response, "model", None),
            cached_input_tokens=_prompt_tokens_detail(getattr(response, "usage", None), "cached_tokens"),
            cache_write_input_tokens=_prompt_tokens_detail(getattr(response, "usage", None), "cache_write_tokens"),
        )

    def _client(self, request: LlmRequest) -> Any:
        resolved_key = request.api_key or os.getenv("OPENAI_API_KEY")
        if request.base_url is not None:
            resolved_key = resolved_key or "EMPTY"
        if not resolved_key:
            raise LlmProviderError(
                f"OPENAI_API_KEY is required for LLM provider {request.provider!r}"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LlmProviderError("openai package is required for LLM provider 'openai'") from exc
        kwargs: dict[str, Any] = {"api_key": resolved_key, "max_retries": 0}
        if request.base_url is not None:
            kwargs["base_url"] = request.base_url
        timeout = timeout_for_request(request)
        if timeout is not None:
            kwargs["timeout"] = timeout
        return OpenAI(**kwargs)


def _request_kwargs(request: LlmRequest) -> dict[str, Any]:
    """Build chat-completions kwargs from a request (shared by sync + async)."""
    messages = request.messages
    anthropic = reaches_anthropic(request.provider, request.model)
    if anthropic:
        messages = with_cache_breakpoints(messages)
    kwargs: dict[str, Any] = {"model": request.model, "messages": messages}
    extra_body = dict(request.extra_body or {})
    if request.response_format is not None:
        kwargs["response_format"] = _openai_response_format(request.response_format)
    if request.seed is not None and not anthropic:
        kwargs["seed"] = request.seed
    if request.max_tokens is not None:
        kwargs["max_tokens"] = request.max_tokens
    if request.reasoning_effort is not None:
        if openrouter_model(request.provider, request.model) is not None:
            extra_body["reasoning"] = {"effort": request.reasoning_effort}
        elif _uses_openai_native_options(request):
            kwargs["reasoning_effort"] = request.reasoning_effort
    if _uses_openai_gpt55_latency_defaults(request):
        kwargs["verbosity"] = request.verbosity or "low"
        kwargs["prompt_cache_key"] = request.prompt_cache_key or OPENAI_GPT55_DEFAULT_CACHE_KEY
        extra_body.setdefault("prompt_cache_retention", request.prompt_cache_retention or "24h")
    elif _uses_openai_native_options(request) and request.prompt_cache_key:
        kwargs["prompt_cache_key"] = request.prompt_cache_key
    if extra_body:
        kwargs["extra_body"] = extra_body
    return kwargs


def _uses_openai_gpt55_latency_defaults(request: LlmRequest) -> bool:
    """Return whether OpenAI GPT-5.5 latency/caching knobs should be emitted."""

    return _uses_openai_native_options(request) and request.model.startswith("gpt-5.5")


def _uses_openai_native_options(request: LlmRequest) -> bool:
    """Return whether request fields map to OpenAI-only Chat Completions options."""

    return (
        request.provider == "openai"
        and request.base_url is None
    )


def _openai_response_format(response_format: dict[str, Any] | None) -> dict[str, Any] | None:
    """Map a schema-bearing response_format into OpenAI json_schema structured output.

    A plain ``{"type": "json_object"}`` (or any non-schema format) passes through
    unchanged; only a schema-bearing format (``mime_type`` application/json with a
    concrete ``schema``) is promoted to the OpenAI ``json_schema`` shape so the
    chat-completions endpoint enforces it.
    """

    if not isinstance(response_format, dict):
        return response_format
    schema = response_format.get("schema")
    if response_format.get("mime_type") == "application/json" and isinstance(schema, dict):
        return {
            "type": "json_schema",
            "json_schema": {
                "name": response_format.get("name") or "response",
                "schema": schema,
                "strict": bool(response_format.get("strict")),
            },
        }
    return response_format


def _prompt_tokens_detail(usage: Any, field: str) -> int:
    """Return one usage.prompt_tokens_details count, e.g. cached_tokens (0 when unreported)."""
    if usage is None:
        return 0
    details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else getattr(usage, "prompt_tokens_details", None)
    if details is None:
        return 0
    value = details.get(field) if isinstance(details, dict) else getattr(details, field, None)
    return int(value or 0)


def _raw_request_payload(request: LlmRequest) -> dict[str, Any]:
    """Build JSON payload for direct OpenAI-compatible HTTP endpoints."""
    payload = _request_kwargs(request)
    extra_body = payload.pop("extra_body", None)
    if extra_body is not None:
        payload.update(extra_body)
    return payload


def _headers_for(request: LlmRequest) -> dict[str, str]:
    """Return optional auth headers for OpenAI-compatible direct HTTP."""
    key_env = HOSTED_API_KEY_ENVS.get(request.provider)
    api_key = request.api_key or (os.getenv(key_env) if key_env else None) or None
    if api_key is None:
        if key_env is not None:
            raise LlmProviderError(
                f"{key_env} is required for LLM provider {request.provider!r}"
            )
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def _with_provider_defaults(request: LlmRequest) -> LlmRequest:
    """Route Claude models to the Anthropic API when CLAUDE_API_KEY is set (claude_direct.route_claude),
    then fill provider-owned endpoint defaults that callers can otherwise omit."""
    request = route_claude(request)
    hosted_url = HOSTED_BASE_URLS.get(request.provider)
    if hosted_url is not None and request.base_url is None and request.client is None:
        return replace(request, base_url=hosted_url)
    return request


def _response_from_payload(payload: dict) -> LlmResponse:
    """Coerce an OpenAI-compatible chat completion JSON payload."""
    try:
        choice = payload["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LlmProviderError("LLM provider returned malformed chat completion") from exc
    usage = payload.get("usage") if isinstance(payload, dict) else None
    usage = usage if isinstance(usage, dict) else {}
    return LlmResponse(
        text=message.get("content") or "",
        call_id=payload.get("id") if isinstance(payload.get("id"), str) else None,
        input_tokens=usage.get("prompt_tokens") or 0,
        output_tokens=usage.get("completion_tokens") or 0,
        model=payload.get("model") if isinstance(payload.get("model"), str) else None,
        cached_input_tokens=_prompt_tokens_detail(usage, "cached_tokens"),
        cache_write_input_tokens=_prompt_tokens_detail(usage, "cache_write_tokens"),
    )


def _complete_openai_compatible_http(request: LlmRequest) -> LlmResponse:
    """Call a non-OpenAI OpenAI-compatible endpoint directly with httpx."""
    assert request.base_url is not None
    try:
        import httpx
    except ImportError as exc:
        raise LlmProviderError("httpx package is required for OpenAI-compatible LLMs") from exc
    url = f"{request.base_url.rstrip('/')}/chat/completions"
    body = _raw_request_payload(request)
    headers = _headers_for(request)
    timeout = timeout_for_request(request)
    try:
        if request.provider in STREAMED_PROVIDERS:
            body.update(stream=True, stream_options={"include_usage": True})
            with httpx.stream("POST", url, json=body, headers=headers, timeout=timeout) as response:
                if response.status_code >= 400:
                    response.read()
                    _raise_for_status(response)
                return _response_from_payload(completion_from_stream(response.iter_lines()))
        response = httpx.post(url, json=body, headers=headers, timeout=timeout)
    except httpx.HTTPError as exc:
        raise LlmTransientError(str(exc)) from exc
    _raise_for_status(response)
    try:
        payload = response.json()
    except ValueError as exc:
        raise LlmProviderError("LLM provider returned invalid JSON response") from exc
    return _response_from_payload(payload)


def _raise_for_status(response: Any) -> None:
    if is_budget_refusal(response.status_code, response.text):
        raise budget_error(response.text)
    if response.status_code >= 500:
        raise LlmTransientError(response.text)
    if response.status_code == 429:
        raise LlmRateLimitError(
            response.text, retry_after=retry_after_http_headers(response.headers)
        )
    if response.status_code >= 400:
        raise LlmProviderError(response.text)


def _classify_openai_error(exc: Exception) -> LlmProviderError:
    """Map an OpenAI SDK exception to the LLM error hierarchy (for retry routing).

    A spent key -> LlmBudgetError (terminal, ib.llm.budget); 429 -> LlmRateLimitError (retryable, with retry_after); connection/timeout/5xx
    -> LlmTransientError (retryable); auth/bad-request/permission -> LlmProviderError
    (terminal). Falls back to LlmTransientError when the SDK is unavailable so a
    bare environment retries rather than crashing on an unknown error shape.
    """
    try:
        import openai
    except ImportError:
        return LlmTransientError(str(exc))

    if is_budget_refusal(getattr(exc, "status_code", None), str(exc)):
        return budget_error(str(exc))
    if isinstance(exc, openai.RateLimitError):
        return LlmRateLimitError(str(exc), retry_after=_retry_after(exc))
    if isinstance(exc, (openai.AuthenticationError, openai.BadRequestError, openai.PermissionDeniedError)):
        return LlmProviderError(str(exc))
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError)):
        return LlmTransientError(str(exc))
    if isinstance(exc, openai.APIError):
        return LlmTransientError(str(exc))
    raise exc


def _retry_after(exc: Exception) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        return None
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None


class AsyncOpenAiLlmProvider:
    """Async OpenAI chat-completions adapter used by LlmService for live calls.

    Stateless: the client (a shared ``openai.AsyncOpenAI`` or a test double whose
    ``chat.completions.create`` is awaitable) is owned by LlmService and passed in.
    """

    async def complete(self, request: LlmRequest, client: Any) -> LlmResponse:
        refuse_if_tripped()
        if request.base_url is not None and client is None:
            return await _acomplete_openai_compatible_http(request)
        try:
            response = await client.chat.completions.create(**_request_kwargs(request))
        except LlmProviderError:
            raise
        except Exception as exc:  # noqa: BLE001 — re-raised classified below
            raise _classify_openai_error(exc) from exc
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        return LlmResponse(
            text=choice.message.content or "",
            call_id=getattr(response, "id", None),
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0 if usage else 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0 if usage else 0,
            model=getattr(response, "model", None),
        )


async def _acomplete_openai_compatible_http(request: LlmRequest) -> LlmResponse:
    """Async direct HTTP path for non-OpenAI OpenAI-compatible endpoints."""
    assert request.base_url is not None
    try:
        import httpx
    except ImportError as exc:
        raise LlmProviderError("httpx package is required for OpenAI-compatible LLMs") from exc
    try:
        async with httpx.AsyncClient(timeout=timeout_for_request(request)) as client:
            response = await client.post(
                f"{request.base_url.rstrip('/')}/chat/completions",
                json=_raw_request_payload(request),
                headers=_headers_for(request),
            )
    except httpx.TimeoutException as exc:
        raise LlmTransientError(str(exc)) from exc
    except httpx.HTTPError as exc:
        raise LlmTransientError(str(exc)) from exc
    if is_budget_refusal(response.status_code, response.text):
        raise budget_error(response.text)
    if response.status_code >= 500:
        raise LlmTransientError(response.text)
    if response.status_code >= 400:
        raise LlmProviderError(response.text)
    try:
        payload = response.json()
    except ValueError as exc:
        raise LlmProviderError("LLM provider returned invalid JSON response") from exc
    return _response_from_payload(payload)


class UnsupportedLlmProvider:
    """Declared future provider that fails loudly instead of silently falling back."""

    def __init__(self, name: str) -> None:
        self.name = name

    def complete(self, request: LlmRequest) -> LlmResponse:
        raise LlmProviderError(
            f"LLM provider {self.name!r} is declared for future support but not implemented"
        )


PROVIDER_REGISTRY: dict[str, LlmProvider] = {
    "openai": OpenAiLlmProvider(),
    "gemini": GeminiLlmProvider(),
    OPENROUTER_PROVIDER: OpenAiLlmProvider(),
    "qwen3_5": OpenAiLlmProvider(),
    "qwen3_5_thinking": OpenAiLlmProvider(),
    ANTHROPIC_PROVIDER: ClaudeLlmProvider(),
    "vllm": UnsupportedLlmProvider("vllm"),
    "sglang": UnsupportedLlmProvider("sglang"),
}


def complete(request: LlmRequest) -> LlmResponse:
    """Dispatch one LLM request through the unified provider registry.

    A 429 (LlmRateLimitError) is first waited out for up to rate_limit_wait.RATE_LIMIT_BUDGET_S
    in total (Retry-After, the proxy's "Limit resets at", else jittered backoff) without using an
    attempt. Transient failures (LlmTransientError, incl. a 429 past that budget) are retried
    with exponential backoff up to the configured retry_max_attempts calls (default 3); terminal
    LlmProviderError propagates immediately.
    """
    request = _with_provider_defaults(request)
    try:
        provider = PROVIDER_REGISTRY[request.provider]
    except KeyError as exc:
        raise LlmProviderError(f"unknown LLM provider {request.provider!r}") from exc
    from ib.llm.retry import RetryPolicy  # lazy: retry imports this module's error types

    policy = RetryPolicy(
        max_attempts=_sync_retry_max_attempts,
        base_delay_s=SYNC_RETRY_BASE_DELAY_S,
        max_delay_s=SYNC_RETRY_MAX_DELAY_S,
    )
    attempt, rate_tries, rate_waited = 0, 0, 0.0
    while True:
        try:
            return provider.complete(request)
        except LlmTransientError as exc:
            wait = rate_limit_wait(exc, rate_tries, rate_waited)
            if wait is not None:  # a 429 within its wait budget: no transient attempt used
                _sleep(wait)
                rate_tries, rate_waited = rate_tries + 1, rate_waited + wait
                continue
            attempt += 1
            if attempt >= policy.max_attempts:
                raise
            retry_after = exc.retry_after if isinstance(exc, LlmRateLimitError) else None
            _sleep(policy.delay_for(attempt - 1, retry_after))


def _sleep(seconds: float) -> None:
    """Blocking backoff sleep (module-level so tests can stub it)."""
    time.sleep(seconds)


def complete_json(request: LlmRequest) -> dict:
    """Dispatch one JSON-mode LLM request and parse the response as an object.

    An unparseable response retries the single provider call once (with a
    nudged seed) instead of failing the caller's whole attempt.
    """
    response = complete(request)
    try:
        return parse_json_object(response.text)
    except LlmProviderError:
        retry = request if request.seed is None else replace(request, seed=request.seed + 1)
        response = complete(retry)
        return parse_json_object(response.text)


def parse_json_object(text: str) -> dict:
    """Parse a JSON object, falling back to the first balanced object in text."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        extracted = _extract_json_object(text)
        if extracted is None:
            raise LlmProviderError("LLM provider returned invalid JSON") from exc
        try:
            payload = json.loads(extracted)
        except json.JSONDecodeError as nested_exc:
            raise LlmProviderError("LLM provider returned invalid JSON") from nested_exc
    if not isinstance(payload, dict):
        raise LlmProviderError("LLM provider response must be a JSON object")
    return payload


def _extract_json_object(text: str) -> str | None:
    """Return the first balanced JSON object substring, respecting strings."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None
