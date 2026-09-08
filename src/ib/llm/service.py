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
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from typing import Any, Protocol

from ib.llm.cache_keys import OPENAI_GPT55_DEFAULT_CACHE_KEY
from ib.llm.openrouter import OPENROUTER_API_KEY_ENV, OPENROUTER_BASE_URL, OPENROUTER_PROVIDER

DEFAULT_SYNC_LLM_TIMEOUT_S = 180.0


class LlmProviderError(ValueError):
    """Raised when a unified LLM provider is unavailable or returns invalid output.

    Treated as terminal by the async retry policy (no retry) unless it is one of
    the retryable subclasses below.
    """


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
    messages: list[dict[str, str]]
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


class LlmProvider(Protocol):
    """Adapter contract for provider-specific synchronous completions."""

    def complete(self, request: LlmRequest) -> LlmResponse: ...


class OpenAiLlmProvider:
    """OpenAI chat-completions adapter used by all text LLM call sites."""

    def complete(self, request: LlmRequest) -> LlmResponse:
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
            input_tokens=_get_usage_value(getattr(response, "usage", None), "prompt_tokens", "promptTokens"),
            output_tokens=_get_usage_value(
                getattr(response, "usage", None),
                "completion_tokens",
                "completionTokens",
            ),
            model=getattr(response, "model", None),
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
        timeout = _timeout_for_request(request)
        if timeout is not None:
            kwargs["timeout"] = timeout
        return OpenAI(**kwargs)


class GeminiLlmProvider:
    """Gemini Interactions API adapter for text/JSON completions."""

    def complete(self, request: LlmRequest) -> LlmResponse:
        if request.client is not None:
            interaction = request.client.interactions.create(**_gemini_interaction_kwargs(request))
            return _gemini_response_from_interaction(interaction, fallback_model=request.model)
        payload = _complete_gemini_rest(request)
        return _gemini_response_from_payload(payload, fallback_model=request.model)


def _request_kwargs(request: LlmRequest) -> dict[str, Any]:
    """Build chat-completions kwargs from a request (shared by sync + async)."""
    kwargs: dict[str, Any] = {"model": request.model, "messages": request.messages}
    extra_body = dict(request.extra_body or {})
    if request.response_format is not None:
        kwargs["response_format"] = _openai_response_format(request.response_format)
    if request.seed is not None:
        kwargs["seed"] = request.seed
    if request.max_tokens is not None:
        kwargs["max_tokens"] = request.max_tokens
    if _uses_openai_native_options(request) and request.reasoning_effort is not None:
        kwargs["reasoning_effort"] = request.reasoning_effort
    elif request.provider == OPENROUTER_PROVIDER and request.reasoning_effort is not None:
        extra_body["reasoning"] = {"effort": request.reasoning_effort}
    if _uses_openai_gpt55_latency_defaults(request):
        kwargs["verbosity"] = request.verbosity or "low"
        kwargs["prompt_cache_key"] = request.prompt_cache_key or OPENAI_GPT55_DEFAULT_CACHE_KEY
        extra_body.setdefault("prompt_cache_retention", request.prompt_cache_retention or "24h")
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


def _gemini_interaction_kwargs(request: LlmRequest) -> dict[str, Any]:
    """Build Google GenAI SDK Interactions kwargs from a unified request."""

    kwargs: dict[str, Any] = {
        "model": request.model,
        "input": _gemini_input_text(request.messages),
    }
    system_instruction = _gemini_system_instruction(request.messages)
    if system_instruction:
        kwargs["system_instruction"] = system_instruction
    response_format = _gemini_response_format(request.response_format)
    if response_format is not None:
        kwargs["response_format"] = response_format
    generation_config: dict[str, Any] = {}
    if request.max_tokens is not None:
        generation_config["max_output_tokens"] = request.max_tokens
    extra_body = request.extra_body or {}
    extra_generation_config = extra_body.get("generation_config")
    if isinstance(extra_generation_config, dict):
        generation_config.update(extra_generation_config)
    if _gemini_json_mode(response_format) and _is_gemini_3_model(request.model):
        generation_config.setdefault("thinking_level", "low")
    if generation_config:
        kwargs["generation_config"] = generation_config
    for key, value in extra_body.items():
        if key == "generation_config":
            continue
        kwargs[key] = value
    return kwargs


def _gemini_input_text(messages: list[dict[str, str]]) -> str:
    """Flatten chat messages into the stateless Interactions text input."""

    chunks = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "system":
            continue
        if len(messages) == 1 and role == "user":
            chunks.append(content)
        else:
            chunks.append(f"{role.upper()}:\n{content}")
    return "\n\n".join(chunks)


def _gemini_system_instruction(messages: list[dict[str, str]]) -> str | None:
    instructions = [message.get("content", "") for message in messages if message.get("role") == "system"]
    return "\n\n".join(item for item in instructions if item) or None


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


def _gemini_response_format(response_format: dict[str, Any] | None) -> dict[str, Any] | None:
    """Map OpenAI JSON mode into Gemini Interactions structured-output format."""

    if response_format is None:
        return None
    if response_format.get("mime_type") == "application/json":
        schema = response_format.get("schema")
        if isinstance(schema, dict):
            return {"type": "text", "mime_type": "application/json", "schema": schema}
        return dict(response_format)
    if response_format.get("type") == "json_object":
        return {
            "type": "text",
            "mime_type": "application/json",
            "schema": {"type": "object", "additionalProperties": True},
        }
    return dict(response_format)


def _gemini_json_mode(response_format: dict[str, Any] | None) -> bool:
    return (
        isinstance(response_format, dict)
        and response_format.get("type") == "text"
        and response_format.get("mime_type") == "application/json"
    )


def _is_gemini_3_model(model: str) -> bool:
    return model.startswith("gemini-3")


def _gemini_response_from_interaction(interaction: Any, *, fallback_model: str) -> LlmResponse:
    usage = getattr(interaction, "usage_metadata", None) or getattr(interaction, "usageMetadata", None)
    return LlmResponse(
        text=getattr(interaction, "output_text", "") or "",
        call_id=getattr(interaction, "id", None),
        input_tokens=_get_usage_value(usage, "prompt_token_count", "promptTokenCount"),
        output_tokens=_get_usage_value(usage, "candidates_token_count", "candidatesTokenCount"),
        model=getattr(interaction, "model_version", None)
        or getattr(interaction, "modelVersion", None)
        or fallback_model,
    )


def _get_usage_value(usage: Any, snake: str, camel: str) -> int:
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return int(usage.get(snake) or usage.get(camel) or 0)
    return int(getattr(usage, snake, None) or getattr(usage, camel, None) or 0)


def _raw_request_payload(request: LlmRequest) -> dict[str, Any]:
    """Build JSON payload for direct OpenAI-compatible HTTP endpoints."""
    payload = _request_kwargs(request)
    extra_body = payload.pop("extra_body", None)
    if extra_body is not None:
        payload.update(extra_body)
    return payload


def _gemini_rest_payload(request: LlmRequest) -> dict[str, Any]:
    """Build the REST payload for Gemini Interactions API."""

    payload = _gemini_interaction_kwargs(request)
    payload["store"] = False
    return payload


def _gemini_api_key(request: LlmRequest) -> str:
    api_key = request.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise LlmProviderError("GEMINI_API_KEY is required for LLM provider 'gemini'")
    return api_key


def _timeout_for_request(request: LlmRequest) -> float | None:
    if request.timeout_s is not None:
        return request.timeout_s if request.timeout_s > 0 else None
    raw = os.getenv("IB_LLM_TIMEOUT_S")
    if raw:
        try:
            value = float(raw)
        except ValueError as exc:
            raise LlmProviderError("IB_LLM_TIMEOUT_S must be a number") from exc
        return value if value > 0 else None
    return DEFAULT_SYNC_LLM_TIMEOUT_S


def _headers_for(request: LlmRequest) -> dict[str, str]:
    """Return optional auth headers for OpenAI-compatible direct HTTP."""
    api_key = request.api_key or None
    if api_key is None and request.provider == OPENROUTER_PROVIDER:
        api_key = os.getenv(OPENROUTER_API_KEY_ENV) or None
    if api_key is None:
        if request.provider == OPENROUTER_PROVIDER:
            raise LlmProviderError(
                f"{OPENROUTER_API_KEY_ENV} is required for LLM provider "
                f"{OPENROUTER_PROVIDER!r}"
            )
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def _with_provider_defaults(request: LlmRequest) -> LlmRequest:
    """Fill provider-owned endpoint defaults that callers can otherwise omit."""
    if (
        request.provider == OPENROUTER_PROVIDER
        and request.base_url is None
        and request.client is None
    ):
        return replace(request, base_url=OPENROUTER_BASE_URL)
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
    )


def _complete_openai_compatible_http(request: LlmRequest) -> LlmResponse:
    """Call a non-OpenAI OpenAI-compatible endpoint directly with httpx."""
    assert request.base_url is not None
    try:
        import httpx
    except ImportError as exc:
        raise LlmProviderError("httpx package is required for OpenAI-compatible LLMs") from exc
    try:
        response = httpx.post(
            f"{request.base_url.rstrip('/')}/chat/completions",
            json=_raw_request_payload(request),
            headers=_headers_for(request),
            timeout=_timeout_for_request(request),
        )
    except httpx.TimeoutException as exc:
        raise LlmTransientError(str(exc)) from exc
    except httpx.HTTPError as exc:
        raise LlmTransientError(str(exc)) from exc
    if response.status_code >= 500:
        raise LlmTransientError(response.text)
    if response.status_code >= 400:
        raise LlmProviderError(response.text)
    try:
        payload = response.json()
    except ValueError as exc:
        raise LlmProviderError("LLM provider returned invalid JSON response") from exc
    return _response_from_payload(payload)


def _complete_gemini_rest(request: LlmRequest) -> dict:
    """Call Gemini Interactions API directly using the stdlib HTTP client."""

    body = json.dumps(_gemini_rest_payload(request)).encode("utf-8")
    http_request = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/interactions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": _gemini_api_key(request),
            "Api-Revision": "2026-05-20",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(http_request, timeout=_timeout_for_request(request)) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        if exc.code == 429:
            raise LlmRateLimitError(message, retry_after=_retry_after_http_headers(exc.headers)) from exc
        if exc.code >= 500:
            raise LlmTransientError(message) from exc
        raise LlmProviderError(message) from exc
    except urllib.error.URLError as exc:
        raise LlmTransientError(str(exc)) from exc
    except (TimeoutError, socket.timeout) as exc:
        raise LlmTransientError(str(exc)) from exc
    except json.JSONDecodeError as exc:
        raise LlmProviderError("Gemini LLM provider returned invalid JSON response") from exc
    if not isinstance(payload, dict):
        raise LlmProviderError("Gemini LLM provider returned malformed response")
    return payload


def _gemini_response_from_payload(payload: dict, *, fallback_model: str) -> LlmResponse:
    usage = payload.get("usageMetadata") if isinstance(payload.get("usageMetadata"), dict) else {}
    if not usage and isinstance(payload.get("usage"), dict):
        usage = payload["usage"]
    return LlmResponse(
        text=_gemini_output_text(payload),
        call_id=payload.get("id") or payload.get("responseId"),
        input_tokens=usage.get("promptTokenCount") or usage.get("total_input_tokens") or 0,
        output_tokens=(
            usage.get("candidatesTokenCount")
            or usage.get("outputTokenCount")
            or usage.get("total_output_tokens")
            or 0
        ),
        model=payload.get("modelVersion") or fallback_model,
    )


def _gemini_output_text(payload: dict) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    if isinstance(payload.get("outputText"), str):
        return payload["outputText"]
    text = _gemini_text_from_steps(payload.get("steps"))
    if text:
        return text
    return _gemini_text_from_candidates(payload.get("candidates"))


def _gemini_text_from_steps(steps: object) -> str:
    if not isinstance(steps, list):
        return ""
    for step in reversed(steps):
        if not isinstance(step, dict):
            continue
        content = step.get("content")
        if isinstance(content, str):
            return content
        parts = content if isinstance(content, list) else []
        texts = [part.get("text") for part in parts if isinstance(part, dict)]
        joined = "".join(text for text in texts if isinstance(text, str))
        if joined:
            return joined
    return ""


def _gemini_text_from_candidates(candidates: object) -> str:
    if not isinstance(candidates, list) or not candidates:
        return ""
    content = candidates[0].get("content") if isinstance(candidates[0], dict) else None
    if isinstance(content, dict):
        parts = content.get("parts") if isinstance(content.get("parts"), list) else []
        texts = [part.get("text") for part in parts if isinstance(part, dict)]
        return "".join(text for text in texts if isinstance(text, str))
    return ""


def _classify_openai_error(exc: Exception) -> LlmProviderError:
    """Map an OpenAI SDK exception to the LLM error hierarchy (for retry routing).

    429 -> LlmRateLimitError (retryable, with retry_after); connection/timeout/5xx
    -> LlmTransientError (retryable); auth/bad-request/permission -> LlmProviderError
    (terminal). Falls back to LlmTransientError when the SDK is unavailable so a
    bare environment retries rather than crashing on an unknown error shape.
    """
    try:
        import openai
    except ImportError:
        return LlmTransientError(str(exc))

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


def _retry_after_http_headers(headers: Any) -> float | None:
    raw = headers.get("retry-after") if headers is not None else None
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


class AsyncGeminiLlmProvider:
    """Async Gemini Interactions API adapter used by LlmService."""

    async def complete(self, request: LlmRequest, client: Any) -> LlmResponse:
        if client is not None:
            result = client.interactions.create(**_gemini_interaction_kwargs(request))
            if hasattr(result, "__await__"):
                result = await result
            return _gemini_response_from_interaction(result, fallback_model=request.model)
        payload = await _acomplete_gemini_rest(request)
        return _gemini_response_from_payload(payload, fallback_model=request.model)


async def _acomplete_openai_compatible_http(request: LlmRequest) -> LlmResponse:
    """Async direct HTTP path for non-OpenAI OpenAI-compatible endpoints."""
    assert request.base_url is not None
    try:
        import httpx
    except ImportError as exc:
        raise LlmProviderError("httpx package is required for OpenAI-compatible LLMs") from exc
    try:
        async with httpx.AsyncClient(timeout=_timeout_for_request(request)) as client:
            response = await client.post(
                f"{request.base_url.rstrip('/')}/chat/completions",
                json=_raw_request_payload(request),
                headers=_headers_for(request),
            )
    except httpx.TimeoutException as exc:
        raise LlmTransientError(str(exc)) from exc
    except httpx.HTTPError as exc:
        raise LlmTransientError(str(exc)) from exc
    if response.status_code >= 500:
        raise LlmTransientError(response.text)
    if response.status_code >= 400:
        raise LlmProviderError(response.text)
    try:
        payload = response.json()
    except ValueError as exc:
        raise LlmProviderError("LLM provider returned invalid JSON response") from exc
    return _response_from_payload(payload)


async def _acomplete_gemini_rest(request: LlmRequest) -> dict:
    """Async direct HTTP path for Gemini Interactions API."""

    try:
        import httpx
    except ImportError as exc:
        raise LlmProviderError("httpx package is required for Gemini LLMs") from exc
    try:
        async with httpx.AsyncClient(timeout=_timeout_for_request(request)) as client:
            response = await client.post(
                "https://generativelanguage.googleapis.com/v1beta/interactions",
                json=_gemini_rest_payload(request),
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": _gemini_api_key(request),
                    "Api-Revision": "2026-05-20",
                },
            )
    except httpx.TimeoutException as exc:
        raise LlmTransientError(str(exc)) from exc
    except httpx.HTTPError as exc:
        raise LlmTransientError(str(exc)) from exc
    if response.status_code == 429:
        raise LlmRateLimitError(
            response.text, retry_after=_retry_after_http_headers(response.headers)
        )
    if response.status_code >= 500:
        raise LlmTransientError(response.text)
    if response.status_code >= 400:
        raise LlmProviderError(response.text)
    try:
        payload = response.json()
    except ValueError as exc:
        raise LlmProviderError("Gemini LLM provider returned invalid JSON response") from exc
    if not isinstance(payload, dict):
        raise LlmProviderError("Gemini LLM provider returned malformed response")
    return payload


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
    "anthropic": UnsupportedLlmProvider("anthropic"),
    "vllm": UnsupportedLlmProvider("vllm"),
    "sglang": UnsupportedLlmProvider("sglang"),
}


def complete(request: LlmRequest) -> LlmResponse:
    """Dispatch one LLM request through the unified provider registry."""
    request = _with_provider_defaults(request)
    try:
        provider = PROVIDER_REGISTRY[request.provider]
    except KeyError as exc:
        raise LlmProviderError(f"unknown LLM provider {request.provider!r}") from exc
    return provider.complete(request)


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
