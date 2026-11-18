"""gemini — Gemini adapters (Interactions SDK and REST), sync and async.

Calling spec:
    GeminiLlmProvider().complete(request) -> LlmResponse
    await AsyncGeminiLlmProvider().acomplete(request, ...) -> LlmResponse
Side effects: network calls to the Gemini API; reads the Gemini API key.
"""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from typing import Any

from ib.llm.transport import retry_after_http_headers, timeout_for_request, usage_value
from ib.llm.types import (
    LlmProviderError,
    LlmRateLimitError,
    LlmRequest,
    LlmResponse,
    LlmTransientError,
)


class GeminiLlmProvider:
    """Gemini Interactions API adapter for text/JSON completions."""

    def complete(self, request: LlmRequest) -> LlmResponse:
        if request.client is not None:
            interaction = request.client.interactions.create(**_gemini_interaction_kwargs(request))
            return _gemini_response_from_interaction(interaction, fallback_model=request.model)
        payload = _complete_gemini_rest(request)
        return _gemini_response_from_payload(payload, fallback_model=request.model)


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
        input_tokens=usage_value(usage, "prompt_token_count", "promptTokenCount"),
        output_tokens=usage_value(usage, "candidates_token_count", "candidatesTokenCount"),
        model=getattr(interaction, "model_version", None)
        or getattr(interaction, "modelVersion", None)
        or fallback_model,
    )


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
        with urllib.request.urlopen(http_request, timeout=timeout_for_request(request)) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        if exc.code == 429:
            raise LlmRateLimitError(message, retry_after=retry_after_http_headers(exc.headers)) from exc
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


async def _acomplete_gemini_rest(request: LlmRequest) -> dict:
    """Async direct HTTP path for Gemini Interactions API."""

    try:
        import httpx
    except ImportError as exc:
        raise LlmProviderError("httpx package is required for Gemini LLMs") from exc
    try:
        async with httpx.AsyncClient(timeout=timeout_for_request(request)) as client:
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
            response.text, retry_after=retry_after_http_headers(response.headers)
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
