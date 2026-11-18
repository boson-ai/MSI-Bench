"""claude_direct — send Claude-model requests straight to the Anthropic API when CLAUDE_API_KEY is set.

Calling spec:
    request = route_claude(request)              pure given the env: a Claude request -> provider "anthropic"
    model_id = claude_model_id(model)            pure: "anthropic/claude-haiku-5.5" -> "claude-haiku-5-5"; None if not Claude
    params = anthropic_params(request)           pure: the messages.stream(**params) arguments
    response = ClaudeLlmProvider().complete(request)                  sync, one streamed call
    response = await AsyncClaudeLlmProvider().complete(request, client)   async; client = async_client(request)

route_claude moves a request to provider ANTHROPIC_PROVIDER (model id without the route prefix, dots
in the version turned to dashes) when CLAUDE_API_KEY is set, the model is a Claude model on any route
(OpenRouter "anthropic/claude-x.y", bare "claude-..."), and the caller
pinned nothing the direct API cannot honour: no injected client, explicit base_url or api_key, and no
extra_body (provider-specific options). Without the anthropic package (an image built before it was
added) nothing is rerouted. Other requests pass through unchanged.
anthropic_params maps the OpenAI-style request: leading system messages -> `system`, later system
messages stay in `messages` (mid-conversation system messages); text content parts keep their
cache_control; any other part type is an LlmProviderError. reasoning_effort -> output_config.effort
("minimal"/"none" -> "low"); a schema response_format -> output_config.format json_schema
(json_object has no API equivalent: the prompt asks for JSON and parse_json_object reads it);
top-level cache_control (automatic prompt caching); max_tokens defaults to DEFAULT_MAX_TOKENS.
seed, verbosity and prompt_cache_* are OpenAI options and are dropped. Thinking is left to the model's
default. The reply text is the concatenated text blocks; stop_reason "refusal" is an LlmProviderError.
Usage follows ib.llm.pricing: input_tokens is the whole prompt (uncached + cache reads + cache writes).
Errors: 429 -> LlmRateLimitError (retry-after), 5xx/overloaded/connection/timeout -> LlmTransientError,
spend limit -> the budget error, other 4xx -> LlmProviderError. SDK retries are off; ib.llm retries.
Side effects: HTTPS calls to api.anthropic.com with CLAUDE_API_KEY.
"""

from __future__ import annotations

import importlib.util
import os
import re
from dataclasses import replace
from typing import Any

from ib.llm.budget import budget_error, refuse_if_tripped
from ib.llm.types import LlmProviderError, LlmRateLimitError, LlmRequest, LlmResponse, LlmTransientError

ANTHROPIC_PROVIDER = "anthropic"
CLAUDE_API_KEY_ENV = "CLAUDE_API_KEY"
DEFAULT_MAX_TOKENS = 32_000
EFFORTS = ("low", "medium", "high", "xhigh", "max")
_CLAUDE = re.compile(r"^(?:anthropic/)?(claude-[a-z]+-[0-9][0-9.\-]*)$")


def claude_model_id(model: str) -> str | None:
    match = _CLAUDE.match(model.strip())
    return re.sub(r"(\d)\.(\d)", r"\1-\2", match.group(1)) if match else None


def _sdk_installed() -> bool:
    return importlib.util.find_spec("anthropic") is not None


def route_claude(request: LlmRequest) -> LlmRequest:
    if request.provider == ANTHROPIC_PROVIDER or not os.getenv(CLAUDE_API_KEY_ENV) or not _sdk_installed():
        return request
    model = claude_model_id(request.model)
    if model is None or request.client is not None or request.base_url or request.api_key or request.extra_body:
        return request
    return replace(request, provider=ANTHROPIC_PROVIDER, model=model)


def _content(content: Any) -> str | list[dict]:
    if isinstance(content, str):
        return content
    parts = []
    for part in content or []:
        if not isinstance(part, dict) or part.get("type") != "text":
            raise LlmProviderError(f"Claude direct route supports text content only, got {part!r:.80}")
        parts.append({k: part[k] for k in ("type", "text", "cache_control") if k in part})
    return parts


def _text_blocks(content: Any) -> list[dict]:
    content = _content(content)
    return [{"type": "text", "text": content}] if isinstance(content, str) else content


def anthropic_params(request: LlmRequest) -> dict[str, Any]:
    messages = list(request.messages)
    system: list[dict] = []
    while messages and messages[0].get("role") == "system":
        system += _text_blocks(messages.pop(0).get("content"))
    params: dict[str, Any] = {
        "model": request.model,
        "max_tokens": request.max_tokens or DEFAULT_MAX_TOKENS,
        "messages": [{"role": m["role"], "content": _content(m.get("content"))} for m in messages],
        "cache_control": {"type": "ephemeral"},
    }
    if system:
        params["system"] = system
    output_config: dict[str, Any] = {}
    if request.reasoning_effort:
        effort = request.reasoning_effort if request.reasoning_effort in EFFORTS else "low"
        output_config["effort"] = effort
    fmt = request.response_format or {}
    schema = fmt.get("schema") or (fmt.get("json_schema") or {}).get("schema")
    if isinstance(schema, dict):
        output_config["format"] = {"type": "json_schema", "schema": schema}
    if output_config:
        params["output_config"] = output_config
    return params


def _response(message: Any) -> LlmResponse:
    if message.stop_reason == "refusal":
        details = getattr(message, "stop_details", None)
        raise LlmProviderError(f"Claude refused the request ({getattr(details, 'category', None)})")
    usage = message.usage
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    return LlmResponse(
        text="".join(block.text for block in message.content if block.type == "text"),
        call_id=message.id,
        input_tokens=(usage.input_tokens or 0) + cache_read + cache_write,
        output_tokens=usage.output_tokens or 0,
        model=message.model,
        cached_input_tokens=cache_read,
        cache_write_input_tokens=cache_write,
    )


def _classified(exc: Exception) -> Exception:
    import anthropic

    if isinstance(exc, anthropic.EnforcedSpendLimitReachedError):
        return budget_error(str(exc))
    if isinstance(exc, anthropic.RateLimitError):
        retry_after = exc.response.headers.get("retry-after")
        return LlmRateLimitError(str(exc), retry_after=float(retry_after) if retry_after else None)
    if isinstance(exc, (anthropic.APIConnectionError, anthropic.APITimeoutError)):
        return LlmTransientError(str(exc))
    if isinstance(exc, anthropic.APIStatusError):
        return LlmTransientError(str(exc)) if exc.status_code >= 500 or exc.status_code in (408, 409) else LlmProviderError(str(exc))
    return exc


def _api_key() -> str:
    key = os.getenv(CLAUDE_API_KEY_ENV)
    if not key:
        raise LlmProviderError(f"{CLAUDE_API_KEY_ENV} is required for LLM provider {ANTHROPIC_PROVIDER!r}")
    return key


class ClaudeLlmProvider:
    """Sync adapter: one streamed Messages call per request (no SDK retries)."""

    def complete(self, request: LlmRequest) -> LlmResponse:
        import anthropic

        refuse_if_tripped()
        client = anthropic.Anthropic(api_key=_api_key(), max_retries=0, timeout=request.timeout_s or 600.0)
        try:
            with client.messages.stream(**anthropic_params(request)) as stream:
                message = stream.get_final_message()
        except anthropic.AnthropicError as exc:
            raise _classified(exc) from exc
        return _response(message)


def async_client(request: LlmRequest) -> Any:
    import anthropic

    return anthropic.AsyncAnthropic(api_key=_api_key(), max_retries=0)


class AsyncClaudeLlmProvider:
    """Async adapter: the client (anthropic.AsyncAnthropic from async_client) is owned by LlmService."""

    async def complete(self, request: LlmRequest, client: Any) -> LlmResponse:
        import anthropic

        refuse_if_tripped()
        try:
            async with client.messages.stream(**anthropic_params(request)) as stream:
                message = await stream.get_final_message()
        except anthropic.AnthropicError as exc:
            raise _classified(exc) from exc
        return _response(message)
