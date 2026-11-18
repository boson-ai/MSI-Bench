"""openrouter — constants and pure wire helpers for the OpenRouter OpenAI-compatible endpoint.

Calling spec:
    from ib.llm.openrouter import OPENROUTER_PROVIDER, OPENROUTER_BASE_URL
    openrouter_model(provider, model) -> str | None         OpenRouter model id a call reaches, else None
    reaches_anthropic(provider, model) -> bool              served by Anthropic via OpenRouter
    with_cache_breakpoints(messages) -> list[dict]          wire messages with Anthropic cache marks

Everything here is deterministic and has no side effects. Authentication is
resolved at request time from OPENROUTER_API_KEY by the LLM service.
"""

from __future__ import annotations

from typing import Any

OPENROUTER_PROVIDER = "openrouter"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_DEFAULT_MODEL = "openai/gpt-5.5"

# Anthropic rejects a request carrying more than four cache_control marks.
ANTHROPIC_MAX_CACHE_BREAKPOINTS = 4


def openrouter_model(provider: str, model: str) -> str | None:
    """Return the OpenRouter model id a call reaches; None otherwise."""

    if provider == OPENROUTER_PROVIDER:
        return model
    return None


def reaches_anthropic(provider: str, model: str) -> bool:
    """Return whether a call is served by Anthropic, which caches only on explicit marks and rejects `seed`.

    Covers OpenRouter's anthropic/ models.
    """

    routed = openrouter_model(provider, model)
    if routed is not None:
        return routed.startswith("anthropic/")
    return False


def with_cache_breakpoints(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark the end of the system message and of the last message as cache breakpoints.

    Only string contents are marked; list contents are the caller's own parts and pass
    through unchanged. Automatic marks are dropped (last message first, then system) so
    the caller's marks plus ours stay within ANTHROPIC_MAX_CACHE_BREAKPOINTS.
    """

    if not messages:
        return messages
    candidates = [0] if messages[0].get("role") == "system" else []
    if len(messages) - 1 not in candidates:
        candidates.append(len(messages) - 1)
    candidates = [index for index in candidates if isinstance(messages[index].get("content"), str)]
    budget = max(0, ANTHROPIC_MAX_CACHE_BREAKPOINTS - _caller_marks(messages))
    marked = set(candidates[:budget])
    return [
        {**message, "content": [{"type": "text", "text": message["content"], "cache_control": {"type": "ephemeral"}}]}
        if index in marked else message
        for index, message in enumerate(messages)
    ]


def _caller_marks(messages: list[dict[str, Any]]) -> int:
    return sum(
        isinstance(part, dict) and "cache_control" in part
        for message in messages
        if isinstance(message.get("content"), list)
        for part in message["content"]
    )
