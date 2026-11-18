"""transport — request timeout, retry-after and usage helpers shared by provider adapters; pure.

Calling spec:
    timeout_for_request(request) -> float | None     request.timeout_s, else $IB_LLM_TIMEOUT_S, else 180 s
    retry_after_http_headers(headers) -> float | None
    usage_value(usage, "prompt_tokens", "promptTokens") -> int   dict or object usage records
    completion_from_stream(sse_lines) -> dict   a streamed chat completion as the non-streamed payload
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from typing import Any

from ib.llm.types import LlmProviderError, LlmRequest, LlmTransientError

DEFAULT_SYNC_LLM_TIMEOUT_S = 180.0


def usage_value(usage: Any, snake: str, camel: str) -> int:
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return int(usage.get(snake) or usage.get(camel) or 0)
    return int(getattr(usage, snake, None) or getattr(usage, camel, None) or 0)


def timeout_for_request(request: LlmRequest) -> float | None:
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


def retry_after_http_headers(headers: Any) -> float | None:
    raw = headers.get("retry-after") if headers is not None else None
    if raw is None:
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None


def completion_from_stream(sse_lines: Iterable[str]) -> dict[str, Any]:
    """Join the `data:` chunks of a streamed chat completion into a non-streamed payload.

    The content deltas of choice 0 are concatenated; id and model come from the first chunk that
    has them, usage from the chunk that carries it (stream_options.include_usage). A chunk with
    an `error` (an upstream failure after the stream opened) or a stream that ends before
    `[DONE]` raises LlmTransientError.
    """
    parts: list[str] = []
    payload: dict[str, Any] = {}
    for line in sse_lines:
        if not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except ValueError as exc:
            raise LlmProviderError("LLM provider streamed an invalid JSON chunk") from exc
        if chunk.get("error"):
            raise LlmTransientError(f"LLM stream failed: {chunk['error']}")
        for key in ("id", "model"):
            if isinstance(chunk.get(key), str):
                payload.setdefault(key, chunk[key])
        if chunk.get("usage"):
            payload["usage"] = chunk["usage"]
        for choice in chunk.get("choices") or []:
            if choice.get("index", 0) == 0:
                parts.append((choice.get("delta") or {}).get("content") or "")
    else:
        raise LlmTransientError("LLM stream ended before [DONE]")
    payload["choices"] = [{"message": {"content": "".join(parts)}}]
    return payload
