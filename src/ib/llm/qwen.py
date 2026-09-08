"""qwen — constants for the Qwen 3.5 OpenAI-compatible endpoint.

Calling spec:
    qwen_request_options(enable_thinking=False) -> dict

Deterministic: returns provider metadata and per-request OpenAI-compatible
``extra_body`` options. No side effects.
"""

from __future__ import annotations

QWEN35_PROVIDER = "qwen3_5"
QWEN35_THINKING_PROVIDER = "qwen3_5_thinking"
DEFAULT_LLM_PROVIDER = "openai"
QWEN35_MODEL = "qwen3.5-397b"
QWEN35_BASE_URL = "http://h100-46.canada.boson.ai:26000/v1"
QWEN35_MAX_CONTEXT_TOKENS = 120_000
QWEN35_MAX_OUTPUT_TOKENS = 8_192


def qwen_request_options(*, enable_thinking: bool) -> dict:
    """Return chat-template options for Qwen's per-request thinking mode."""
    return {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
