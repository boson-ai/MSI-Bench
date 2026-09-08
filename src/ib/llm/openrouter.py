"""openrouter — constants for the OpenRouter OpenAI-compatible endpoint.

Calling spec:
    from ib.llm.openrouter import OPENROUTER_PROVIDER, OPENROUTER_BASE_URL

Constants are deterministic and have no side effects. Authentication is resolved
at request time from OPENROUTER_API_KEY by the LLM service.
"""

from __future__ import annotations

OPENROUTER_PROVIDER = "openrouter"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_DEFAULT_MODEL = "openai/gpt-5.5"
