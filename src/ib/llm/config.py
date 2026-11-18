"""config — tuning knobs for the async LLM service.

Calling spec:
    config = LlmConfig()                          # sane defaults
    config = LlmConfig(max_concurrency=16, rpm=120.0)

LlmConfig is a strict IbBaseModel (extra='forbid'); every field is bounded so a
typo or out-of-range value fails loud at construction.

Side effects: none.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from ib.models.base import IbBaseModel


class LlmConfig(IbBaseModel):
    """Concurrency, rate-limit and retry settings for LlmService."""

    max_concurrency: int = Field(default=8, ge=1, le=256)
    rpm: float = Field(default=60.0, gt=0.0, description="Requests per minute (token bucket).")
    request_timeout_s: float = Field(default=100.0, gt=0.0, le=600.0)

    retry_max_attempts: int = Field(default=3, ge=1, le=10)
    retry_base_delay_s: float = Field(default=2.0, gt=0.0, le=30.0)
    retry_max_delay_s: float = Field(default=30.0, gt=0.0, le=300.0)

    @model_validator(mode="after")
    def _validate_retry_delays(self) -> LlmConfig:
        if self.retry_base_delay_s > self.retry_max_delay_s:
            raise ValueError(
                f"retry_base_delay_s ({self.retry_base_delay_s}) must be <= "
                f"retry_max_delay_s ({self.retry_max_delay_s})"
            )
        return self
