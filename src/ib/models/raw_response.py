"""raw_response — schema for one persisted provider response.

Calling spec:
    row = RawProviderResponseRow(...)

The schema contains provider response data after lossless JSON conversion.
Validation is deterministic and has no side effects.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from ib.models.base import IbBaseModel


RAW_PROVIDER_RESPONSE_SCHEMA_VERSION = "ib.raw_provider_response.v1"
RawResponsePhase = Literal["prediction", "base_prediction", "live_probe"]
EvalMode = Literal["turn_based", "fullduplex"]


class RawProviderResponseRow(IbBaseModel):
    """One provider response associated with a prediction attempt."""

    schema_version: Literal["ib.raw_provider_response.v1"] = (
        RAW_PROVIDER_RESPONSE_SCHEMA_VERSION
    )
    cell_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    eval_mode: EvalMode
    phase: RawResponsePhase
    attempt: int = Field(default=1, ge=1)
    response_id: str | None = None
    response_type: str = Field(min_length=1)
    response: Any
    metadata: dict[str, Any] = Field(default_factory=dict)
