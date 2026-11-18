"""predictions — strict schema and ingestion for model output JSONL."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field

from ib.io import read_jsonl
from ib.models.base import IbBaseModel
from ib.models.logic import ExpectedToolCall


class PredictionAction(str, Enum):
    """Canonical model-emitted action labels."""

    respond = "respond"
    silent = "silent"

    def __str__(self) -> str:
        return self.value


class PredictionRow(IbBaseModel):
    """One model prediction row aligned to a manifest cell."""

    cell_id: str
    predicted_action: PredictionAction
    answer_text: str | None = None
    tool_calls: list[ExpectedToolCall] = Field(default_factory=list)
    confidence: float | None = None
    metadata: dict[str, Any] | None = None


def read_predictions(path: str | Path) -> list[PredictionRow]:
    """Read and validate model predictions JSONL, rejecting duplicate cell IDs."""
    rows = read_jsonl(path, PredictionRow)
    seen: set[str] = set()
    duplicates: list[str] = []
    for row in rows:
        if row.cell_id in seen:
            duplicates.append(row.cell_id)
        seen.add(row.cell_id)
    if duplicates:
        raise ValueError(f"duplicate prediction cell_id(s): {sorted(set(duplicates))}")
    return rows
