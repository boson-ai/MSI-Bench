"""io — deterministic JSON and JSONL helpers.

Calling spec:
    write_jsonl(path, models) -> None
    read_jsonl(path, model_type) -> list[model_type]
    write_json(path, payload) -> None

Side effects: explicit filesystem writes only. JSON output is canonical and UTF-8.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, TypeVar

from pydantic import BaseModel, ValidationError

ModelT = TypeVar("ModelT", bound=BaseModel)


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible data using a stable byte representation."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def write_jsonl(path: str | Path, values: Iterable[Any]) -> None:
    """Write one canonical JSON object per line."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(f"{canonical_json(value)}\n" for value in values)
    destination.write_text(text, encoding="utf-8")


def read_jsonl(path: str | Path, model_type: type[ModelT]) -> list[ModelT]:
    """Parse and validate a JSONL artifact, identifying malformed line numbers."""
    source = Path(path)
    rows: list[ModelT] = []
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read {source}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            rows.append(model_type.model_validate_json(line))
        except (ValidationError, ValueError) as exc:
            raise ValueError(f"invalid {source} line {line_number}: {exc}") from exc
    return rows


def write_json(path: str | Path, payload: Any) -> None:
    """Write a canonical JSON document with a trailing newline."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(f"{canonical_json(payload)}\n", encoding="utf-8")
