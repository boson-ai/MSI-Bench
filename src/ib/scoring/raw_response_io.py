"""raw_response_io — serialize and append provider responses.

Calling spec:
    raw_response_path(predictions_path) -> sibling raw_responses.jsonl
    json_safe_response(value) -> JSON-compatible value
    append_raw_response(path, cell_id=..., provider=..., model=...,
                        eval_mode=..., phase=..., response=...) -> None

Serialization is deterministic and does not mutate inputs. ``append_raw_response``
is the only side-effecting operation and appends one validated UTF-8 JSONL row.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, time
from enum import Enum
from pathlib import Path
from typing import Any

from ib.io import canonical_json
from ib.models.raw_response import (
    EvalMode,
    RawProviderResponseRow,
    RawResponsePhase,
)


RAW_RESPONSES_FILENAME = "raw_responses.jsonl"


def raw_response_path(predictions_path: str | Path) -> Path:
    """Return the raw-response artifact adjacent to a predictions artifact."""
    return Path(predictions_path).with_name(RAW_RESPONSES_FILENAME)


def json_safe_response(value: Any) -> Any:
    """Return a lossless JSON-compatible representation of an SDK value."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, bytes | bytearray | memoryview):
        return {
            "$type": "bytes",
            "$encoding": "base64",
            "data": base64.b64encode(bytes(value)).decode("ascii"),
        }
    if isinstance(value, Enum):
        return json_safe_response(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime | date | time):
        return value.isoformat()
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return json_safe_response(model_dump(mode="json"))
        except (TypeError, ValueError):
            return json_safe_response(model_dump())
    if is_dataclass(value) and not isinstance(value, type):
        return json_safe_response(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_safe_response(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe_response(item) for item in value]
    if isinstance(value, set | frozenset):
        return [json_safe_response(item) for item in sorted(value, key=repr)]
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return {
            key: json_safe_response(item)
            for key, item in attributes.items()
            if isinstance(key, str) and not key.startswith("_")
        }
    return {"$type": _qualified_type_name(value), "$repr": repr(value)}


def append_raw_response(
    output: str | Path,
    *,
    cell_id: str,
    provider: str,
    model: str,
    eval_mode: EvalMode,
    phase: RawResponsePhase,
    response: Any,
    attempt: int = 1,
    response_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Append one validated raw provider response row."""
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    row = RawProviderResponseRow(
        cell_id=cell_id,
        provider=provider,
        model=model,
        eval_mode=eval_mode,
        phase=phase,
        attempt=attempt,
        response_id=response_id or _response_id(response),
        response_type=_qualified_type_name(response),
        response=json_safe_response(response),
        metadata=json_safe_response(dict(metadata or {})),
    )
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(f"{canonical_json(row)}\n")
        handle.flush()


def _response_id(response: Any) -> str | None:
    for key in ("response_id", "id"):
        value = response.get(key) if isinstance(response, Mapping) else getattr(response, key, None)
        if isinstance(value, str) and value:
            return value
    return None


def _qualified_type_name(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"
