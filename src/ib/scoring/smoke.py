"""smoke — JSONL row readers shared by the eval adapters and the answer judge.

Calling spec:
    rows = _read_rows(path)               every JSON object line; ValueError on unreadable or non-object lines
    ids = _rubric_references(manifest_row)  rubric ids from rubric_ids, answer_judge_inputs and judge_inputs

Side effects: _read_rows reads the file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read {source}: {exc}") from exc
    rows = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid {source} line {number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"invalid {source} line {number}: expected object")
        rows.append(row)
    return rows


def _rubric_references(row: dict[str, Any]) -> list[str]:
    references: list[str] = []
    flat_ids = row.get("rubric_ids", [])
    if flat_ids is None:
        flat_ids = []
    if not isinstance(flat_ids, list):
        raise ValueError("manifest field 'rubric_ids' must be a list when present")
    for rubric_id in flat_ids:
        if not isinstance(rubric_id, str) or not rubric_id:
            raise ValueError("manifest field 'rubric_ids' must contain non-empty strings")
        references.append(rubric_id)

    for field in ("answer_judge_inputs", "judge_inputs"):
        judge_inputs = row.get(field, [])
        if judge_inputs is None:
            continue
        if not isinstance(judge_inputs, list):
            raise ValueError(f"manifest field {field!r} must be a list when present")
        for item in judge_inputs:
            if not isinstance(item, dict):
                raise ValueError(f"manifest field {field!r} entries must be objects")
            rubric_id = item.get("rubric_id")
            if rubric_id is not None:
                if not isinstance(rubric_id, str) or not rubric_id:
                    raise ValueError(f"manifest field {field!r} rubric_id must be non-empty")
                references.append(rubric_id)
    return references
