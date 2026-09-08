"""smoke — minimal exact-action scorer for end-to-end pipeline validation.

Calling spec:
    score_predictions(manifest_path, predictions_path|None, rubrics_path|None) -> dict

Predictions are JSONL rows with ``cell_id`` and ``predicted_action``. When no
file is supplied, oracle predictions are derived from the manifest so the build
recipe validates artifact plumbing without evaluating a model.

Manifest rows may reference planner-generated answer rubrics by ``rubric_ids``
or nested judge-input rows containing ``rubric_id``. Scoring validates those IDs
against the S1 planner rubric artifact; it never creates scoring-time rubrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ib.predictions import read_predictions


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


def _required_str(row: dict[str, Any], key: str, source: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source} row missing non-empty {key!r}")
    return value


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


def _infer_rubrics_path(manifest_path: str | Path) -> Path | None:
    source = Path(manifest_path)
    # Canonical run layout: <run>/manifest/manifest.jsonl -> <run>/plans/rubrics.jsonl.
    candidate = source.parent.parent / "plans" / "rubrics.jsonl"
    if source.parent.name == "manifest" and candidate.exists():
        return candidate
    sibling = source.with_name("rubrics.jsonl")
    if sibling.exists():
        return sibling
    return None


def _rubric_ids_from(path: str | Path) -> set[str]:
    ids: set[str] = set()
    for row in _read_rows(path):
        rubric_id = _required_str(row, "rubric_id", "rubrics")
        if rubric_id in ids:
            raise ValueError(f"duplicate rubric_id {rubric_id!r} in {path}")
        ids.add(rubric_id)
    return ids


def _validate_rubric_references(
    manifest_path: str | Path, manifest_rows: list[dict[str, Any]], rubrics_path: str | Path | None
) -> tuple[list[str], Path | None]:
    references = sorted({rid for row in manifest_rows for rid in _rubric_references(row)})
    if not references:
        return [], None
    resolved_path = (
        Path(rubrics_path) if rubrics_path is not None else _infer_rubrics_path(manifest_path)
    )
    if resolved_path is None:
        raise ValueError(
            "manifest references answer-layer rubric IDs, but no rubrics artifact was supplied "
            "and no canonical rubrics.jsonl could be inferred"
        )
    known = _rubric_ids_from(resolved_path)
    missing = sorted(set(references) - known)
    if missing:
        raise ValueError(f"unknown rubric_id references in manifest: {missing}")
    return references, resolved_path


def score_predictions(
    manifest_path: str | Path,
    predictions_path: str | Path | None = None,
    rubrics_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return exact expected-action accuracy, row counts, and rubric validation evidence."""
    manifest = _read_rows(manifest_path)
    validated_rubrics, resolved_rubrics_path = _validate_rubric_references(
        manifest_path, manifest, rubrics_path
    )
    predictions = (
        [row.model_dump(mode="json") for row in read_predictions(predictions_path)]
        if predictions_path
        else [
            {
                "cell_id": _required_str(row, "cell_id", "manifest"),
                "predicted_action": _required_str(row, "expected_action", "manifest"),
            }
            for row in manifest
        ]
    )
    by_id = {
        _required_str(row, "cell_id", "predictions"): row.get("predicted_action")
        for row in predictions
    }
    missing = sorted(
        _required_str(row, "cell_id", "manifest") for row in manifest if row["cell_id"] not in by_id
    )
    correct = sum(by_id.get(row["cell_id"]) == row["expected_action"] for row in manifest)
    total = len(manifest)
    errors = []
    if missing:
        errors.append(
            f"missing predictions for {len(missing)} manifest row(s): {', '.join(missing)}"
        )
    return {
        "schema_version": "ib.v1",
        "metric": "expected_action_exact_match",
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "missing_prediction_cell_ids": missing,
        "errors": errors,
        "rubric_reference_count": len(validated_rubrics),
        "validated_rubric_ids": validated_rubrics,
        "rubrics_path": str(resolved_rubrics_path) if resolved_rubrics_path else None,
    }
