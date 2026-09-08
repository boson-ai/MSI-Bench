"""S3.A3 — smoke scoring consumes planner-generated rubric references."""

from __future__ import annotations

import json

import pytest

from ib.scoring.smoke import score_predictions


def _write_jsonl(path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def test_oracle_smoke_scoring_is_stable(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            {"cell_id": "c1", "expected_action": "ignore"},
            {"cell_id": "c2", "expected_action": "respond", "rubric_ids": ["c2:o1"]},
        ],
    )
    rubrics = tmp_path / "rubrics.jsonl"
    _write_jsonl(rubrics, [{"cell_id": "c2", "rubric_id": "c2:o1", "layer": "O1"}])

    first = score_predictions(manifest, rubrics_path=rubrics)
    same = score_predictions(manifest, rubrics_path=rubrics)

    assert first == same
    assert first["total"] == 2
    assert first["correct"] == 2
    assert first["accuracy"] == 1.0
    assert first["rubric_reference_count"] == 1
    assert first["validated_rubric_ids"] == ["c2:o1"]


def test_missing_predictions_are_reported_clearly(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(
        manifest,
        [
            {"cell_id": "c1", "expected_action": "silent"},
            {"cell_id": "c2", "expected_action": "respond"},
        ],
    )
    _write_jsonl(predictions, [{"cell_id": "c1", "predicted_action": "silent"}])

    result = score_predictions(manifest, predictions)

    assert result["correct"] == 1
    assert result["missing_prediction_cell_ids"] == ["c2"]
    assert result["errors"] == ["missing predictions for 1 manifest row(s): c2"]


def test_referenced_rubric_ids_must_exist(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    rubrics = tmp_path / "rubrics.jsonl"
    _write_jsonl(
        manifest, [{"cell_id": "c1", "expected_action": "respond", "rubric_ids": ["c1:o1"]}]
    )
    _write_jsonl(rubrics, [{"cell_id": "other", "rubric_id": "other:o1", "layer": "O1"}])

    with pytest.raises(ValueError, match="unknown rubric_id references"):
        score_predictions(manifest, rubrics_path=rubrics)


def test_nested_answer_judge_inputs_are_validated(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    rubrics = tmp_path / "rubrics.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "cell_id": "c1",
                "expected_action": "respond",
                "answer_judge_inputs": [{"layer": "O1", "rubric_id": "c1:o1"}],
            }
        ],
    )
    _write_jsonl(rubrics, [{"cell_id": "c1", "rubric_id": "c1:o1", "layer": "O1"}])

    result = score_predictions(manifest, rubrics_path=rubrics)

    assert result["validated_rubric_ids"] == ["c1:o1"]
