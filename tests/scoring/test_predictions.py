"""S8.A1 — prediction JSONL schema and ingestion."""

from __future__ import annotations

import json

import pytest

from ib.cli.predictions import run as run_validate_predictions
from ib.predictions import PredictionAction, PredictionRow, read_predictions
from ib.scoring.realtime_prediction_tool import emit_prediction_tool
from ib.scoring.smoke import score_predictions


def _write_jsonl(path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def test_prediction_row_schema_accepts_model_output() -> None:
    row = PredictionRow(
        cell_id="c1",
        predicted_action="respond",
        answer_text="The review starts at three.",
        confidence=0.75,
        metadata={"model": "demo"},
    )

    assert row.predicted_action.value == "respond"
    assert row.answer_text == "The review starts at three."


def test_prediction_action_schema_exposes_only_binary_actions() -> None:
    assert [action.value for action in PredictionAction] == ["respond", "silent"]


@pytest.mark.parametrize("legacy_action", ["ignore", "wait", "clarify", "incorporate", "refuse"])
def test_prediction_row_rejects_legacy_action(legacy_action: str) -> None:
    with pytest.raises(ValueError, match="Input should be 'respond' or 'silent'"):
        PredictionRow(cell_id="c1", predicted_action=legacy_action)


def test_realtime_prediction_tool_exposes_only_binary_actions() -> None:
    action_schema = emit_prediction_tool()["parameters"]["properties"]["predicted_action"]

    assert action_schema["enum"] == ["respond", "silent"]


def test_prediction_ingestion_rejects_malformed_action(tmp_path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(predictions, [{"cell_id": "c1", "predicted_action": "do_magic"}])

    with pytest.raises(ValueError, match="invalid"):
        read_predictions(predictions)


def test_prediction_ingestion_rejects_duplicate_cell_ids(tmp_path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions,
        [
            {"cell_id": "c1", "predicted_action": "silent"},
            {"cell_id": "c1", "predicted_action": "respond"},
        ],
    )

    with pytest.raises(ValueError, match="duplicate prediction"):
        read_predictions(predictions)


def test_score_smoke_consumes_valid_prediction_schema(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(manifest, [{"cell_id": "c1", "expected_action": "respond"}])
    _write_jsonl(predictions, [{"cell_id": "c1", "predicted_action": "respond"}])

    result = score_predictions(manifest, predictions)

    assert result["accuracy"] == 1.0
    assert result["errors"] == []


def test_validate_predictions_cli_writes_report(tmp_path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "report.json"
    _write_jsonl(predictions, [{"cell_id": "c1", "predicted_action": "silent"}])

    report = run_validate_predictions(str(predictions), str(output))

    assert report["passed"] is True
    assert report["prediction_count"] == 1
    assert json.loads(output.read_text(encoding="utf-8"))["cell_ids"] == ["c1"]
