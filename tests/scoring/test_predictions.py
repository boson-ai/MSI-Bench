"""S8.A1 — prediction JSONL schema and ingestion."""

from __future__ import annotations

import json

import pytest

from ib.scoring.predictions import PredictionAction, read_predictions
from ib.scoring.realtime_prediction_tool import emit_prediction_tool


def _write_jsonl(path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def test_prediction_action_schema_exposes_only_binary_actions() -> None:
    assert [action.value for action in PredictionAction] == ["respond", "silent"]


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

