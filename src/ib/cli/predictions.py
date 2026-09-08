"""CLI wrapper for prediction JSONL schema validation."""

from __future__ import annotations

from ib.io import write_json
from ib.predictions import read_predictions


def run(predictions: str, output: str) -> dict:
    rows = read_predictions(predictions)
    result = {
        "schema_version": "ib.predictions.validation.v1",
        "passed": True,
        "prediction_count": len(rows),
        "cell_ids": [row.cell_id for row in rows],
    }
    write_json(output, result)
    return result
