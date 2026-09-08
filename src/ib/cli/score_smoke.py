"""CLI wrapper for minimal score-smoke evaluation."""

from __future__ import annotations

from ib.io import write_json
from ib.scoring.smoke import score_predictions


def run(
    manifest: str,
    output: str,
    predictions: str | None = None,
    rubrics: str | None = None,
) -> dict:
    result = score_predictions(manifest, predictions, rubrics)
    write_json(output, result)
    return result
