"""openai_realtime_error_io — persist Realtime prediction parse failures.

Calling spec:
    write_prediction_format_failure(error_dir, provider=..., model=...,
                                    cell_id=..., attempt=..., capture=..., error=...) -> None
    error_artifacts_for(error_dir, cell_id) -> artifact path strings

Functions only read or write the explicit error directory.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ib.scoring.openai_realtime_capture import RealtimeCapture


def write_prediction_format_failure(
    error_dir: Path | None,
    *,
    provider: str,
    model: str,
    cell_id: str,
    attempt: int,
    capture: RealtimeCapture,
    error: Exception,
) -> None:
    """Persist one Realtime schema/JSON failure when a directory is configured."""
    if error_dir is None:
        return
    error_dir.mkdir(parents=True, exist_ok=True)
    path = error_dir / f"{_safe_artifact_name(cell_id)}__attempt_{attempt}.json"
    payload = {
        "schema_version": "ib.openai_fullduplex.prediction_error.v1",
        "provider": provider,
        "model": model,
        "cell_id": cell_id,
        "attempt": attempt,
        "error": str(error),
        "response_id": capture.response_id,
        "response_status": capture.response_status,
        "emit_prediction_arguments": capture.emit_prediction_arguments,
        "text": capture.text(),
        "text_parts": capture.text_parts,
        "audio_transcripts": capture.audio_transcripts,
        "tool_calls": capture.tool_calls,
        "usage": capture.usage,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def error_artifacts_for(error_dir: Path | None, cell_id: str) -> list[str]:
    """Return sorted parse/format error artifact paths for one cell."""
    if error_dir is None:
        return []
    pattern = f"{_safe_artifact_name(cell_id)}__attempt_*.json"
    return [str(path) for path in sorted(error_dir.glob(pattern))]


def _safe_artifact_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._=-]+", "_", value).strip("._")
    return name or "unnamed"
