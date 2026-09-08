"""realtime_prediction_tool — forced Realtime tool for structured predictions.

Calling spec:
    emit_prediction_tool() -> dict
    emit_prediction_tool_choice() -> dict
    raw_arguments_text(raw) -> str

Outputs are deterministic JSON-compatible Realtime tool specs and normalized
function-call argument strings. Side effects: none.
"""

from __future__ import annotations

import json
from typing import Any

from ib.predictions import PredictionAction

EMIT_PREDICTION_TOOL_NAME = "emit_prediction"


def emit_prediction_tool_choice() -> dict[str, str]:
    """Return a Realtime tool_choice that forces the prediction emitter."""

    return {"type": "function", "name": EMIT_PREDICTION_TOOL_NAME}


def emit_prediction_tool() -> dict[str, Any]:
    """Return the schema-constrained tool used to emit PredictionRow payloads."""

    return {
        "type": "function",
        "name": EMIT_PREDICTION_TOOL_NAME,
        "description": "Return the final MSI-Bench prediction. Do not speak JSON.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "predicted_action": {
                    "type": "string",
                    "enum": [action.value for action in PredictionAction],
                },
                "answer_text": {"type": ["string", "null"]},
                "tool_calls": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "name": {"type": "string"},
                            "arguments": {"type": "object"},
                        },
                        "required": ["name", "arguments"],
                    },
                },
                "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
            },
            "required": ["predicted_action", "answer_text", "tool_calls"],
        },
    }


def raw_arguments_text(raw: Any) -> str:
    """Return function-call arguments as a JSON string suitable for schema parsing."""

    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        return json.dumps(raw, ensure_ascii=False)
    return ""
