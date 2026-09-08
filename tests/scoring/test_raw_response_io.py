"""Raw provider-response serialization and JSONL persistence tests.

Calling spec: tests pass SDK-like objects and temporary output paths; no network
calls occur and all writes stay under ``tmp_path``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from ib.scoring.raw_response_io import append_raw_response, raw_response_path


def test_append_raw_response_preserves_nested_sdk_fields_and_bytes(tmp_path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    output = raw_response_path(predictions)
    response = SimpleNamespace(
        response_id="gemini-response-1",
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(text="internal summary", thought=True),
                        SimpleNamespace(inline_data=b"\x00\x01"),
                    ]
                )
            )
        ],
    )

    append_raw_response(
        output,
        cell_id="c1",
        provider="gemini",
        model="gemini-3.5-flash",
        eval_mode="turn_based",
        phase="prediction",
        attempt=2,
        response=response,
        metadata={"thinking_mode": "enabled"},
    )

    row = json.loads(output.read_text(encoding="utf-8"))
    assert output == tmp_path / "raw_responses.jsonl"
    assert row["schema_version"] == "ib.raw_provider_response.v1"
    assert row["cell_id"] == "c1"
    assert row["response_id"] == "gemini-response-1"
    assert row["attempt"] == 2
    assert row["metadata"] == {"thinking_mode": "enabled"}
    parts = row["response"]["candidates"][0]["content"]["parts"]
    assert parts[0] == {"text": "internal summary", "thought": True}
    assert parts[1]["inline_data"] == {
        "$encoding": "base64",
        "$type": "bytes",
        "data": "AAE=",
    }


def test_append_raw_response_appends_one_json_object_per_call(tmp_path) -> None:
    output = tmp_path / "raw_responses.jsonl"

    for cell_id in ("c1", "c2"):
        append_raw_response(
            output,
            cell_id=cell_id,
            provider="openai",
            model="gpt-audio-1.5",
            eval_mode="turn_based",
            phase="prediction",
            response=SimpleNamespace(id=f"response-{cell_id}", output_text="ok"),
        )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["cell_id"] for row in rows] == ["c1", "c2"]
    assert [row["response_id"] for row in rows] == ["response-c1", "response-c2"]
