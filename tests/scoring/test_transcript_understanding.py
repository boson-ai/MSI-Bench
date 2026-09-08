"""Transcript-input control for turn-based Gemini understanding evals.

These tests pin the experiment invariant: an ``input_mode="transcript"`` run
differs from the audio run in exactly one variable — the model sees a labeled
transcript instead of audio, while the system prompt, tool protocol, and output
schema stay identical.
"""

from __future__ import annotations

import json
import wave
from io import BytesIO
from types import SimpleNamespace

from ib.io import read_jsonl
from ib.predictions import PredictionRow
from ib.scoring.gemini_understanding import run_gemini_understanding_predictions
from ib.scoring.transcript_render import render_visible_transcript


def _wav_bytes() -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24_000)
        handle.writeframes((b"\x00\x00") * 16)
    return buffer.getvalue()


def _attribution_row() -> dict:
    return {
        "cell_id": "commerce-attr-1",
        "scene": "commerce_service",
        "logic_pattern": "constraint_attribution_under_interleaving",
        "expected_action": "respond",
        "mixed_testcase_audio_paths": ["audio/final.wav"],
        "user_turn_1_audio": "audio/final.wav",
        "visible_context_lines": [
            {
                "line_id": "L1",
                "speaker": "Mark",
                "addressed_to": "assistant",
                "text": "Assistant, refund that forty-five dollar spa pass.",
            },
            {
                "line_id": "L2",
                "speaker": "assistant",
                "addressed_to": "Mark",
                "text": "Got it, the forty-five from Saturday, flagged.",
            },
            {
                "line_id": "L3",
                "speaker": "Dana",
                "addressed_to": "Mark",
                "text": "And refund my thirty-eight dollar sweater from the gift shop.",
            },
        ],
        "logic_available_functions": [
            {
                "name": "dispute_folio_charge",
                "arguments": {
                    "charge_item": "string",
                    "disputed_amount": "number range[0,500]",
                    "requester": "enum[Mark,Dana]",
                },
            }
        ],
    }


def _manifest(tmp_path, row: dict):
    run_root = tmp_path / "run"
    (run_root / "manifest").mkdir(parents=True)
    (run_root / "audio").mkdir()
    (run_root / "audio" / "final.wav").write_bytes(_wav_bytes())
    manifest = run_root / "manifest" / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return manifest


class _GeminiModels:
    def __init__(self, response) -> None:
        self.response = response
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class _GeminiClient:
    def __init__(self, response) -> None:
        self.models = _GeminiModels(response)


def _native_response():
    return SimpleNamespace(
        response_id="gemini_txt_1",
        text=None,
        function_calls=[
            SimpleNamespace(
                name="dispute_folio_charge",
                args={"charge_item": "spa pass", "disputed_amount": 45, "requester": "Mark"},
            )
        ],
        candidates=[],
        usage_metadata=SimpleNamespace(
            prompt_token_count=11, candidates_token_count=6, total_token_count=17
        ),
    )


def _part_texts(contents) -> list[str]:
    texts: list[str] = []
    for content in contents:
        for part in content.parts:
            value = getattr(part, "text", None)
            if isinstance(value, str):
                texts.append(value)
    return texts


def _has_audio_bytes(contents) -> bool:
    for content in contents:
        for part in content.parts:
            blob = getattr(part, "inline_data", None)
            if blob is not None and getattr(blob, "data", None):
                return True
    return False


def test_render_visible_transcript_labels_speaker_and_addressee() -> None:
    transcript = render_visible_transcript(_attribution_row())
    assert transcript.splitlines() == [
        "L1 Mark -> assistant: Assistant, refund that forty-five dollar spa pass.",
        "L2 assistant -> Mark: Got it, the forty-five from Saturday, flagged.",
        "L3 Dana -> Mark: And refund my thirty-eight dollar sweater from the gift shop.",
    ]


def test_transcript_mode_swaps_audio_for_labeled_text_only(tmp_path) -> None:
    manifest = _manifest(tmp_path, _attribution_row())

    audio_client = _GeminiClient(_native_response())
    audio_report = run_gemini_understanding_predictions(
        manifest,
        tmp_path / "audio.jsonl",
        model="gemini-3.5-flash",
        client=audio_client,
        tool_protocol="native",
    )
    text_client = _GeminiClient(_native_response())
    text_report = run_gemini_understanding_predictions(
        manifest,
        tmp_path / "text.jsonl",
        model="gemini-3.5-flash",
        client=text_client,
        tool_protocol="native",
        input_mode="transcript",
    )

    audio_call = audio_client.models.calls[0]
    text_call = text_client.models.calls[0]

    # The controlled variable: system prompt, tools, and schema are byte-identical;
    # only the request contents differ (audio bytes vs labeled transcript text).
    assert audio_call["config"].system_instruction == text_call["config"].system_instruction
    audio_decl = audio_call["config"].tools[0].function_declarations[0].name
    text_decl = text_call["config"].tools[0].function_declarations[0].name
    assert audio_decl == text_decl == "dispute_folio_charge"

    assert _has_audio_bytes(audio_call["contents"]) is True
    assert _has_audio_bytes(text_call["contents"]) is False
    text_blob = "\n".join(_part_texts(text_call["contents"]))
    assert "L1 Mark -> assistant" in text_blob
    assert "L3 Dana -> Mark" in text_blob

    assert audio_report["provider"] == "gemini"
    assert audio_report["input_mode"] == "audio"
    assert text_report["provider"] == "gemini_text"
    assert text_report["input_mode"] == "transcript"
    raw_rows = [
        json.loads(line)
        for line in (tmp_path / "raw_responses.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["provider"] for row in raw_rows] == ["gemini", "gemini_text"]
    assert all(row["response_id"] == "gemini_txt_1" for row in raw_rows)

    prediction = read_jsonl(tmp_path / "text.jsonl", PredictionRow)[0]
    assert prediction.predicted_action.value == "respond"
    assert prediction.tool_calls[0].name == "dispute_folio_charge"
    assert prediction.tool_calls[0].arguments == {
        "charge_item": "spa pass",
        "disputed_amount": 45,
        "requester": "Mark",
    }
    assert prediction.metadata["provider"] == "gemini_text"
    assert prediction.metadata["input_mode"] == "transcript"
    assert prediction.metadata["audio_clip_count"] == 0
