"""Gemini turn-based understanding generation and thinking-variant tests.

Calling spec: pytest supplies temporary paths; fake Gemini clients receive the
generated request and return deterministic text. Tests write only under tmp_path.
"""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace
import wave

import pytest

from ib.io import read_jsonl
from ib.predictions import PredictionRow
from ib.scoring.gemini_understanding import _response_text, run_gemini_understanding_predictions
from ib.scoring.openai_understanding import UnderstandingProviderError


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def _wav_bytes() -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24_000)
        handle.writeframes((b"\x00\x00") * 16)
    return buffer.getvalue()


def _manifest_row(cell_id: str = "c1") -> dict:
    return {
        "cell_id": cell_id,
        "expected_action": "respond",
        "sibling_group": "g1",
        "mixed_testcase_audio_paths": ["audio/final.wav"],
        "assistant_turn_1_transcript": "I previously gave the relevant setup steps.",
        "user_turn_2_audio": "audio/final.wav",
        "user_turn_2_transcript": "Please answer the current request.",
        "rubric_ids": [f"{cell_id}:o1"],
        "labels": {
            "perception": {
                "scene": "work_professional",
                "addressee_cue": "wake_word",
                "acoustic_conditions": {
                    "reverb": "dry",
                    "competing_speech": "none",
                    "channel": "clean_close",
                },
            },
            "interaction": {
                "expected_action": "respond",
                "expected_addressed_speaker": "user",
                "participation_frame": "user_addressed",
                "contract_required": False,
            },
            "answer": {
                "answerable": True,
                "rubric_ids": [f"{cell_id}:o1"],
                "contract_text": None,
            },
        },
    }


def _manifest(tmp_path: Path, rows: list[dict], *, include_audio: bool = True) -> Path:
    run_root = tmp_path / "run"
    (run_root / "manifest").mkdir(parents=True)
    if include_audio:
        (run_root / "audio").mkdir()
        (run_root / "audio" / "final.wav").write_bytes(_wav_bytes())
    manifest = run_root / "manifest" / "manifest.jsonl"
    _write_jsonl(manifest, rows)
    return manifest


class _GeminiModels:
    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = payloads
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        payload = self.payloads.pop(0)
        return SimpleNamespace(
            response_id="gemini_1",
            text=json.dumps(payload),
            usage_metadata=SimpleNamespace(
                prompt_token_count=11,
                candidates_token_count=6,
                total_token_count=17,
            ),
        )


class _GeminiClient:
    def __init__(self, payloads: list[dict]) -> None:
        self.models = _GeminiModels(payloads)


class _RawTextGeminiModels:
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            response_id="gemini_raw",
            text=self.texts.pop(0),
            usage_metadata=SimpleNamespace(
                prompt_token_count=11,
                candidates_token_count=6,
                total_token_count=17,
            ),
        )


class _RawTextGeminiClient:
    def __init__(self, texts: list[str]) -> None:
        self.models = _RawTextGeminiModels(texts)


def test_gemini_response_text_prefers_visible_parts_over_combined_sdk_text() -> None:
    final_text = '{"predicted_action":"respond","answer_text":"Limit updated."}'
    response = SimpleNamespace(
        text=f"Processing the request internally.\n{final_text}",
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(text="Processing the request internally.", thought=True),
                        SimpleNamespace(text=final_text, thought=False),
                    ]
                )
            )
        ],
    )

    assert _response_text(response) == final_text


def test_gemini_understanding_generates_thinking_prediction(tmp_path, capsys) -> None:
    manifest = _manifest(tmp_path, [_manifest_row()])
    output = tmp_path / "predictions.jsonl"
    client = _GeminiClient(
        [{"predicted_action": "respond", "answer_text": "Yes, that is right.", "confidence": 0.9}]
    )

    report = run_gemini_understanding_predictions(
        manifest,
        output,
        model="gemini-3-flash-preview",
        client=client,
        thinking_mode="enabled",
    )
    progress = capsys.readouterr().err

    rows = read_jsonl(output, PredictionRow)
    raw_row = json.loads((tmp_path / "raw_responses.jsonl").read_text(encoding="utf-8"))
    call = client.models.calls[0]
    contents = call["contents"]
    assert report["provider"] == "gemini"
    assert report["model"] == "gemini-3-flash-preview"
    assert report["thinking_mode"] == "enabled"
    assert report["raw_responses"] == str(tmp_path / "raw_responses.jsonl")
    assert raw_row["response_id"] == "gemini_1"
    assert raw_row["response"]["text"].startswith('{"predicted_action": "respond"')
    assert "gemini:gemini-3-flash-preview case 1/1" in progress
    assert report["usage"] == {"input_tokens": 11, "output_tokens": 6, "total_tokens": 17}
    assert rows[0].predicted_action.value == "respond"
    assert contents[0].role == "model"
    assert contents[0].parts[0].text == "I previously gave the relevant setup steps."
    assert contents[1].parts[0].inline_data.mime_type == "audio/wav"
    assert rows[0].metadata["thinking_mode"] == "enabled"
    assert call["config"].response_mime_type == "application/json"
    assert call["config"].max_output_tokens == 4096
    assert call["config"].thinking_config.include_thoughts is True
    assert call["config"].thinking_config.thinking_budget == 8192
    assert "voice assistant participating" in call["config"].system_instruction
    assert "confidence" not in call["config"].system_instruction
    assert rows[0].metadata["prompts"]["source"] == "captured_at_generation"


def test_gemini_understanding_disables_thinking_for_plain_variant(tmp_path) -> None:
    manifest = _manifest(tmp_path, [_manifest_row()])
    client = _GeminiClient([{"predicted_action": "respond", "answer_text": "Done."}])

    report = run_gemini_understanding_predictions(
        manifest,
        tmp_path / "predictions.jsonl",
        model="gemini-3-flash-preview",
        client=client,
        thinking_mode="disabled",
    )

    config = client.models.calls[0]["config"]
    assert report["thinking_mode"] == "disabled"
    assert config.max_output_tokens == 1400
    assert config.thinking_config.include_thoughts is False
    assert config.thinking_config.thinking_budget == 0


def test_gemini_understanding_resumes_existing_predictions(tmp_path) -> None:
    manifest = _manifest(tmp_path, [_manifest_row("c1"), _manifest_row("c2")])
    output = tmp_path / "predictions.jsonl"
    _write_jsonl(
        output,
        [{"cell_id": "c1", "predicted_action": "respond", "answer_text": "cached"}],
    )
    client = _GeminiClient(
        [{"predicted_action": "respond", "answer_text": "fresh", "confidence": 0.8}]
    )

    report = run_gemini_understanding_predictions(
        manifest, output, model="gemini-3.5-flash", client=client
    )

    rows = read_jsonl(output, PredictionRow)
    assert report["predicted_count"] == 2
    assert report["llm_call_count"] == 1
    assert [row.cell_id for row in rows] == ["c1", "c2"]
    assert [row.answer_text for row in rows] == ["cached", "fresh"]
    assert client.models.calls[0]["config"].thinking_config is None


def test_gemini_understanding_records_format_failure_as_silent_row(tmp_path) -> None:
    manifest = _manifest(tmp_path, [_manifest_row("c1"), _manifest_row("c2")])
    output = tmp_path / "predictions.jsonl"
    malformed = '{\n "predicted_action": "respond",\n "tool_calls": []\n []\n}'
    client = _RawTextGeminiClient(
        [malformed] * 5 + [json.dumps({"predicted_action": "respond", "answer_text": "ok"})]
    )

    report = run_gemini_understanding_predictions(
        manifest, output, model="gemini-3.5-flash", client=client
    )

    rows = read_jsonl(output, PredictionRow)
    raw_rows = [
        json.loads(line)
        for line in (tmp_path / "raw_responses.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert report["predicted_count"] == 2
    assert [(row["cell_id"], row["attempt"]) for row in raw_rows] == [
        ("c1", 1),
        ("c1", 2),
        ("c1", 3),
        ("c1", 4),
        ("c1", 5),
        ("c2", 1),
    ]
    assert raw_rows[0]["response"]["text"] == malformed
    assert rows[0].predicted_action.value == "silent"
    error = rows[0].metadata["provider_error"]
    assert error["type"] == "prediction_format_error"
    assert "invalid JSON prediction for c1" in error["message"]
    assert rows[1].predicted_action.value == "respond"


def test_gemini_understanding_transport_error_still_aborts(tmp_path) -> None:
    row = dict(_manifest_row("c1"))
    row["final_audio_path"] = "audio/missing.wav"
    manifest = _manifest(tmp_path, [row], include_audio=False)

    with pytest.raises(UnderstandingProviderError):
        run_gemini_understanding_predictions(
            manifest,
            tmp_path / "predictions.jsonl",
            model="gemini-3.5-flash",
            client=_RawTextGeminiClient([]),
        )


def test_retryable_gemini_errors_cover_transport_failures() -> None:
    import ssl

    import httpx

    from ib.scoring.gemini_understanding import _is_retryable_gemini_error

    assert _is_retryable_gemini_error(httpx.ReadTimeout("timed out"))
    assert _is_retryable_gemini_error(httpx.ConnectError("connection refused"))
    assert _is_retryable_gemini_error(httpx.RemoteProtocolError("server disconnected"))
    assert _is_retryable_gemini_error(ssl.SSLError(8, "UNEXPECTED_EOF_WHILE_READING"))
    assert _is_retryable_gemini_error(ValueError("[SSL: UNEXPECTED_EOF_WHILE_READING] unexpected eof while reading"))
    assert not _is_retryable_gemini_error(ValueError("invalid request payload"))


def test_gemini_client_sets_request_timeout() -> None:
    from ib.scoring.gemini_understanding import (
        GEMINI_REQUEST_TIMEOUT_MS,
        GeminiUnderstandingProvider,
    )

    provider = GeminiUnderstandingProvider(model="gemini-3.5-flash", api_key="test-key")
    client = provider._client()
    assert client._api_client._http_options.timeout == GEMINI_REQUEST_TIMEOUT_MS
