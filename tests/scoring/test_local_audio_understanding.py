"""Registered local audio endpoint request contracts."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import wave

import pytest

from ib.io import read_jsonl
from ib.predictions import PredictionRow
from ib.scoring.local_understanding import run_local_understanding_predictions
from ib.scoring.openai_understanding import UnderstandingProviderError


class _Completions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        payload = {
            "predicted_action": "respond",
            "answer_text": "Local answer.",
            "confidence": 0.8,
        }
        return SimpleNamespace(
            id="local-response",
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))],
            usage=None,
        )


class _Client:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_Completions())


def _wav_bytes() -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24_000)
        handle.writeframes((b"\x00\x00") * 16)
    return buffer.getvalue()


def _manifest(tmp_path: Path) -> Path:
    run_root = tmp_path / "run"
    manifest_dir = run_root / "manifest"
    audio_dir = run_root / "audio"
    manifest_dir.mkdir(parents=True)
    audio_dir.mkdir()
    (audio_dir / "final.wav").write_bytes(_wav_bytes())
    row = {
        "cell_id": "c1",
        "expected_action": "respond",
        "mixed_testcase_audio_paths": ["audio/final.wav"],
        "assistant_turn_1_transcript": "Earlier assistant context.",
        "user_turn_2_audio": "audio/final.wav",
        "labels": {
            "interaction": {"expected_action": "respond"},
            "answer": {"answerable": True, "rubric_ids": []},
        },
    }
    manifest = manifest_dir / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return manifest


def test_gemma_local_request_forwards_thinking_mode(tmp_path, monkeypatch) -> None:
    manifest = _manifest(tmp_path)
    output = tmp_path / "predictions.jsonl"
    client = _Client()
    monkeypatch.setenv("GEMMA4_UNDERSTANDING_BASE_URL", "http://gemma.test/v1")

    report = run_local_understanding_predictions(
        manifest,
        output,
        model="gemma-4-12B-it",
        thinking_mode="enabled",
        client=client,
    )

    call = client.chat.completions.calls[0]
    prediction = read_jsonl(output, PredictionRow)[0]
    raw_row = json.loads((tmp_path / "raw_responses.jsonl").read_text(encoding="utf-8"))
    assert report["provider"] == "local"
    assert report["raw_responses"] == str(tmp_path / "raw_responses.jsonl")
    assert raw_row["response_id"] == "local-response"
    assert report["metadata"]["base_url"] == "http://gemma.test/v1"
    assert call["messages"][0]["role"] == "system"
    assert call["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
    assert prediction.answer_text == "Local answer."
    assert prediction.metadata["provider"] == "local"


def test_voxtral_inlines_system_in_first_user_message(tmp_path) -> None:
    manifest = _manifest(tmp_path)
    client = _Client()

    run_local_understanding_predictions(
        manifest,
        tmp_path / "predictions.jsonl",
        model="Voxtral-Small-24B-2507",
        client=client,
    )

    messages = client.chat.completions.calls[0]["messages"]
    assert all(message["role"] != "system" for message in messages)
    assert [message["role"] for message in messages] == ["user", "assistant", "user"]
    assert messages[0]["content"][0]["type"] == "text"
    assert "voice assistant participating" in messages[0]["content"][0]["text"]
    assert messages[2]["content"][0]["type"] == "audio_url"


def test_kimi_transcription_only_route_fails_closed(tmp_path) -> None:
    with pytest.raises(UnderstandingProviderError, match="transcription only"):
        run_local_understanding_predictions(
            tmp_path / "unused-manifest.jsonl",
            tmp_path / "predictions.jsonl",
            model="Kimi-Audio-7B-Instruct",
            client=_Client(),
        )


class _ScriptedCompletions:
    def __init__(self, texts: list[str]) -> None:
        self.texts = list(texts)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            id="local-response",
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.texts.pop(0)))],
            usage=None,
        )


class _ScriptedClient:
    def __init__(self, texts: list[str]) -> None:
        self.chat = SimpleNamespace(completions=_ScriptedCompletions(texts))


def test_local_invalid_json_degrades_to_error_prediction(tmp_path) -> None:
    manifest = _manifest(tmp_path)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    second = dict(rows[0], cell_id="c2")
    manifest.write_text(
        "\n".join(json.dumps(row) for row in [rows[0], second]) + "\n", encoding="utf-8"
    )
    output = tmp_path / "predictions.jsonl"
    client = _ScriptedClient(
        [
            "{'spoken_text': \"transcription instead of a prediction\"}",
            json.dumps({"predicted_action": "respond", "answer_text": "ok"}),
        ]
    )

    report = run_local_understanding_predictions(
        manifest, output, model="Qwen2-Audio-7B-Instruct", client=client
    )

    predictions = read_jsonl(output, PredictionRow)
    raw_rows = [
        json.loads(line)
        for line in (tmp_path / "raw_responses.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert report["predicted_count"] == 2
    assert [row["cell_id"] for row in raw_rows] == ["c1", "c2"]
    assert raw_rows[0]["response"]["choices"][0]["message"]["content"] == (
        "{'spoken_text': \"transcription instead of a prediction\"}"
    )
    assert predictions[0].predicted_action.value == "silent"
    assert predictions[0].confidence == 0.0
    assert predictions[0].metadata["provider_error"]["type"] == "prediction_format_error"
    assert "invalid JSON prediction for c1" in predictions[0].metadata["provider_error"]["message"]
    assert predictions[1].predicted_action.value == "respond"


def test_voxtral_rejects_rows_beyond_five_clips_before_endpoint_call(tmp_path) -> None:
    manifest = _manifest(tmp_path)
    row = json.loads(manifest.read_text(encoding="utf-8"))
    row["mixed_testcase_audio_paths"] = ["audio/final.wav"] * 6
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    client = _Client()

    with pytest.raises(UnderstandingProviderError, match="at most 5 audio clip"):
        run_local_understanding_predictions(
            manifest,
            tmp_path / "predictions.jsonl",
            model="Voxtral-Small-24B-2507",
            client=client,
        )
    assert client.chat.completions.calls == []


def test_local_text_transcript_control_matches_audio_prompt(tmp_path, monkeypatch) -> None:
    manifest = _manifest(tmp_path)
    row = json.loads(manifest.read_text(encoding="utf-8"))
    row["visible_context_lines"] = [
        {"line_id": "L1", "speaker": "Mara", "addressed_to": "assistant", "text": "Book the noon slot."}
    ]
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setenv("GEMMA4_UNDERSTANDING_BASE_URL", "http://gemma.test/v1")

    audio_client = _Client()
    run_local_understanding_predictions(
        tmp_path / "run" / "manifest" / "manifest.jsonl",
        tmp_path / "audio_out" / "predictions.jsonl",
        model="gemma-4-12B-it",
        thinking_mode="enabled",
        client=audio_client,
    )
    text_client = _Client()
    report = run_local_understanding_predictions(
        tmp_path / "run" / "manifest" / "manifest.jsonl",
        tmp_path / "text_out" / "predictions.jsonl",
        model="gemma-4-12B-it",
        thinking_mode="enabled",
        input_mode="transcript",
        client=text_client,
    )

    audio_call = audio_client.chat.completions.calls[0]
    text_call = text_client.chat.completions.calls[0]
    # Control invariant: identical system prompt; only the user content differs.
    assert audio_call["messages"][0] == text_call["messages"][0]
    assert text_call["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
    user_messages = [m for m in text_call["messages"] if m["role"] == "user"]
    assert len(user_messages) == 1
    (text_part,) = user_messages[0]["content"]
    assert text_part["type"] == "text"
    assert "L1 Mara -> assistant: Book the noon slot." in text_part["text"]
    assert all(
        part.get("type") != "audio_url"
        for message in text_call["messages"]
        if isinstance(message.get("content"), list)
        for part in message["content"]
    )
    assert report["provider"] == "local_text"
    prediction = read_jsonl(tmp_path / "text_out" / "predictions.jsonl", PredictionRow)[0]
    assert prediction.metadata["provider"] == "local_text"
    assert prediction.metadata["input_mode"] == "transcript"
