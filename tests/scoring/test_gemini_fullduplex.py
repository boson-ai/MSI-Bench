"""Gemini Live full-duplex prediction generation tests."""

from __future__ import annotations

import asyncio
import json
import wave
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from ib.io import read_jsonl
from ib.predictions import PredictionRow
from ib.scoring.gemini_fullduplex import (
    GeminiLiveCapture,
    _append_function_call,
    _gemini_tools,
    run_gemini_fullduplex_predictions,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _wav_bytes(duration_samples: int = 32) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24_000)
        handle.writeframes((b"\x00\x00") * duration_samples)
    return buffer.getvalue()


def test_gemini_live_tools_use_canonical_strict_schema() -> None:
    tools = _gemini_tools(
        [
            {
                "name": "set_count",
                "arguments": {"count": "integer", "note": "string_or_null"},
            }
        ]
    )

    parameters = tools[0]["function_declarations"][0]["parameters_json_schema"]
    assert parameters["additionalProperties"] is False
    assert parameters["required"] == ["count"]
    assert parameters["properties"]["count"] == {"type": "integer"}


def test_gemini_live_restores_sanitized_function_names() -> None:
    capture = GeminiLiveCapture()

    _append_function_call(
        capture,
        SimpleNamespace(name="tool_1", args={"time": "19:00"}),
        tool_name_map={"tool_1": "修改预订时段"},
    )

    assert capture.tool_calls == [{"name": "修改预订时段", "arguments": {"time": "19:00"}}]


def _message(text: str, *, turn_complete: bool = True):
    return SimpleNamespace(
        text=None,
        usage_metadata=None,
        tool_call=None,
        server_content=SimpleNamespace(
            interrupted=False,
            turn_complete=turn_complete,
            output_transcription=None,
            model_turn=SimpleNamespace(
                parts=[SimpleNamespace(text=text, inline_data=None, function_call=None)]
            ),
        ),
    )


def _message_with_thought(thought: str, final_text: str):
    return SimpleNamespace(
        text=f"{thought}\n{final_text}",
        usage_metadata=None,
        tool_call=None,
        server_content=SimpleNamespace(
            interrupted=False,
            turn_complete=True,
            output_transcription=None,
            model_turn=SimpleNamespace(
                parts=[
                    SimpleNamespace(
                        text=thought,
                        thought=True,
                        inline_data=None,
                        function_call=None,
                    ),
                    SimpleNamespace(
                        text=final_text,
                        thought=False,
                        inline_data=None,
                        function_call=None,
                    ),
                ]
            ),
        ),
    )


class _Session:
    def __init__(self, messages: list[object], *, never_end: bool = False) -> None:
        self.messages = messages
        self.never_end = never_end
        self.sent = []

    async def send_client_content(self, **kwargs) -> None:
        self.sent.append(("send_client_content", kwargs))

    async def send_realtime_input(self, **kwargs) -> None:
        self.sent.append(("send_realtime_input", kwargs))

    async def receive(self):
        for message in self.messages:
            yield message
        if self.never_end:
            await asyncio.sleep(3600)
        return


class _ConnectManager:
    def __init__(self, session: _Session) -> None:
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb


class _Live:
    def __init__(self, sessions: list[_Session]) -> None:
        self.sessions = list(sessions)
        self.calls: list[dict] = []

    def connect(self, **kwargs):
        self.calls.append(kwargs)
        return _ConnectManager(self.sessions.pop(0))


class _Aio:
    def __init__(self, sessions: list[_Session]) -> None:
        self.live = _Live(sessions)


class _Client:
    def __init__(self, sessions: list[_Session]) -> None:
        self.aio = _Aio(sessions)


def _manifest(tmp_path: Path, rows: list[dict]) -> Path:
    run_root = tmp_path / "run"
    (run_root / "manifest").mkdir(parents=True)
    (run_root / "audio").mkdir()
    (run_root / "audio" / "final.wav").write_bytes(_wav_bytes())
    manifest = run_root / "manifest" / "manifest.jsonl"
    _write_jsonl(manifest, rows)
    return manifest


def test_gemini_live_stops_on_turn_complete_without_stream_close(tmp_path) -> None:
    manifest = _manifest(
        tmp_path,
        [
            {
                "cell_id": "c1",
                "expected_action": "respond",
                "mixed_testcase_audio_paths": ["audio/final.wav"],
            }
        ],
    )
    payload = json.dumps(
        {
            "predicted_action": "respond",
            "answer_text": "Done.",
            "tool_calls": [],
            "confidence": 0.8,
        }
    )
    client = _Client([_Session([_message(payload, turn_complete=True)], never_end=True)])

    report = run_gemini_fullduplex_predictions(
        manifest,
        tmp_path / "predictions.jsonl",
        model="gemini-3.1-flash-live-preview",
        client=client,
        recv_timeout_s=0.01,
        thinking_mode="enabled",
    )

    rows = read_jsonl(tmp_path / "predictions.jsonl", PredictionRow)
    raw_row = json.loads((tmp_path / "raw_responses.jsonl").read_text(encoding="utf-8"))
    assert report["predicted_count"] == 1
    assert report["raw_responses"] == str(tmp_path / "raw_responses.jsonl")
    assert raw_row["phase"] == "base_prediction"
    assert raw_row["response"]["messages"][0]["server_content"]["turn_complete"] is True
    assert report["thinking_mode"] == "enabled"
    assert rows[0].predicted_action.value == "respond"
    assert rows[0].answer_text == "Done."
    assert rows[0].metadata["prompts"]["source"] == "captured_at_generation"
    assert (
        "voice assistant participating in a multi-party conversation"
        in rows[0].metadata["prompts"]["system"]
    )
    assert (
        "The assistant already has this conversation context"
        in rows[0].metadata["prompts"]["system"]
    )
    assert "confidence" not in rows[0].metadata["prompts"]["system"]
    assert rows[0].metadata["prompts"]["request_rendering"] == ""
    assert "benchmark" not in rows[0].metadata["prompts"]["system"].lower()
    assert rows[0].metadata["thinking_mode"] == "enabled"
    connect_config = client.aio.live.calls[0]["config"]
    assert connect_config["thinking_config"] == {"include_thoughts": True}
    assert connect_config["max_output_tokens"] == 4096


def test_gemini_live_excludes_thought_from_prediction_but_keeps_it_raw(tmp_path) -> None:
    manifest = _manifest(
        tmp_path,
        [
            {
                "cell_id": "c1",
                "expected_action": "respond",
                "mixed_testcase_audio_paths": ["audio/final.wav"],
            }
        ],
    )
    thought = "I should check the account authority first."
    payload = json.dumps({"predicted_action": "respond", "answer_text": "Done."})
    client = _Client([_Session([_message_with_thought(thought, payload)])])

    run_gemini_fullduplex_predictions(
        manifest,
        tmp_path / "predictions.jsonl",
        model="gemini-live-test",
        client=client,
        thinking_mode="enabled",
    )

    prediction = read_jsonl(tmp_path / "predictions.jsonl", PredictionRow)[0]
    raw_row = json.loads((tmp_path / "raw_responses.jsonl").read_text(encoding="utf-8"))
    raw_parts = raw_row["response"]["messages"][0]["server_content"]["model_turn"]["parts"]
    assert raw_parts[0]["text"] == thought
    assert raw_parts[0]["thought"] is True
    assert prediction.answer_text == "Done."


def test_gemini_live_plain_variant_disables_thoughts(tmp_path) -> None:
    manifest = _manifest(
        tmp_path,
        [
            {
                "cell_id": "c1",
                "expected_action": "respond",
                "mixed_testcase_audio_paths": ["audio/final.wav"],
            }
        ],
    )
    payload = json.dumps({"predicted_action": "respond", "answer_text": "Done."})
    client = _Client([_Session([_message(payload)])])

    report = run_gemini_fullduplex_predictions(
        manifest,
        tmp_path / "predictions.jsonl",
        model="gemini-3.1-flash-live-preview",
        client=client,
        thinking_mode="disabled",
    )

    config = client.aio.live.calls[0]["config"]
    assert report["thinking_mode"] == "disabled"
    assert config["thinking_config"] == {"include_thoughts": False}
    assert config["max_output_tokens"] == 1400


def test_gemini_live_writes_timeout_prediction_and_continues(tmp_path) -> None:
    manifest = _manifest(
        tmp_path,
        [
            {
                "cell_id": "c1",
                "expected_action": "respond",
                "mixed_testcase_audio_paths": ["audio/final.wav"],
            },
            {
                "cell_id": "c2",
                "expected_action": "respond",
                "mixed_testcase_audio_paths": ["audio/final.wav"],
            },
        ],
    )
    payload = json.dumps(
        {
            "predicted_action": "respond",
            "answer_text": "Second done.",
            "tool_calls": [],
            "confidence": 0.8,
        }
    )
    client = _Client(
        [
            _Session([], never_end=True),
            _Session([_message(payload, turn_complete=True)]),
        ]
    )

    report = run_gemini_fullduplex_predictions(
        manifest,
        tmp_path / "predictions.jsonl",
        model="gemini-live-test",
        client=client,
        recv_timeout_s=0.01,
    )

    rows = read_jsonl(tmp_path / "predictions.jsonl", PredictionRow)
    assert report["predicted_count"] == 2
    assert rows[0].cell_id == "c1"
    assert rows[0].predicted_action.value == "silent"
    assert rows[0].metadata["provider_error"]["type"] == "gemini_live_timeout"
    assert rows[1].cell_id == "c2"
    assert rows[1].answer_text == "Second done."
