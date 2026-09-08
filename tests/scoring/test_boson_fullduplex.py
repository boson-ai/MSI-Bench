"""Boson Realtime full-duplex speak-probe prediction generation tests."""

from __future__ import annotations

import json
import wave
from io import BytesIO
from pathlib import Path

import base64

import pytest

import websockets.sync.client

from ib.io import read_jsonl
from ib.predictions import PredictionRow
from ib.scoring.boson_fullduplex import run_boson_fullduplex_predictions
from ib.scoring.openai_understanding import UnderstandingProviderError


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


class _FakeBosonSocket:
    """Scripted Boson websocket: records sends, replays queued server events."""

    def __init__(self, events: list[dict]) -> None:
        self.events = list(events)
        self.sent: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb

    def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))

    def recv(self, timeout=None):
        del timeout
        if not self.events:
            raise AssertionError("fake Boson event stream exhausted")
        return json.dumps(self.events.pop(0))


def _speak_probe_manifest(tmp_path: Path) -> Path:
    run_root = tmp_path / "run"
    (run_root / "manifest").mkdir(parents=True)
    (run_root / "audio").mkdir()
    (run_root / "audio" / "final.wav").write_bytes(_wav_bytes())
    (run_root / "audio" / "live.wav").write_bytes(_wav_bytes())
    manifest = run_root / "manifest" / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "cell_id": "c1s",
                "logic_case_mode": "speak_time_probe",
                "expected_action": "respond",
                "mixed_testcase_audio_paths": ["audio/final.wav"],
                "logic_live_expansion_type": "bystander_interference_suppression",
                "logic_speak_inject_offset_s": 1.0,
                "logic_live_event_audio_paths": ["audio/live.wav"],
                "logic_expected_tool_calls": [
                    {"name": "修改预订时段", "arguments": {"time": "19:00"}}
                ],
                "logic_available_functions": [
                    {"name": "修改预订时段", "arguments": {"time": "string_or_null"}}
                ],
            }
        ],
    )
    return manifest


def test_boson_speak_probe_injects_and_maps_tool_calls(tmp_path, monkeypatch) -> None:
    manifest = _speak_probe_manifest(tmp_path)
    output_audio = base64.b64encode((b"\x00\x00") * int(24_000 * 1.5)).decode("ascii")
    socket = _FakeBosonSocket(
        [
            {"type": "input_audio_buffer.committed"},
            {"type": "response.created", "response": {"id": "live"}},
            {"type": "response.output_audio.delta", "delta": output_audio},
            {"type": "response.output_audio_transcript.done", "transcript": "马上帮您改到七点。"},
            {
                "type": "response.done",
                "response": {
                    "id": "live",
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "name": "tool_1",
                            "arguments": json.dumps({"time": "19:00"}),
                        }
                    ],
                    "usage": {"input_tokens": 40, "output_tokens": 9, "total_tokens": 49},
                },
            },
        ]
    )
    monkeypatch.setenv("BOSONAI_API_KEY", "test-key")
    monkeypatch.setattr(websockets.sync.client, "connect", lambda *args, **kwargs: socket)

    report = run_boson_fullduplex_predictions(
        manifest,
        tmp_path / "predictions.jsonl",
        model="higgs-realtime",
        live_probe=True,
    )

    rows = read_jsonl(tmp_path / "predictions.jsonl", PredictionRow)
    assert report["live_probe_count"] == 1
    assert report["predicted_count"] == 1
    assert rows[0].cell_id == "c1s"
    assert rows[0].predicted_action.value == "respond"
    assert rows[0].tool_calls[0].name == "修改预订时段"
    live_probe = rows[0].metadata["live_probe"]
    assert live_probe["executed"] is True
    assert live_probe["injection_started"] is True
    assert live_probe["injection_scheduled_offset_s"] == 1.0
    assert live_probe["assistant_audio_ms_before_injection"] == 1500
    assert live_probe["interrupted"] is False
    assert live_probe["followed_bystander"] is False
    assert live_probe["tool_calls"] == [{"name": "修改预订时段", "arguments": {"time": "19:00"}}]

    session = next(m for m in socket.sent if m["type"] == "session.update")["session"]
    assert session["tools"][0]["name"] == "tool_1"
    assert session["audio"]["input"] == {"turn_detection": None}
    assert "emit_prediction" not in json.dumps(session)
    commits = [m for m in socket.sent if m["type"] == "input_audio_buffer.commit"]
    assert len(commits) == 2  # history turn + live injection
    creates = [m for m in socket.sent if m["type"] == "response.create"]
    assert len(creates) == 1

    raw_rows = [
        json.loads(line)
        for line in (tmp_path / "raw_responses.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["phase"] for row in raw_rows] == ["live_probe"]


def test_boson_speak_probe_requires_live_probe_flag(tmp_path, monkeypatch) -> None:
    manifest = _speak_probe_manifest(tmp_path)
    monkeypatch.setenv("BOSONAI_API_KEY", "test-key")

    with pytest.raises(UnderstandingProviderError, match="require --live-probe"):
        run_boson_fullduplex_predictions(
            manifest,
            tmp_path / "predictions.jsonl",
            model="higgs-realtime",
            live_probe=False,
        )
