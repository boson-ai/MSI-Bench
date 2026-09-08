"""OpenAI Realtime full-duplex prediction generation tests."""

from __future__ import annotations

import base64
import json
import wave
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from ib.io import read_jsonl
from ib.predictions import PredictionRow
from ib.scoring.openai_fullduplex import _realtime_tools, run_openai_fullduplex_predictions


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


def test_realtime_tools_use_canonical_strict_schema() -> None:
    tools, name_map = _realtime_tools(
        [
            {
                "name": "set_count",
                "arguments": {"count": "integer", "note": "string_or_null"},
            }
        ]
    )

    assert name_map == {"set_count": "set_count"}
    assert tools[0]["parameters"]["additionalProperties"] is False
    assert tools[0]["parameters"]["required"] == ["count"]
    assert tools[0]["parameters"]["properties"]["count"] == {"type": "integer"}


def _ns(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _ns(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_ns(item) for item in value]
    return value


class _FakeWebsocket:
    def __init__(self, events: list[dict]) -> None:
        self.events = list(events)

    def recv(self, *, timeout=None, decode=False):
        del timeout, decode
        if not self.events:
            raise AssertionError("fake realtime event stream exhausted")
        return json.dumps(self.events.pop(0)).encode("utf-8")


class _Session:
    def __init__(self, conn) -> None:
        self.conn = conn

    def update(self, **kwargs) -> None:
        self.conn.sent.append(("session.update", kwargs))


class _Item:
    def __init__(self, conn) -> None:
        self.conn = conn

    def create(self, **kwargs) -> None:
        self.conn.sent.append(("conversation.item.create", kwargs))


class _Conversation:
    def __init__(self, conn) -> None:
        self.item = _Item(conn)


class _Response:
    def __init__(self, conn) -> None:
        self.conn = conn

    def create(self, **kwargs) -> None:
        self.conn.sent.append(("response.create", kwargs))


class _InputAudioBuffer:
    def __init__(self, conn) -> None:
        self.conn = conn

    def append(self, **kwargs) -> None:
        self.conn.sent.append(("input_audio_buffer.append", kwargs))

    def commit(self, **kwargs) -> None:
        self.conn.sent.append(("input_audio_buffer.commit", kwargs))


class _Connection:
    def __init__(self, events: list[dict]) -> None:
        self._connection = _FakeWebsocket(events)
        self.sent: list[tuple[str, dict]] = []
        self.session = _Session(self)
        self.conversation = _Conversation(self)
        self.response = _Response(self)
        self.input_audio_buffer = _InputAudioBuffer(self)

    def parse_event(self, raw: bytes):
        return _ns(json.loads(raw))


class _ConnectManager:
    def __init__(self, conn: _Connection) -> None:
        self.conn = conn

    def __enter__(self):
        return self.conn

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb


class _Realtime:
    def __init__(self, scripts: list[list[dict]]) -> None:
        self.scripts = list(scripts)
        self.connections: list[_Connection] = []

    def connect(self, **kwargs):
        del kwargs
        conn = _Connection(self.scripts.pop(0))
        self.connections.append(conn)
        return _ConnectManager(conn)


class _Client:
    def __init__(self, scripts: list[list[dict]]) -> None:
        self.realtime = _Realtime(scripts)


@pytest.mark.parametrize("reasoning_effort", ["medium", "xhigh"])
def test_openai_fullduplex_generates_base_prediction_and_live_probe(
    tmp_path, reasoning_effort
) -> None:
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
                "cell_id": "c1",
                "expected_action": "respond",
                "mixed_testcase_audio_paths": ["audio/final.wav"],
                "logic_live_expansion_type": "bystander_interference_suppression",
                "logic_live_expansion_types": ["bystander_interference_suppression"],
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
    output_audio = base64.b64encode((b"\x00\x00") * int(24_000 * 1.5)).decode("ascii")
    client = _Client(
        [
            [
                {
                    "type": "response.function_call_arguments.done",
                    "name": "emit_prediction",
                    "arguments": json.dumps(
                        {
                            "predicted_action": "respond",
                            "answer_text": "已为您修改预订时段。",
                            "tool_calls": [
                                {"name": "修改预订时段", "arguments": {"time": "19:00"}}
                            ],
                            "confidence": 0.9,
                        },
                        ensure_ascii=False,
                    ),
                },
                {"type": "response.done", "response": {"id": "base", "status": "completed"}},
            ],
            [
                {"type": "response.audio.delta", "delta": output_audio},
                {"type": "response.audio_transcript.done", "transcript": "已为您修改预订时段。"},
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
                    },
                },
            ],
        ]
    )

    report = run_openai_fullduplex_predictions(
        manifest,
        tmp_path / "predictions.jsonl",
        model="gpt-realtime-2",
        client=client,
        live_probe=True,
        reasoning_effort=reasoning_effort,
    )

    rows = read_jsonl(tmp_path / "predictions.jsonl", PredictionRow)
    raw_rows = [
        json.loads(line)
        for line in (tmp_path / "raw_responses.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    base_conn = client.realtime.connections[0]
    live_conn = client.realtime.connections[1]
    assert report["provider"] == "openai_realtime"
    assert report["raw_responses"] == str(tmp_path / "raw_responses.jsonl")
    assert [row["phase"] for row in raw_rows] == ["base_prediction", "live_probe"]
    assert raw_rows[0]["response"]["events"][-1]["type"] == "response.done"
    assert report["reasoning_effort"] == reasoning_effort
    assert report["live_probe_count"] == 1
    assert rows[0].predicted_action.value == "respond"
    assert rows[0].metadata["eval_mode"] == "fullduplex"
    assert rows[0].metadata["reasoning_effort"] == reasoning_effort
    assert rows[0].metadata["prompts"]["source"] == "captured_at_generation"
    assert rows[0].metadata["prompts"]["system"] == base_conn.sent[0][1]["session"]["instructions"]
    assert "confidence" not in rows[0].metadata["prompts"]["system"]
    assert rows[0].metadata["prompts"]["request_rendering"] == ""
    assert (
        rows[0].metadata["prompts"]["live_system"]
        == live_conn.sent[0][1]["session"]["instructions"]
    )
    assert "confidence" not in rows[0].metadata["prompts"]["live_system"]
    assert "benchmark" not in rows[0].metadata["prompts"]["live_system"].lower()
    assert "benchmark" not in rows[0].metadata["prompts"]["live_request_rendering"].lower()
    assert rows[0].metadata["live_probe"]["executed"] is True
    assert rows[0].metadata["live_probe"]["predicted_action"] == "respond"
    assert rows[0].metadata["live_probe"]["tool_calls"] == [
        {"name": "修改预订时段", "arguments": {"time": "19:00"}}
    ]
    assert base_conn.sent[0][1]["session"]["type"] == "realtime"
    assert base_conn.sent[0][1]["session"]["output_modalities"] == ["text"]
    assert base_conn.sent[0][1]["session"]["reasoning"] == {"effort": reasoning_effort}
    assert base_conn.sent[0][1]["session"]["tools"][0]["name"] == "emit_prediction"
    assert base_conn.sent[-1][1]["response"]["tool_choice"]["name"] == "emit_prediction"
    assert base_conn.sent[-1][1]["response"]["max_output_tokens"] == (
        4_096 if reasoning_effort == "xhigh" else 700
    )
    assert len(base_conn.sent[-1][1]["response"]["tools"]) == 1
    assert base_conn.sent[1][1]["item"]["role"] == "user"
    assert base_conn.sent[1][1]["item"]["content"][0]["type"] == "input_audio"
    assert live_conn.sent[0][1]["session"]["output_modalities"] == ["audio"]
    assert live_conn.sent[0][1]["session"]["reasoning"] == {"effort": reasoning_effort}
    assert live_conn.sent[1][1]["item"]["role"] == "user"
    assert live_conn.sent[1][1]["item"]["content"][0]["type"] == "input_audio"
    assert live_conn.sent[0][1]["session"]["tools"][0]["name"] == "tool_1"
    assert any(name == "input_audio_buffer.append" for name, _kwargs in live_conn.sent)
    assert any(name == "input_audio_buffer.commit" for name, _kwargs in live_conn.sent)


def test_openai_fullduplex_writes_error_prediction_for_invalid_json(tmp_path) -> None:
    run_root = tmp_path / "run"
    (run_root / "manifest").mkdir(parents=True)
    (run_root / "audio").mkdir()
    (run_root / "audio" / "final.wav").write_bytes(_wav_bytes())
    manifest = run_root / "manifest" / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "cell_id": "c1",
                "expected_action": "respond",
                "mixed_testcase_audio_paths": ["audio/final.wav"],
            }
        ],
    )
    invalid_script = [
        {
            "type": "response.text.done",
            "text": 'I should ignore this. {"note":"not a prediction"}',
        },
        {"type": "response.done", "response": {"id": "bad", "status": "completed"}},
    ]
    client = _Client([invalid_script, invalid_script, invalid_script])
    output = tmp_path / "predictions.jsonl"

    report = run_openai_fullduplex_predictions(
        manifest, output, model="gpt-realtime-test", client=client
    )

    rows = read_jsonl(output, PredictionRow)
    error_files = sorted((tmp_path / "prediction_errors").glob("c1__attempt_*.json"))
    raw_rows = [
        json.loads(line)
        for line in (tmp_path / "raw_responses.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert report["predicted_count"] == 1
    assert [row["attempt"] for row in raw_rows] == [1, 2, 3]
    assert all(row["response"]["events"][-1]["type"] == "response.done" for row in raw_rows)
    assert rows[0].cell_id == "c1"
    assert rows[0].predicted_action.value == "silent"
    assert rows[0].confidence == 0.0
    assert rows[0].metadata["provider_error"]["type"] == "prediction_format_error"
    assert len(error_files) == 3
    error_payload = json.loads(error_files[0].read_text(encoding="utf-8"))
    assert error_payload["cell_id"] == "c1"
    assert error_payload["text"] == 'I should ignore this. {"note":"not a prediction"}'


def test_openai_fullduplex_skips_live_probe_by_default(tmp_path) -> None:
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
                "cell_id": "c1",
                "expected_action": "respond",
                "mixed_testcase_audio_paths": ["audio/final.wav"],
                "logic_live_expansion_type": "bystander_interference_suppression",
                "logic_live_event_audio_paths": ["audio/live.wav"],
            }
        ],
    )
    client = _Client(
        [
            [
                {
                    "type": "response.text.done",
                    "text": json.dumps(
                        {"predicted_action": "respond", "answer_text": "ok", "tool_calls": []}
                    ),
                },
                {"type": "response.done", "response": {"id": "base", "status": "completed"}},
            ]
        ]
    )

    report = run_openai_fullduplex_predictions(
        manifest, tmp_path / "predictions.jsonl", model="gpt-live-1.5", client=client
    )

    rows = read_jsonl(tmp_path / "predictions.jsonl", PredictionRow)
    assert report["live_probe_enabled"] is False
    assert report["live_probe_count"] == 0
    assert len(client.realtime.connections) == 1
    assert "live_probe" not in rows[0].metadata
