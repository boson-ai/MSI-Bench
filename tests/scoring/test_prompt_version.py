"""prompt_version selects the setup prompt and the Realtime output cap a row is evaluated with.

v1 is the MSI-Bench v1 paper's prompt, v2 the published v2 run's prompt, and a row without
the field keeps the current prompt (a v1 release cell id alone also means v1).
"""

from __future__ import annotations

import wave
from pathlib import Path

import pytest
from pydantic import ValidationError

import ib.scoring.openai_fullduplex as fd
from ib.scoring.multiturn_audio import setup_prompt_for
from ib.scoring.prompt_version import prompt_version_for

FUNCTIONS = [{"name": "prepare_wheelchair_zone",
              "arguments": {"door_area": "enum[front_door,rear_door]"}}]


def _row(**extra) -> dict:
    return {"cell_id": "commerce_service-scene_logic_constraints-1234-000",
            "logic_pattern": "distributed_parameters", "scene": "commerce_service",
            "logic_scene_date": "2026-03-20", "logic_available_functions": FUNCTIONS, **extra}


def test_v1_renders_the_v1_paper_setup_prompt():
    prompt = setup_prompt_for(_row(prompt_version="v1"))
    assert prompt.startswith("Assistant authorization policy:\n- None\n")
    assert "- scene assistant role: service assistant" in prompt
    assert "settled service setting constraints, include tool_calls with name and arguments." in prompt
    assert "- prepare_wheelchair_zone arguments: door_area: enum[front_door,rear_door]" in prompt
    assert "Users:" not in prompt


def test_v2_renders_the_current_setup_prompt():
    prompt = setup_prompt_for(_row(prompt_version="v2"))
    assert prompt.startswith("The assistant already has this conversation context.")
    assert "Users:" in prompt and "Available functions:\n[" in prompt
    assert "Assistant authorization policy" not in prompt and "scene assistant role" not in prompt


def test_absent_version_renders_exactly_the_current_prompt():
    assert setup_prompt_for(_row()) == setup_prompt_for(_row(prompt_version="v2"))
    assert prompt_version_for(_row()) is None


def test_absent_version_on_a_v1_release_cell_id_is_v1():
    v1_ids = ["msi-en-scope-commerce-00", "msi-zh-retr-workprof-95h", "msi-en-auth-edu-07s"]
    for cell_id in v1_ids:
        assert prompt_version_for({"cell_id": cell_id}) == "v1"
    assert setup_prompt_for(_row(cell_id=v1_ids[0])) == setup_prompt_for(_row(prompt_version="v1"))
    for cell_id in ["msi2-en-scope-commerce-00", "msi-en-scope-commerce-00x", "msi-en-other-edu-01"]:
        assert prompt_version_for({"cell_id": cell_id}) is None


def test_unknown_version_is_a_validation_error():
    with pytest.raises(ValidationError):
        setup_prompt_for(_row(prompt_version="v3"))


class _Connection:
    def __init__(self, sent: list) -> None:
        self.session = type("S", (), {"update": staticmethod(lambda **kw: None)})()
        self.conversation = type("C", (), {})()
        self.conversation.item = type("I", (), {"create": staticmethod(lambda **kw: None)})()
        self.response = type("R", (), {"create": staticmethod(lambda **kw: sent.append(kw))})()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


def _sent_cap(tmp_path: Path, monkeypatch, effort: str, **extra) -> int:
    wav = tmp_path / "turn.wav"
    with wave.open(str(wav), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(24_000)
        fh.writeframes(b"\x00\x00" * 2_400)
    sent: list = []
    client = type("Client", (), {})()
    client.realtime = type("RT", (), {"connect": staticmethod(lambda **kw: _Connection(sent))})()
    monkeypatch.setattr(fd, "_collect_response", lambda connection, **kw: fd.RealtimeCapture())
    provider = fd.OpenAiFullDuplexProvider(model="gpt-realtime-2.1", client=client,
                                           reasoning_effort=effort)
    row = _row(audio_paths=["turn.wav"], user_turn_1_audio="turn.wav", **extra)
    provider._base_capture(row, tmp_path / "manifest.jsonl", attempt=1)
    return sent[0]["response"]["max_output_tokens"]


def test_realtime_medium_cap_follows_the_version(tmp_path, monkeypatch):
    assert _sent_cap(tmp_path, monkeypatch, "medium", prompt_version="v1") == 700
    assert _sent_cap(tmp_path, monkeypatch, "xhigh", prompt_version="v1") == 4096
    assert _sent_cap(tmp_path, monkeypatch, "medium", prompt_version="v2") == 4096
    assert _sent_cap(tmp_path, monkeypatch, "medium") == 4096
