"""Eval replay of a timeline benchmark row: audio_history drives history_messages, in the row's history style."""

from __future__ import annotations

from pathlib import Path

import pytest

from ib.scoring.multiturn_audio import HISTORY_STYLES, history_messages

ROOT = Path("/run")


def _row(block_count: int, style: str | None = None) -> tuple[dict, list[Path]]:
    history = []
    for k in range(block_count):
        history.append({"role": "user", "audio": f"audio/cases/c__block_{k:02d}.wav", "block_index": k})
        if k < block_count - 1:
            history.append({"role": "assistant", "text": f"answer {k + 1}", "block_index": k})
    row = {
        "audio_history": history,
        "mixed_testcase_audio_paths": [e["audio"] for e in history if e["role"] == "user"],
        "case_mix": {"room": "audio/cases/c__room.wav", "heard": "audio/cases/c__heard.wav"},
        # Stale v1 columns must lose to audio_history.
        "user_turn_1_audio": "audio/perturbed/old.wav",
        "assistant_turn_1_transcript": "stale",
        **({"history_style": style} if style else {}),
    }
    return row, [ROOT / e["audio"] for e in history if e["role"] == "user"]


def test_text_turns_replays_blocks_and_assistant_text_in_order_beyond_eight_slots() -> None:
    row, paths = _row(10)
    messages = history_messages(row, paths)
    assert [m.role for m in messages] == ["user", "assistant"] * 9 + ["user"]
    assert [m.audio_path for m in messages if m.role == "user"] == paths
    assert [m.assistant_text for m in messages if m.role == "assistant"] == [f"answer {k}" for k in range(1, 10)]
    assert [m.final_user_turn for m in messages] == [False] * 18 + [True]
    assert "stale" not in {m.assistant_text for m in messages}


def test_single_mix_yields_one_audio_message_with_the_room_mix(monkeypatch) -> None:
    row, paths = _row(3, "single_mix")
    monkeypatch.setattr(Path, "exists", lambda self: self == ROOT / "audio/cases/c__room.wav")
    [message] = history_messages(row, paths)
    assert message.role == "user" and message.final_user_turn
    assert message.audio_path == ROOT / "audio/cases/c__room.wav"


def test_unknown_style_and_unresolved_block_raise() -> None:
    row, paths = _row(2, "heard_mix")
    with pytest.raises(ValueError, match="unknown history_style"):
        history_messages(row, paths)
    row, paths = _row(2)
    with pytest.raises(ValueError, match="not among the row's resolved audio paths"):
        history_messages(row, paths[:1])


def test_registry_lists_both_styles() -> None:
    assert set(HISTORY_STYLES) == {"text_turns", "single_mix"}


def test_rows_without_audio_history_keep_the_wide_column_walk() -> None:
    row = {"user_turn_1_audio": "a.wav", "assistant_turn_1_transcript": "hi", "user_turn_2_audio": "b.wav"}
    messages = history_messages(row, [ROOT / "a.wav", ROOT / "b.wav"])
    assert [(m.role, m.final_user_turn) for m in messages] == [("user", False), ("assistant", False), ("user", True)]
