"""Tests for turn-based barge-in rendering of speak_time_probe rows."""

from pathlib import Path

import pytest

from ib.scoring.multiturn_audio import history_messages, setup_prompt_for
from ib.scoring.turnbased_speak_probe import (
    SPEAK_PROBE_RENDER_FIELD,
    barge_in_probe_row,
    is_speak_probe_row,
)


def _probe_row() -> dict:
    return {
        "cell_id": "scene-pattern-0001-000s",
        "logic_case_mode": "speak_time_probe",
        "logic_standard_answer": (
            "Got it, I'm removing the ninety-five dollar spa charge from Dana's "
            "account and swapping the medium robe for a large right away."
        ),
        "logic_live_anchor": {"overlap_fraction": 0.45, "start_offset_seconds": 0.8},
        "logic_live_event_audio_paths": ["audio/perturbed/case__line_20000.wav"],
        "audio_paths": [
            "audio/perturbed/case__line_00.wav",
            "audio/perturbed/case__line_06.wav",
        ],
        "mixed_testcase_audio_paths": [
            "audio/perturbed/case__line_00.wav",
            "audio/perturbed/case__line_06.wav",
        ],
        "user_turn_1_audio": "audio/perturbed/case__line_00.wav",
        "user_turn_1_transcript": "Assistant, remove the spa charge.",
        "user_turn_2_audio": "audio/perturbed/case__line_06.wav",
        "user_turn_2_transcript": "And swap the robe, the shuttle is waiting.",
    }


def test_barge_in_probe_row_places_prefix_and_live_event() -> None:
    row = barge_in_probe_row(_probe_row())
    prefix = row["assistant_turn_2_transcript"]
    assert prefix.endswith(" <STOPPED>")
    assert prefix != row["logic_standard_answer"]
    assert row["logic_standard_answer"].startswith(prefix[: -len(" <STOPPED>")])
    assert row["user_turn_3_audio"] == "audio/perturbed/case__line_20000.wav"
    assert row["audio_paths"][-1] == "audio/perturbed/case__line_20000.wav"
    assert row["mixed_testcase_audio_paths"][-1] == "audio/perturbed/case__line_20000.wav"
    render = row[SPEAK_PROBE_RENDER_FIELD]
    assert render["variant"] == "barge_in_resume"
    assert render["interrupted_turn_index"] == 2
    assert 0.1 <= render["truncate_fraction"] <= 0.5


def test_truncate_fraction_samples_per_cell_and_clamps_dedicated_override() -> None:
    base = _probe_row()
    # The builder anchor's uniform overlap_fraction must not steer truncation:
    # the fraction is a stable per-cell sample in [0.1, 0.5].
    first = barge_in_probe_row(base)[SPEAK_PROBE_RENDER_FIELD]["truncate_fraction"]
    assert 0.1 <= first <= 0.5
    assert barge_in_probe_row(base)[SPEAK_PROBE_RENDER_FIELD]["truncate_fraction"] == first
    other = dict(base, cell_id="scene-pattern-0001-001s")
    assert barge_in_probe_row(other)[SPEAK_PROBE_RENDER_FIELD]["truncate_fraction"] != first
    low = dict(base, logic_speak_truncate_fraction=0.01)
    assert barge_in_probe_row(low)[SPEAK_PROBE_RENDER_FIELD]["truncate_fraction"] == 0.1
    high = dict(base, logic_speak_truncate_fraction=0.9)
    assert barge_in_probe_row(high)[SPEAK_PROBE_RENDER_FIELD]["truncate_fraction"] == 0.5
    mid = dict(base, logic_speak_truncate_fraction=0.42)
    assert barge_in_probe_row(mid)[SPEAK_PROBE_RENDER_FIELD]["truncate_fraction"] == 0.42


def test_barge_in_probe_row_keeps_expected_fields_untouched() -> None:
    source = _probe_row()
    source["logic_expected_tool_calls"] = [{"name": "remove_folio_charge", "arguments": {}}]
    row = barge_in_probe_row(source)
    assert row["logic_expected_tool_calls"] == source["logic_expected_tool_calls"]
    assert row["logic_case_mode"] == "speak_time_probe"
    assert source.get(SPEAK_PROBE_RENDER_FIELD) is None


def test_history_messages_order_ends_with_truncated_turn_then_live_event() -> None:
    row = barge_in_probe_row(_probe_row())
    audio_paths = [Path(value) for value in row["mixed_testcase_audio_paths"]]
    messages = history_messages(row, audio_paths)
    roles = [message.role for message in messages]
    assert roles == ["user", "user", "assistant", "user"]
    assert messages[2].assistant_text.endswith(" <STOPPED>")
    assert messages[3].audio_path == Path("audio/perturbed/case__line_20000.wav")
    assert messages[3].final_user_turn
    assert not messages[1].final_user_turn


def test_unspaced_answer_truncates_by_characters() -> None:
    source = _probe_row()
    source["logic_standard_answer"] = "好的我现在就把九十五美元的水疗费用从账单里移除并更换浴袍尺码"
    row = barge_in_probe_row(source)
    prefix = row["assistant_turn_2_transcript"]
    assert prefix.endswith(" <STOPPED>")
    assert 4 <= len(prefix) - 1 < len(source["logic_standard_answer"])


def test_setup_prompt_mentions_barge_in_only_for_probe_rows() -> None:
    row = barge_in_probe_row(_probe_row())
    assert "Barge-in context" in setup_prompt_for(row)
    assert "Barge-in context" not in setup_prompt_for(_probe_row())


def test_rejects_rows_without_probe_material() -> None:
    assert not is_speak_probe_row({"logic_case_mode": "base_semantic"})
    with pytest.raises(ValueError):
        barge_in_probe_row({"logic_case_mode": "base_semantic"})
    missing_answer = _probe_row()
    missing_answer["logic_standard_answer"] = None
    with pytest.raises(ValueError):
        barge_in_probe_row(missing_answer)
    missing_live = _probe_row()
    missing_live["logic_live_event_audio_paths"] = []
    with pytest.raises(ValueError):
        barge_in_probe_row(missing_live)
    occupied_slot = _probe_row()
    occupied_slot["assistant_turn_2_transcript"] = "Already answered."
    with pytest.raises(ValueError):
        barge_in_probe_row(occupied_slot)
