"""turnbased_speak_probe — render speak_time_probe rows for turn-based barge-in eval.

Calling spec:
    is_speak_probe_row(row) -> bool
    barge_in_probe_row(row) -> dict

Inputs are built speak_time_probe manifest rows (relative audio paths). Output is
a transformed copy of the row that any turn-based understanding provider can
render: the truncated ``logic_standard_answer`` becomes the trailing assistant
turn (the utterance the assistant had spoken when barge-in cut it off) and the
live bystander event audio becomes the new final user turn. The transform is
deterministic, has no side effects, and raises ValueError when a row lacks the
required probe material.

Design note: the truncation point is a benchmark parameter, not a timing
reconstruction. A timing-faithful prefix (~0.8s of speech) would carry no
commitment and the probe would collapse into hear-time; the prefix must contain
the in-flight commitment whose preservation is being measured.
"""

from __future__ import annotations

from typing import Any

from ib.scoring.probe_sampling import truncate_fraction_for_row

SPEAK_PROBE_RENDER_FIELD = "logic_speak_probe_render"
BARGE_IN_VARIANT = "barge_in_resume"
BARGE_IN_POLICY_VERSION = "ib.speak_probe.barge_in.v3"
# Prefix ends with this marker; the model emits only the continuation after it,
# which is joined back to the prefix and judged against the base rubric.
STOPPED_MARKER = "<STOPPED>"
_MAX_TURN_INDEX = 8
_AUDIO_LIST_FIELDS = ("audio_paths", "mixed_testcase_audio_paths")


def is_speak_probe_row(row: dict[str, Any]) -> bool:
    """Return whether the manifest row is a dedicated speak_time_probe row."""
    return row.get("logic_case_mode") == "speak_time_probe"


def barge_in_probe_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a speak_time_probe row rendered for turn-based barge-in."""
    if not is_speak_probe_row(row):
        raise ValueError("barge_in_probe_row requires logic_case_mode=speak_time_probe")
    cell_id = row.get("cell_id")
    live_paths = _live_event_paths(row)
    prefix = _truncated_standard_answer(row)
    updated = dict(row)
    anchor_turn = _last_user_turn_index(updated)
    if anchor_turn is None:
        raise ValueError(f"speak_time_probe row {cell_id!r} has no user audio turns")
    if _clean(updated.get(f"assistant_turn_{anchor_turn}_transcript")) is not None:
        raise ValueError(
            f"speak_time_probe row {cell_id!r} already has an assistant turn "
            f"after its final user turn"
        )
    updated[f"assistant_turn_{anchor_turn}_transcript"] = prefix
    _place_live_event_turns(updated, anchor_turn, live_paths)
    for field in _AUDIO_LIST_FIELDS:
        values = updated.get(field)
        if isinstance(values, list):
            updated[field] = list(values) + list(live_paths)
    updated[SPEAK_PROBE_RENDER_FIELD] = {
        "variant": BARGE_IN_VARIANT,
        "policy_version": BARGE_IN_POLICY_VERSION,
        "truncated_assistant_text": prefix,
        "truncate_fraction": _truncate_fraction(row),
        "interrupted_turn_index": anchor_turn,
    }
    return updated


def _live_event_paths(row: dict[str, Any]) -> list[str]:
    paths = row.get("logic_live_event_audio_paths")
    if not isinstance(paths, list) or not paths:
        raise ValueError(
            f"speak_time_probe row {row.get('cell_id')!r} has no live event audio"
        )
    cleaned = [path for path in paths if isinstance(path, str) and path]
    if not cleaned:
        raise ValueError(
            f"speak_time_probe row {row.get('cell_id')!r} has invalid live event audio paths"
        )
    return cleaned


def _truncated_standard_answer(row: dict[str, Any]) -> str:
    answer = _clean(row.get("logic_standard_answer"))
    if answer is None:
        raise ValueError(
            f"speak_time_probe row {row.get('cell_id')!r} has no logic_standard_answer"
        )
    return _truncate_utterance(answer, _truncate_fraction(row))


def _truncate_fraction(row: dict[str, Any]) -> float:
    # Benchmark parameter, not a timing reconstruction: sampled per case (built
    # anchors carry a uniform builder-default overlap_fraction that never
    # encoded per-case intent). See probe_sampling for the override field.
    return truncate_fraction_for_row(row)


def _truncate_utterance(text: str, fraction: float) -> str:
    """Cut an utterance at a word boundary (character boundary for unspaced text).

    The prefix ends with the ``<STOPPED>`` marker: the model is instructed to
    emit only the continuation from that point, which is joined back to this
    prefix and judged against the base rubric.
    """
    words = text.split()
    if len(words) >= 6:
        keep = min(max(round(len(words) * fraction), 3), len(words) - 1)
        return " ".join(words[:keep]) + " " + STOPPED_MARKER
    keep = min(max(round(len(text) * fraction), 4), len(text) - 1)
    return text[:keep].rstrip() + " " + STOPPED_MARKER


def _last_user_turn_index(row: dict[str, Any]) -> int | None:
    last = None
    for turn_index in range(1, _MAX_TURN_INDEX + 1):
        value = row.get(f"user_turn_{turn_index}_audio")
        if isinstance(value, str) and value:
            last = turn_index
    return last


def _place_live_event_turns(
    row: dict[str, Any], anchor_turn: int, live_paths: list[str]
) -> None:
    """Bind live event clips to explicit user turns after the interrupted one.

    Clips that do not fit in the 8-turn field window stay unbound; providers
    append unreferenced audio after the transcribed turns, which preserves
    chronological order.
    """
    turn_index = anchor_turn + 1
    for path in live_paths:
        if turn_index > _MAX_TURN_INDEX:
            break
        row[f"user_turn_{turn_index}_audio"] = path
        row[f"user_turn_{turn_index}_transcript"] = None
        turn_index += 1


def _clean(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None
