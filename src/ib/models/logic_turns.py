"""logic_turns — deterministic hard-logic dialogue turn-shape checks.

Calling spec:
    assert_assistant_replies_after_nonfinal_addressed_line(lines, final_human_line_count=1)
    assert_assistant_speaks_only_after_addressed_line(lines)
    assert_no_immediate_repeated_final_assistant_handoff(lines, final_human_line_count=1)

Inputs: any sequence of objects/dicts with ``line_id``, ``speaker``, and
``addressed_to`` fields. Output: None, or ValueError on invalid turn shape.
Side effects: none.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def assert_assistant_replies_after_nonfinal_addressed_line(
    lines: Sequence[Any],
    *,
    final_human_line_count: int = 1,
) -> None:
    """Require every assistant-addressed line before the final human span to get a reply."""
    if final_human_line_count < 1:
        raise ValueError("final_human_line_count must be >= 1")
    final_start = max(0, len(lines) - final_human_line_count)
    for index, line in enumerate(lines[:final_start]):
        if _field(line, "addressed_to") != "assistant":
            continue
        next_line = lines[index + 1]
        if _field(next_line, "speaker") == "assistant":
            continue
        raise ValueError(
            "non-final assistant-addressed line must be followed immediately "
            "by an assistant reply: "
            f"{_line_label(line)} is followed by {_line_label(next_line)}"
        )


def assert_assistant_speaks_only_after_addressed_line(lines: Sequence[Any]) -> None:
    """Require assistant turns to answer the immediately preceding assistant-addressed line."""
    for index, line in enumerate(lines):
        if _field(line, "speaker") != "assistant":
            continue
        if index == 0:
            raise ValueError(
                "assistant-spoken line must not initiate dialogue: "
                f"{_line_label(line)} has no previous line"
            )
        previous = lines[index - 1]
        if (
            _field(previous, "speaker") != "assistant"
            and _field(previous, "addressed_to") == "assistant"
        ):
            continue
        raise ValueError(
            "assistant-spoken line must immediately follow a human line addressed "
            "to assistant: "
            f"{_line_label(previous)} -> {_line_label(line)}"
        )


def assert_no_immediate_repeated_final_assistant_handoff(
    lines: Sequence[Any],
    *,
    final_human_line_count: int = 1,
) -> None:
    """Reject human→assistant, assistant→human, final human→assistant repetition."""
    if final_human_line_count != 1:
        return
    final_start = len(lines) - 1
    if final_start < 2:
        return
    prior_request = lines[final_start - 2]
    prior_reply = lines[final_start - 1]
    final_request = lines[final_start]
    if (
        _field(prior_request, "speaker") != "assistant"
        and _field(prior_request, "addressed_to") == "assistant"
        and _field(prior_reply, "speaker") == "assistant"
        and _field(final_request, "speaker") != "assistant"
        and _field(final_request, "addressed_to") == "assistant"
    ):
        raise ValueError(
            "final assistant request must not immediately repeat a just-answered "
            "assistant handoff: "
            f"{_line_label(prior_request)} -> {_line_label(prior_reply)} -> "
            f"{_line_label(final_request)}"
        )


def _field(line: Any, name: str) -> Any:
    if isinstance(line, dict):
        return line.get(name)
    return getattr(line, name, None)


def _line_label(line: Any) -> str:
    line_id = _field(line, "line_id") or "unknown_line"
    speaker = _field(line, "speaker") or "unknown_speaker"
    addressed_to = _field(line, "addressed_to") or "unknown_addressee"
    return f"{line_id}({speaker}->{addressed_to})"
