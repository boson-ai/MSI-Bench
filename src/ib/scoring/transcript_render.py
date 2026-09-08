"""transcript_render — render a manifest row's visible context as labeled text.

Calling spec:
    render_visible_transcript(row) -> str

Turns a built manifest row's ``visible_context_lines`` into a speaker/addressee
labeled transcript for text-input understanding evals, e.g.::

    L1 Mark -> assistant: I want that spa pass refunded.
    L2 assistant -> Mark: Got it, flagged the forty-five.
    L3 Dana -> Mark: Trade my sweater for a large before we go.

This is the text-modality analogue of the interleaved audio the model would
otherwise hear: the *same* dialogue lines, presented as clean labeled text so a
transcript-input control isolates attribution reasoning from voice perception.

Deterministic and side-effect free. Speech-control and SFX markup is stripped
with the same cleaner the answer judge uses, so the transcript matches the
model-visible dialogue exactly.
"""

from __future__ import annotations

from typing import Any

from ib.audio.acoustic_markup import clean_spoken_text


def render_visible_transcript(row: dict[str, Any]) -> str:
    """Return a labeled transcript for one manifest row's visible context lines."""
    lines = row.get("visible_context_lines")
    if not isinstance(lines, list) or not lines:
        raise ValueError("manifest row has no visible_context_lines list")
    rendered: list[str] = []
    for index, line in enumerate(lines, start=1):
        text = _line_text(line)
        if not text:
            continue
        rendered.append(_format_line(line, index, text))
    if not rendered:
        raise ValueError("visible_context_lines produced no usable transcript text")
    return "\n".join(rendered)


def _line_text(line: Any) -> str:
    if not isinstance(line, dict):
        return ""
    raw = line.get("text", line.get("transcript"))
    return clean_spoken_text(raw) if isinstance(raw, str) else ""


def _format_line(line: dict[str, Any], index: int, text: str) -> str:
    line_id = line.get("line_id")
    if not isinstance(line_id, str) or not line_id:
        line_id = f"L{index}"
    speaker = _str_or_none(line.get("speaker"))
    addressed_to = _str_or_none(line.get("addressed_to"))
    if speaker and addressed_to:
        prefix = f"{line_id} {speaker} -> {addressed_to}"
    elif speaker:
        prefix = f"{line_id} {speaker}"
    else:
        prefix = line_id
    return f"{prefix}: {text}"


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
