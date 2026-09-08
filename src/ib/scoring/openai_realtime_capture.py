"""openai_realtime_capture — collected state for one Realtime response.

Calling spec:
    capture = RealtimeCapture()
    capture.prediction_text() -> provider prediction payload text
    capture.answer_text() -> visible assistant text/transcript or None

The schema only stores response observations. It performs no I/O and has no
side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RealtimeCapture:
    """Collected fields and raw events from one Realtime response."""

    response_id: str | None = None
    response_status: str | None = None
    emit_prediction_arguments: str | None = None
    text_parts: list[str] = field(default_factory=list)
    audio_transcripts: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    raw_events: list[Any] = field(default_factory=list)
    assistant_audio_ms: int = 0
    injection_started: bool = False
    injected_at_audio_ms: int | None = None

    def text(self) -> str:
        return "\n".join(part for part in self.text_parts if part).strip()

    def prediction_text(self) -> str:
        return (self.emit_prediction_arguments or self.text()).strip()

    def answer_text(self) -> str | None:
        text = self.text() or "\n".join(self.audio_transcripts).strip()
        return text or None

    def has_output(self) -> bool:
        return bool(self.answer_text() or self.tool_calls or self.assistant_audio_ms > 0)
