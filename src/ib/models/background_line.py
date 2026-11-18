"""background_line — background-line schema for generation targets, plans and contracts.

Calling spec:
    target.background = BackgroundSlot(speaker_count=2, level_db=14.2)
    contract.background = BackgroundLine(seed_id="self_zh_hiking_017", language="zh-CN", ...)

A background line is a separate 2- or 3-person chat in the same room, unrelated to
the main topic and quieter than every main speaker. It only distracts the
model: it holds no task, so it never changes a target of the main line or the side
branches. Its speakers are separate people: they are not participants and appear
neither in the system prompt nor in `messages`. Its language is
the language of its case's main line.

BackgroundSlot is the planner's draw for one target before any line exists: how many
people chat and how many dB below the loudest main speaker they sit (the sampled level; the
render lowers it further when needed to stay 5 dB under the quietest main speaker).
BackgroundLine is the written line of a case (the background writer fills it from a
topic seed); the synthesis step fills each speaker's voice.

Side effects: none.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import model_validator

from ib.models.base import IbBaseModel


MIN_BACKGROUND_LINES = 3
MAX_BACKGROUND_LINES = 8


@dataclass(frozen=True)
class BackgroundSlot:
    """A target's sampled background line: speaker_count 2 or 3 and level_db, the sampled dB below the loudest main speaker (> 0)."""

    speaker_count: int
    level_db: float
    seed_id: str | None = None  # the topic seed picked for this target before generation; None until assigned


class BackgroundSpeaker(IbBaseModel):
    """One person of the background line; voice is filled by synthesis."""

    name: str
    gender: Literal["female", "male"]
    voice: str | None = None


class BackgroundUtterance(IbBaseModel):
    """One spoken line of the background line."""

    speaker: str
    text: str


class BackgroundLine(IbBaseModel):
    """A case's written background line: the seed it came from, its speakers and its utterances."""

    seed_id: str
    language: Literal["zh-CN", "en"]
    category: str
    topic: str
    speaker_count: int
    level_db: float
    speakers: list[BackgroundSpeaker]
    lines: list[BackgroundUtterance]

    @model_validator(mode="after")
    def _fits_itself(self) -> "BackgroundLine":
        """Raise ValueError unless the line is consistent.

        Rejects: speaker_count not 2 or 3; len(speakers) != speaker_count; duplicate
        speaker names; a line whose speaker is not in speakers; fewer than 3 or more
        than 8 lines; a listed speaker with no line; level_db <= 0. A speaker with exactly
        one line is valid.
        """
        if self.speaker_count not in (2, 3):
            raise ValueError(f"background speaker_count must be 2 or 3, got {self.speaker_count}")
        if len(self.speakers) != self.speaker_count:
            raise ValueError(
                f"background has {len(self.speakers)} speakers but speaker_count is {self.speaker_count}"
            )
        names = [speaker.name for speaker in self.speakers]
        if len(set(names)) != len(names):
            raise ValueError(f"background speaker names must be unique: {names}")
        strangers = sorted({line.speaker for line in self.lines} - set(names))
        if strangers:
            raise ValueError(f"background lines by unlisted speakers: {strangers}")
        if not MIN_BACKGROUND_LINES <= len(self.lines) <= MAX_BACKGROUND_LINES:
            raise ValueError(
                f"background must have {MIN_BACKGROUND_LINES} to {MAX_BACKGROUND_LINES} "
                f"lines, got {len(self.lines)}"
            )
        silent = sorted(set(names) - {line.speaker for line in self.lines})
        if silent:
            raise ValueError(f"background speakers with no line: {silent}")
        if not self.level_db > 0:
            raise ValueError(f"background level_db must be > 0, got {self.level_db}")
        return self
