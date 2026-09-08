"""sound_plan — structured acoustic intent schema for realism rendering.

Calling spec:
    SoundPlan(mode="tta", line_plans=[...])
    SoundPlanLine(clean_text=..., events=[SoundPlanEvent(kind="event", ...)])

The schema stores non-speech acoustic intent separately from spoken text so TTS
requests receive only clean speech while render/provenance layers retain SFX and
ambient instructions.

Side effects: none.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from ib.models.base import IbBaseModel

SOUND_PLAN_SCHEMA_VERSION = "ib.sound_plan.v1"

SoundMode = Literal["off", "tta", "hybrid", "assets"]
SoundBackend = Literal["auto", "tta", "assets"]
SoundEventKind = Literal["ambient", "event"]
SoundAnchorPolicy = Literal["scene", "line", "prefix", "suffix", "infix"]
SoundLoudnessClass = Literal["quiet", "medium", "loud"]


class SoundPlanEvent(IbBaseModel):
    """One ambient or burst acoustic intent item."""

    event_id: str
    kind: SoundEventKind
    description: str = Field(min_length=1)
    asset_id: str | None = Field(default=None, min_length=1)
    backend: SoundBackend = "auto"
    anchor_policy: SoundAnchorPolicy = "infix"
    anchor_text_before: str | None = None
    anchor_text_after: str | None = None
    loudness_class: SoundLoudnessClass | None = None
    duration_ms: int | None = Field(default=None, gt=0)


class SoundPlanLine(IbBaseModel):
    """Acoustic plan for one dialogue line; `clean_text` is safe for TTS."""

    line_index: int = Field(ge=0)
    clean_text: str
    events: list[SoundPlanEvent] = Field(default_factory=list)


class SoundPlan(IbBaseModel):
    """Per-cell acoustic rendering plan."""

    schema_version: Literal["ib.sound_plan.v1"] = SOUND_PLAN_SCHEMA_VERSION
    mode: SoundMode = "tta"
    tta_prompt: str | None = Field(default=None, min_length=1)
    stable_audio3_prompt: str | None = Field(default=None, min_length=1)
    scene_ambient: SoundPlanEvent | None = None
    line_plans: list[SoundPlanLine] = Field(default_factory=list)
