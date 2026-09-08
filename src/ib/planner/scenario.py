"""scenario — planner output models (Stage 10 line/turn transcript schema).

Calling spec:
    line = ScriptLine(line_index/turn_index=..., dialogue_turn_index=..., ...)
    turn = ScriptTurn(turn_index=..., line_indices=[...])
    outline = ScriptOutline(cell_id=..., turn_count=..., line_count=..., turns=[...], beats=[...])
    transcript = Transcript(cell_id=..., turns=[TranscriptTurn(...), ...])
    frozen = FrozenScenario(cell=<Cell>, outline=outline, transcript=transcript, content_hash=...)
    rubric = AnswerRubric(cell_id=..., rubric_id=..., ...)
    tts_turns(transcript)  # human non-held-out turns (synthesized audio)
    scripted_turns(transcript)  # all non-held-out turns incl. prior assistant text

These are pure schema definitions plus compatibility normalization. A dialogue
turn is an exchange: context turns contain an initiator line plus a responder
line; the final turn contains final human line(s) plus the held-out assistant
target. `beats` is kept as the legacy name for ordered lines. Earlier
`assistant` lines may carry actor-written transcript (no TTS). Only the final
assistant line is held out for the model-under-test, so it carries empty text.
All models inherit IbBaseModel, so unknown keys fail loudly.

Side effects: none.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from ib.models.base import IbBaseModel
from ib.models.enums import ExpectedAction
from ib.models.scenario import Cell
from ib.models.sound_plan import SoundPlan

# Reserved synthesis/render line-index bases for non-dialogue speech sources.
COMPETING_OVERLAY_LINE_INDEX = 10_000
LIVE_EVENT_LINE_INDEX = 20_000


class CompetingOverlayPlan(IbBaseModel):
    """One independently voiced side-speech source."""

    speaker: str
    role: str
    intent: str
    anchor_line_id: str | None = None
    overlap_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    interruption_delay_seconds: float = Field(default=0.3, ge=0.0)
    gain: float = Field(default=0.24, ge=0.0, le=1.0)
    target_snr_db: float | None = None


class CompetingOverlayLine(IbBaseModel):
    """One actor-rendered competing-speech source."""

    speaker: str
    role: str
    text: str
    anchor_line_id: str | None = None
    overlap_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    interruption_delay_seconds: float = Field(default=0.3, ge=0.0)
    gain: float = Field(default=0.24, ge=0.0, le=1.0)
    target_snr_db: float | None = None


LineKind = Literal[
    "context_initiation",
    "context_reply",
    "final_prompt",
    "held_out_reply",
]

TurnKind = Literal["context", "final"]


class ScriptUtterance(IbBaseModel):
    """One generated utterance that can be merged into a single planned line."""

    turn_index: int
    speaker: str
    role: str
    speaker_field: str | None = None
    addressed_to: str
    semantic_goal: str | None = None


class TranscriptUtterance(ScriptUtterance):
    """One realised utterance inside a merged transcript line."""

    text: str


class ScriptLine(IbBaseModel):
    """One planned utterance line in a Director outline (no realised text yet)."""

    turn_index: int
    speaker: str
    role: str
    speaker_field: str | None = None
    addressed_to: str
    intent: str
    held_out: bool = False
    dialogue_turn_index: int | None = None
    line_kind: LineKind | None = None
    semantic_goal: str | None = None
    merged_utterances: list[ScriptUtterance] = Field(default_factory=list)

    @property
    def line_index(self) -> int:
        """Compatibility alias: historic `turn_index` is now the line index."""

        return self.turn_index


# Backward-compatible import name. Existing code/tests may still construct
# ScriptBeat, but it is semantically a line.
ScriptBeat = ScriptLine


class ScriptTurn(IbBaseModel):
    """One dialogue turn/exchange, represented by ordered line indices."""

    turn_index: int
    initiator: str
    responder: str
    addressed_to: str
    intent: str
    line_indices: list[int] = Field(min_length=1)
    turn_kind: TurnKind = "context"
    semantic_goal: str | None = None
    overlap_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    interruption_delay_seconds: float | None = Field(default=None, ge=0.0)


class ScriptOutline(IbBaseModel):
    """The Director's plan for one cell: topic, exchange turns, and lines."""

    cell_id: str
    turn_count: int
    line_count: int | None = None
    roster: list[str]
    beats: list[ScriptLine]
    turns: list[ScriptTurn] = Field(default_factory=list)
    competing_overlays: list[CompetingOverlayPlan] = Field(default_factory=list)
    topic_id: str | None = None
    topic: str | None = None
    topic_category: str | None = None
    conversation_premise: str | None = None
    seed_customer_line: str | None = None
    seed_agent_line: str | None = None
    context_topology_id: str | None = None
    context_topology: list[tuple[str, str]] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _strip_retired_fields(cls, data):
        """Ignore outline fields retired with the authorization_scope plane."""
        if isinstance(data, dict) and "authorization" in data:
            migrated = dict(data)
            migrated.pop("authorization", None)
            return migrated
        return data

    @model_validator(mode="after")
    def _normalize_counts_and_turns(self) -> "ScriptOutline":
        """Fill line_count/turns while preserving legacy outline payloads."""

        line_count = self.line_count if self.line_count is not None else len(self.beats)
        if line_count != len(self.beats):
            raise ValueError("line_count must equal number of beats/lines")
        self.line_count = line_count
        if self.turns:
            if self.turn_count != len(self.turns):
                raise ValueError("turn_count must equal number of ScriptTurn entries")
            return self
        inferred = _infer_legacy_turns(self.beats)
        if self.turn_count == len(self.beats):
            # Legacy payloads used turn_count as line count. Keep them valid but
            # normalize the stored value to exchange-count semantics.
            self.turn_count = len(inferred)
        elif self.turn_count != len(inferred):
            raise ValueError("turn_count must equal number of dialogue turns")
        self.turns = inferred
        return self


class TranscriptTurn(IbBaseModel):
    """One realised utterance line. `text` is empty when `held_out` is True."""

    turn_index: int
    speaker: str
    role: str
    speaker_field: str | None = None
    addressed_to: str
    text: str
    held_out: bool = False
    dialogue_turn_index: int | None = None
    line_kind: LineKind | None = None
    semantic_goal: str | None = None
    speech_realism: dict[str, object] | None = None
    merged_utterances: list[TranscriptUtterance] = Field(default_factory=list)


class Transcript(IbBaseModel):
    """The assembled multi-line dialogue for one cell (final reply held out)."""

    cell_id: str
    turns: list[TranscriptTurn]
    competing_overlays: list[CompetingOverlayLine] = Field(default_factory=list)


class FrozenScenario(IbBaseModel):
    """An immutable, fully-resolved scenario: its cell, outline, transcript, and hash."""

    cell: Cell
    outline: ScriptOutline
    transcript: Transcript
    content_hash: str
    sound_plan: SoundPlan | None = None


class AtomicAnswerCriterion(IbBaseModel):
    """One atomic answer-rubric criterion."""

    criterion_id: str | None = None
    criterion_index: int = Field(ge=0)
    dimension: str = "legacy.unspecified"
    criterion: str
    source_index: int | None = None


class AnswerRubric(IbBaseModel):
    """Planner-generated O1 situated-answer rubric for one cell.

    `target_turn_index` is the held-out assistant turn the rubric scores.
    """

    cell_id: str
    rubric_id: str
    layer: str
    applies_to_actions: list[ExpectedAction]
    judge_instruction: str
    atomic_criteria: list[AtomicAnswerCriterion] = Field(min_length=1)
    provider: str
    target_turn_index: int | None = None


ASSISTANT_SPEAKER = "assistant"


def scripted_turns(transcript: Transcript) -> list[TranscriptTurn]:
    """Return every non-held-out turn, including prior scripted assistant replies."""
    return [turn for turn in transcript.turns if not turn.held_out]


def tts_turns(transcript: Transcript) -> list[TranscriptTurn]:
    """Return human, non-held-out turns — the ones that receive synthesized audio."""
    return [
        turn
        for turn in transcript.turns
        if not turn.held_out
        and turn.role != ASSISTANT_SPEAKER
        and turn.speaker != ASSISTANT_SPEAKER
    ]


def audible_turns(transcript: Transcript) -> list[TranscriptTurn]:
    """Alias for `tts_turns` (human lines with audio)."""
    return tts_turns(transcript)


def competing_overlay_lines(transcript: Transcript) -> list[CompetingOverlayLine]:
    """Return independently voiced competing-speech lines."""
    return transcript.competing_overlays


def competing_overlay_line(transcript: Transcript) -> CompetingOverlayLine | None:
    """Compatibility helper returning the first competing-speech line."""
    return transcript.competing_overlays[0] if transcript.competing_overlays else None


def live_event_synthesis_lines(frozen: FrozenScenario) -> list[tuple[int, str, str, str]]:
    """Return standalone TTS rows for live-injected speech probes."""
    contract = frozen.cell.logic_contract
    if contract is None:
        return []
    expansions = []
    if contract.live_expansion is not None:
        expansions.append(contract.live_expansion)
    expansions.extend(contract.live_expansions)
    lines: list[tuple[int, str, str, str]] = []
    for expansion in expansions:
        for event in expansion.events:
            lines.append(
                (
                    LIVE_EVENT_LINE_INDEX + len(lines),
                    event.speaker,
                    event.role,
                    event.text,
                )
            )
    return lines


def _infer_legacy_turns(lines: list[ScriptLine]) -> list[ScriptTurn]:
    """Infer one-line turns for legacy payloads that predate ScriptTurn."""

    turns: list[ScriptTurn] = []
    for line in lines:
        kind: TurnKind = "final" if line.held_out or line.intent == "final_user_context" else "context"
        turns.append(
            ScriptTurn(
                turn_index=line.dialogue_turn_index
                if line.dialogue_turn_index is not None
                else len(turns),
                initiator=line.speaker,
                responder=line.addressed_to,
                addressed_to=line.addressed_to,
                intent=line.intent,
                line_indices=[line.turn_index],
                turn_kind=kind,
                semantic_goal=line.semantic_goal,
            )
        )
    return turns


def synthesis_lines(frozen: FrozenScenario) -> list[tuple[int, str, str, str]]:
    """Return (line_index, speaker, role, text) rows to synthesize as source WAVs."""
    lines = []
    for turn in tts_turns(frozen.transcript):
        if turn.merged_utterances:
            lines.extend(
                (utterance.turn_index, utterance.speaker, utterance.role, utterance.text)
                for utterance in turn.merged_utterances
            )
            continue
        lines.append((turn.turn_index, turn.speaker, turn.role, turn.text))
    for offset, overlay in enumerate(competing_overlay_lines(frozen.transcript)):
        lines.append(
            (
                COMPETING_OVERLAY_LINE_INDEX + offset,
                overlay.speaker,
                overlay.role,
                overlay.text,
            )
        )
    lines.extend(live_event_synthesis_lines(frozen))
    return lines
