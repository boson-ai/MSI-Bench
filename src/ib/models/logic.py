"""logic — machine-readable contracts for hard interaction-logic cases.

Calling spec:
    contract = LogicContract(...)

A LogicContract carries the hidden oracle for cases where the answer target is a
multi-constraint situated plan rather than a generic dialogue answer. It is pure
schema: generation, freezing, and scoring logic live in planner/scoring modules.

Side effects: none.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import Field, model_validator

from ib.models.base import IbBaseModel
from ib.models.enums import CompetingSpeechLevel, ExpectedAction, ParticipationFrame
from ib.models.tool_call_args import gold_tool_arg_policy, validate_gold_tool_arguments
from ib.models.logic_turns import (
    assert_assistant_replies_after_nonfinal_addressed_line,
    assert_assistant_speaks_only_after_addressed_line,
    assert_no_immediate_repeated_final_assistant_handoff,
)

LogicPattern = str
VoiceGender = Literal["male", "female"]
SpeakerDistance = Literal["near", "mid", "far"]
IdentityMode = Literal["in_text", "in_audio"]
TIERED_LOGIC_PATTERNS = {"authority_gated_override"}

FORBIDDEN_EXPLANATION_TOOL_NAMES = {"explain_only", "explain_infeasible_options"}
BRACKETED_EVENT_MARKER_RE = re.compile(r"\[[^\]]+\]")

LEGACY_LOGIC_PATTERN_ALIASES = {
    "conflicting_interruption_stiching": "eavesdropping",
    "conflicting_interruption_stitching": "eavesdropping",
    "conflicting_interruption": "eavesdropping",
    "interruption_stiching": "eavesdropping",
    "interruption_stitching": "eavesdropping",
    "eavesdrop": "eavesdropping",
    "overheard_context": "eavesdropping",
    "authority_override": "authority_gated_override",
    "scope_binding": "distributed_parameters",
    "distributed_parameter": "distributed_parameters",
    "cross_speaker_amendment": "authority_gated_override",
    "cross_speaker_amendments": "authority_gated_override",
    "hardness_ranking": "hard_vs_soft_constraint",
}


class LogicLine(IbBaseModel):
    """One human play line in a hard-logic dialogue."""

    line_id: str
    speaker: str
    addressed_to: str
    text: str
    stage_directions: list[str] = Field(default_factory=list)
    hear_time_probe: bool = Field(default=False, exclude_if=lambda value: not value)


class LogicLinePlan(IbBaseModel):
    """One planned dialogue beat before spoken text is rendered."""

    line_id: str
    speaker: str
    addressed_to: str
    beat: str
    subtext: str = ""
    must_surface: list[str] = Field(default_factory=list)
    must_not_say: list[str] = Field(default_factory=list)


class LogicParticipant(IbBaseModel):
    """One human participant available to the actor dialogue writer."""

    name: str
    display_name: str = Field(default="", exclude_if=lambda value: not value)
    role: str = ""
    gender: VoiceGender | None = Field(default=None, exclude_if=lambda value: value is None)
    age_bucket: Literal["teen", "young_adult", "middle_aged", "senior"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    speaker_distance: SpeakerDistance = "near"
    role_tier: int | None = Field(default=None, ge=1, exclude_if=lambda value: value is None)
    capabilities: list[str] = Field(default_factory=list, exclude_if=lambda value: not value)

    @model_validator(mode="before")
    @classmethod
    def _migrate_participant_aliases(cls, data):
        """Accept legacy display/capability aliases while preferring name+permission input."""
        if not isinstance(data, dict):
            return data
        migrated = dict(data)
        if "display_name" not in migrated and isinstance(migrated.get("displayName"), str):
            migrated["display_name"] = migrated.pop("displayName")
        if "name" not in migrated and isinstance(migrated.get("display_name"), str):
            migrated["name"] = migrated["display_name"]
        if "gender" in migrated:
            normalized_gender = _normalize_voice_gender(migrated.get("gender"))
            if normalized_gender is not None:
                migrated["gender"] = normalized_gender
        else:
            for alias in ("voice_gender", "voiceGender", "voice_sex"):
                normalized = _normalize_voice_gender(migrated.get(alias))
                if normalized is not None:
                    migrated["gender"] = normalized
                    break
        if "speaker_distance" in migrated:
            normalized_distance = _normalize_speaker_distance(
                migrated.get("speaker_distance")
            )
            if normalized_distance is not None:
                migrated["speaker_distance"] = normalized_distance
        else:
            for alias in ("distance", "role_distance", "speakerDistance", "speaker_field"):
                normalized_distance = _normalize_speaker_distance(migrated.get(alias))
                if normalized_distance is not None:
                    migrated["speaker_distance"] = normalized_distance
                    break
        if "capabilities" not in migrated:
            for alias in ("permission", "permissions", "allowed_capabilities"):
                if isinstance(migrated.get(alias), list):
                    migrated["capabilities"] = migrated.pop(alias)
                    break
        for alias in (
            "voice_gender",
            "voiceGender",
            "voice_sex",
            "distance",
            "role_distance",
            "speakerDistance",
            "speaker_field",
            "permission",
            "permissions",
            "allowed_capabilities",
        ):
            migrated.pop(alias, None)
        return migrated


class LogicCompetingOverlay(IbBaseModel):
    """One intelligible background side-speech source for interaction cases."""

    speaker: str
    role: str = "background_speaker"
    intent: str = "unrelated side conversation"
    text: str
    anchor_line_id: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    overlap_fraction: float = Field(default=0.55, ge=0.0, le=1.0)
    interruption_delay_seconds: float = Field(default=0.3, ge=0.0)
    gain: float = Field(default=0.24, ge=0.0, le=1.0)
    target_snr_db: float | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def _validate_spoken_text_only(self) -> "LogicCompetingOverlay":
        _validate_no_bracketed_event_marker(self.text, "competing_overlays[].text")
        return self


LogicCaseMode = Literal["base_semantic", "live_expansion", "hear_time_probe"]
LiveExpansionType = Literal[
    "side_conversation_filtering",
    "bystander_interference_suppression",
    "new_participant_joining",
]


class LogicLiveAnchor(IbBaseModel):
    """Where a live expansion is inserted relative to the model's spoken reply."""

    target: Literal["assistant_response"] = "assistant_response"
    start_offset_seconds: float = Field(default=1.0, ge=0.0)
    overlap_fraction: float = Field(default=0.55, ge=0.0, le=1.0)


class LogicLiveEvent(IbBaseModel):
    """One live speech event injected while the assistant is replying."""

    speaker: str
    role: str
    text: str
    addressed_to: str | None = None
    is_assistant_relevant: bool = False
    authority: str | None = Field(default=None, exclude_if=lambda value: value is None)
    gain: float = Field(default=0.24, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_spoken_text_only(self) -> "LogicLiveEvent":
        _validate_no_bracketed_event_marker(self.text, "live_expansions[].events[].text")
        return self


class LogicExpectedLiveBehavior(IbBaseModel):
    """Oracle for full-duplex behavior under a live speech injection."""

    should_interrupt: bool
    should_use_interference_content: bool = False
    should_preserve_base_answer: bool = True
    should_resume_after_overlap: bool = True
    expected_action: ExpectedAction | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    should_follow_bystander: bool | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    should_not_directly_execute_new_request: bool | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class LogicLiveExpansion(IbBaseModel):
    """A live anti-interference variant wrapped around a semantic base case."""

    type: LiveExpansionType
    subtype: str | None = Field(default=None, exclude_if=lambda value: value is None)
    base_pattern: LogicPattern
    anchor: LogicLiveAnchor = Field(default_factory=LogicLiveAnchor)
    events: list[LogicLiveEvent] = Field(min_length=1)
    expected_live_behavior: LogicExpectedLiveBehavior


def _validate_no_bracketed_event_marker(text: str, field_name: str) -> None:
    """Reject non-spoken square-bracket cue markers in overlay speech text."""
    if BRACKETED_EVENT_MARKER_RE.search(text):
        raise ValueError(
            f"{field_name} must be spoken words only; do not include square-bracket "
            "acoustic event markers"
        )


class LogicHearTimeProbe(IbBaseModel):
    """A human-to-human line reused as a final no-response subtest."""

    type: Literal["bystander_interference_suppression"] = (
        "bystander_interference_suppression"
    )
    subtype: Literal["human_to_human_final_line_no_response"] = (
        "human_to_human_final_line_no_response"
    )
    line_id: str
    speaker: str
    addressed_to: str
    text: str
    expected_action: ExpectedAction = ExpectedAction.ignore


def _normalize_voice_gender(value: Any) -> VoiceGender | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in {"male", "m", "man", "男", "男性", "男声"}:
        return "male"
    if normalized in {"female", "f", "woman", "女", "女性", "女声"}:
        return "female"
    return None


def _normalize_speaker_distance(value: Any) -> SpeakerDistance | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases: dict[str, SpeakerDistance] = {
        "near": "near",
        "near_field": "near",
        "close": "near",
        "close_static": "near",
        "near_static": "near",
        "near_moving": "near",
        "near_field_moving": "near",
        "close_moving": "near",
        "moving_near": "near",
        "mid": "mid",
        "middle": "mid",
        "medium": "mid",
        "mid_field": "mid",
        "far_static": "mid",
        "far_field_static": "mid",
        "far_field": "mid",
        "far": "far",
        "distant": "far",
        "far_moving": "far",
        "far_field_moving": "far",
        "moving_far": "far",
    }
    return aliases.get(normalized)


class LogicCausalFrame(IbBaseModel):
    """Scene-native grounding plan that dialogue must realize."""

    setting_pressure: str
    primary_goal: str
    causal_bridges: list[str] = Field(min_length=1)
    pragmatic_subtext: list[str] = Field(default_factory=list)
    ordinary_objects: list[str] = Field(default_factory=list)
    scope_or_timing: list[str] = Field(default_factory=list)


class ExpectedToolCall(IbBaseModel):
    """One expected assistant tool/action call."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_gold_arguments(self) -> "ExpectedToolCall":
        if gold_tool_arg_policy() == "strict":
            validate_gold_tool_arguments(self.arguments, call_name=self.name)
        return self


class LogicRubricAtom(IbBaseModel):
    """One public atomic rubric criterion for logic-case judging."""

    dimension: str = "legacy.unspecified"
    criterion: str

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_polarity(cls, data):
        """Accept old satisfy/avoid atoms but persist one neutral criterion string."""
        if not isinstance(data, dict):
            return data
        if "polarity" not in data:
            return data
        migrated = dict(data)
        migrated.pop("polarity")
        return migrated


def _context_human_speakers(lines: list[Any], final_human_line_count: int = 1) -> set[str]:
    """Return human speakers from context lines, excluding the final handoff span."""
    cutoff = _final_span_start(lines, final_human_line_count)
    return {line.speaker for line in lines[:cutoff] if line.speaker != "assistant"}


def _final_span_start(lines: list[Any], final_human_line_count: int) -> int:
    """Return the first index of the final human decision span."""
    return max(0, len(lines) - final_human_line_count)


class LogicResolution(IbBaseModel):
    """Hidden oracle response for a hard-logic case."""

    response_summary: str
    standard_answer: str = ""
    available_functions: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[ExpectedToolCall] = Field(default_factory=list)
    state_updates: dict[str, Any] = Field(default_factory=dict)


class LogicHarness(IbBaseModel):
    """Harness-generated tools, oracle answer, and grading atoms."""

    response_summary: str
    standard_answer: str = ""
    available_functions: list[dict[str, Any]] = Field(min_length=1)
    tool_calls: list[ExpectedToolCall] = Field(default_factory=list)
    atomic_rubric: list[LogicRubricAtom] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_tool_schema_alignment(self) -> "LogicHarness":
        names = {str(item.get("name")) for item in self.available_functions if isinstance(item, dict)}
        if not names:
            raise ValueError("available_functions must name at least one function")
        forbidden = sorted(names & FORBIDDEN_EXPLANATION_TOOL_NAMES)
        if forbidden:
            raise ValueError(f"available_functions must not include explanation-only tools: {forbidden}")
        missing = sorted({call.name for call in self.tool_calls} - names)
        if missing:
            raise ValueError(f"tool_calls reference unavailable functions: {missing}")
        forbidden_calls = sorted({call.name for call in self.tool_calls} & FORBIDDEN_EXPLANATION_TOOL_NAMES)
        if forbidden_calls:
            raise ValueError(f"tool_calls must not include explanation-only tools: {forbidden_calls}")
        distractors = names - {call.name for call in self.tool_calls}
        if not distractors:
            raise ValueError("available_functions must include at least one unused distractor function")
        return self


class LogicContract(IbBaseModel):
    """One deterministic hard-logic testcase contract."""

    contract_id: str
    case_mode: LogicCaseMode = "base_semantic"
    pattern: LogicPattern = Field(min_length=1)
    subpattern: str = Field(default="", exclude_if=lambda value: not value)
    base_pattern: LogicPattern | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    live_expansion: LogicLiveExpansion | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    live_expansions: list[LogicLiveExpansion] = Field(default_factory=list)
    hear_time_probes: list[LogicHearTimeProbe] = Field(default_factory=list)
    speaker_count: int = Field(ge=2, le=3)
    title: str
    scene_date: str = Field(default="", exclude_if=lambda value: not value)
    scene_context: str
    private_memory: str = Field(default="", exclude_if=lambda value: not value)
    acoustic_environment: str | None = Field(
        default=None, min_length=1, exclude_if=lambda value: value is None
    )
    acoustic_asset_id: str | None = Field(default=None, min_length=1)
    primary_goal: str = ""
    participants: list[LogicParticipant] = Field(default_factory=list)
    dialogue_outline: list[str] = Field(default_factory=list, max_length=5)
    causal_frame: LogicCausalFrame | None = None
    line_plans: list[LogicLinePlan] = Field(default_factory=list)
    interaction_flow: str | None = Field(default=None, exclude_if=lambda value: value is None)
    participation_frame: ParticipationFrame = ParticipationFrame.user_addressed
    expected_action: ExpectedAction = ExpectedAction.respond
    expected_addressed_speaker: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    final_human_line_count: int = Field(default=1, ge=1, le=2)
    competing_speech: CompetingSpeechLevel = CompetingSpeechLevel.none
    competing_overlay_count: int = Field(default=0, ge=0)
    identity_mode: IdentityMode = "in_text"
    competing_overlays: list[LogicCompetingOverlay] = Field(default_factory=list)
    lines: list[LogicLine] = Field(min_length=2)
    resolution: LogicResolution
    atomic_rubric: list[LogicRubricAtom] = Field(default_factory=list)
    traps: list[str] = Field(default_factory=list)
    ability_tags: list[str] = Field(default_factory=list)
    provenance: str = "generated_from_public_logic_blueprint"

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_pattern_aliases(cls, data):
        """Accept prior clean-room aliases but persist Gold canonical names."""
        if not isinstance(data, dict):
            return data
        pattern = data.get("pattern")
        if pattern not in LEGACY_LOGIC_PATTERN_ALIASES:
            return data
        migrated = dict(data)
        migrated["pattern"] = LEGACY_LOGIC_PATTERN_ALIASES[pattern]
        return migrated

    @model_validator(mode="after")
    def _validate_dialogue_shape(self) -> "LogicContract":
        _validate_acoustic_fields(
            self.acoustic_environment, self.acoustic_asset_id, self.participants
        )
        if self.causal_frame is not None:
            if not self.primary_goal:
                self.primary_goal = self.causal_frame.primary_goal
        validate_contract_dialogue_shape(
            pattern=self.pattern,
            base_pattern=self.base_pattern,
            subpattern=self.subpattern,
            case_mode=self.case_mode,
            participants=self.participants,
            lines=self.lines,
            speaker_count=self.speaker_count,
            participation_frame=self.participation_frame,
            expected_action=self.expected_action,
            final_human_line_count=self.final_human_line_count,
            competing_speech=self.competing_speech,
            competing_overlay_count=self.competing_overlay_count,
            competing_overlays=self.competing_overlays,
            private_memory=self.private_memory,
            identity_mode=self.identity_mode,
            live_expansion=self.live_expansion,
            live_expansions=self.live_expansions,
            hear_time_probes=self.hear_time_probes,
            atomic_rubric=self.atomic_rubric,
        )
        return self


def validate_contract_dialogue_shape(
    *,
    pattern: str,
    base_pattern: str | None,
    subpattern: str,
    case_mode: LogicCaseMode,
    participants: list[LogicParticipant],
    lines: list[LogicLine],
    speaker_count: int,
    participation_frame: ParticipationFrame,
    expected_action: ExpectedAction,
    final_human_line_count: int,
    competing_speech: CompetingSpeechLevel,
    competing_overlay_count: int,
    competing_overlays: list[LogicCompetingOverlay],
    private_memory: str,
    identity_mode: IdentityMode,
    live_expansion: LogicLiveExpansion | None,
    live_expansions: list[LogicLiveExpansion],
    hear_time_probes: list[LogicHearTimeProbe],
    atomic_rubric: list[LogicRubricAtom] | None = None,
) -> None:
    """Validate plan+lines contract shape without requiring harness output.

    Pass atomic_rubric=None to run every check that does not depend on the
    harness; the full contract validator passes the real rubric.
    """
    _validate_interaction_axis_pair(participation_frame, expected_action)
    participant_names = {participant.name for participant in participants}
    if participants:
        _validate_participant_count(speaker_count, participant_names, participation_frame)
    _validate_tiered_participants(pattern, participants)
    if pattern == "eavesdropping":
        _validate_eavesdropping_overlay_shape(
            participants,
            lines,
            competing_overlays,
            final_human_line_count=final_human_line_count,
            atomic_rubric=atomic_rubric,
        )
    context_human_speakers = _context_human_speakers(lines, final_human_line_count)
    if pattern != "eavesdropping":
        if len(context_human_speakers) != speaker_count:
            raise ValueError("speaker_count must match unique human speakers in context lines")
    final_lines = lines[_final_span_start(lines, final_human_line_count):]
    if pattern != "eavesdropping":
        _validate_participants_match_speakers(
            participant_names,
            context_human_speakers,
            [line.speaker for line in final_lines],
            participation_frame,
        )
    for final in final_lines:
        if final.speaker == "assistant":
            raise ValueError("final hard-logic line must be spoken by a human")
        if participants and final.speaker not in participant_names:
            raise ValueError("final hard-logic speaker must be a participant")
    final_must_address_assistant = not (
        participation_frame == ParticipationFrame.not_addressed
        and expected_action == ExpectedAction.ignore
    )
    if final_must_address_assistant and lines[-1].addressed_to != "assistant":
        raise ValueError("final hard-logic line must address the assistant")
    if case_mode != "hear_time_probe":
        _validate_disclosure_memory_and_context(
            pattern,
            base_pattern,
            subpattern,
            private_memory,
            lines,
            final_human_line_count,
            participants,
        )
        validate_identity_exposure(identity_mode, participants, lines)
    _validate_hear_time_probes(lines, hear_time_probes)
    _validate_competing_overlay_shape(
        competing_speech,
        competing_overlay_count,
        competing_overlays,
        pattern=pattern,
    )
    assert_assistant_replies_after_nonfinal_addressed_line(
        lines,
        final_human_line_count=final_human_line_count,
    )
    assert_assistant_speaks_only_after_addressed_line(lines)
    assert_no_immediate_repeated_final_assistant_handoff(
        lines,
        final_human_line_count=final_human_line_count,
    )
    _validate_live_expansion_shape(
        case_mode,
        pattern,
        base_pattern,
        live_expansion,
        live_expansions,
    )
    _validate_live_expansion_speaker_binding(lines, live_expansion, live_expansions)


class LogicContractPlan(IbBaseModel):
    """Coarse story outline before dialogue and harness rendering."""

    contract_id: str | None = None
    case_mode: LogicCaseMode = "base_semantic"
    pattern: LogicPattern = Field(min_length=1)
    subpattern: str = Field(default="", exclude_if=lambda value: not value)
    base_pattern: LogicPattern | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    live_expansion: LogicLiveExpansion | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    live_expansions: list[LogicLiveExpansion] = Field(default_factory=list)
    hear_time_probes: list[LogicHearTimeProbe] = Field(default_factory=list)
    speaker_count: int = Field(ge=2, le=3)
    title: str
    scene_date: str = Field(default="", exclude_if=lambda value: not value)
    scene_context: str
    private_memory: str = Field(default="", exclude_if=lambda value: not value)
    acoustic_environment: str | None = Field(
        default=None, min_length=1, exclude_if=lambda value: value is None
    )
    acoustic_asset_id: str | None = Field(default=None, min_length=1)
    primary_goal: str
    participants: list[LogicParticipant] = Field(default_factory=list)
    dialogue_outline: list[str] = Field(default_factory=list, max_length=5)
    line_plans: list[LogicLinePlan] = Field(default_factory=list)
    interaction_flow: str | None = Field(default=None, exclude_if=lambda value: value is None)
    participation_frame: ParticipationFrame = ParticipationFrame.user_addressed
    expected_action: ExpectedAction = ExpectedAction.respond
    expected_addressed_speaker: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    final_human_line_count: int = Field(default=1, ge=1, le=2)
    competing_speech: CompetingSpeechLevel = CompetingSpeechLevel.none
    competing_overlay_count: int = Field(default=0, ge=0)
    identity_mode: IdentityMode = "in_text"
    competing_overlays: list[LogicCompetingOverlay] = Field(default_factory=list)
    resolution: LogicResolution | None = None
    atomic_rubric: list[LogicRubricAtom] = Field(default_factory=list)
    traps: list[str] = Field(default_factory=list)
    ability_tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_script_shape(self) -> "LogicContractPlan":
        _validate_acoustic_fields(
            self.acoustic_environment, self.acoustic_asset_id, self.participants
        )
        _validate_interaction_axis_pair(self.participation_frame, self.expected_action)
        if not self.participants and self.line_plans:
            self.participants = [
                LogicParticipant(name=name)
                for name in sorted({line.speaker for line in self.line_plans})
            ]
        participant_names = {participant.name for participant in self.participants}
        if self.participants:
            _validate_participant_count(
                self.speaker_count, participant_names, self.participation_frame
            )
        elif not self.line_plans:
            raise ValueError("contract plans require participants or legacy line_plans")
        _validate_tiered_participants(self.pattern, self.participants)
        if self.line_plans:
            context_human_speakers = _context_human_speakers(
                self.line_plans, self.final_human_line_count
            )
            if self.pattern != "eavesdropping" and len(context_human_speakers) != self.speaker_count:
                raise ValueError("speaker_count must match unique human speakers in context line_plans")
            final_plans = self.line_plans[
                _final_span_start(self.line_plans, self.final_human_line_count):
            ]
            if self.pattern != "eavesdropping":
                _validate_participants_match_speakers(
                    participant_names,
                    context_human_speakers,
                    [line.speaker for line in final_plans],
                    self.participation_frame,
                )
            for final in final_plans:
                if final.speaker == "assistant":
                    raise ValueError("final hard-logic line plan must be spoken by a human")
                if self.participants and final.speaker not in participant_names:
                    raise ValueError("final hard-logic line plan speaker must be a participant")
            if self.line_plans[-1].addressed_to != "assistant":
                raise ValueError("final hard-logic line plan must address the assistant")
            assert_assistant_replies_after_nonfinal_addressed_line(
                self.line_plans,
                final_human_line_count=self.final_human_line_count,
            )
            assert_assistant_speaks_only_after_addressed_line(self.line_plans)
            assert_no_immediate_repeated_final_assistant_handoff(
                self.line_plans,
                final_human_line_count=self.final_human_line_count,
            )
        _validate_disclosure_memory_plan(
            self.pattern, self.base_pattern, self.subpattern, self.private_memory, self.participants
        )
        _validate_competing_overlay_shape(
            self.competing_speech,
            self.competing_overlay_count,
            self.competing_overlays,
            pattern=self.pattern,
        )
        if self.pattern == "eavesdropping":
            if self.line_plans:
                _validate_eavesdropping_overlay_shape(
                    self.participants,
                    self.line_plans,
                    self.competing_overlays,
                    final_human_line_count=self.final_human_line_count,
                )
            else:
                _validate_eavesdropping_overlay_plan_shape(
                    self.participants,
                    self.competing_overlays,
                )
        _validate_live_expansion_shape(
            self.case_mode,
            self.pattern,
            self.base_pattern,
            self.live_expansion,
            self.live_expansions,
        )
        if not self.dialogue_outline and not self.line_plans:
            raise ValueError("contract plans require dialogue_outline")
        return self


def _validate_disclosure_memory_plan(
    pattern: str,
    base_pattern: str | None,
    subpattern: str,
    private_memory: str,
    participants: list[LogicParticipant] | None = None,
) -> None:
    """Validate disclosure private-memory placement without needing rendered lines."""
    is_disclosure_case = (
        pattern == "disclosure_clause_in_instruction"
        or base_pattern == "disclosure_clause_in_instruction"
    )
    if not is_disclosure_case:
        if private_memory.strip():
            raise ValueError("private_memory is only supported for disclosure_clause_in_instruction")
        return
    if subpattern == "audio_stated_secret_with_later_probe":
        if private_memory.strip():
            raise ValueError("audio-stated disclosure cases must not use private_memory")
        return
    if not private_memory.strip():
        raise ValueError("disclosure private-memory subpatterns require private_memory")
    _validate_private_memory_fact_capsule(private_memory, participants or [])


def _validate_private_memory_fact_capsule(
    private_memory: str, participants: list[LogicParticipant]
) -> None:
    """Keep private_memory as values only, not testcase narrative or secrecy policy."""
    text = private_memory.strip()
    leaked_names = []
    for participant in participants:
        name = participant.name.strip() if participant.name else ""
        if not name:
            continue
        if len(name) < 2 and all(ord(char) < 128 for char in name):
            continue
        if name in text:
            leaked_names.append(name)
    if leaked_names:
        raise ValueError(f"private_memory must not include participant names: {leaked_names}")
    forbidden_terms = (
        "助手",
        "助理",
        "assistant",
        "保密",
        "秘密",
        "透露",
        "告诉",
        "不得",
        "不要",
        "不能",
        "惊喜",
        "surprise",
        "secret",
        "withhold",
        "disclose",
        "tell",
        "audience",
    )
    found = [term for term in forbidden_terms if term.lower() in text.lower()]
    if found:
        raise ValueError(
            "private_memory must contain only concrete withheld values, not "
            f"secrecy policy or narrative terms: {found}"
        )
    if _private_memory_is_categories_only(text):
        raise ValueError("private_memory must contain exact withheld values, not categories only")


def _private_memory_is_categories_only(text: str) -> bool:
    """Return true when private_memory names slots instead of slot values."""
    fragments = [
        fragment
        for fragment in re.split(r"[,，;；、|'\"“”‘’：:\s]+", _normalize_disclosure_text(text))
        if fragment
    ]
    return bool(fragments) and all(_private_memory_fragment_is_category(fragment) for fragment in fragments)


def _private_memory_fragment_is_category(fragment: str) -> bool:
    category_terms = (
        "内容",
        "地址",
        "金额",
        "价格",
        "价位",
        "时间",
        "细节",
        "名字",
        "名称",
        "信息",
        "通知",
        "提醒",
        "余额",
        "变动",
        "触发",
        "留言",
        "地点",
        "款式",
        "哪儿",
        "哪里",
    )
    if fragment in _PRIVATE_MEMORY_GENERIC_TERMS:
        return True
    return any(term in fragment for term in category_terms)


def _validate_disclosure_memory_and_context(
    pattern: str,
    base_pattern: str | None,
    subpattern: str,
    private_memory: str,
    lines: list[Any],
    final_human_line_count: int,
    participants: list[LogicParticipant] | None = None,
) -> None:
    """Require disclosure cases to establish secrecy with the assistant in context."""
    _validate_disclosure_memory_plan(pattern, base_pattern, subpattern, private_memory, participants)
    if pattern != "disclosure_clause_in_instruction" and base_pattern != "disclosure_clause_in_instruction":
        return
    context_lines = lines[:_final_span_start(lines, final_human_line_count)]
    if not any(line.speaker == "assistant" or line.addressed_to == "assistant" for line in context_lines):
        raise ValueError("disclosure cases require a non-final assistant context turn")
    if subpattern != "audio_stated_secret_with_later_probe":
        _validate_private_memory_not_spoken(private_memory, lines)
        _validate_private_memory_secret_lines_are_category_only(context_lines)


def _validate_private_memory_not_spoken(private_memory: str, lines: list[Any]) -> None:
    """Reject private disclosure cases that leak exact private-memory values in audio."""
    fragments = _private_memory_fragments(private_memory)
    if not fragments:
        return
    leaked: list[str] = []
    for line in lines:
        text = _normalize_disclosure_text(line.text)
        leaked.extend(fragment for fragment in fragments if fragment in text)
    if leaked:
        raise ValueError(
            "private-memory disclosure exact values must not appear in spoken lines: "
            f"{sorted(set(leaked))}"
        )


def _validate_private_memory_secret_lines_are_category_only(lines: list[Any]) -> None:
    """Ensure private-memory disclosure speech names categories, not substitute values."""
    offending = [
        line.line_id
        for line in lines
        if _has_secrecy_term(line.text) and _has_concrete_value_marker(line.text)
    ]
    if offending:
        raise ValueError(
            "private-memory disclosure secrecy lines must mention only categories, "
            f"not exact substitute values: {offending}"
        )


def _private_memory_fragments(private_memory: str) -> set[str]:
    """Return salient private-memory value fragments for spoken-line leak checks."""
    text = _normalize_disclosure_text(private_memory)
    raw_parts = re.split(r"[,，;；、|'\"“”‘’：:\s]+", text)
    fragments = {
        part
        for part in raw_parts
        if len(part) >= 2 and part not in _PRIVATE_MEMORY_GENERIC_TERMS
    }
    fragments.update(
        match.group(0)
        for match in re.finditer(r"\d+(?:\.\d+)?[\u4e00-\u9fffa-z%]*", text)
    )
    # Bare single digits match ubiquitous spoken numbers (times, counts) and
    # produce false leak positives; require at least two characters.
    return {fragment for fragment in fragments if len(fragment) >= 2}


_PRIVATE_MEMORY_GENERIC_TERMS = {
    "价格",
    "实付",
    "折后",
    "原价",
    "内容",
    "留言",
    "地址",
    "时间",
    "金额",
    "品牌",
    "花束",
    "贺卡",
    "房间",
    "包间",
    "餐厅",
}


def _normalize_disclosure_text(text: str) -> str:
    """Return compact text with speech/acoustic tags removed for value matching."""
    without_tags = re.sub(r"<\|[^<>]*?\|>|<[^<>]+>|\[[^\[\]]+\]", " ", text)
    return re.sub(r"\s+", "", without_tags.lower())


def _has_secrecy_term(text: str) -> bool:
    normalized = _normalize_disclosure_text(text)
    terms = (
        "保密",
        "别告诉",
        "不要告诉",
        "不能告诉",
        "别跟",
        "不要跟",
        "别提",
        "先别提",
        "别说",
        "不要说",
        "瞒",
        "惊喜",
        "secret",
        "keep",
        "withhold",
        "don'ttell",
        "donottell",
    )
    return any(term in normalized for term in terms)


def _has_concrete_value_marker(text: str) -> bool:
    normalized = _normalize_disclosure_text(text)
    if re.search(r"\d", normalized):
        return True
    # English contractions/possessives (don't, it's, parents') are punctuation,
    # not quoted substitute values.
    without_contractions = re.sub(r"(?<=\w)['’](?=\w)|(?<=s)['’](?!\w)", "", text)
    if re.search(r"['\"“”‘’]", without_contractions):
        return True
    chinese_number = "零〇一二两三四五六七八九十百千万"
    value_units = (
        "元",
        "块",
        "万",
        "折",
        "克",
        "盎司",
        "支",
        "朵",
        "束",
        "号",
        "楼",
        "栋",
        "单元",
        "室",
        "包间",
        "桌",
        "点",
        "分",
        "张",
        "份",
        "人均",
    )
    return bool(re.search(f"[{chinese_number}]+(?:{'|'.join(value_units)})", normalized))


def _validate_acoustic_fields(
    acoustic_environment: str | None,
    acoustic_asset_id: str | None,
    participants: list[LogicParticipant],
) -> None:
    """Require free-form acoustic text only when no local ambience asset is selected."""
    if acoustic_asset_id:
        return
    if acoustic_environment is None:
        raise ValueError("acoustic_environment is required without acoustic_asset_id")
    _validate_acoustic_environment(acoustic_environment, participants)


def _validate_acoustic_environment(
    acoustic_environment: str,
    participants: list[LogicParticipant],
) -> None:
    """Reject semantic/testcase content in the audio-generator sound bed field."""
    text = acoustic_environment.strip()
    if not text:
        raise ValueError("acoustic_environment is required")
    if any(ord(char) > 127 for char in text):
        raise ValueError("acoustic_environment must be English/ASCII for Stable Audio prompts")
    if any(char.isdigit() for char in text):
        raise ValueError("acoustic_environment must not include numbers, prices, or times")
    lowered = text.lower()
    forbidden_terms = (
        "assistant",
        "dialogue",
        "speech",
        "price",
        "yuan",
        "dollar",
        "secret",
        "birthday",
        "tool",
        "route",
        "destination",
        "navigation",
    )
    found_terms = [term for term in forbidden_terms if term in lowered]
    if found_terms:
        raise ValueError(
            "acoustic_environment must describe only non-speech background sound; "
            f"forbidden terms: {found_terms}"
        )
    participant_names = [
        participant.name
        for participant in participants
        if participant.name and len(participant.name.strip()) >= 3
    ]
    leaked_names = [name for name in participant_names if name.lower() in lowered]
    if leaked_names:
        raise ValueError(
            "acoustic_environment must not include participant names: "
            f"{leaked_names}"
        )


def _validate_participant_count(
    speaker_count: int,
    participant_names: set[str],
    participation_frame: ParticipationFrame,
) -> None:
    """Validate participant roster count against the context speaker count."""
    expected = (
        {speaker_count, speaker_count + 1}
        if participation_frame == ParticipationFrame.new_participant_joins
        else {speaker_count}
    )
    if len(participant_names) not in expected:
        raise ValueError("speaker_count must match unique participants")


def _validate_interaction_axis_pair(
    participation_frame: ParticipationFrame,
    expected_action: ExpectedAction,
) -> None:
    """Mirror the Cell-level incorporate/new-participant pairing in logic contracts."""
    if (
        expected_action == ExpectedAction.incorporate
        and participation_frame != ParticipationFrame.new_participant_joins
    ):
        raise ValueError(
            "expected_action 'incorporate' requires participation_frame 'new_participant_joins'"
        )


def _validate_participants_match_speakers(
    participant_names: set[str],
    context_human_speakers: set[str],
    final_speakers: list[str],
    participation_frame: ParticipationFrame,
) -> None:
    """Validate context/final speaker membership for ordinary and joiner cases."""
    if not participant_names:
        return
    final_humans = {speaker for speaker in final_speakers if speaker != "assistant"}
    if participation_frame == ParticipationFrame.new_participant_joins:
        expected_participants = context_human_speakers | final_humans
        if expected_participants != participant_names:
            raise ValueError("new participant roster must equal context speakers plus final joiner")
        if context_human_speakers & final_humans:
            raise ValueError("new participant final speaker must not appear in context lines")
        return
    if context_human_speakers != participant_names:
        raise ValueError("dialogue context human speakers must match participants")


def _foreground_participant_names(participants: list[LogicParticipant]) -> set[str]:
    return {
        participant.name
        for participant in participants
        if participant.speaker_distance in {"near", "mid"}
    }


def _far_participant_names(participants: list[LogicParticipant]) -> set[str]:
    return {
        participant.name
        for participant in participants
        if participant.speaker_distance == "far"
    }


def _validate_overlay_anchor_consistency(overlays: list[LogicCompetingOverlay]) -> None:
    anchors = [overlay.anchor_line_id for overlay in overlays]
    if not any(anchors):
        return
    if not all(anchors):
        raise ValueError("competing_overlays must all include anchor_line_id when any overlay is anchored")
    for anchor in anchors:
        if anchor is None or not re.fullmatch(r"L\d+", anchor):
            raise ValueError("competing_overlays anchor_line_id must look like L1, L2, ...")


def _validate_eavesdropping_overlay_plan_shape(
    participants: list[LogicParticipant],
    overlays: list[LogicCompetingOverlay],
) -> None:
    """Director plan: far speakers carry testcase speech in anchored overlays."""
    if not participants:
        raise ValueError("eavesdropping cases require participants with speaker_distance")
    far_speakers = _far_participant_names(participants)
    expected_far_count = 1 if len(participants) == 2 else 2
    foreground_speakers = _foreground_participant_names(participants)
    if len(far_speakers) != expected_far_count or len(foreground_speakers) != 1:
        raise ValueError(
            "eavesdropping cases require one foreground near/mid speaker and "
            f"{expected_far_count} far speaker(s)"
        )
    if not overlays:
        raise ValueError("eavesdropping cases require competing_overlays for far-field speech")
    _validate_overlay_anchor_consistency(overlays)
    if not all(overlay.anchor_line_id for overlay in overlays):
        raise ValueError("eavesdropping competing_overlays require anchor_line_id")
    overlay_speakers = {overlay.speaker for overlay in overlays}
    if overlay_speakers - far_speakers:
        raise ValueError(
            "eavesdropping competing_overlays may only use far participants as speakers: "
            f"{sorted(overlay_speakers - far_speakers)}"
        )
    if not far_speakers <= overlay_speakers:
        raise ValueError("eavesdropping far participants must speak in competing_overlays")


def _validate_eavesdropping_overlay_shape(
    participants: list[LogicParticipant],
    lines: list[Any],
    overlays: list[LogicCompetingOverlay],
    *,
    final_human_line_count: int,
    atomic_rubric: list[LogicRubricAtom] | None = None,
) -> None:
    """Final contract: dialogue is free-form; testcase facts live in far-field overlays."""
    _validate_eavesdropping_overlay_plan_shape(participants, overlays)
    if atomic_rubric:
        _validate_eavesdropping_far_field_exclusive_content(lines, overlays, atomic_rubric)
    participant_names = {participant.name for participant in participants}
    far_speakers = _far_participant_names(participants)
    final_start = _final_span_start(lines, final_human_line_count)
    context_lines = lines[:final_start]
    context_human_speakers = {
        line.speaker for line in context_lines if line.speaker != "assistant"
    }
    if context_human_speakers != participant_names:
        raise ValueError(
            "eavesdropping context human speakers must match all participants: "
            f"expected={sorted(participant_names)} actual={sorted(context_human_speakers)}"
        )
    overlay_speakers = {overlay.speaker for overlay in overlays}
    if overlay_speakers - far_speakers:
        raise ValueError(
            "eavesdropping competing_overlays may only use far participants as speakers: "
            f"{sorted(overlay_speakers - far_speakers)}"
        )
    if not far_speakers <= overlay_speakers:
        raise ValueError("eavesdropping far participants must speak in competing_overlays")
    context_line_ids = {line.line_id for line in context_lines}
    anchored_ids = {overlay.anchor_line_id for overlay in overlays if overlay.anchor_line_id}
    unknown_anchors = sorted(anchored_ids - context_line_ids)
    if unknown_anchors:
        raise ValueError(
            "eavesdropping overlay anchor_line_id must reference a context line: "
            + ", ".join(unknown_anchors)
        )
    _validate_eavesdropping_overlay_anchors(
        participants,
        context_lines,
        overlays,
    )


EAVESDROPPING_ANCHOR_LENGTH_RATIO = 1.2


def _leak_bigrams(text: str) -> set[str]:
    """Character bigrams over tag- and punctuation-stripped text for leak matching."""
    compact = re.sub(r"[^\w一-鿿]+", "", _normalize_disclosure_text(text))
    return {compact[index : index + 2] for index in range(len(compact) - 1)}


def _validate_eavesdropping_far_field_exclusive_content(
    lines: list[Any],
    overlays: list[LogicCompetingOverlay],
    atomic_rubric: list[LogicRubricAtom],
) -> None:
    """At least one overlay must be the only spoken source of rubric-required content."""
    line_bigrams: set[str] = set()
    for line in lines:
        line_bigrams |= _leak_bigrams(line.text)
    rubric_bigrams: set[str] = set()
    for atom in atomic_rubric:
        rubric_bigrams |= _leak_bigrams(atom.criterion)
    if not any(
        (_leak_bigrams(overlay.text) - line_bigrams) & rubric_bigrams
        for overlay in overlays
    ):
        raise ValueError(
            "eavesdropping decisive content leaked into spoken lines: no overlay carries "
            "rubric-referenced content that is absent from the foreground dialogue; the "
            "final prompt must reference the far-field detail without restating it"
        )


def _validate_eavesdropping_overlay_anchors(
    participants: list[LogicParticipant],
    context_lines: list[Any],
    overlays: list[LogicCompetingOverlay],
) -> None:
    """Ensure far-field overlays happen while the foreground speaker is talking."""
    foreground_speakers = _foreground_participant_names(participants)
    lines_by_id = {line.line_id: line for line in context_lines}
    for overlay in overlays:
        if overlay.anchor_line_id is None:
            continue
        anchor = lines_by_id.get(overlay.anchor_line_id)
        if anchor is None:
            continue
        if anchor.speaker == overlay.speaker:
            raise ValueError(
                "eavesdropping overlay speaker cannot match anchor line speaker: "
                f"{overlay.anchor_line_id} uses {overlay.speaker}"
            )
        if anchor.speaker == "assistant":
            raise ValueError(
                "eavesdropping overlay anchor_line_id must reference a foreground "
                f"human line, not assistant line {overlay.anchor_line_id}"
            )
        if anchor.speaker not in foreground_speakers:
            raise ValueError(
                "eavesdropping overlay anchor_line_id must reference a near/mid "
                f"foreground line: {overlay.anchor_line_id}"
            )
        anchor_chars = len(_normalize_disclosure_text(anchor.text))
        overlay_chars = len(_normalize_disclosure_text(overlay.text))
        if anchor_chars < EAVESDROPPING_ANCHOR_LENGTH_RATIO * overlay_chars:
            raise ValueError(
                "eavesdropping anchor line must be at least "
                f"{EAVESDROPPING_ANCHOR_LENGTH_RATIO}x the overlay text length so the "
                f"overlay stays masked by foreground speech: {overlay.anchor_line_id} "
                f"has {anchor_chars} chars vs overlay {overlay_chars} chars"
            )


def _validate_competing_overlay_shape(
    competing_speech: CompetingSpeechLevel,
    competing_overlay_count: int,
    overlays: list[LogicCompetingOverlay],
    *,
    pattern: LogicPattern | str | None = None,
) -> None:
    """Require side-speech transcript rows only when a competing-speech level is active."""
    if competing_speech == CompetingSpeechLevel.none:
        if pattern == "eavesdropping":
            return
        if overlays or competing_overlay_count:
            raise ValueError("competing_overlays require competing_speech to be non-none")
        return
    if not overlays:
        raise ValueError("competing_speech cases require competing_overlays")
    if competing_overlay_count and len(overlays) != competing_overlay_count:
        raise ValueError("competing_overlay_count must match competing_overlays length")
    if pattern != "eavesdropping":
        _validate_overlay_anchor_consistency(overlays)


def _validate_live_expansion_shape(
    case_mode: LogicCaseMode,
    pattern: LogicPattern,
    base_pattern: LogicPattern | None,
    live_expansion: LogicLiveExpansion | None,
    live_expansions: list[LogicLiveExpansion],
) -> None:
    """Keep test-time live probes tied to their semantic base pattern."""
    for item in live_expansions:
        if item.base_pattern != pattern:
            raise ValueError("live_expansions[].base_pattern must match pattern")
    if live_expansion is None:
        if base_pattern is not None and case_mode != "hear_time_probe":
            raise ValueError("base_pattern is only valid with singular live_expansion")
        return
    if base_pattern != pattern:
        raise ValueError("live_expansion base_pattern must match pattern")
    if live_expansion.base_pattern != pattern:
        raise ValueError("live_expansion.base_pattern must match pattern")
    if case_mode != "live_expansion":
        raise ValueError("singular live_expansion requires case_mode=live_expansion")


def validate_identity_exposure(
    identity_mode: IdentityMode,
    participants: list[LogicParticipant],
    lines: list[LogicLine],
) -> None:
    """Require audio-visible name grounding when identity_mode is in_audio.

    A participant counts as grounded when their name appears in the spoken text
    of any line spoken by someone else; a dedicated addressed_to name-calling
    line is not required.
    """
    if identity_mode == "in_text":
        return
    missing: list[str] = []
    for participant in participants:
        name = participant.name.strip()
        if not name:
            continue
        named_by_other = any(
            line.speaker != name and name in line.text for line in lines
        )
        if not named_by_other:
            missing.append(name)
    if missing:
        raise ValueError(
            "identity_mode=in_audio requires every participant to be named "
            f"in another speaker's spoken text: {missing}"
        )


def _validate_live_expansion_speaker_binding(
    lines: list[LogicLine],
    live_expansion: LogicLiveExpansion | None,
    live_expansions: list[LogicLiveExpansion],
) -> None:
    """Require bystander-interference live speech to reuse prior human speakers."""
    human_speakers = {line.speaker for line in lines if line.speaker != "assistant"}
    expansions = ([live_expansion] if live_expansion is not None else []) + list(live_expansions)
    for expansion in expansions:
        if expansion.type != "bystander_interference_suppression":
            continue
        if len(human_speakers) < 2:
            raise ValueError(
                "bystander_interference_suppression requires at least two prior human speakers"
            )
        for event in expansion.events:
            if event.speaker not in human_speakers:
                raise ValueError(
                    "bystander_interference_suppression speaker must be a prior human speaker"
                )
            if event.addressed_to not in human_speakers:
                raise ValueError(
                    "bystander_interference_suppression addressed_to must be a prior human speaker"
                )
            if event.addressed_to == event.speaker:
                raise ValueError(
                    "bystander_interference_suppression must be speaker-to-other-human"
                )
            if event.is_assistant_relevant:
                raise ValueError(
                    "bystander_interference_suppression live event must not be assistant-relevant"
                )


def _validate_hear_time_probes(
    lines: list[LogicLine], probes: list[LogicHearTimeProbe]
) -> None:
    """Ensure marked hear-time probes point to human-to-human dialogue lines."""
    marked = [line for line in lines if line.hear_time_probe]
    if len(marked) > 1:
        raise ValueError("at most one hear_time_probe line is allowed")
    if len(probes) > 1:
        raise ValueError("at most one hear_time_probe metadata row is allowed")
    line_by_id = {line.line_id: line for line in lines}
    for probe in probes:
        line = line_by_id.get(probe.line_id)
        if line is None:
            raise ValueError(f"hear_time_probe line_id not found: {probe.line_id}")
        if line.speaker != probe.speaker or line.addressed_to != probe.addressed_to:
            raise ValueError("hear_time_probe metadata must match the marked line")
        if line.addressed_to == "assistant" or line.speaker == "assistant":
            raise ValueError("hear_time_probe must be a human-to-human line")
    if marked and probes and marked[0].line_id != probes[0].line_id:
        raise ValueError("marked hear_time_probe line must match probe metadata")


def _validate_tiered_participants(
    pattern: LogicPattern, participants: list[LogicParticipant]
) -> None:
    """Require role-tier metadata for tiered speaker patterns."""
    if pattern not in TIERED_LOGIC_PATTERNS:
        return
    if not participants:
        raise ValueError("tiered logic pattern requires participants")
    missing: list[str] = []
    for participant in participants:
        fields: list[str] = []
        if participant.role_tier is None:
            fields.append("role_tier")
        if not participant.capabilities:
            fields.append("permission")
        if fields:
            missing.append(f"{participant.name}: {', '.join(fields)}")
    if missing:
        raise ValueError(
            "tiered logic pattern participants require role_tier and permission: "
            f"{'; '.join(missing)}"
        )
