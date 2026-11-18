"""logic — machine-readable contracts for hard interaction-logic cases.

Calling spec:
    contract = LogicContract(...)
    tier_capabilities(tiers: list[LogicTier]) -> dict[int, list[str]]

A LogicContract carries the hidden oracle for cases where the answer target is a
multi-constraint situated plan rather than a generic dialogue answer. It is pure
schema: generation, freezing, and scoring logic live in planner/scoring modules.
Validators only normalize, apart from the probe check (each probe's voices are
participants) and the tier-table check (tiers numbered 1..k in order, each granting
capabilities no other tier grants). `probes` are the addressee probes (ib.models.probe)
the planner drew and the director bound; on a contract the harness has filled their
decisions.
Account authority lives on the tier, never on the person: tier_capabilities gives
each tier its cumulative capabilities, and a participant holds those of its role_tier.
Tiers are generation metadata for the director, actor and harness; the evaluated model
never sees them and learns who holds authority only from the dialogue.
The text.rule gates are called explicitly by the stage that produces a plan or
contract (ib.gates), so loading a stored contract never re-gates it.

Side effects: none.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from ib.models.background_line import BackgroundLine
from ib.models.base import IbBaseModel
from ib.models.floor import FloorPlan
from ib.models.enums import CompetingSpeechLevel, ExpectedAction, ParticipationFrame
from ib.models.probe import Probe
from ib.models.tool_call_args import gold_tool_arg_policy, validate_gold_tool_arguments

LogicPattern = str
VoiceGender = Literal["male", "female"]
SpeakerDistance = Literal["near", "mid", "far"]

# Patterns whose contracts carry tiers and whose participants carry role_tier (account authority).
AUTHORITY_PATTERN = "authority_gated_override"
DISCLOSURE_PATTERN = "disclosure_clause_in_instruction"
TIERED_LOGIC_PATTERNS = {AUTHORITY_PATTERN}
# The planner-drawn outcome of a disclosure case's final probe: the secret_instructor's own voice
# probes (disclose: the assistant may tell them) or any other participant does (withhold).
DisclosureOutcome = Literal["withhold", "disclose"]
# The planner-drawn way an authority case says whose the changed thing is: the authorized
# person says it ("self"), or another voice says it to them right after they spoke
# ("other_after") or calling them by name, the authorized person answering next ("other_call").
OwnershipCue = Literal["self", "other_after", "other_call"]
OWNERSHIP_CUES: tuple[OwnershipCue, ...] = ("self", "other_after", "other_call")


# Dialogue length limits a contract must respect (targets sample within them).
MIN_LOGIC_LINE_COUNT = 6
MAX_LOGIC_LINE_COUNT = 15
# Two people cannot naturally sustain twelve turns of one conflict; cap them
# lower so the freed line count cannot be filled with repeated beats.
MAX_LOGIC_LINE_COUNT_BY_SPEAKER_COUNT = {2: 9, 3: 12, 4: 12, 5: 20}
# target.line_count is the minimum; the actor chooses the actual count up to the
# ceiling so information can spread across breathable turns. The keys are the
# supported human speaker counts (the assistant is not counted).
MIN_LOGIC_LINE_COUNT_BY_SPEAKER_COUNT = {
    2: 6,
    3: 8,
    4: 10,
    5: 15,
}
SUPPORTED_LOGIC_SPEAKER_COUNTS = tuple(MIN_LOGIC_LINE_COUNT_BY_SPEAKER_COUNT)
MIN_SPEAKER_COUNT = min(SUPPORTED_LOGIC_SPEAKER_COUNTS)
MAX_SPEAKER_COUNT = max(SUPPORTED_LOGIC_SPEAKER_COUNTS)


def max_logic_line_count(speaker_count: int) -> int:
    """Return the dialogue line-count ceiling for one speaker count."""
    return MAX_LOGIC_LINE_COUNT_BY_SPEAKER_COUNT.get(speaker_count, MAX_LOGIC_LINE_COUNT)


# Contract and plan fields dropped from the schema that old frozen and resume rows still carry.
_RETIRED_CONTRACT_KEYS = ("identity_mode", "protected_audience")
# Line fields of retired in-dialogue side branches that old rows still carry.
_RETIRED_LINE_KEYS = ("side_branch", "side_branch_reply")

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
    """One play line in a hard-logic dialogue.

    `probe` is the 1-based number of the addressee probe (contract.probes) the line belongs
    to, None for a main-dialogue line (ib.models.probe.main_lines).
    """

    line_id: str
    speaker: str
    addressed_to: str
    text: str
    stage_directions: list[str] = Field(default_factory=list)
    hear_time_probe: bool = Field(default=False, exclude_if=lambda value: not value)
    probe: int | None = Field(default=None, ge=1, exclude_if=lambda value: value is None)

    @model_validator(mode="before")
    @classmethod
    def _strip_retired_line_fields(cls, data):
        """Drop the retired side_branch and side_branch_reply keys (_RETIRED_LINE_KEYS) from an old row, so it loads; the loaded line has neither field."""
        if not isinstance(data, dict) or not any(key in data for key in _RETIRED_LINE_KEYS):
            return data
        return {key: value for key, value in data.items() if key not in _RETIRED_LINE_KEYS}


class LogicLinePlan(IbBaseModel):
    """One planned dialogue beat before spoken text is rendered."""

    line_id: str
    speaker: str
    addressed_to: str
    beat: str
    subtext: str = ""
    must_surface: list[str] = Field(default_factory=list)
    must_not_say: list[str] = Field(default_factory=list)


class LogicTier(IbBaseModel):
    """One account-authority tier; `grants` are the capabilities it adds over the tier below."""

    tier: int = Field(ge=1)
    grants: list[str] = Field(min_length=1)


def tier_capabilities(tiers: list[LogicTier]) -> dict[int, list[str]]:
    """Map each tier to its cumulative capabilities: the grants of tiers 1..t, in tier then grant order."""
    capabilities: dict[int, list[str]] = {}
    held: list[str] = []
    for tier in sorted(tiers, key=lambda item: item.tier):
        held = [*held, *tier.grants]
        capabilities[tier.tier] = held
    return capabilities


def _validate_probe_voices(probes: list[Probe], participants: list["LogicParticipant"]) -> None:
    """Reject a probe whose speaker or other_speaker is not a participant (plans without participants are not checked)."""
    names = {participant.name for participant in participants}
    if not names:
        return
    for probe in probes:
        for voice in (probe.speaker, probe.other_speaker):
            if voice is not None and voice not in names:
                raise ValueError(f"probe after line {probe.after_line}: {voice!r} is not a participant")


def _validate_tiers(tiers: list[LogicTier]) -> list[LogicTier]:
    """Reject tiers not numbered 1..k in order, or a capability granted more than once."""
    numbers = [tier.tier for tier in tiers]
    if numbers != list(range(1, len(tiers) + 1)):
        raise ValueError(f"tiers must be numbered 1..{len(tiers)} in order, got {numbers}")
    granted = [grant for tier in tiers for grant in tier.grants]
    repeated = sorted({grant for grant in granted if granted.count(grant) > 1})
    if repeated:
        raise ValueError(f"capabilities granted more than once across tiers: {repeated}")
    return tiers


def _strip_retired_contract_keys(data: Any) -> Any:
    """Return a raw contract or plan row without the _RETIRED_CONTRACT_KEYS keys; other input unchanged."""
    if not isinstance(data, dict) or not any(key in data for key in _RETIRED_CONTRACT_KEYS):
        return data
    return {key: value for key, value in data.items() if key not in _RETIRED_CONTRACT_KEYS}


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

    @model_validator(mode="before")
    @classmethod
    def _migrate_participant_aliases(cls, data):
        """Accept legacy display aliases; drop per-person capability keys (capabilities live on tiers)."""
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
        for alias in (
            "voice_gender",
            "voiceGender",
            "voice_sex",
            "distance",
            "role_distance",
            "speakerDistance",
            "speaker_field",
            "capabilities",
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


class LogicResolution(IbBaseModel):
    """Hidden oracle response for a hard-logic case."""

    response_summary: str
    standard_answer: str = ""
    thinking: str = ""
    available_functions: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[ExpectedToolCall] = Field(default_factory=list)
    state_updates: dict[str, Any] = Field(default_factory=dict)

    @field_validator("available_functions")
    @classmethod
    def _arguments_in_name_order(cls, functions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Order each function's arguments and argument_descriptions by name.

        Stored contracts are written with sorted keys, so a contract read back has its
        arguments in name order; holding that order in memory too keeps the tool
        definitions of a training sample (ib.manifest.tool_definitions) the same
        whether the contract was just built or read back.
        """
        return [
            {
                **function,
                **{
                    key: dict(sorted(function[key].items()))
                    for key in ("arguments", "argument_descriptions")
                    if isinstance(function.get(key), dict)
                },
            }
            for function in functions
        ]


class HarnessProbeDecision(IbBaseModel):
    """The harness's gold for one probe decision point: the probe's 1-based number, the human line decided on and the thinking; tool_call stays None (text.rule.probe_targets)."""

    probe: int = Field(ge=1)
    line_id: str
    tool_call: dict[str, Any] | None = None
    thinking: str = ""


class LogicHarness(IbBaseModel):
    """Harness-generated tools, oracle answer, grading atoms and probe decisions."""

    response_summary: str
    standard_answer: str = ""
    thinking: str = ""
    available_functions: list[dict[str, Any]] = Field(min_length=1)
    tool_calls: list[ExpectedToolCall] = Field(default_factory=list)
    atomic_rubric: list[LogicRubricAtom] = Field(default_factory=list)
    probe_decisions: list[HarnessProbeDecision] = Field(default_factory=list)


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
    speaker_count: int = Field(ge=MIN_SPEAKER_COUNT, le=MAX_SPEAKER_COUNT)
    title: str
    scene_date: str = Field(default="", exclude_if=lambda value: not value)
    # The assistant's name in this case, carried from the target like scene_date; the
    # speakers call the assistant by it and no participant has it.
    assistant_name: str = Field(default="", exclude_if=lambda value: not value)
    scene_context: str
    private_memory: str = Field(default="", exclude_if=lambda value: not value)
    secret_instructor: str = Field(default="", exclude_if=lambda value: not value)
    disclosure_outcome: DisclosureOutcome | None = Field(default=None, exclude_if=lambda value: value is None)
    ownership_cue: OwnershipCue | None = Field(default=None, exclude_if=lambda value: value is None)
    acoustic_asset_id: str | None = Field(default=None, min_length=1)
    acoustic_facets: dict[str, str] = Field(default_factory=dict, exclude_if=lambda value: not value)
    acoustic_retrieval: dict[str, Any] | None = Field(default=None, exclude_if=lambda value: value is None)
    primary_goal: str = ""
    participants: list[LogicParticipant] = Field(default_factory=list)
    tiers: list[LogicTier] = Field(default_factory=list, exclude_if=lambda value: not value)
    probes: list[Probe] = Field(default_factory=list)
    # Background line (see ib.models.background_line); the background writer fills it.
    background: BackgroundLine | None = None
    dropped_background: dict[str, Any] | None = None
    # Floor plan (see ib.models.floor); the writer's full-duplex markup fills it.
    floor: FloorPlan | None = None
    dropped_floor: dict[str, Any] | None = None
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
    competing_overlays: list[LogicCompetingOverlay] = Field(default_factory=list)
    lines: list[LogicLine] = Field(min_length=2)
    resolution: LogicResolution
    atomic_rubric: list[LogicRubricAtom] = Field(default_factory=list)
    traps: list[str] = Field(default_factory=list)
    ability_tags: list[str] = Field(default_factory=list)
    provenance: str = "generated_from_public_logic_blueprint"

    @model_validator(mode="before")
    @classmethod
    def _strip_retired_fields(cls, data):
        """Ignore contract fields retired from the schema."""
        return _strip_retired_contract_keys(data)

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

    @field_validator("tiers")
    @classmethod
    def _tiers_are_contiguous_and_disjoint(cls, tiers: list[LogicTier]) -> list[LogicTier]:
        return _validate_tiers(tiers)

    @model_validator(mode="after")
    def _default_primary_goal(self) -> "LogicContract":
        if self.causal_frame is not None and not self.primary_goal:
            self.primary_goal = self.causal_frame.primary_goal
        return self

    @model_validator(mode="after")
    def _probes_fit_participants(self) -> "LogicContract":
        _validate_probe_voices(self.probes, self.participants)
        return self

    @model_validator(mode="after")
    def _background_fits_contract(self) -> "LogicContract":
        """Reject a background speaker named like a participant, or a background in another language.

        The contract's language is resolution.state_updates["language"]. A contract
        with no background is not checked.
        """
        if self.background is not None:
            participant_names = {participant.name for participant in self.participants}
            clashes = sorted(
                speaker.name for speaker in self.background.speakers
                if speaker.name in participant_names
            )
            if clashes:
                raise ValueError(f"background speakers {clashes} are named like participants")
            language = self.resolution.state_updates.get("language")
            if self.background.language != language:
                raise ValueError(
                    f"background language {self.background.language!r} differs from "
                    f"contract language {language!r}"
                )
        return self


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
    speaker_count: int = Field(ge=MIN_SPEAKER_COUNT, le=MAX_SPEAKER_COUNT)
    title: str
    scene_date: str = Field(default="", exclude_if=lambda value: not value)
    # The assistant's name in this case, carried from the target like scene_date; the
    # speakers call the assistant by it and no participant has it.
    assistant_name: str = Field(default="", exclude_if=lambda value: not value)
    scene_context: str
    private_memory: str = Field(default="", exclude_if=lambda value: not value)
    secret_instructor: str = Field(default="", exclude_if=lambda value: not value)
    disclosure_outcome: DisclosureOutcome | None = Field(default=None, exclude_if=lambda value: value is None)
    ownership_cue: OwnershipCue | None = Field(default=None, exclude_if=lambda value: value is None)
    acoustic_asset_id: str | None = Field(default=None, min_length=1)
    acoustic_facets: dict[str, str] = Field(default_factory=dict, exclude_if=lambda value: not value)
    acoustic_retrieval: dict[str, Any] | None = Field(default=None, exclude_if=lambda value: value is None)
    primary_goal: str
    participants: list[LogicParticipant] = Field(default_factory=list)
    tiers: list[LogicTier] = Field(default_factory=list, exclude_if=lambda value: not value)
    probes: list[Probe] = Field(default_factory=list)
    # The case's functions when the director designs them (two_call flow); [] in the staged flow.
    functions: list[dict[str, Any]] = Field(default_factory=list, exclude_if=lambda value: not value)
    background: BackgroundLine | None = None
    dropped_background: dict[str, Any] | None = None
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
    competing_overlays: list[LogicCompetingOverlay] = Field(default_factory=list)
    resolution: LogicResolution | None = None
    atomic_rubric: list[LogicRubricAtom] = Field(default_factory=list)
    traps: list[str] = Field(default_factory=list)
    ability_tags: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _strip_retired_fields(cls, data):
        """Ignore plan fields retired from the schema."""
        return _strip_retired_contract_keys(data)

    @field_validator("tiers")
    @classmethod
    def _tiers_are_contiguous_and_disjoint(cls, tiers: list[LogicTier]) -> list[LogicTier]:
        return _validate_tiers(tiers)

    @model_validator(mode="after")
    def _probes_fit_participants(self) -> "LogicContractPlan":
        _validate_probe_voices(self.probes, self.participants)
        return self

    @model_validator(mode="after")
    def _default_participants_from_line_plans(self) -> "LogicContractPlan":
        if not self.participants and self.line_plans:
            self.participants = [
                LogicParticipant(name=name)
                for name in sorted({line.speaker for line in self.line_plans})
            ]
        return self
