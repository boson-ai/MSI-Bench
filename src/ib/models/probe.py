"""probe — addressee probes: everyday questions inside the main dialogue that test who is addressed.

Calling spec:
    slot = ProbeSlot(after_line=3, kind="convert", addressee="ambiguous", follow_up="yes")
    turns = probe_script(addressee, follow_up)        -> (("speaker", "clarify"), ("assistant", None), ...)
    probe = Probe(after_line=3, kind="convert", addressee="ambiguous", follow_up="yes", speaker="柏霖")
    points = probe_decision_points(probe, lines, number) -> [ProbePoint(line_index, expected_action, reply_index), ...]
    main = main_lines(lines)                             -> the lines with no probe tag
    contract.probes = [Probe(..., decisions=[ProbeDecision(...)])]

A probe is a short run of lines inside the main dialogue, written by the actor and
marked with the probe's 1-based number on each line (LogicLine.probe): a participant
asks a short everyday question, unrelated to the main task, that the assistant answers
from general knowledge in a sentence or two, with no tool call; the assistant decides by
who is addressed. Only the final turn of a case may call a tool: a probe call would sit
in later history with no tool result ever following it. The kind (GENERIC_KINDS or
SCENE_KIND) says what sort of question; ib.planner.generation.probe_prompt.QUESTIONS words
each one for the actor. LEGACY_TOOL_KINDS (and a scene probe's scene_request) are the
tool-backed requests of contracts written before that rule; they are only read, never
drawn. The planner draws the slot (gap, kind, addressee, follow-up), the director binds
the voices (speaker; other_speaker for "other" and "other_yes"), the actor writes the
lines after probe_script, and the harness writes each decision's thinking. Every human
probe line is a decision point of the training data; the assistant line that follows it,
when the script has one, is the gold reply.

The script per addressee (PROBE_SCRIPTS), each turn (who speaks, the action the
assistant owes after a human turn):
    assistant        the question calls the assistant by name         -> respond (the spoken answer)
    other            the question is said to other_speaker by name    -> ignore
    ambiguous        the question names nobody                        -> clarify ("are you asking me?")
      follow_up yes        the speaker confirms                       -> respond (the spoken answer)
      follow_up no         the speaker meant someone else             -> ignore
      follow_up other_yes  other_speaker confirms instead             -> ignore (only the asker's own yes counts)
                           then the speaker confirms                  -> respond (the spoken answer)

Side effects: none.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import model_validator

from ib.models.base import IbBaseModel

ProbeKind = Literal["fact", "convert", "word", "tip", "scene", "weather", "time", "timer", "music"]
# The four generic question kinds the planner draws; "scene" asks about something in the setting.
GENERIC_KINDS: tuple[ProbeKind, ...] = ("fact", "convert", "word", "tip")
SCENE_KIND: ProbeKind = "scene"
# Tool-backed request kinds of older contracts: still read, never drawn.
LEGACY_TOOL_KINDS: tuple[ProbeKind, ...] = ("weather", "time", "timer", "music")
ProbeAddressee = Literal["assistant", "other", "ambiguous"]
PROBE_ADDRESSEES: tuple[ProbeAddressee, ...] = ("assistant", "other", "ambiguous")
ProbeFollowUp = Literal["yes", "no", "other_yes"]
PROBE_FOLLOW_UPS: tuple[ProbeFollowUp, ...] = ("yes", "no", "other_yes")
ProbeAction = Literal["respond", "ignore", "clarify"]
ProbeVoice = Literal["speaker", "other", "assistant"]
# One scripted turn: who speaks (ProbeVoice) and, for a human turn, what the assistant must do next.
ProbeTurn = tuple[ProbeVoice, ProbeAction | None]

# (addressee, follow_up) -> the turns the actor writes for the probe, in order.
PROBE_SCRIPTS: dict[tuple[ProbeAddressee, ProbeFollowUp | None], tuple[ProbeTurn, ...]] = {
    ("assistant", None): (("speaker", "respond"), ("assistant", None)),
    ("other", None): (("speaker", "ignore"),),
    ("ambiguous", "yes"): (("speaker", "clarify"), ("assistant", None), ("speaker", "respond"), ("assistant", None)),
    ("ambiguous", "no"): (("speaker", "clarify"), ("assistant", None), ("speaker", "ignore")),
    ("ambiguous", "other_yes"): (
        ("speaker", "clarify"), ("assistant", None), ("other", "ignore"), ("speaker", "respond"), ("assistant", None),
    ),
}


def probe_script(addressee: ProbeAddressee, follow_up: ProbeFollowUp | None) -> tuple[ProbeTurn, ...]:
    """Return the turns of a probe with this addressee and follow-up; raise KeyError for a pair no script has."""
    return PROBE_SCRIPTS[addressee, follow_up]


def needs_other_speaker(addressee: ProbeAddressee, follow_up: ProbeFollowUp | None) -> bool:
    """Whether the probe binds a second participant: it is addressed to one, or one wrongly confirms."""
    return addressee == "other" or follow_up == "other_yes"


def _check_slot(after_line: int, kind: str, addressee: str, follow_up: str | None) -> None:
    if after_line < 1:
        raise ValueError(f"probe after_line must be >= 1, got {after_line}")
    if kind not in (*GENERIC_KINDS, SCENE_KIND, *LEGACY_TOOL_KINDS):
        raise ValueError(f"probe kind must be one of {(*GENERIC_KINDS, SCENE_KIND)} (or a legacy {LEGACY_TOOL_KINDS}), got {kind!r}")
    if addressee not in PROBE_ADDRESSEES:
        raise ValueError(f"probe addressee must be one of {PROBE_ADDRESSEES}, got {addressee!r}")
    if (addressee, follow_up) not in PROBE_SCRIPTS:
        raise ValueError(
            f"probe addressee {addressee!r} takes follow_up "
            f"{'one of ' + str(PROBE_FOLLOW_UPS) if addressee == 'ambiguous' else 'None'}, got {follow_up!r}"
        )


@dataclass(frozen=True)
class ProbeSlot:
    """Target-level probe drawn before any line exists.

    after_line is 1-based: the probe's lines follow the first `after_line` main lines
    (the actor places them after about that many main lines).
    """

    after_line: int
    kind: ProbeKind
    addressee: ProbeAddressee
    follow_up: ProbeFollowUp | None = None

    def __post_init__(self) -> None:
        """Raise ValueError when after_line < 1, the kind or addressee is unknown, or follow_up does not fit the addressee (ambiguous needs one, the others none)."""
        _check_slot(self.after_line, self.kind, self.addressee, self.follow_up)


class ProbeDecision(IbBaseModel):
    """One decision point of a probe: the human line decided on and the assistant's gold for it.

    `reply` is the text of the assistant line that follows (None when the action is ignore),
    `reply_line_id` that line's id, `tool_call` None (a legacy contract's respond decision may
    hold {"name", "arguments"}), `thinking` the harness's rationale (blank outside traindata).
    """

    line_id: str
    expected_action: ProbeAction
    reply: str | None = None
    reply_line_id: str | None = None
    tool_call: dict[str, Any] | None = None
    thinking: str = ""

    @model_validator(mode="after")
    def _fits_action(self) -> "ProbeDecision":
        """Reject a tool call outside respond, a reply on ignore, or a missing reply on respond or clarify."""
        if self.expected_action == "ignore":
            if self.reply is not None or self.reply_line_id is not None or self.tool_call is not None:
                raise ValueError("an ignore decision has no reply and no tool_call")
            return self
        if not (self.reply or "").strip() or not self.reply_line_id:
            raise ValueError(f"a {self.expected_action} decision needs the assistant's reply line")
        if self.expected_action == "clarify" and self.tool_call is not None:
            raise ValueError("a clarify decision has no tool_call")
        return self


class Probe(IbBaseModel):
    """Plan- and contract-level probe: the slot, its bound voices and, once the harness ran, its decisions."""

    after_line: int
    kind: ProbeKind
    scene_request: str | None = None  # legacy: the catalog request (order_drink) of an older scene probe
    addressee: ProbeAddressee
    follow_up: ProbeFollowUp | None = None
    speaker: str
    other_speaker: str | None = None  # the participant addressed (other) or wrongly confirming (other_yes)
    decisions: list[ProbeDecision] = []

    @model_validator(mode="after")
    def _slot_and_voices_fit(self) -> "Probe":
        """Reject a bad slot, a scene_request on a probe that is not a scene probe, a missing other_speaker when the script needs one, or one equal to the speaker."""
        _check_slot(self.after_line, self.kind, self.addressee, self.follow_up)
        if self.scene_request and self.kind != SCENE_KIND:
            raise ValueError("scene_request is set only on a scene probe")
        if needs_other_speaker(self.addressee, self.follow_up):
            if not self.other_speaker or self.other_speaker == self.speaker:
                raise ValueError(f"a {self.addressee}/{self.follow_up} probe needs another participant than {self.speaker!r}")
        elif self.other_speaker is not None:
            raise ValueError("other_speaker is bound only for other or other_yes probes")
        actions = [action for _, action in probe_script(self.addressee, self.follow_up) if action]
        if self.decisions and [d.expected_action for d in self.decisions] != actions:
            raise ValueError(f"probe decisions must follow the script actions {actions}")
        return self

    @property
    def script(self) -> tuple[ProbeTurn, ...]:
        return probe_script(self.addressee, self.follow_up)

    def voice(self, turn: ProbeVoice) -> str:
        """Return the participant name (or "assistant") speaking a scripted turn."""
        return {"speaker": self.speaker, "other": self.other_speaker, "assistant": "assistant"}[turn]


@dataclass(frozen=True)
class ProbePoint:
    """One decision point located in the lines: the human line's index, the action, and the reply line's index (None for ignore)."""

    line_index: int
    expected_action: ProbeAction
    reply_index: int | None


def main_lines(lines: Sequence[Any]) -> list[Any]:
    """Return the lines of the main dialogue: those whose `probe` is None."""
    return [line for line in lines if getattr(line, "probe", None) is None]


def probe_line_indexes(lines: Sequence[Any], number: int) -> list[int]:
    """Return the indexes of the lines tagged with probe `number` (1-based), in order."""
    return [index for index, line in enumerate(lines) if getattr(line, "probe", None) == number]


def probe_decision_points(probe: Probe, lines: Sequence[Any], number: int) -> list[ProbePoint]:
    """Return the decision points of probe `number` (1-based) from its tagged lines and script.

    Raises ValueError when the tagged lines are not exactly as many as the script's turns.
    Each human turn gives a ProbePoint whose reply_index is the following assistant turn's
    index when the script has one, else None. Pure.
    """
    indexes = probe_line_indexes(lines, number)
    script = probe.script
    if len(indexes) != len(script):
        raise ValueError(f"probe {number} has {len(indexes)} tagged lines, its script has {len(script)} turns")
    points = []
    for position, (voice, action) in enumerate(script):
        if voice == "assistant":
            continue
        follows = position + 1 < len(script) and script[position + 1][0] == "assistant"
        points.append(ProbePoint(indexes[position], action, indexes[position + 1] if follows else None))
    return points
