"""floor — floor-layer schema: the planner's slot, the written acts and the contract's floor plan.

Calling spec:
    target.floor = FloorSlot(backchannels=2, failed_claims=1, interruptions=1, holds=1)
    contract.floor = FloorPlan(acts=[FloorAct(...)], holds=["L4"])
    contract.floor = FloorPlan(acts=[...], holds=["L4"], hold_after={"L4": "so what we need is"},
                               gaps={"L6": "latched"})          the script markup's extras
    contract.dropped_floor = {"reason": str, "failed_gates": [gate_id, ...]}
    contract.dropped_floor = {..., "dropped_acts": [{"kind": "interruption", "holder_line_id",
                              "speaker", "tail", "reason"}, ...]}   markup acts dropped, the rest kept
                              (kinds: interruption, backchannel, failed_claim, hold, gap;
                              ib.planner.generation.markup_salvage lists each record's fields)

The floor layer makes a training sample full duplex. The contract lines stay as written;
the layer adds, on top of them, short acts by other participants during a holder line and
pauses inside a line:
    backchannel     a listener's "mm-hm" during the holder line; the holder keeps the floor
    failed_claim    a listener's cut-off bid for the floor ("wait, which—") that fails
    interruption    the holder's own droppable `tail` after its line is cut by the next
                    speaker's lead-in `text`; the next contract line (the interrupter's) follows
    hold            a mid-line pause of the holder (holds: line ids), the floor kept
A holder may be a context assistant line; its tail is then spoken in the assistant voice.
The speaker of an act is a participant, never the holder; "assistant" only for a backchannel
(the assistant's listening word during a line said to it, spoken in the assistant voice).

Two sources fill a plan. The floor writer (scheduled acts): one act per holder line, an
interruption with a lead-in `text`, holds split where the timeline finds a pause. The script
markup (ib.planner.generation.floor_markup): a backchannel or failed claim carries `after`,
the quote of its holder line it follows (several such acts may share a line); an
interruption carries no lead-in (`text` ""), the interrupter's own next contract line is the
barge-in; `hold_after[line]` is the words before a hold's pause; `gaps[line]` is latched or
long_pause, or "departure" on the line after a disclose case's departure_line (set by the
planner, never written as markup). The script writer and the staged and two_call actors write that markup
(ib.planner.generation.markup_check); markup the gate rejects, and a
staged barge-in whose tail the harness's gold then hears, are dropped and listed in
dropped_floor["dropped_acts"] (and, with ids M1, M2, ..., in the training record's
floor.dropped_acts, ib.audio.floor_events.markup_drops). A line holds at most one unanchored act (one without `after`), and a held line
carries none.

FloorSlot is the planner's draw for one target (how many of each to schedule).
FloorAct is one written act (ib.planner.generation.floor_writer fills text and tail into the
slots ib.planner.generation.floor_schedule picked). FloorPlan sits on the contract like
contract.background; dropped_floor records why a plan was dropped (the case still passes).

Side effects: none.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import Field, model_validator

from ib.models.base import IbBaseModel

FloorKind = Literal["backchannel", "failed_claim", "interruption"]
FLOOR_KINDS: tuple[FloorKind, ...] = ("backchannel", "failed_claim", "interruption")


@dataclass(frozen=True)
class FloorSlot:
    """How many acts of each kind and how many holds the scheduler places for one target."""

    backchannels: int = 0
    failed_claims: int = 0
    interruptions: int = 0
    holds: int = 0

    @property
    def empty(self) -> bool:
        return not (self.backchannels or self.failed_claims or self.interruptions or self.holds)


class FloorAct(IbBaseModel):
    """One written floor act during `holder_line_id`, spoken by `speaker`; `tail` only for an interruption.

    `after` (script markup) is the quote of the holder line the act follows; an interruption
    with no `text` has no lead-in. `unsaid`, only on a failed claim, is the rest of the sentence
    its speaker was about to say: never spoken or heard, it is the text a speaker the model plays
    had planned (`tail` is the holder's audible continuation that a barge-in cuts off).
    """

    act_id: str
    kind: FloorKind
    holder_line_id: str
    speaker: str
    text: str
    tail: str = Field(default="", exclude_if=lambda value: not value)
    after: str = Field(default="", exclude_if=lambda value: not value)
    unsaid: str = Field(default="", exclude_if=lambda value: not value)

    @model_validator(mode="after")
    def _tail_fits_kind(self) -> "FloorAct":
        """Reject an interruption without a tail, a tail on another kind, an `after` on an interruption,
        speaker "assistant" on anything but a backchannel, an empty text on a backchannel or failed claim,
        or an `unsaid` on anything but a failed claim."""
        if self.kind == "interruption" and not self.tail.strip():
            raise ValueError(f"floor act {self.act_id}: an interruption needs the holder's tail")
        if self.kind != "interruption" and self.tail:
            raise ValueError(f"floor act {self.act_id}: only an interruption carries a tail")
        if self.speaker == "assistant" and self.kind != "backchannel":
            raise ValueError(f"floor act {self.act_id}: the assistant performs no floor act but a backchannel")
        if self.kind == "interruption" and self.after:
            raise ValueError(f"floor act {self.act_id}: an interruption follows its holder line, it has no after quote")
        if self.kind != "interruption" and not self.text.strip():
            raise ValueError(f"floor act {self.act_id}: text is empty")
        if self.kind != "failed_claim" and self.unsaid:
            raise ValueError(f"floor act {self.act_id}: only a failed claim carries an unsaid rest")
        return self


class FloorPlan(IbBaseModel):
    """A case's floor plan: its written acts, the line ids that get a mid-line hold, and the script's extras."""

    acts: list[FloorAct] = Field(default_factory=list)
    holds: list[str] = Field(default_factory=list)
    hold_after: dict[str, str] = Field(default_factory=dict, exclude_if=lambda value: not value)
    gaps: dict[str, Literal["latched", "long_pause", "departure"]] = Field(default_factory=dict, exclude_if=lambda value: not value)

    @model_validator(mode="after")
    def _one_act_per_holder(self) -> "FloorPlan":
        """Reject duplicate act ids, two unanchored acts on one holder line, a hold on an unanchored act's holder
        line, a hold_after for a line without a hold, or an empty plan."""
        ids = [act.act_id for act in self.acts]
        if len(set(ids)) != len(ids):
            raise ValueError(f"floor act ids must be unique: {ids}")
        if not set(self.hold_after) <= set(self.holds):
            raise ValueError(f"hold_after names lines without a hold: {sorted(set(self.hold_after) - set(self.holds))}")
        holders = [act.holder_line_id for act in self.acts if not act.after]
        if len(set(holders)) != len(holders):
            raise ValueError(f"at most one floor act per holder line: {holders}")
        if len(set(self.holds)) != len(self.holds):
            raise ValueError(f"floor holds must be unique: {self.holds}")
        clash = sorted(set(holders) & set(self.holds))
        if clash:
            raise ValueError(f"a line cannot both hold and carry a floor act: {clash}")
        if not self.acts and not self.holds and not self.gaps:
            raise ValueError("a floor plan needs at least one act, hold or gap")
        return self
