"""The probe model: scripts, slots, bound probes, decisions and decision points in tagged lines."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ib.models.probe import (
    Probe,
    ProbeDecision,
    ProbePoint,
    ProbeSlot,
    GENERIC_KINDS,
    LEGACY_TOOL_KINDS,
    probe_decision_points,
    probe_script,
)


def _lines(*rows):
    return [
        SimpleNamespace(line_id=f"L{i}", speaker=speaker, addressed_to=addressed, text=text, probe=probe)
        for i, (speaker, addressed, text, probe) in enumerate(rows, 1)
    ]


@pytest.mark.parametrize(
    ("addressee", "follow_up", "actions"),
    [
        ("assistant", None, ["respond"]),
        ("other", None, ["ignore"]),
        ("ambiguous", "yes", ["clarify", "respond"]),
        ("ambiguous", "no", ["clarify", "ignore"]),
        ("ambiguous", "other_yes", ["clarify", "ignore", "respond"]),
    ],
)
def test_each_script_lists_the_decisions_the_assistant_owes(addressee, follow_up, actions) -> None:
    script = probe_script(addressee, follow_up)
    assert [action for _, action in script if action] == actions
    assert all(action is None for voice, action in script if voice == "assistant")


@pytest.mark.parametrize(
    "fields",
    [
        {"after_line": 0, "kind": "fact", "addressee": "assistant"},
        {"after_line": 1, "kind": "coffee", "addressee": "assistant"},
        {"after_line": 1, "kind": "fact", "addressee": "nobody"},
        {"after_line": 1, "kind": "fact", "addressee": "assistant", "follow_up": "yes"},
        {"after_line": 1, "kind": "fact", "addressee": "ambiguous"},
        {"after_line": 1, "kind": "fact", "addressee": "ambiguous", "follow_up": "maybe"},
    ],
)
def test_slots_reject_bad_values_and_follow_ups_that_do_not_fit(fields) -> None:
    with pytest.raises(ValueError):
        ProbeSlot(**fields)


def test_probe_binds_other_speaker_exactly_when_the_script_needs_one() -> None:
    Probe(after_line=2, kind="fact", addressee="other", speaker="Ann", other_speaker="Bob")
    Probe(after_line=2, kind="fact", addressee="ambiguous", follow_up="other_yes", speaker="Ann", other_speaker="Bob")
    Probe(after_line=2, kind="fact", addressee="ambiguous", follow_up="yes", speaker="Ann")
    for fields in (
        {"addressee": "other", "speaker": "Ann"},
        {"addressee": "other", "speaker": "Ann", "other_speaker": "Ann"},
        {"addressee": "assistant", "speaker": "Ann", "other_speaker": "Bob"},
        {"addressee": "ambiguous", "follow_up": "other_yes", "speaker": "Ann"},
    ):
        with pytest.raises(ValueError):
            Probe(after_line=2, kind="fact", **fields)


def test_the_drawn_kinds_are_questions_and_a_scene_probe_needs_no_request() -> None:
    assert GENERIC_KINDS == ("fact", "convert", "word", "tip")
    assert not set(GENERIC_KINDS) & set(LEGACY_TOOL_KINDS)
    assert Probe(after_line=1, kind="scene", addressee="assistant", speaker="Ann").scene_request is None
    with pytest.raises(ValueError):
        Probe(after_line=1, kind="convert", scene_request="order_drink", addressee="assistant", speaker="Ann")


def test_a_legacy_tool_probe_still_loads() -> None:
    """Contracts written before probes stopped calling tools still validate, call and all."""
    decision = ProbeDecision(
        line_id="L3", expected_action="respond", reply="Ordering a latte.", reply_line_id="L4",
        tool_call={"name": "order_drink", "arguments": {"drink": "latte"}},
    )
    probe = Probe(after_line=1, kind="scene", scene_request="order_drink", addressee="assistant", speaker="Ann", decisions=[decision])
    assert probe.decisions[0].tool_call["name"] == "order_drink"
    assert Probe(after_line=1, kind="timer", addressee="assistant", speaker="Ann").kind == "timer"


def test_decisions_fit_their_action() -> None:
    ProbeDecision(line_id="L3", expected_action="ignore")
    ProbeDecision(line_id="L3", expected_action="clarify", reply="Are you asking me?", reply_line_id="L4")
    ProbeDecision(line_id="L3", expected_action="respond", reply="About 454 grams.", reply_line_id="L4")
    for fields in (
        {"expected_action": "ignore", "reply": "x", "reply_line_id": "L4"},
        {"expected_action": "clarify", "reply": "x", "reply_line_id": "L4", "tool_call": {"name": "a", "arguments": {}}},
        {"expected_action": "respond", "reply": "", "reply_line_id": "L4"},
        {"expected_action": "respond", "reply": "x"},
    ):
        with pytest.raises(ValueError):
            ProbeDecision(line_id="L3", **fields)


def test_decisions_must_follow_the_script_order() -> None:
    with pytest.raises(ValueError, match="script actions"):
        Probe(
            after_line=2, kind="fact", addressee="ambiguous", follow_up="yes", speaker="Ann",
            decisions=[ProbeDecision(line_id="L3", expected_action="ignore")],
        )


def test_decision_points_pair_each_human_probe_line_with_the_assistant_line_after_it() -> None:
    lines = _lines(
        ("Ann", "Bob", "Dinner at seven?", None),
        ("Bob", "Ann", "Sure.", None),
        ("Ann", "group", "How many grams in a pound again?", 1),
        ("assistant", "Ann", "Are you asking me?", 1),
        ("Ann", "assistant", "Yes, you.", 1),
        ("assistant", "Ann", "About 454 grams.", 1),
        ("Bob", "Ann", "Mia, how do you spell rhythm?", 2),
        ("Bob", "assistant", "Nova, book it.", None),
    )
    yes = Probe(after_line=2, kind="convert", addressee="ambiguous", follow_up="yes", speaker="Ann")
    other = Probe(after_line=3, kind="word", addressee="other", speaker="Bob", other_speaker="Mia")
    assert probe_decision_points(yes, lines, 1) == [ProbePoint(2, "clarify", 3), ProbePoint(4, "respond", 5)]
    assert probe_decision_points(other, lines, 2) == [ProbePoint(6, "ignore", None)]
    with pytest.raises(ValueError, match="tagged lines"):
        probe_decision_points(yes, lines, 2)
