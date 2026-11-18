"""Account authority lives on the tier: cumulative capabilities and the tier-table invariants."""

import pytest
from pydantic import ValidationError

from ib.models.logic import LogicContract, LogicContractPlan, LogicTier, tier_capabilities

THREE_TIERS = [
    {"tier": 1, "grants": ["set_climate", "play_music"]},
    {"tier": 2, "grants": ["add_stop"]},
    {"tier": 3, "grants": ["unlock_trunk", "share_location"]},
]


def _plan(**updates):
    data = {
        "pattern": "authority_gated_override",
        "speaker_count": 2,
        "title": "Trunk unlock",
        "scene_context": "Two riders by the car.",
        "primary_goal": "Open the trunk.",
        "participants": [{"name": "Maya", "role_tier": 3}, {"name": "Jon", "role_tier": 1}],
        "tiers": THREE_TIERS,
    }
    data.update(updates)
    return LogicContractPlan(**data)


def test_each_tier_holds_its_own_grants_and_every_lower_tiers_grants_in_order():
    tiers = [LogicTier(**tier) for tier in THREE_TIERS]

    assert tier_capabilities(tiers) == {
        1: ["set_climate", "play_music"],
        2: ["set_climate", "play_music", "add_stop"],
        3: ["set_climate", "play_music", "add_stop", "unlock_trunk", "share_location"],
    }


def test_tier_capabilities_of_no_tiers_is_empty():
    assert tier_capabilities([]) == {}


def test_plan_keeps_its_tier_table_and_dumps_it():
    plan = _plan()

    assert [tier.tier for tier in plan.tiers] == [1, 2, 3]
    assert plan.model_dump(mode="json")["tiers"] == THREE_TIERS


def test_a_plan_without_tiers_dumps_no_tiers_key():
    plan = _plan(pattern="distributed_parameters", participants=[{"name": "Maya"}, {"name": "Jon"}], tiers=[])

    assert "tiers" not in plan.model_dump(mode="json")


@pytest.mark.parametrize(
    "tiers",
    [
        [{"tier": 1, "grants": ["a"]}, {"tier": 3, "grants": ["b"]}],
        [{"tier": 2, "grants": ["a"]}, {"tier": 1, "grants": ["b"]}],
        [{"tier": 2, "grants": ["a"]}],
    ],
    ids=["gap", "out_of_order", "not_from_one"],
)
def test_tiers_must_be_numbered_one_to_k_in_order(tiers):
    with pytest.raises(ValidationError, match=r"tiers must be numbered 1\.\.\d+ in order"):
        _plan(tiers=tiers)


def test_a_tier_must_grant_something():
    with pytest.raises(ValidationError, match="grants"):
        _plan(tiers=[{"tier": 1, "grants": ["a"]}, {"tier": 2, "grants": []}])


def test_a_capability_granted_by_two_tiers_is_rejected():
    with pytest.raises(ValidationError, match=r"granted more than once across tiers: \['a'\]"):
        _plan(tiers=[{"tier": 1, "grants": ["a"]}, {"tier": 2, "grants": ["b", "a"]}])


def test_the_contract_enforces_the_same_tier_invariants():
    with pytest.raises(ValidationError, match="granted more than once"):
        LogicContract(
            contract_id="demo",
            pattern="authority_gated_override",
            speaker_count=2,
            title="Demo",
            scene_context="In a car.",
            participants=[{"name": "a", "role_tier": 2}, {"name": "b", "role_tier": 1}],
            tiers=[{"tier": 1, "grants": ["a"]}, {"tier": 2, "grants": ["a"]}],
            lines=[
                {"line_id": "L1", "speaker": "a", "addressed_to": "b", "text": "open it"},
                {"line_id": "L2", "speaker": "b", "addressed_to": "assistant", "text": "open the trunk"},
            ],
            resolution={"response_summary": "Open the trunk."},
        )
