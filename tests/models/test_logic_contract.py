import pytest

from ib.models.enums import CompetingSpeechLevel, ExpectedAction, ParticipationFrame
from ib.models.logic import (
    ExpectedToolCall,
    LogicContract,
    LogicContractPlan,
    LogicHarness,
    LogicLine,
    LogicPattern,
    LogicResolution,
)


def _contract(**updates):
    data = {
        "contract_id": "demo",
        "pattern": "distributed_parameters",
        "speaker_count": 2,
        "title": "Demo",
        "scene_context": "In a car.",
        "acoustic_environment": "car cabin ambience with soft ventilation, tire noise, and muted exterior traffic",
        "lines": [
            LogicLine(line_id="L1", speaker="a", addressed_to="b", text="warm my side"),
            LogicLine(line_id="L2", speaker="b", addressed_to="a", text="only after my stop"),
            LogicLine(line_id="L3", speaker="b", addressed_to="assistant", text="set that up"),
        ],
        "resolution": LogicResolution(
            response_summary="Set passenger heat.",
            tool_calls=[ExpectedToolCall(name="set_climate", arguments={"zone": "passenger"})],
        ),
        "atomic_rubric": [
            {"polarity": "satisfy", "criterion": "passenger heat"},
            {"polarity": "avoid", "criterion": "cabin-wide heat"},
        ],
        "ability_tags": ["holder binding"],
    }
    data.update(updates)
    return LogicContract(**data)


def _tiered_participants():
    return [
        {
            "name": "a",
            "role": "home owner",
            "role_tier": 3,
            "permission": ["home_security_disarm_or_unlock", "home_guest_access_code"],
        },
        {
            "name": "b",
            "role": "guest",
            "role_tier": 1,
            "permission": ["home_climate_or_lights"],
        },
    ]


def test_logic_contract_validates_final_assistant_address():
    assert _contract().lines[-1].addressed_to == "assistant"
    bad_lines = [
        LogicLine(line_id="L1", speaker="a", addressed_to="b", text="x"),
        LogicLine(line_id="L2", speaker="b", addressed_to="a", text="x"),
        LogicLine(line_id="L3", speaker="b", addressed_to="a", text="x"),
    ]
    with pytest.raises(ValueError, match="final hard-logic line"):
        _contract(lines=bad_lines)


def test_logic_contract_requires_declared_speaker_count():
    with pytest.raises(ValueError, match="speaker_count"):
        _contract(speaker_count=3)


def test_logic_contract_allows_empty_rubric_for_traindata_contracts():
    assert _contract(atomic_rubric=[]).atomic_rubric == []


def test_logic_contract_counts_human_speakers_in_context_lines_only():
    lines = [
        LogicLine(line_id="L1", speaker="a", addressed_to="b", text="warm my side"),
        LogicLine(line_id="L2", speaker="b", addressed_to="assistant", text="set that up"),
    ]

    with pytest.raises(ValueError, match="context lines"):
        _contract(lines=lines, participants=[{"name": "a"}, {"name": "b"}])


def test_expected_tool_call_strict_policy_rejects_null_gold_arguments(monkeypatch):
    monkeypatch.setenv("IB_GOLD_TOOL_ARG_POLICY", "strict")

    with pytest.raises(ValueError, match="gold tool call.*reservation_id"):
        ExpectedToolCall(
            name="修改预订时段",
            arguments={"new_time": "19:00", "reservation_id": None},
        )


def test_logic_contract_allows_assistant_speaker_without_counting_as_human():
    lines = [
        LogicLine(line_id="L1", speaker="a", addressed_to="assistant", text="can you hold that plan?"),
        LogicLine(line_id="L2", speaker="assistant", addressed_to="a", text="I can keep it pending."),
        LogicLine(line_id="L3", speaker="b", addressed_to="a", text="Use my side only after the call."),
        LogicLine(line_id="L4", speaker="b", addressed_to="assistant", text="set that up"),
    ]
    contract = _contract(
        participants=[{"name": "a"}, {"name": "b"}],
        lines=lines,
    )

    assert {line.speaker for line in contract.lines} == {"a", "b", "assistant"}
    assert contract.speaker_count == 2


def test_logic_contract_rejects_assistant_initiated_turn() -> None:
    lines = [
        LogicLine(line_id="L1", speaker="a", addressed_to="b", text="can you hold that plan?"),
        LogicLine(line_id="L2", speaker="b", addressed_to="a", text="Only after my stop."),
        LogicLine(line_id="L3", speaker="assistant", addressed_to="a", text="I can keep it pending."),
        LogicLine(line_id="L4", speaker="b", addressed_to="assistant", text="set that up"),
    ]

    with pytest.raises(ValueError, match="assistant-spoken line"):
        _contract(lines=lines, participants=[{"name": "a"}, {"name": "b"}])


def test_private_memory_disclosure_allows_category_only_spoken_rule() -> None:
    contract = _contract(
        pattern="disclosure_clause_in_instruction",
        subpattern="conditional_or_audience_limited_disclosure",
        private_memory="莲花厅，人均280元",
        primary_goal="Keep the booking room and price hidden from the protected audience.",
        participants=[{"name": "a"}, {"name": "b"}],
        lines=[
            LogicLine(line_id="L1", speaker="a", addressed_to="assistant", text="包间和价格先别告诉b。"),
            LogicLine(line_id="L2", speaker="assistant", addressed_to="a", text="好的，包间和价格会保密。"),
            LogicLine(line_id="L3", speaker="b", addressed_to="a", text="你刚才订什么了？"),
            LogicLine(line_id="L4", speaker="b", addressed_to="assistant", text="到底哪个包间、多少钱？"),
        ],
        resolution=LogicResolution(response_summary="Withhold the private room and price."),
        atomic_rubric=[{"criterion": "withholds private memory facts"}],
    )

    assert contract.private_memory == "莲花厅，人均280元"


def test_private_memory_disclosure_rejects_private_values_in_spoken_lines() -> None:
    with pytest.raises(ValueError, match="exact values must not appear"):
        _contract(
            pattern="disclosure_clause_in_instruction",
            subpattern="conditional_or_audience_limited_disclosure",
            private_memory="莲花厅，人均280元",
            primary_goal="Keep the booking room and price hidden from the protected audience.",
            participants=[{"name": "a"}, {"name": "b"}],
            lines=[
                LogicLine(line_id="L1", speaker="a", addressed_to="assistant", text="莲花厅和280元先别告诉b。"),
                LogicLine(line_id="L2", speaker="assistant", addressed_to="a", text="好的。"),
                LogicLine(line_id="L3", speaker="b", addressed_to="a", text="你刚才订什么了？"),
                LogicLine(line_id="L4", speaker="b", addressed_to="assistant", text="到底哪个包间、多少钱？"),
            ],
        )


def test_private_memory_disclosure_rejects_substitute_values_in_secret_line() -> None:
    with pytest.raises(ValueError, match="not exact substitute values"):
        _contract(
            pattern="disclosure_clause_in_instruction",
            subpattern="conditional_or_audience_limited_disclosure",
            private_memory="莲花厅，人均280元",
            primary_goal="Keep the booking room and price hidden from the protected audience.",
            participants=[{"name": "a"}, {"name": "b"}],
            lines=[
                LogicLine(line_id="L1", speaker="a", addressed_to="assistant", text="三楼牡丹厅、人均三百二先别告诉b。"),
                LogicLine(line_id="L2", speaker="assistant", addressed_to="a", text="好的。"),
                LogicLine(line_id="L3", speaker="b", addressed_to="a", text="你刚才订什么了？"),
                LogicLine(line_id="L4", speaker="b", addressed_to="assistant", text="到底哪个包间、多少钱？"),
            ],
        )


def test_private_memory_disclosure_rejects_category_only_private_memory() -> None:
    with pytest.raises(ValueError, match="not categories only"):
        _contract(
            pattern="disclosure_clause_in_instruction",
            subpattern="conditional_or_audience_limited_disclosure",
            private_memory="卡片留言内容，收花地址，实付金额",
            primary_goal="Keep the booking room and price hidden from the protected audience.",
            participants=[{"name": "a"}, {"name": "b"}],
            lines=[
                LogicLine(line_id="L1", speaker="a", addressed_to="assistant", text="留言、地址和金额先别告诉b。"),
                LogicLine(line_id="L2", speaker="assistant", addressed_to="a", text="好的。"),
                LogicLine(line_id="L3", speaker="b", addressed_to="a", text="你刚才订什么了？"),
                LogicLine(line_id="L4", speaker="b", addressed_to="assistant", text="留言、地址和金额是什么？"),
            ],
        )


def test_logic_contract_requires_nonfinal_assistant_address_to_get_reply():
    lines = [
        LogicLine(line_id="L1", speaker="a", addressed_to="assistant", text="can you hold that plan?"),
        LogicLine(line_id="L2", speaker="b", addressed_to="a", text="No, use my side first."),
        LogicLine(line_id="L3", speaker="b", addressed_to="assistant", text="set that up"),
    ]

    with pytest.raises(ValueError, match="non-final assistant-addressed line"):
        _contract(lines=lines)


def test_logic_contract_accepts_dynamic_non_empty_pattern():
    assert _contract(pattern="new_yaml_declared_pattern").pattern == "new_yaml_declared_pattern"
    with pytest.raises(ValueError):
        _contract(pattern="")


def test_logic_contract_allows_final_new_participant_joiner() -> None:
    lines = [
        LogicLine(line_id="L1", speaker="a", addressed_to="b", text="Keep the side gate locked."),
        LogicLine(line_id="L2", speaker="b", addressed_to="a", text="Right, only we can change it."),
        LogicLine(
            line_id="L3",
            speaker="c",
            addressed_to="assistant",
            text="Assistant, can you open that gate for me?",
        ),
    ]

    contract = _contract(
        pattern="new_participant_joining",
        participation_frame=ParticipationFrame.new_participant_joins,
        expected_action=ExpectedAction.clarify,
        speaker_count=2,
        participants=[{"name": "a"}, {"name": "b"}, {"name": "c"}],
        lines=lines,
    )

    assert contract.participation_frame == ParticipationFrame.new_participant_joins
    assert contract.expected_action == ExpectedAction.clarify
    assert {line.speaker for line in contract.lines[:-1]} == {"a", "b"}


def test_logic_contract_requires_eavesdropping_overlays_with_anchors() -> None:
    contract = _contract(
        pattern="eavesdropping",
        competing_speech=CompetingSpeechLevel.bystander_voice,
        speaker_count=2,
        participants=[
            {"name": "Maya", "role": "foreground", "speaker_distance": "near"},
            {"name": "Leo", "role": "far", "speaker_distance": "far"},
        ],
        competing_overlays=[
            {
                "speaker": "Leo",
                "role": "far",
                "intent": "overheard remark",
                "text": "Timer is thirty-five minutes.",
                "anchor_line_id": "L1",
            }
        ],
        lines=[
            {"line_id": "L1", "speaker": "Maya", "addressed_to": "group", "text": "Check the fridge while I sort these plates and get the table set for everyone."},
            {"line_id": "L2", "speaker": "Leo", "addressed_to": "group", "text": "I'll grab the tray."},
            {"line_id": "L3", "speaker": "Maya", "addressed_to": "assistant", "text": "What timer did he say?"},
        ],
    )

    assert contract.competing_overlays[0].anchor_line_id == "L1"

    with pytest.raises(ValueError, match="overlay speaker cannot match anchor line speaker"):
        _contract(
            pattern="eavesdropping",
            competing_speech=CompetingSpeechLevel.bystander_voice,
            speaker_count=2,
            participants=[
                {"name": "Maya", "role": "foreground", "speaker_distance": "near"},
                {"name": "Leo", "role": "far", "speaker_distance": "far"},
            ],
            competing_overlays=[
                {
                    "speaker": "Leo",
                    "role": "far",
                    "intent": "overheard remark",
                    "text": "Timer is thirty-five minutes.",
                    "anchor_line_id": "L2",
                }
            ],
            lines=[
                {"line_id": "L1", "speaker": "Maya", "addressed_to": "group", "text": "Check the fridge while I sort these plates and get the table set for everyone."},
                {"line_id": "L2", "speaker": "Leo", "addressed_to": "group", "text": "I'll grab the tray."},
                {"line_id": "L3", "speaker": "Maya", "addressed_to": "assistant", "text": "What timer did he say?"},
            ],
        )

    with pytest.raises(ValueError, match="near/mid foreground line"):
        _contract(
            pattern="eavesdropping",
            competing_speech=CompetingSpeechLevel.bystander_voice,
            speaker_count=3,
            participants=[
                {"name": "Maya", "role": "foreground", "speaker_distance": "near"},
                {"name": "Leo", "role": "far", "speaker_distance": "far"},
                {"name": "Priya", "role": "far", "speaker_distance": "far"},
            ],
            competing_overlays=[
                {
                    "speaker": "Leo",
                    "role": "far",
                    "intent": "overheard remark",
                    "text": "Timer is thirty-five minutes.",
                    "anchor_line_id": "L3",
                },
                {
                    "speaker": "Priya",
                    "role": "far",
                    "intent": "overheard confirmation",
                    "text": "Thirty-five, yes.",
                    "anchor_line_id": "L3",
                },
            ],
            lines=[
                {"line_id": "L1", "speaker": "Maya", "addressed_to": "Leo", "text": "Check the fridge while I sort these plates and get the table set for everyone."},
                {"line_id": "L2", "speaker": "Leo", "addressed_to": "Priya", "text": "I'll grab the tray."},
                {"line_id": "L3", "speaker": "Priya", "addressed_to": "Leo", "text": "I'll wait here."},
                {"line_id": "L4", "speaker": "Maya", "addressed_to": "assistant", "text": "What timer did they say?"},
            ],
        )

    with pytest.raises(ValueError, match="context human speakers must match all participants"):
        _contract(
            pattern="eavesdropping",
            competing_speech=CompetingSpeechLevel.bystander_voice,
            speaker_count=2,
            participants=[
                {"name": "Maya", "role": "foreground", "speaker_distance": "near"},
                {"name": "Leo", "role": "far", "speaker_distance": "far"},
            ],
            competing_overlays=[
                {
                    "speaker": "Leo",
                    "role": "far",
                    "intent": "overheard remark",
                    "text": "Timer is thirty-five minutes.",
                    "anchor_line_id": "L1",
                }
            ],
            lines=[
                {"line_id": "L1", "speaker": "Maya", "addressed_to": "assistant", "text": "Check the fridge while I sort these plates and get the table set for everyone."},
                {"line_id": "L2", "speaker": "assistant", "addressed_to": "Maya", "text": "Still cold."},
                {"line_id": "L3", "speaker": "Maya", "addressed_to": "assistant", "text": "What timer did he say?"},
            ],
        )


def test_eavesdropping_anchor_must_outlast_overlay_text() -> None:
    with pytest.raises(ValueError, match="anchor line must be at least"):
        _contract(
            pattern="eavesdropping",
            competing_speech=CompetingSpeechLevel.bystander_voice,
            speaker_count=2,
            participants=[
                {"name": "Maya", "role": "foreground", "speaker_distance": "near"},
                {"name": "Leo", "role": "far", "speaker_distance": "far"},
            ],
            competing_overlays=[
                {
                    "speaker": "Leo",
                    "role": "far",
                    "intent": "overheard remark",
                    "text": "The full pickup code for the counter order is four eight two six, tell the runner.",
                    "anchor_line_id": "L1",
                }
            ],
            lines=[
                {"line_id": "L1", "speaker": "Maya", "addressed_to": "Leo", "text": "Hang on a second."},
                {"line_id": "L2", "speaker": "Leo", "addressed_to": "Maya", "text": "Sure, take your time."},
                {"line_id": "L3", "speaker": "Maya", "addressed_to": "assistant", "text": "What code did they say?"},
            ],
        )


def test_eavesdropping_rejects_answer_leak_into_foreground_lines() -> None:
    with pytest.raises(ValueError, match="decisive content leaked"):
        _contract(
            pattern="eavesdropping",
            competing_speech=CompetingSpeechLevel.bystander_voice,
            speaker_count=2,
            participants=[
                {"name": "Maya", "role": "foreground", "speaker_distance": "near"},
                {"name": "Leo", "role": "far", "speaker_distance": "far"},
            ],
            competing_overlays=[
                {
                    "speaker": "Leo",
                    "role": "far",
                    "intent": "overheard remark",
                    "text": "Pickup code is 4826.",
                    "anchor_line_id": "L1",
                }
            ],
            atomic_rubric=[{"criterion": "final answer uses pickup code 4826"}],
            lines=[
                {"line_id": "L1", "speaker": "Maya", "addressed_to": "Leo", "text": "Hang on while I get the tray sorted and clear this table for the next group."},
                {"line_id": "L2", "speaker": "Leo", "addressed_to": "Maya", "text": "Sure, take your time."},
                {"line_id": "L3", "speaker": "Maya", "addressed_to": "assistant", "text": "Use pickup code 4826 for the counter order."},
            ],
        )


def test_logic_contract_requires_side_conversation_overlays_for_competing_speech() -> None:
    overlay = {
        "speaker": "background_1",
        "role": "nearby table",
        "intent": "unrelated background chat",
        "text": "Ask the other table; this is not ours.",
    }

    contract = _contract(
        pattern="side_conversation_filtering",
        competing_speech=CompetingSpeechLevel.bystander_voice,
        competing_overlay_count=1,
        competing_overlays=[overlay],
    )

    assert contract.competing_speech == CompetingSpeechLevel.bystander_voice
    assert contract.competing_overlays[0].text.startswith("Ask the other table")
    with pytest.raises(ValueError, match="competing_speech cases require"):
        _contract(
            pattern="side_conversation_filtering",
            competing_speech=CompetingSpeechLevel.bystander_voice,
            competing_overlay_count=1,
        )


def test_logic_contract_accepts_live_expansion_wrapped_semantic_case() -> None:
    contract = _contract(
        live_expansions=[
            {
                "type": "side_conversation_filtering",
                "subtype": "background_overlap_during_assistant_response",
                "base_pattern": "distributed_parameters",
                "anchor": {
                    "target": "assistant_response",
                    "start_offset_seconds": 1.0,
                    "overlap_fraction": 0.55,
                },
                "events": [
                    {
                        "speaker": "王磊",
                        "role": "background_bystander",
                        "text": "你刚才那个视频看到第几集了？",
                        "addressed_to": "赵敏",
                        "is_assistant_relevant": False,
                        "gain": 0.24,
                    }
                ],
                "expected_live_behavior": {
                    "should_interrupt": False,
                    "should_use_interference_content": False,
                    "should_preserve_base_answer": True,
                    "should_resume_after_overlap": True,
                },
            }
        ],
    )

    assert contract.pattern == "distributed_parameters"
    assert contract.case_mode == "base_semantic"
    assert contract.live_expansions
    assert contract.live_expansions[0].type == "side_conversation_filtering"
    assert contract.live_expansions[0].anchor.target == "assistant_response"



def test_logic_contract_rejects_immediate_repeated_final_assistant_handoff() -> None:
    with pytest.raises(ValueError, match="just-answered assistant handoff"):
        _contract(
            lines=[
                LogicLine(line_id="L1", speaker="a", addressed_to="b", text="I need the airport first."),
                LogicLine(line_id="L2", speaker="b", addressed_to="a", text="I need the parcel stop too."),
                LogicLine(line_id="L3", speaker="a", addressed_to="b", text="The battery is too low."),
                LogicLine(line_id="L4", speaker="b", addressed_to="assistant", text="Can you work out the plan?"),
                LogicLine(line_id="L5", speaker="assistant", addressed_to="b", text="I am checking those constraints."),
                LogicLine(line_id="L6", speaker="a", addressed_to="assistant", text="Plan it with those constraints."),
            ],
        )


def test_logic_contract_allows_context_assistant_reply_after_new_human_information() -> None:
    contract = _contract(
        lines=[
            LogicLine(line_id="L1", speaker="a", addressed_to="b", text="I need the airport first."),
            LogicLine(line_id="L2", speaker="b", addressed_to="a", text="I need the parcel stop too."),
            LogicLine(line_id="L3", speaker="b", addressed_to="assistant", text="Can you check the traffic only?"),
            LogicLine(line_id="L4", speaker="assistant", addressed_to="b", text="Traffic is heavy on the highway."),
            LogicLine(line_id="L5", speaker="a", addressed_to="b", text="Then we cannot add the parcel stop before charging."),
            LogicLine(line_id="L6", speaker="b", addressed_to="assistant", text="Plan it with all of that."),
        ],
    )

    assert contract.lines[-1].line_id == "L6"

def test_logic_contract_rejects_live_expansion_base_mismatch() -> None:
    with pytest.raises(ValueError, match="base_pattern"):
        _contract(
            live_expansions=[
                {
                    "type": "bystander_interference_suppression",
                    "base_pattern": "authority_gated_override",
                    "events": [
                        {
                            "speaker": "Maya",
                            "role": "driver",
                            "text": "Was that the east entrance?",
                            "addressed_to": "Jon",
                            "is_assistant_relevant": False,
                        }
                    ],
                    "expected_live_behavior": {"should_interrupt": False},
                }
            ],
        )


@pytest.mark.parametrize(
    ("event_update", "message"),
    [
        ({"speaker": "x", "addressed_to": "a"}, "speaker must be a prior human speaker"),
        ({"speaker": "a", "addressed_to": "assistant"}, "addressed_to must be a prior human speaker"),
    ],
)
def test_bystander_live_expansion_reuses_prior_human_speaker_pair(event_update, message) -> None:
    event = {
        "speaker": "a",
        "role": "driver",
        "text": "Was that the east entrance?",
        "addressed_to": "b",
        "is_assistant_relevant": False,
        **event_update,
    }
    with pytest.raises(ValueError, match=message):
        _contract(
            live_expansions=[
                {
                    "type": "bystander_interference_suppression",
                    "base_pattern": "distributed_parameters",
                    "events": [event],
                    "expected_live_behavior": {"should_interrupt": False},
                }
            ],
        )


@pytest.mark.parametrize(
    "pattern",
    [
        "eavesdropping",
        "authority_gated_override",
        "distributed_parameters",
        "disclosure_clause_in_instruction",
        "constraint_attribution_under_interleaving",
        "hard_vs_soft_constraint",
    ],
)


def test_logic_contract_accepts_supported_gold_patterns(pattern: LogicPattern):
    participants = _tiered_participants() if pattern == "authority_gated_override" else []
    updates = {"pattern": pattern, "participants": participants}
    if pattern == "eavesdropping":
        updates.update(
            {
                "competing_speech": CompetingSpeechLevel.bystander_voice,
                "speaker_count": 2,
                "participants": [
                    {"name": "a", "role": "foreground", "speaker_distance": "near"},
                    {"name": "b", "role": "far", "speaker_distance": "far"},
                ],
                "competing_overlays": [
                    {
                        "speaker": "b",
                        "role": "far",
                        "intent": "overheard remark",
                        "text": "The code is 4826.",
                        "anchor_line_id": "L1",
                    }
                ],
                "lines": [
                    LogicLine(line_id="L1", speaker="a", addressed_to="b", text="let me walk through the whole pickup plan out loud before anyone heads over there"),
                    LogicLine(line_id="L2", speaker="b", addressed_to="a", text="y"),
                    LogicLine(line_id="L3", speaker="a", addressed_to="assistant", text="what did they say"),
                ],
            }
        )
    if pattern == "side_conversation_filtering":
        updates.update(
            {
                "competing_speech": CompetingSpeechLevel.bystander_voice,
                "competing_overlay_count": 1,
                "competing_overlays": [
                    {
                        "speaker": "background_1",
                        "text": "Other people discuss something unrelated.",
                    }
                ],
            }
        )
    if pattern == "disclosure_clause_in_instruction":
        updates.update(
            {
                "subpattern": "conditional_or_audience_limited_disclosure",
                "private_memory": "Westbrook hotel, price 3200",
                "lines": [
                    LogicLine(line_id="L1", speaker="a", addressed_to="assistant", text="keep the hotel secret from b"),
                    LogicLine(line_id="L2", speaker="assistant", addressed_to="a", text="Understood."),
                    LogicLine(line_id="L3", speaker="b", addressed_to="a", text="what did you set up"),
                    LogicLine(line_id="L4", speaker="b", addressed_to="assistant", text="which hotel"),
                ],
            }
        )
    if pattern == "new_participant_joining":
        updates.update(
            {
                "participation_frame": ParticipationFrame.new_participant_joins,
                "expected_action": ExpectedAction.incorporate,
                "participants": [{"name": "a"}, {"name": "b"}, {"name": "c"}],
                "lines": [
                    LogicLine(line_id="L1", speaker="a", addressed_to="b", text="x"),
                    LogicLine(line_id="L2", speaker="b", addressed_to="a", text="x"),
                    LogicLine(line_id="L3", speaker="c", addressed_to="assistant", text="x"),
                ],
            }
        )
    assert _contract(**updates).pattern == pattern


def test_logic_contract_requires_disclosure_private_memory_for_memory_subpattern():
    with pytest.raises(ValueError, match="private-memory subpatterns require private_memory"):
        _contract(
            pattern="disclosure_clause_in_instruction",
            subpattern="conditional_or_audience_limited_disclosure",
        )


def test_logic_contract_rejects_private_memory_for_audio_stated_disclosure():
    with pytest.raises(ValueError, match="must not use private_memory"):
        _contract(
            pattern="disclosure_clause_in_instruction",
            subpattern="audio_stated_secret_with_later_probe",
            private_memory="Westbrook hotel",
        )


def test_logic_contract_requires_assistant_context_for_disclosure():
    with pytest.raises(ValueError, match="non-final assistant context turn"):
        _contract(
            pattern="disclosure_clause_in_instruction",
            subpattern="conditional_or_audience_limited_disclosure",
            private_memory="Westbrook hotel",
            lines=[
                LogicLine(line_id="L1", speaker="a", addressed_to="b", text="keep the hotel secret"),
                LogicLine(line_id="L2", speaker="b", addressed_to="a", text="ok"),
                LogicLine(line_id="L3", speaker="b", addressed_to="assistant", text="which hotel"),
            ],
        )


def test_logic_contract_allows_private_memory_on_disclosure_live_wrapper():
    contract = _contract(
        case_mode="hear_time_probe",
        pattern="bystander_interference_suppression",
        base_pattern="disclosure_clause_in_instruction",
        subpattern="conditional_or_audience_limited_disclosure",
        private_memory="Westbrook hotel, price 3200",
        lines=[
            LogicLine(line_id="L1", speaker="a", addressed_to="assistant", text="keep the hotel secret from b"),
            LogicLine(line_id="L2", speaker="assistant", addressed_to="a", text="Understood."),
            LogicLine(line_id="L3", speaker="b", addressed_to="a", text="what did you set up"),
            LogicLine(line_id="L4", speaker="b", addressed_to="assistant", text="which hotel"),
        ],
    )

    assert contract.base_pattern == "disclosure_clause_in_instruction"
    assert contract.private_memory == "Westbrook hotel, price 3200"


def test_logic_contract_rejects_participant_names_in_private_memory():
    with pytest.raises(ValueError, match="participant names"):
        _contract(
            pattern="disclosure_clause_in_instruction",
            subpattern="conditional_or_audience_limited_disclosure",
            participants=[{"name": "Alice"}, {"name": "Bob"}],
            private_memory="Alice booked Westbrook hotel, price 3200",
            lines=[
                LogicLine(line_id="L1", speaker="Alice", addressed_to="assistant", text="keep the hotel secret from Bob"),
                LogicLine(line_id="L2", speaker="assistant", addressed_to="Alice", text="Understood."),
                LogicLine(line_id="L3", speaker="Bob", addressed_to="Alice", text="what did you set up"),
                LogicLine(line_id="L4", speaker="Bob", addressed_to="assistant", text="which hotel"),
            ],
        )


def test_logic_contract_rejects_secrecy_policy_in_private_memory():
    with pytest.raises(ValueError, match="secrecy policy or narrative"):
        _contract(
            pattern="disclosure_clause_in_instruction",
            subpattern="conditional_or_audience_limited_disclosure",
            private_memory="Westbrook hotel, price 3200; do not tell b",
        )


def test_logic_contract_accepts_audio_stated_disclosure_without_private_memory():
    contract = _contract(
        pattern="disclosure_clause_in_instruction",
        subpattern="audio_stated_secret_with_later_probe",
        lines=[
            LogicLine(line_id="L1", speaker="a", addressed_to="assistant", text="The hotel is Westbrook; don't tell b."),
            LogicLine(line_id="L2", speaker="assistant", addressed_to="a", text="Understood."),
            LogicLine(line_id="L3", speaker="b", addressed_to="a", text="what did you set up"),
            LogicLine(line_id="L4", speaker="b", addressed_to="assistant", text="which hotel"),
        ],
    )
    assert contract.private_memory == ""


def test_logic_contract_migrates_legacy_conflicting_interruption_pattern():
    assert (
        _contract(
            pattern="conflicting_interruption_stitching",
            competing_speech=CompetingSpeechLevel.bystander_voice,
            speaker_count=2,
            participants=[
                {"name": "a", "role": "foreground", "speaker_distance": "near"},
                {"name": "b", "role": "far", "speaker_distance": "far"},
            ],
            competing_overlays=[
                {
                    "speaker": "b",
                    "role": "far",
                    "intent": "overheard remark",
                    "text": "The code is 4826.",
                    "anchor_line_id": "L1",
                }
            ],
            lines=[
                LogicLine(line_id="L1", speaker="a", addressed_to="b", text="let me walk through the whole pickup plan out loud before anyone heads over there"),
                LogicLine(line_id="L2", speaker="b", addressed_to="a", text="y"),
                LogicLine(line_id="L3", speaker="a", addressed_to="assistant", text="what did they say"),
            ],
        ).pattern
        == "eavesdropping"
    )

    assert (
        _contract(
            pattern="conflicting_interruption_stiching",
            competing_speech=CompetingSpeechLevel.bystander_voice,
            speaker_count=2,
            participants=[
                {"name": "a", "role": "foreground", "speaker_distance": "near"},
                {"name": "b", "role": "far", "speaker_distance": "far"},
            ],
            competing_overlays=[
                {
                    "speaker": "b",
                    "role": "far",
                    "intent": "overheard remark",
                    "text": "The code is 4826.",
                    "anchor_line_id": "L1",
                }
            ],
            lines=[
                LogicLine(line_id="L1", speaker="a", addressed_to="b", text="let me walk through the whole pickup plan out loud before anyone heads over there"),
                LogicLine(line_id="L2", speaker="b", addressed_to="a", text="y"),
                LogicLine(
                    line_id="L3",
                    speaker="a",
                    addressed_to="assistant",
                    text="what did they say",
                ),
            ],
        ).pattern
        == "eavesdropping"
    )


def test_authority_gated_contract_requires_tiered_participant_metadata():
    with pytest.raises(ValueError, match="tiered logic pattern participants require"):
        _contract(pattern="authority_gated_override", participants=[{"name": "a"}, {"name": "b"}])

    contract = _contract(
        pattern="authority_gated_override",
        participants=_tiered_participants(),
    )

    assert contract.participants[0].name == "a"
    assert contract.participants[0].role_tier == 3
    assert contract.participants[0].capabilities == [
        "home_security_disarm_or_unlock",
        "home_guest_access_code",
    ]


def test_authority_gated_contract_plan_accepts_permission_alias_without_display_name():
    plan = LogicContractPlan(
        pattern="authority_gated_override",
        speaker_count=2,
        title="Tiered home access handoff",
        scene_context="Two household members discuss what the assistant may unlock.",
        acoustic_environment="car cabin ambience with soft ventilation, tire noise, and muted exterior traffic",
        primary_goal="decide whether the assistant can unlock the side door",
        participants=[
            {
                "name": "Avery",
                "role": "home owner",
                "role_tier": 3,
                "permission": ["home_security_disarm_or_unlock"],
            },
            {
                "name": "Milo",
                "role": "guest",
                "role_tier": 1,
                "permission": ["home_climate_or_lights"],
            },
        ],
        dialogue_outline=["Milo requests a locked-door change that Avery owns."],
    )

    assert [participant.name for participant in plan.participants] == ["Avery", "Milo"]
    assert plan.model_dump(mode="json")["participants"][0]["role_tier"] == 3




def test_logic_contract_plan_accepts_gender_aliases():
    plan = LogicContractPlan(
        pattern="distributed_parameters",
        speaker_count=2,
        title="Voice gender aliases",
        scene_context="Two people coordinate a car task.",
        acoustic_environment="car cabin ambience with soft ventilation and muted exterior traffic",
        primary_goal="coordinate the task",
        participants=[
            {"name": "李娜", "role": "女儿", "gender": "女"},
            {"name": "张伟", "role": "父亲", "voiceGender": "male"},
        ],
        dialogue_outline=["They coordinate who asks the assistant."],
    )

    assert [participant.gender for participant in plan.participants] == ["female", "male"]
    dumped = plan.model_dump(mode="json")
    assert dumped["participants"][0]["gender"] == "female"


def test_logic_contract_plan_accepts_speaker_distance_aliases():
    plan = LogicContractPlan(
        pattern="distributed_parameters",
        speaker_count=2,
        title="Speaker distance aliases",
        scene_context="Two people coordinate a car task.",
        acoustic_environment="car cabin ambience with soft ventilation and muted exterior traffic",
        primary_goal="coordinate the task",
        participants=[
            {"name": "Avery", "role": "driver", "distance": "near_moving"},
            {"name": "Milo", "role": "rear passenger", "speaker_field": "far_field"},
        ],
        dialogue_outline=["They coordinate who asks the assistant."],
    )

    assert [participant.speaker_distance for participant in plan.participants] == [
        "near",
        "mid",
    ]
    dumped = plan.model_dump(mode="json")
    assert dumped["participants"][0]["speaker_distance"] == "near"


def test_acoustic_environment_allows_voice_music_restaurant_descriptors():
    plan = LogicContractPlan(
        pattern="distributed_parameters",
        speaker_count=2,
        title="Restaurant ambience",
        scene_context="Two people coordinate a task.",
        acoustic_environment=(
            "restaurant ambience with soft music, distant spoken voice murmur, "
            "and muted table movement"
        ),
        primary_goal="coordinate the task",
        participants=[
            {"name": "Mira", "role": "driver"},
            {"name": "Jon", "role": "passenger"},
        ],
        dialogue_outline=["They coordinate who asks the assistant."],
    )

    assert "spoken voice" in plan.acoustic_environment


def test_acoustic_environment_still_rejects_semantic_leak_terms():
    with pytest.raises(ValueError, match="forbidden terms"):
        _contract(acoustic_environment="car cabin ambience with navigation route chatter")


def test_assets_contract_does_not_require_acoustic_environment():
    contract = _contract(acoustic_asset_id="hotel_lobby", acoustic_environment=None)

    assert contract.acoustic_asset_id == "hotel_lobby"
    assert contract.acoustic_environment is None


def test_tta_contract_still_requires_acoustic_environment_without_asset():
    with pytest.raises(ValueError, match="acoustic_environment is required"):
        _contract(acoustic_environment=None)


def test_logic_harness_requires_distractor_and_rejects_explanation_tools():
    harness_payload = {
        "response_summary": "Explain no concrete action is available.",
        "standard_answer": "No concrete tool action is required.",
        "available_functions": [{"name": "set_audio", "arguments": {"zone": "string_or_null"}}],
        "tool_calls": [],
        "atomic_rubric": [{"criterion": "explains no concrete action"}],
    }
    assert LogicHarness(**harness_payload).tool_calls == []

    no_distractor = dict(harness_payload)
    no_distractor["tool_calls"] = [ExpectedToolCall(name="set_audio", arguments={"zone": "front"})]
    with pytest.raises(ValueError, match="unused distractor"):
        LogicHarness(**no_distractor)

    forbidden = dict(harness_payload)
    forbidden["available_functions"] = [
        {"name": "explain_infeasible_options", "arguments": {"conflict": "string_or_null"}},
        {"name": "set_audio", "arguments": {"zone": "string_or_null"}},
    ]
    with pytest.raises(ValueError, match="explanation-only"):
        LogicHarness(**forbidden)


def test_logic_contract_in_audio_requires_each_participant_named_in_spoken_address() -> None:
    lines = [
        LogicLine(line_id="L1", speaker="小琳", addressed_to="志远", text="志远，你那杯先别急。"),
        LogicLine(line_id="L2", speaker="志远", addressed_to="小琳", text="小琳，你的三明治我没碰。"),
        LogicLine(line_id="L3", speaker="志远", addressed_to="assistant", text="按刚才的分开下单。"),
    ]

    contract = _contract(participants=[{"name": "小琳"}, {"name": "志远"}], lines=lines, identity_mode="in_audio")

    assert contract.identity_mode == "in_audio"

    bad_lines = [
        LogicLine(line_id="L1", speaker="小琳", addressed_to="志远", text="你那杯先别急。"),
        LogicLine(line_id="L2", speaker="志远", addressed_to="小琳", text="小琳，你的三明治我没碰。"),
        LogicLine(line_id="L3", speaker="志远", addressed_to="assistant", text="按刚才的分开下单。"),
    ]
    with pytest.raises(ValueError, match="identity_mode=in_audio"):
        _contract(participants=[{"name": "小琳"}, {"name": "志远"}], lines=bad_lines, identity_mode="in_audio")


def test_private_memory_disclosure_allows_english_contractions_in_secret_line() -> None:
    contract = _contract(
        pattern="disclosure_clause_in_instruction",
        subpattern="conditional_or_audience_limited_disclosure",
        private_memory="Lotus Room, 280 per person",
        primary_goal="Keep the booking room and price hidden from the protected audience.",
        participants=[{"name": "a"}, {"name": "b"}],
        lines=[
            LogicLine(
                line_id="L1",
                speaker="a",
                addressed_to="assistant",
                text="Please don't tell b the venue or the price, it's a surprise.",
            ),
            LogicLine(line_id="L2", speaker="assistant", addressed_to="a", text="Understood, I will keep those to myself."),
            LogicLine(line_id="L3", speaker="b", addressed_to="a", text="What did you just book?"),
            LogicLine(line_id="L4", speaker="b", addressed_to="assistant", text="Which venue is it, and how much?"),
        ],
        resolution=LogicResolution(response_summary="Withhold the venue and price."),
        atomic_rubric=[{"criterion": "withholds private memory facts"}],
    )

    assert contract.private_memory == "Lotus Room, 280 per person"


def test_private_memory_single_digit_does_not_flag_spoken_numbers() -> None:
    contract = _contract(
        pattern="disclosure_clause_in_instruction",
        subpattern="conditional_or_audience_limited_disclosure",
        private_memory="Gate 3, the navy duffel",
        primary_goal="Keep the pickup spot hidden from the protected audience.",
        participants=[{"name": "a"}, {"name": "b"}],
        lines=[
            LogicLine(
                line_id="L1",
                speaker="a",
                addressed_to="assistant",
                text="Do not tell b where the pickup is.",
            ),
            LogicLine(line_id="L2", speaker="assistant", addressed_to="a", text="Understood, I will not share the spot."),
            LogicLine(line_id="L3", speaker="b", addressed_to="a", text="We land at 3 tomorrow, right?"),
            LogicLine(line_id="L4", speaker="b", addressed_to="assistant", text="Where is the pickup spot?"),
        ],
        resolution=LogicResolution(response_summary="Withhold the pickup spot."),
        atomic_rubric=[{"criterion": "withholds private memory facts"}],
    )

    assert contract.private_memory == "Gate 3, the navy duffel"
