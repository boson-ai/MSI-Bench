"""Read-only tool calls: surplus lookups are ignored, state-changing calls stay strict."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ib.models.tool_validation import (
    ExactArgumentMatcher,
    ExactMultisetValidation,
    NoCallsValidation,
    RequiredToolCall,
)
from ib.scoring.tool_validation import (
    effective_tool_calls,
    resolve_tool_validation_spec,
    validate_tool_calls,
)


def _exact_call(name: str, **arguments: object) -> RequiredToolCall:
    return RequiredToolCall(
        name=name,
        arguments={key: ExactArgumentMatcher(value=value) for key, value in arguments.items()},
    )


def _call(name: str, **arguments: object) -> dict:
    return {"name": name, "arguments": arguments}


def test_no_calls_ignores_a_read_only_lookup_but_not_a_state_changing_call() -> None:
    spec = NoCallsValidation(read_only_call_names=["get_current_time"])

    lookup = validate_tool_calls(spec, [_call("get_current_time")])
    action = validate_tool_calls(spec, [_call("get_current_time"), _call("send_message", to="ana")])

    assert lookup.passed is True
    assert lookup.ignored_predicted_indices == [0]
    assert lookup.unmatched_predicted_indices == []
    assert lookup.predicted_call_count == 1
    assert action.passed is False
    assert action.ignored_predicted_indices == [0]
    assert action.unmatched_predicted_indices == [1]


def test_exact_multiset_ignores_an_extra_lookup_and_rejects_an_extra_action() -> None:
    spec = ExactMultisetValidation(
        required_calls=[_exact_call("set_alarm", time="07:00")],
        read_only_call_names=["get_weather"],
    )

    with_lookup = validate_tool_calls(
        spec, [_call("get_weather", city="Oslo"), _call("set_alarm", time="07:00")]
    )
    with_action = validate_tool_calls(
        spec, [_call("set_alarm", time="07:00"), _call("send_message", to="ana")]
    )

    assert with_lookup.passed is True
    assert with_lookup.ignored_predicted_indices == [0]
    assert [(p.required_index, p.predicted_index) for p in with_lookup.matched_pairs] == [(0, 1)]
    assert with_lookup.rationale().endswith("; 1 ignored read-only call(s)")
    assert with_action.passed is False
    assert with_action.ignored_predicted_indices == []
    assert with_action.unmatched_predicted_indices == [1]


def test_a_required_read_only_tool_is_judged_by_its_last_call() -> None:
    spec = ExactMultisetValidation(
        required_calls=[_exact_call("get_weather", city="Oslo")],
        read_only_call_names=["get_weather"],
    )

    last_right = validate_tool_calls(
        spec, [_call("get_weather", city="Rome"), _call("get_weather", city="Oslo")]
    )
    first_right = validate_tool_calls(
        spec, [_call("get_weather", city="Oslo"), _call("get_weather", city="Rome")]
    )

    assert last_right.passed is True
    assert last_right.ignored_predicted_indices == [0]
    assert first_right.passed is False
    assert first_right.ignored_predicted_indices == [0]
    assert first_right.unmatched_required_indices == [0]
    assert first_right.unmatched_predicted_indices == [1]


def test_a_read_only_tool_required_twice_keeps_its_last_two_calls() -> None:
    spec = ExactMultisetValidation(
        required_calls=[
            _exact_call("get_weather", city="Oslo"),
            _exact_call("get_weather", city="Rome"),
        ],
        read_only_call_names=["get_weather"],
    )
    rome, oslo, paris = (_call("get_weather", city=c) for c in ("Rome", "Oslo", "Paris"))

    last_two_right = validate_tool_calls(spec, [paris, rome, oslo])
    first_two_right = validate_tool_calls(spec, [rome, oslo, paris])

    assert last_two_right.passed is True
    assert last_two_right.ignored_predicted_indices == [0]
    assert first_two_right.passed is False
    assert first_two_right.ignored_predicted_indices == [0]


def test_a_call_with_an_unknown_name_is_never_ignored() -> None:
    spec = NoCallsValidation(read_only_call_names=["get_weather"])

    result = validate_tool_calls(spec, [_call("get_weather_v2"), _call("get_weather")])

    assert result.passed is False
    assert result.ignored_predicted_indices == [1]
    assert result.unmatched_predicted_indices == [0]


def test_a_malformed_read_only_call_still_fails() -> None:
    spec = NoCallsValidation(read_only_call_names=["get_weather"])

    result = validate_tool_calls(spec, [{"name": "get_weather", "arguments": ["Oslo"]}])

    assert result.passed is False
    assert result.ignored_predicted_indices == []
    assert result.invalid_predicted_indices == [0]


@pytest.mark.parametrize(
    "spec, message",
    [
        (
            {
                "call_policy": "exact_multiset",
                "required_calls": [{"name": "save"}],
                "forbidden_call_names": ["delete_all"],
                "read_only_call_names": ["delete_all"],
            },
            "cannot also be read-only",
        ),
        ({"call_policy": "no_calls", "read_only_call_names": [" "]}, "must not be blank"),
        (
            {"call_policy": "no_calls", "read_only_call_names": ["get_time", "get_time"]},
            "must be unique",
        ),
    ],
)
def test_spec_rejects_invalid_read_only_names(spec: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        resolve_tool_validation_spec({"logic_tool_validation": spec})


def test_required_subset_does_not_accept_read_only_names() -> None:
    with pytest.raises(ValidationError, match="read_only_call_names"):
        resolve_tool_validation_spec(
            {
                "logic_tool_validation": {
                    "call_policy": "required_subset",
                    "required_calls": [{"name": "save"}],
                    "read_only_call_names": ["get_time"],
                }
            }
        )


def test_empty_read_only_names_are_left_out_of_the_dump_and_change_nothing() -> None:
    spec = ExactMultisetValidation(required_calls=[_exact_call("save", value=1)])

    result = validate_tool_calls(spec, [_call("get_time"), _call("save", value=1)])

    assert "read_only_call_names" not in spec.model_dump(mode="json")
    assert "read_only_call_names" not in NoCallsValidation().model_dump(mode="json")
    assert result.passed is False
    assert result.ignored_predicted_indices == []
    assert result.unmatched_predicted_indices == [0]
    assert "ignored" not in result.rationale()


def test_effective_tool_calls_follow_the_row_spec() -> None:
    calls = [_call("get_time"), _call("save", value=1)]
    stamped = {
        "logic_tool_validation": {
            "call_policy": "exact_multiset",
            "required_calls": [{"name": "save", "arguments": {"value": {"op": "exact", "value": 1}}}],
            "read_only_call_names": ["get_time"],
        }
    }

    assert effective_tool_calls(stamped, calls) == [calls[1]]
    assert effective_tool_calls({}, calls) == calls
