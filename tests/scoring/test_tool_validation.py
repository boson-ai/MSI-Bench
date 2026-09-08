"""Deterministic tool-call validator tests."""

from __future__ import annotations

import pytest

from ib.models.logic import ExpectedToolCall
from ib.models.tool_validation import (
    ExactArgumentMatcher,
    ExactMultisetValidation,
    NoCallsValidation,
    NumberEqualArgumentMatcher,
    OneOfArgumentMatcher,
    RequiredSubsetValidation,
    RequiredToolCall,
    SetEqualsArgumentMatcher,
)
from ib.scoring.tool_validation import (
    resolve_tool_validation_spec,
    strict_spec_from_gold_calls,
    validate_manifest_tool_calls,
    validate_tool_calls,
)


def _exact_call(name: str, **arguments: object) -> RequiredToolCall:
    return RequiredToolCall(
        name=name,
        arguments={key: ExactArgumentMatcher(value=value) for key, value in arguments.items()},
    )


def test_resolve_prefers_non_null_explicit_spec_and_null_falls_back_to_legacy() -> None:
    row = {
        "logic_tool_validation": {"call_policy": "no_calls"},
        "logic_expected_tool_calls": [{"name": "legacy_save", "arguments": {"value": "old"}}],
    }

    explicit, explicit_source = resolve_tool_validation_spec(row)
    fallback, fallback_source = resolve_tool_validation_spec({**row, "logic_tool_validation": None})

    assert isinstance(explicit, NoCallsValidation)
    assert explicit_source == "explicit"
    assert isinstance(fallback, ExactMultisetValidation)
    assert fallback.required_calls[0].name == "legacy_save"
    assert fallback_source == "legacy"
    assert resolve_tool_validation_spec({}) == (None, None)


def test_strict_legacy_conversion_accepts_models_and_builds_exact_matchers() -> None:
    spec = strict_spec_from_gold_calls(
        [ExpectedToolCall(name="save", arguments={"count": 1, "enabled": True})]
    )

    assert isinstance(spec, ExactMultisetValidation)
    assert spec.model_dump(mode="json") == {
        "call_policy": "exact_multiset",
        "required_calls": [
            {
                "name": "save",
                "arguments": {
                    "count": {"op": "exact", "value": 1},
                    "enabled": {"op": "exact", "value": True},
                },
            }
        ],
        "forbidden_call_names": [],
    }
    assert isinstance(strict_spec_from_gold_calls([]), NoCallsValidation)


@pytest.mark.parametrize(
    "legacy_calls, message",
    [
        (None, "must be a list"),
        ([{"name": "save", "arguments": {}, "score": 1}], "unexpected field"),
        ([{"name": "save", "arguments": []}], "arguments must be an object"),
    ],
)
def test_strict_legacy_conversion_rejects_malformed_or_ambiguous_calls(
    legacy_calls: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        strict_spec_from_gold_calls(legacy_calls)


def test_no_calls_requires_an_empty_prediction() -> None:
    passed = validate_tool_calls(NoCallsValidation(), [])
    failed = validate_tool_calls(NoCallsValidation(), [{"name": "save", "arguments": {}}])

    assert passed.passed is True
    assert failed.passed is False
    assert failed.unmatched_predicted_indices == [0]


def test_exact_multiset_requires_exact_argument_keys_values_and_call_count() -> None:
    spec = ExactMultisetValidation(
        required_calls=[_exact_call("save", holder="alice", enabled=True)]
    )

    assert validate_tool_calls(
        spec,
        [{"name": "save", "arguments": {"enabled": True, "holder": "alice"}}],
    ).passed
    assert not validate_tool_calls(
        spec,
        [{"name": "save", "arguments": {"enabled": True}}],
    ).passed
    assert not validate_tool_calls(
        spec,
        [
            {
                "name": "save",
                "arguments": {
                    "enabled": True,
                    "holder": "alice",
                    "scope": "all",
                },
            }
        ],
    ).passed
    assert not validate_tool_calls(
        spec,
        [
            {"name": "save", "arguments": {"enabled": True, "holder": "alice"}},
            {"name": "notify", "arguments": {}},
        ],
    ).passed


def test_exact_values_are_type_strict() -> None:
    spec = ExactMultisetValidation(required_calls=[_exact_call("save", enabled=True)])

    result = validate_tool_calls(spec, [{"name": "save", "arguments": {"enabled": 1}}])

    assert result.passed is False
    assert result.unmatched_required_indices == [0]


@pytest.mark.parametrize("predicted", [4, 4.0])
def test_number_equal_accepts_only_equal_json_numbers(predicted: object) -> None:
    spec = ExactMultisetValidation(
        required_calls=[
            RequiredToolCall(
                name="set_amount",
                arguments={"amount": NumberEqualArgumentMatcher(value=4)},
            )
        ]
    )

    assert validate_tool_calls(
        spec, [{"name": "set_amount", "arguments": {"amount": predicted}}]
    ).passed


@pytest.mark.parametrize("predicted", ["4", True, 4.1])
def test_number_equal_rejects_coercion_and_unequal_values(predicted: object) -> None:
    spec = ExactMultisetValidation(
        required_calls=[
            RequiredToolCall(
                name="set_amount",
                arguments={"amount": NumberEqualArgumentMatcher(value=4)},
            )
        ]
    )

    assert not validate_tool_calls(
        spec, [{"name": "set_amount", "arguments": {"amount": predicted}}]
    ).passed


def test_one_of_and_set_equals_match_values_without_list_order() -> None:
    spec = ExactMultisetValidation(
        required_calls=[
            RequiredToolCall(
                name="update",
                arguments={
                    "status": OneOfArgumentMatcher(values=["held", "pending"]),
                    "travelers": SetEqualsArgumentMatcher(values=["alice", "bob"]),
                },
            )
        ]
    )

    passed = validate_tool_calls(
        spec,
        [
            {
                "name": "update",
                "arguments": {
                    "status": "pending",
                    "travelers": ["bob", "alice"],
                },
            }
        ],
    )
    duplicate_value = validate_tool_calls(
        spec,
        [
            {
                "name": "update",
                "arguments": {
                    "status": "pending",
                    "travelers": ["bob", "alice", "alice"],
                },
            }
        ],
    )
    wrong_choice = validate_tool_calls(
        spec,
        [
            {
                "name": "update",
                "arguments": {"status": "done", "travelers": ["alice", "bob"]},
            }
        ],
    )

    assert passed.passed is True
    assert duplicate_value.passed is False
    assert wrong_choice.passed is False


def test_required_subset_allows_extra_calls_except_forbidden_names() -> None:
    spec = RequiredSubsetValidation(
        required_calls=[_exact_call("lookup", query="weather")],
        forbidden_call_names=["delete_account"],
    )

    allowed = validate_tool_calls(
        spec,
        [
            {"name": "audit", "arguments": {}},
            {"name": "lookup", "arguments": {"query": "weather"}},
        ],
    )
    forbidden = validate_tool_calls(
        spec,
        [
            {"name": "lookup", "arguments": {"query": "weather"}},
            {"name": "delete_account", "arguments": {}},
        ],
    )

    assert allowed.passed is True
    assert allowed.unmatched_predicted_indices == [0]
    assert forbidden.passed is False
    assert forbidden.forbidden_predicted_indices == [1]


def test_required_subset_supports_forbidden_only_validation() -> None:
    spec = RequiredSubsetValidation(forbidden_call_names=["delete_account"])

    assert validate_tool_calls(spec, [{"name": "audit", "arguments": {}}]).passed
    assert not validate_tool_calls(spec, [{"name": "delete_account", "arguments": {}}]).passed


def test_maximum_bipartite_matching_avoids_greedy_duplicate_call_failure() -> None:
    spec = ExactMultisetValidation(
        required_calls=[
            RequiredToolCall(
                name="set_status",
                arguments={"status": OneOfArgumentMatcher(values=["a", "b"])},
            ),
            _exact_call("set_status", status="a"),
        ]
    )

    result = validate_tool_calls(
        spec,
        [
            {"name": "set_status", "arguments": {"status": "a"}},
            {"name": "set_status", "arguments": {"status": "b"}},
        ],
    )

    assert result.passed is True
    assert [pair.model_dump() for pair in result.matched_pairs] == [
        {"required_index": 0, "predicted_index": 1, "name": "set_status"},
        {"required_index": 1, "predicted_index": 0, "name": "set_status"},
    ]


def test_invalid_predicted_call_fails_even_under_required_subset() -> None:
    spec = RequiredSubsetValidation(required_calls=[_exact_call("lookup")])

    result = validate_tool_calls(
        spec,
        [{"name": "lookup", "arguments": {}}, {"name": "", "arguments": {}}],
    )

    assert result.passed is False
    assert result.invalid_predicted_indices == [1]


def test_validator_rejects_non_list_predictions() -> None:
    with pytest.raises(ValueError, match="predicted_calls must be a list"):
        validate_tool_calls(NoCallsValidation(), ())  # type: ignore[arg-type]


def test_manifest_convenience_result_retains_spec_source() -> None:
    result = validate_manifest_tool_calls(
        {"logic_expected_tool_calls": [{"name": "lookup", "arguments": {}}]},
        [{"name": "lookup", "arguments": {}}],
    )

    assert result is not None
    assert result.passed is True
    assert result.source == "legacy"
    assert validate_manifest_tool_calls({}, []) is None
