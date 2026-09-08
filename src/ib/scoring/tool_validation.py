"""tool_validation - deterministic validation of predicted tool calls.

Calling spec:
    resolve_tool_validation_spec(row) -> (spec | None, "explicit" | "legacy" | None)
    strict_spec_from_gold_calls(calls) -> ToolValidationSpec
    validate_tool_calls(spec, calls) -> ToolValidationResult
    validate_manifest_tool_calls(row, calls) -> ToolValidationResult | None

All matching is exact and order-independent. The module has no side effects.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import TypeAdapter

from ib.models.tool_validation import (
    TOOL_VALIDATOR_POLICY_VERSION,
    ArgumentMatcher,
    ExactArgumentMatcher,
    ExactMultisetValidation,
    NoCallsValidation,
    NumberEqualArgumentMatcher,
    OneOfArgumentMatcher,
    RequiredSubsetValidation,
    RequiredToolCall,
    SetEqualsArgumentMatcher,
    ToolCallMatch,
    ToolValidationResult,
    ToolValidationSource,
    ToolValidationSpec,
)


_SPEC_ADAPTER = TypeAdapter(ToolValidationSpec)

# Legacy sentinel for historical hybrid reports whose tool.call_args result was
# stamped by the deterministic validator. Current judges never write it; the
# leaderboard reader excludes those old atoms from LLM-vs-validator calibration.
DETERMINISTIC_TOOL_VALIDATOR_PREFIX = "deterministic tool validator:"

__all__ = [
    "DETERMINISTIC_TOOL_VALIDATOR_PREFIX",
    "TOOL_VALIDATOR_POLICY_VERSION",
    "resolve_tool_validation_spec",
    "strict_spec_from_gold_calls",
    "validate_manifest_tool_calls",
    "validate_tool_calls",
]


def resolve_tool_validation_spec(
    manifest_row: Mapping[str, Any],
) -> tuple[ToolValidationSpec | None, ToolValidationSource | None]:
    """Resolve explicit validation first, then strictly convert legacy gold calls."""
    explicit = manifest_row.get("logic_tool_validation")
    if explicit is not None:
        return _SPEC_ADAPTER.validate_python(explicit), "explicit"
    if "logic_expected_tool_calls" in manifest_row:
        return strict_spec_from_gold_calls(manifest_row["logic_expected_tool_calls"]), "legacy"
    return None, None


def strict_spec_from_gold_calls(expected_calls: Any) -> ToolValidationSpec:
    """Derive an exact validator spec from gold calls."""
    if not isinstance(expected_calls, list):
        raise ValueError("gold tool calls must be a list")
    if not expected_calls:
        return NoCallsValidation()
    required_calls: list[RequiredToolCall] = []
    for index, raw_call in enumerate(expected_calls):
        call = _model_or_mapping(raw_call)
        if call is None:
            raise ValueError(f"gold tool call at index {index} must be an object")
        unexpected = sorted(set(call) - {"name", "arguments"})
        if unexpected:
            raise ValueError(
                f"gold tool call at index {index} has unexpected field(s): {unexpected}"
            )
        name = call.get("name")
        arguments = call.get("arguments", {})
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"gold tool call at index {index} needs a non-blank name")
        if not isinstance(arguments, Mapping):
            raise ValueError(f"gold tool call at index {index} arguments must be an object")
        if any(not isinstance(key, str) or not key.strip() for key in arguments):
            raise ValueError(
                f"gold tool call at index {index} argument names must be non-blank strings"
            )
        required_calls.append(
            RequiredToolCall(
                name=name,
                arguments={
                    key: ExactArgumentMatcher(value=value) for key, value in arguments.items()
                },
            )
        )
    return ExactMultisetValidation(required_calls=required_calls)


def validate_tool_calls(
    spec: ToolValidationSpec,
    predicted_calls: list[Any],
) -> ToolValidationResult:
    """Validate predicted calls using maximum-cardinality bipartite matching."""
    if not isinstance(predicted_calls, list):
        raise ValueError("predicted_calls must be a list")
    required_calls = _required_calls(spec)
    normalized = [_model_or_mapping(call) for call in predicted_calls]
    invalid_indices = [
        index for index, call in enumerate(normalized) if not _valid_predicted_call(call)
    ]
    forbidden_names = set(getattr(spec, "forbidden_call_names", []))
    forbidden_indices = [
        index
        for index, call in enumerate(normalized)
        if call is not None and call.get("name") in forbidden_names
    ]
    adjacency = [
        [
            predicted_index
            for predicted_index, predicted_call in enumerate(normalized)
            if _call_matches(requirement, predicted_call)
        ]
        for requirement in required_calls
    ]
    pairs = _maximum_bipartite_pairs(adjacency)
    matched_required = {required_index for required_index, _ in pairs}
    matched_predicted = {predicted_index for _, predicted_index in pairs}
    unmatched_required = [
        index for index in range(len(required_calls)) if index not in matched_required
    ]
    unmatched_predicted = [
        index for index in range(len(predicted_calls)) if index not in matched_predicted
    ]
    passed = _validation_passed(
        spec,
        unmatched_required=unmatched_required,
        unmatched_predicted=unmatched_predicted,
        forbidden_indices=forbidden_indices,
        invalid_indices=invalid_indices,
    )
    return ToolValidationResult(
        call_policy=spec.call_policy,
        passed=passed,
        required_call_count=len(required_calls),
        predicted_call_count=len(predicted_calls),
        matched_pairs=[
            ToolCallMatch(
                required_index=required_index,
                predicted_index=predicted_index,
                name=required_calls[required_index].name,
            )
            for required_index, predicted_index in pairs
        ],
        unmatched_required_indices=unmatched_required,
        unmatched_predicted_indices=unmatched_predicted,
        forbidden_predicted_indices=forbidden_indices,
        invalid_predicted_indices=invalid_indices,
    )


def validate_manifest_tool_calls(
    manifest_row: Mapping[str, Any], predicted_calls: list[Any]
) -> ToolValidationResult | None:
    """Resolve one manifest row and validate calls, retaining oracle provenance."""
    spec, source = resolve_tool_validation_spec(manifest_row)
    if spec is None:
        return None
    result = validate_tool_calls(spec, predicted_calls)
    return result.model_copy(update={"source": source})


def _required_calls(spec: ToolValidationSpec) -> list[RequiredToolCall]:
    if isinstance(spec, NoCallsValidation):
        return []
    return spec.required_calls


def _model_or_mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="python")
        return dict(dumped) if isinstance(dumped, Mapping) else None
    return None


def _valid_predicted_call(call: dict[str, Any] | None) -> bool:
    if call is None:
        return False
    name = call.get("name")
    arguments = call.get("arguments", {})
    return (
        isinstance(name, str)
        and bool(name.strip())
        and isinstance(arguments, Mapping)
        and all(isinstance(key, str) and bool(key.strip()) for key in arguments)
    )


def _call_matches(requirement: RequiredToolCall, predicted_call: dict[str, Any] | None) -> bool:
    if not _valid_predicted_call(predicted_call):
        return False
    assert predicted_call is not None
    if predicted_call["name"] != requirement.name:
        return False
    predicted_arguments = predicted_call.get("arguments", {})
    required_names = set(requirement.arguments)
    predicted_names = set(predicted_arguments)
    if not required_names.issubset(predicted_names):
        return False
    extra_names = predicted_names - required_names
    if any(
        name not in requirement.optional_null_arguments or predicted_arguments[name] is not None
        for name in extra_names
    ):
        return False
    return all(
        _argument_matches(matcher, predicted_arguments[name])
        for name, matcher in requirement.arguments.items()
    )


def _argument_matches(matcher: ArgumentMatcher, predicted_value: Any) -> bool:
    match = _ARGUMENT_MATCHERS.get(type(matcher))
    if match is None:
        raise TypeError(f"unsupported argument matcher: {type(matcher).__name__}")
    return match(matcher, predicted_value)


def _match_exact(matcher: ExactArgumentMatcher, predicted_value: Any) -> bool:
    return _strict_equal(matcher.value, predicted_value)


def _match_number_equal(matcher: NumberEqualArgumentMatcher, predicted_value: Any) -> bool:
    return type(predicted_value) in {int, float} and matcher.value == predicted_value


def _match_one_of(matcher: OneOfArgumentMatcher, predicted_value: Any) -> bool:
    return any(_strict_equal(value, predicted_value) for value in matcher.values)


def _match_set_equals(matcher: SetEqualsArgumentMatcher, predicted_value: Any) -> bool:
    return isinstance(predicted_value, list) and _sets_strict_equal(matcher.values, predicted_value)


_ARGUMENT_MATCHERS = {
    ExactArgumentMatcher: _match_exact,
    NumberEqualArgumentMatcher: _match_number_equal,
    OneOfArgumentMatcher: _match_one_of,
    SetEqualsArgumentMatcher: _match_set_equals,
}


def _strict_equal(expected: Any, predicted: Any) -> bool:
    if type(expected) is not type(predicted):
        return False
    if isinstance(expected, dict):
        return set(expected) == set(predicted) and all(
            _strict_equal(value, predicted[key]) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(expected) == len(predicted) and all(
            _strict_equal(left, right) for left, right in zip(expected, predicted)
        )
    return bool(expected == predicted)


def _sets_strict_equal(expected: list[Any], predicted: list[Any]) -> bool:
    return len(expected) == len(predicted) and all(
        any(_strict_equal(item, other) for other in predicted) for item in expected
    )


def _maximum_bipartite_pairs(adjacency: list[list[int]]) -> list[tuple[int, int]]:
    """Return deterministic maximum-cardinality requirement/prediction pairs."""
    predicted_to_required: dict[int, int] = {}

    def augment(required_index: int, visited: set[int]) -> bool:
        for predicted_index in adjacency[required_index]:
            if predicted_index in visited:
                continue
            visited.add(predicted_index)
            previous = predicted_to_required.get(predicted_index)
            if previous is None or augment(previous, visited):
                predicted_to_required[predicted_index] = required_index
                return True
        return False

    for required_index in range(len(adjacency)):
        augment(required_index, set())
    return sorted(
        (
            (required_index, predicted_index)
            for predicted_index, required_index in predicted_to_required.items()
        ),
        key=lambda pair: pair[0],
    )


def _validation_passed(
    spec: ToolValidationSpec,
    *,
    unmatched_required: list[int],
    unmatched_predicted: list[int],
    forbidden_indices: list[int],
    invalid_indices: list[int],
) -> bool:
    if invalid_indices or forbidden_indices or unmatched_required:
        return False
    if isinstance(spec, RequiredSubsetValidation):
        return True
    return not unmatched_predicted
