"""Model schema tests for deterministic tool-call validation contracts."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from ib.models.tool_validation import (
    ExactMultisetValidation,
    NumberEqualArgumentMatcher,
    RequiredSubsetValidation,
    ToolValidationResult,
    ToolValidationSpec,
)


def test_tool_validation_spec_parses_discriminated_argument_matchers() -> None:
    spec = TypeAdapter(ToolValidationSpec).validate_python(
        {
            "call_policy": "exact_multiset",
            "required_calls": [
                {
                    "name": "update_itinerary",
                    "arguments": {
                        "status": {"op": "one_of", "values": ["held", "pending"]},
                        "traveler_ids": {
                            "op": "set_equals",
                            "values": ["a", "b"],
                        },
                        "amount": {"op": "number_equal", "value": 4},
                        "notify": {"op": "exact", "value": False},
                    },
                }
            ],
        }
    )

    assert isinstance(spec, ExactMultisetValidation)
    assert spec.required_calls[0].arguments["status"].op == "one_of"
    assert isinstance(
        spec.required_calls[0].arguments["amount"], NumberEqualArgumentMatcher
    )
    assert spec.model_dump(mode="json")["required_calls"][0]["arguments"] == {
        "status": {"op": "one_of", "values": ["held", "pending"]},
        "traveler_ids": {"op": "set_equals", "values": ["a", "b"]},
        "amount": {"op": "number_equal", "value": 4},
        "notify": {"op": "exact", "value": False},
    }


def test_tool_validation_spec_rejects_unknown_fields_and_operators() -> None:
    adapter = TypeAdapter(ToolValidationSpec)

    with pytest.raises(ValidationError, match="union_tag_invalid"):
        adapter.validate_python(
            {
                "call_policy": "required_subset",
                "required_calls": [
                    {
                        "name": "lookup",
                        "arguments": {"query": {"op": "contains", "value": "x"}},
                    }
                ],
            }
        )
    with pytest.raises(ValidationError, match="extra_forbidden"):
        adapter.validate_python({"call_policy": "no_calls", "required_calls": []})


def test_required_and_forbidden_tool_names_must_be_disjoint() -> None:
    with pytest.raises(ValidationError, match="cannot also be forbidden"):
        RequiredSubsetValidation(
            required_calls=[{"name": "save", "arguments": {}}],
            forbidden_call_names=["save"],
        )


def test_required_subset_allows_forbidden_only_but_rejects_an_empty_rule() -> None:
    spec = RequiredSubsetValidation(forbidden_call_names=["delete_account"])

    assert spec.required_calls == []
    with pytest.raises(ValidationError, match="needs required_calls or forbidden_call_names"):
        RequiredSubsetValidation()


def test_set_equals_spec_rejects_duplicate_values() -> None:
    with pytest.raises(ValidationError, match="set_equals values must be unique"):
        TypeAdapter(ToolValidationSpec).validate_python(
            {
                "call_policy": "exact_multiset",
                "required_calls": [
                    {
                        "name": "update",
                        "arguments": {
                            "travelers": {
                                "op": "set_equals",
                                "values": ["alice", "alice"],
                            }
                        },
                    }
                ],
            }
        )


@pytest.mark.parametrize("value", ["4", True, None])
def test_number_equal_spec_rejects_non_json_numbers(value: object) -> None:
    with pytest.raises(ValidationError, match="number_equal value must be a JSON number"):
        NumberEqualArgumentMatcher(value=value)


def test_validation_result_has_json_diagnostics_and_stable_rationale() -> None:
    result = ToolValidationResult(
        source="explicit",
        call_policy="required_subset",
        passed=True,
        required_call_count=1,
        predicted_call_count=2,
        matched_pairs=[{"required_index": 0, "predicted_index": 1, "name": "save"}],
        unmatched_predicted_indices=[0],
    )

    assert result.model_dump(mode="json")["policy_version"] == "ib.tool_validation.v2"
    assert result.rationale() == (
        "passed: required_subset matched 1/1 required call(s); "
        "1 unmatched predicted call(s); 0 forbidden call(s); 0 invalid call(s)"
    )
