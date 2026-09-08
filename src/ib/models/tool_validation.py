"""tool_validation - schemas for deterministic tool-call validation.

Calling spec:
    spec: ToolValidationSpec
    result = ToolValidationResult(...)

Specs declare required calls, argument match operators, and forbidden tool names.
Results contain only deterministic validation diagnostics. Side effects: none.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from ib.models.base import IbBaseModel


TOOL_VALIDATOR_POLICY_VERSION = "ib.tool_validation.v2"
ToolCallPolicy = Literal["no_calls", "exact_multiset", "required_subset"]
ToolValidationSource = Literal["explicit", "legacy"]


class ExactArgumentMatcher(IbBaseModel):
    """Require one argument value to equal ``value`` exactly."""

    op: Literal["exact"] = "exact"
    value: Any


class NumberEqualArgumentMatcher(IbBaseModel):
    """Require equal JSON numbers without accepting strings or booleans."""

    op: Literal["number_equal"] = "number_equal"
    value: Any

    @model_validator(mode="after")
    def _validate_json_number(self) -> "NumberEqualArgumentMatcher":
        if type(self.value) not in {int, float}:
            raise ValueError("number_equal value must be a JSON number")
        return self


class OneOfArgumentMatcher(IbBaseModel):
    """Require one argument value to equal one of ``values`` exactly."""

    op: Literal["one_of"] = "one_of"
    values: list[Any] = Field(min_length=1)


class SetEqualsArgumentMatcher(IbBaseModel):
    """Require one list argument to contain the same distinct values."""

    op: Literal["set_equals"] = "set_equals"
    values: list[Any]

    @model_validator(mode="after")
    def _validate_distinct_values(self) -> "SetEqualsArgumentMatcher":
        for index, value in enumerate(self.values):
            if any(_strict_schema_equal(value, prior) for prior in self.values[:index]):
                raise ValueError("set_equals values must be unique")
        return self


ArgumentMatcher = Annotated[
    ExactArgumentMatcher
    | NumberEqualArgumentMatcher
    | OneOfArgumentMatcher
    | SetEqualsArgumentMatcher,
    Field(discriminator="op"),
]


class RequiredToolCall(IbBaseModel):
    """One required tool name and its complete argument contract."""

    name: str = Field(min_length=1)
    arguments: dict[str, ArgumentMatcher] = Field(default_factory=dict)
    optional_null_arguments: list[str] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )

    @model_validator(mode="after")
    def _validate_names(self) -> "RequiredToolCall":
        if not self.name.strip():
            raise ValueError("required tool call name must not be blank")
        blank_arguments = sorted(key for key in self.arguments if not key.strip())
        if blank_arguments:
            raise ValueError("required tool argument names must not be blank")
        if any(not name.strip() for name in self.optional_null_arguments):
            raise ValueError("optional null argument names must not be blank")
        if len(self.optional_null_arguments) != len(set(self.optional_null_arguments)):
            raise ValueError("optional null argument names must be unique")
        overlap = sorted(set(self.arguments) & set(self.optional_null_arguments))
        if overlap:
            raise ValueError(
                f"matched arguments cannot also be optional null arguments: {overlap}"
            )
        return self


class NoCallsValidation(IbBaseModel):
    """Reject every predicted tool call."""

    call_policy: Literal["no_calls"] = "no_calls"


class ExactMultisetValidation(IbBaseModel):
    """Require exactly the declared multiset of calls, in any order."""

    call_policy: Literal["exact_multiset"] = "exact_multiset"
    required_calls: list[RequiredToolCall] = Field(min_length=1)
    forbidden_call_names: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_forbidden_names(self) -> "ExactMultisetValidation":
        _validate_forbidden_call_names(self.required_calls, self.forbidden_call_names)
        return self


class RequiredSubsetValidation(IbBaseModel):
    """Require declared calls while allowing other non-forbidden calls."""

    call_policy: Literal["required_subset"] = "required_subset"
    required_calls: list[RequiredToolCall] = Field(default_factory=list)
    forbidden_call_names: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_forbidden_names(self) -> "RequiredSubsetValidation":
        if not self.required_calls and not self.forbidden_call_names:
            raise ValueError("required_subset needs required_calls or forbidden_call_names")
        _validate_forbidden_call_names(self.required_calls, self.forbidden_call_names)
        return self


ToolValidationSpec = Annotated[
    NoCallsValidation | ExactMultisetValidation | RequiredSubsetValidation,
    Field(discriminator="call_policy"),
]


class ToolCallMatch(IbBaseModel):
    """One edge selected by maximum bipartite matching."""

    required_index: int = Field(ge=0)
    predicted_index: int = Field(ge=0)
    name: str


class ToolValidationResult(IbBaseModel):
    """Structured outcome of one deterministic tool-call validation."""

    policy_version: str = TOOL_VALIDATOR_POLICY_VERSION
    source: ToolValidationSource | None = None
    call_policy: ToolCallPolicy
    passed: bool
    required_call_count: int = Field(ge=0)
    predicted_call_count: int = Field(ge=0)
    matched_pairs: list[ToolCallMatch] = Field(default_factory=list)
    unmatched_required_indices: list[int] = Field(default_factory=list)
    unmatched_predicted_indices: list[int] = Field(default_factory=list)
    forbidden_predicted_indices: list[int] = Field(default_factory=list)
    invalid_predicted_indices: list[int] = Field(default_factory=list)

    def rationale(self) -> str:
        """Return a stable, compact explanation suitable for score reports."""
        status = "passed" if self.passed else "failed"
        matched = len(self.matched_pairs)
        summary = (
            f"{status}: {self.call_policy} matched {matched}/"
            f"{self.required_call_count} required call(s)"
        )
        details = [
            f"{len(self.unmatched_predicted_indices)} unmatched predicted call(s)",
            f"{len(self.forbidden_predicted_indices)} forbidden call(s)",
            f"{len(self.invalid_predicted_indices)} invalid call(s)",
        ]
        return f"{summary}; " + "; ".join(details)


def _validate_forbidden_call_names(
    required_calls: list[RequiredToolCall], forbidden_call_names: list[str]
) -> None:
    if any(not name.strip() for name in forbidden_call_names):
        raise ValueError("forbidden tool names must not be blank")
    if len(forbidden_call_names) != len(set(forbidden_call_names)):
        raise ValueError("forbidden tool names must be unique")
    overlap = sorted({call.name for call in required_calls} & set(forbidden_call_names))
    if overlap:
        raise ValueError(f"required tool names cannot also be forbidden: {overlap}")


def _strict_schema_equal(left: Any, right: Any) -> bool:
    """Compare JSON-like schema values without Python's bool/int coercion."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _strict_schema_equal(value, right[key]) for key, value in left.items()
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _strict_schema_equal(a, b) for a, b in zip(left, right)
        )
    return bool(left == right)
