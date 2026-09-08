"""tool_call_args — validate or clean gold tool-call arguments.

Calling spec:
    gold_tool_arg_policy() -> "allow" | "clean" | "strict"
    clean_gold_tool_arguments(args) -> dict
    validate_gold_tool_arguments(args, call_name) -> None

Inputs are expected/gold tool-call argument mappings. Outputs are cleaned
mappings or validation errors. Side effects: reads IB_GOLD_TOOL_ARG_POLICY.
"""

from __future__ import annotations

import os
from typing import Any, Literal


GoldToolArgPolicy = Literal["allow", "clean", "strict"]
_POLICY_ENV = "IB_GOLD_TOOL_ARG_POLICY"
_VALID_POLICIES = {"allow", "clean", "strict"}


def gold_tool_arg_policy() -> GoldToolArgPolicy:
    """Return how gold tool calls should handle null/empty argument values."""
    raw = os.getenv(_POLICY_ENV, "clean").strip().lower()
    if raw not in _VALID_POLICIES:
        raise ValueError(
            f"{_POLICY_ENV} must be one of allow, clean, strict; got {raw!r}"
        )
    return raw  # type: ignore[return-value]


def clean_gold_tool_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Return arguments with null/empty values omitted."""
    return {
        key: value
        for key, value in arguments.items()
        if not _is_invalid_gold_value(value)
    }


def validate_gold_tool_arguments(arguments: dict[str, Any], *, call_name: str) -> None:
    """Raise ValueError if any gold argument carries a null/empty value."""
    invalid = [
        key for key, value in arguments.items() if _is_invalid_gold_value(value)
    ]
    if invalid:
        joined = ", ".join(sorted(invalid))
        raise ValueError(
            f"gold tool call {call_name!r} has null/empty argument(s): {joined}; "
            "omit unknown fields or use an explicit grounded value such as scope='all'"
        )


def _is_invalid_gold_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, dict)) and not value:
        return True
    return False
