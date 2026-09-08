"""eval_tool_metrics - aggregate strict tool results for leaderboard rows.

Calling spec:
    leaderboard_tool_metrics(records) -> metric dict

Consumes per-record validation, strict pass, and fuzzy diagnostics. Side effects: none.
"""

from __future__ import annotations

from typing import Any


def leaderboard_tool_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return overall policy metrics plus required-call-only execution metrics."""
    strict = [
        record["tool_pass"] for record in records if isinstance(record.get("tool_pass"), bool)
    ]
    scores = _numbers(records, "tool_score")
    names = _numbers(records, "tool_name_f1")
    arguments = _numbers(records, "tool_argument_score")
    passed = sum(1 for value in strict if value)
    execution_records = [record for record in records if _requires_tool_execution(record)]
    execution_strict = [
        record["tool_pass"]
        for record in execution_records
        if isinstance(record.get("tool_pass"), bool)
    ]
    execution_passed = sum(1 for value in execution_strict if value)
    return {
        "tool_total": len(strict),
        "tool_score": _mean(scores),
        "tool_passed": passed,
        "tool_pass_rate": passed / len(strict) if strict else None,
        "tool_name_f1": _mean(names),
        "tool_argument_score": _mean(arguments),
        "tool_execution_total": len(execution_strict),
        "tool_execution_passed": execution_passed,
        "tool_execution_pass_rate": (
            execution_passed / len(execution_strict) if execution_strict else None
        ),
        "tool_execution_score": _mean(_numbers(execution_records, "tool_score")),
        "tool_execution_name_f1": _mean(_numbers(execution_records, "tool_name_f1")),
        "tool_execution_argument_score": _mean(_numbers(execution_records, "tool_argument_score")),
    }


def _requires_tool_execution(record: dict[str, Any]) -> bool:
    validation = record.get("tool_validation")
    return isinstance(validation, dict) and validation.get("required_call_count", 0) > 0


def _numbers(records: list[dict[str, Any]], key: str) -> list[float]:
    return [float(record[key]) for record in records if isinstance(record.get(key), int | float)]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
