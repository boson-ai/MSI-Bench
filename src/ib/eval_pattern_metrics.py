"""eval_pattern_metrics — rubric pass metrics for eval leaderboard summaries.

Calling spec:
    apr_ars_metrics(..., validator_pass_by_cell=None) -> dict
    judge_by_cell(answer_judge, validator_pass_by_cell=None) -> dict

Inputs: answer-judge rows, optional cell patterns, and deterministic validator passes.
Outputs: judge-gated semantic+validator APR/ARS plus tool-call calibration observations.
Side effects: none.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from ib.scoring.tool_validation import DETERMINISTIC_TOOL_VALIDATOR_PREFIX
from ib.tool_call_rubric import TOOL_CALL_ARGS_DIMENSION

LEADERBOARD_JUDGE_METRIC_POLICY_VERSION = (
    "ib.leaderboard.judge_metrics.judge_gated_validator_tool.raw_apr.v3"
)
TOOL_VALIDATOR_DIMENSION = "tool.validator"
# Dimensions never folded into ARS/APR regardless of who decided them.
IGNORED_ANSWER_JUDGE_DIMENSIONS = frozenset(
    {
        "authority.request_authorized_confirmation",
    }
)


def apr_ars_metrics(
    answer_judge: dict[str, Any],
    *,
    included_cell_ids: set[str] | None = None,
    cell_patterns: dict[str, str] | None = None,
    validator_pass_by_cell: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """Return direct testcase APR and atom-level ARS for answer-judged cells."""
    judgments = [item for item in answer_judge.get("judgments", []) or [] if isinstance(item, dict)]
    judged_cell_ids = {
        cell_id for item in judgments if isinstance((cell_id := item.get("cell_id")), str)
    }
    by_cell: dict[str, list[bool]] = defaultdict(list)
    rubric_passed = 0
    rubric_total = 0
    for item in judgments:
        cell_id = item.get("cell_id")
        if included_cell_ids is not None and cell_id not in included_cell_ids:
            continue
        atomic_passed, atomic_total = scored_atomic_counts(item)
        rubric_passed += atomic_passed
        rubric_total += atomic_total
        if isinstance(cell_id, str) and atomic_total:
            by_cell[cell_id].append(scored_judgment_passed(item))
    for cell_id, passed in _validator_pass_items(validator_pass_by_cell):
        if cell_id not in judged_cell_ids:
            continue
        if included_cell_ids is not None and cell_id not in included_cell_ids:
            continue
        rubric_passed += int(passed)
        rubric_total += 1
        by_cell[cell_id].append(passed)
    case_results = {cell_id: all(items) for cell_id, items in by_cell.items() if items}
    case_total = len(case_results)
    case_passed = sum(1 for passed in case_results.values() if passed)
    pattern_summary = pattern_apr_summary(case_results, cell_patterns or {})
    apr = case_passed / case_total if case_total else None
    return {
        "apr_case_total": case_total,
        "apr_case_passed": case_passed,
        "apr_raw": apr,
        "apr_pattern_total": len(pattern_summary),
        "apr_by_pattern": pattern_summary,
        "apr": apr,
        "ars_rubric_total": rubric_total,
        "ars_rubric_passed": rubric_passed,
        "ars": rubric_passed / rubric_total if rubric_total else None,
    }


def pattern_apr_summary(
    case_results: dict[str, bool], cell_patterns: dict[str, str]
) -> dict[str, dict[str, Any]]:
    """Return per-pattern APR summaries from case pass booleans."""
    grouped: dict[str, dict[str, int]] = defaultdict(lambda: {"case_total": 0, "case_passed": 0})
    for cell_id, passed in case_results.items():
        pattern = cell_patterns.get(cell_id) or "unknown"
        grouped[pattern]["case_total"] += 1
        grouped[pattern]["case_passed"] += int(passed)
    return {
        pattern: {
            **counts,
            "apr": counts["case_passed"] / counts["case_total"] if counts["case_total"] else None,
        }
        for pattern, counts in sorted(grouped.items())
    }


def judge_by_cell(
    answer_judge: dict[str, Any],
    *,
    validator_pass_by_cell: Mapping[str, bool] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return per-cell semantic+validator scores and LLM tool calibration."""
    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "atomic_count": 0,
            "atomic_passed_count": 0,
            "ignored_atomic_count": 0,
            "judgment_count": 0,
            "passed_count": 0,
            "dimensions": {},
            "llm_tool_call_args_total": 0,
            "llm_tool_call_args_passed": 0,
        }
    )
    for item in answer_judge.get("judgments", []) or []:
        cell_id = item.get("cell_id")
        if not isinstance(cell_id, str):
            continue
        bucket = grouped[cell_id]
        atomic_passed, atomic_total = scored_atomic_counts(item)
        bucket["atomic_count"] += atomic_total
        bucket["atomic_passed_count"] += atomic_passed
        bucket["ignored_atomic_count"] += ignored_atomic_count(item)
        if atomic_total:
            bucket["judgment_count"] += 1
            bucket["passed_count"] += int(scored_judgment_passed(item))
        for atom in llm_tool_call_args_results(item):
            bucket["llm_tool_call_args_total"] += 1
            bucket["llm_tool_call_args_passed"] += int(atom.get("passed") is True)
        for atom in scored_atomic_results(item) or []:
            dimension = str(atom.get("dimension") or "") or "unlabeled"
            passed, total = bucket["dimensions"].get(dimension, (0, 0))
            bucket["dimensions"][dimension] = (
                passed + int(atom.get("passed") is True),
                total + 1,
            )
    for cell_id, passed in _validator_pass_items(validator_pass_by_cell):
        if cell_id not in grouped:
            continue
        bucket = grouped[cell_id]
        bucket["atomic_count"] += 1
        bucket["atomic_passed_count"] += int(passed)
        bucket["judgment_count"] += 1
        bucket["passed_count"] += int(passed)
        dimension_passed, dimension_total = bucket["dimensions"].get(
            TOOL_VALIDATOR_DIMENSION, (0, 0)
        )
        bucket["dimensions"][TOOL_VALIDATOR_DIMENSION] = (
            dimension_passed + int(passed),
            dimension_total + 1,
        )
    for bucket in grouped.values():
        total = bucket["atomic_count"]
        judgments = bucket["judgment_count"]
        llm_tool_total = bucket["llm_tool_call_args_total"]
        bucket["atomic_pass_rate"] = bucket["atomic_passed_count"] / total if total else None
        bucket["passed"] = bucket["passed_count"] == judgments if judgments else None
        bucket["llm_tool_call_args_pass"] = (
            bucket["llm_tool_call_args_passed"] == llm_tool_total if llm_tool_total else None
        )
    return dict(grouped)


def scored_atomic_counts(judgment: dict[str, Any]) -> tuple[int, int]:
    """Return answer-judge atomic pass/total after leaderboard-level filters."""
    atoms = scored_atomic_results(judgment)
    if atoms is None:
        return (
            int(judgment.get("atomic_passed_count", 0) or 0),
            int(judgment.get("atomic_count", 0) or 0),
        )
    return sum(1 for atom in atoms if atom.get("passed") is True), len(atoms)


def scored_judgment_passed(judgment: dict[str, Any]) -> bool:
    """Return case pass after ignored answer-judge dimensions are removed."""
    atoms = scored_atomic_results(judgment)
    if atoms is None:
        return bool(judgment.get("passed"))
    return all(atom.get("passed") is True for atom in atoms)


def ignored_atomic_count(judgment: dict[str, Any]) -> int:
    atoms = judgment.get("atomic_results")
    if not isinstance(atoms, list):
        return 0
    return sum(1 for atom in atoms if is_ignored_atomic_result(atom))


def scored_atomic_results(judgment: dict[str, Any]) -> list[dict[str, Any]] | None:
    atoms = judgment.get("atomic_results")
    if not isinstance(atoms, list):
        return None
    return [atom for atom in atoms if not is_ignored_atomic_result(atom)]


def llm_tool_call_args_results(judgment: dict[str, Any]) -> list[dict[str, Any]]:
    """Return LLM tool-call decisions retained only for validator calibration."""
    atoms = judgment.get("atomic_results")
    if not isinstance(atoms, list):
        return []
    return [atom for atom in atoms if is_llm_tool_call_args_result(atom)]


def is_llm_tool_call_args_result(atom: Any) -> bool:
    if not isinstance(atom, dict) or atom.get("dimension") != TOOL_CALL_ARGS_DIMENSION:
        return False
    rationale = str(atom.get("rationale") or "")
    return not rationale.startswith(DETERMINISTIC_TOOL_VALIDATOR_PREFIX)


def is_ignored_atomic_result(atom: Any) -> bool:
    if not isinstance(atom, dict):
        return False
    dimension = str(atom.get("dimension") or "")
    if dimension in IGNORED_ANSWER_JUDGE_DIMENSIONS:
        return True
    # LLM tool.call_args stays available for calibration but never affects APR/ARS.
    # The deterministic validator contributes one separate tool.validator atom.
    if dimension == TOOL_CALL_ARGS_DIMENSION:
        return True
    return False


def _validator_pass_items(
    values: Mapping[str, bool] | None,
) -> list[tuple[str, bool]]:
    if values is None:
        return []
    return [
        (cell_id, passed)
        for cell_id, passed in values.items()
        if isinstance(cell_id, str) and isinstance(passed, bool)
    ]
