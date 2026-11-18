"""judge_metrics — per-case answer-judge scores, the tool validator folded in.

Calling spec:
    judge_by_cell(answer_judge, validator_pass_by_cell=None) -> {cell_id: scores}

Inputs: an answer-judge report ({"judgments": [...]}) and optional deterministic tool-validator passes per
cell. Each cell gets its scored atomic counts and pass rate, a case pass, per-dimension (passed, total) and the
LLM tool.call_args calibration count. Ignored dimensions (IGNORED_ANSWER_JUDGE_DIMENSIONS, and the LLM
tool.call_args atom, which the deterministic validator replaces with one tool.validator atom) never count.
Used by scripts/release_eval_extras/paper_table_rows.py. Side effects: none.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from ib.scoring.tool_validation import DETERMINISTIC_TOOL_VALIDATOR_PREFIX
from ib.models.tool_call_rubric import TOOL_CALL_ARGS_DIMENSION

TOOL_VALIDATOR_DIMENSION = "tool.validator"
# Dimensions never folded into the scores regardless of who decided them.
IGNORED_ANSWER_JUDGE_DIMENSIONS = frozenset(
    {
        "authority.request_authorized_confirmation",
    }
)


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
    """Return answer-judge atomic pass/total after the ignored dimensions are removed."""
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
