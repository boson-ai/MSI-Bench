"""judge_migration - adapt legacy judge reports to the current answer-judge policy.

Calling spec:
    effective_answer_judge_report(report, manifest_rows, predictions, rubrics) ->
        (report | None, source)

Current reports pass through. Explicitly versioned hybrid reports remain
readable for historical metric filtering. Unversioned legacy reports reuse
every atomic decision verbatim -- including any LLM-judged ``tool.call_args``
atom -- and only re-stamp the policy version. No migration re-runs deterministic
tool validation. Inputs are not modified. Side effects: none.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from ib.scoring.judge import ANSWER_JUDGE_POLICY_VERSION


JudgeMetricSource = Literal[
    "current_report",
    "compatible_previous",
    "migrated_legacy",
    "stale",
    "not_requested",
]
MIGRATED_ANSWER_JUDGE_POLICY_VERSION = "ib.answer_judge.migrated_legacy_semantics.reuse_llm_tool.v1"
PREVIOUS_ANSWER_JUDGE_POLICY_VERSIONS = frozenset({"ib.answer_judge.hybrid_tool_validation.v1"})


def effective_answer_judge_report(
    report: Mapping[str, Any] | None,
    manifest_rows: list[dict[str, Any]],
    predictions: Mapping[str, dict[str, Any]],
    rubrics: Mapping[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any] | None, JudgeMetricSource]:
    """Return current-policy judge metrics without another model call."""
    del manifest_rows, predictions  # reuse-only migration needs neither
    if report is None:
        return None, "not_requested"
    if report.get("policy_version") == ANSWER_JUDGE_POLICY_VERSION:
        return dict(report), "current_report"
    if report.get("policy_version") in PREVIOUS_ANSWER_JUDGE_POLICY_VERSIONS:
        return dict(report), "compatible_previous"
    if (
        report.get("policy_version") is not None
        or report.get("schema_version") != "ib.answer_judge.v1"
    ):
        return None, "stale"
    try:
        return _migrate_legacy_report(report, rubrics or {}), "migrated_legacy"
    except ValueError:
        return None, "stale"


def _migrate_legacy_report(
    report: Mapping[str, Any],
    rubrics: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    raw_judgments = report.get("judgments")
    if not isinstance(raw_judgments, list) or report.get("errors"):
        raise ValueError("legacy judge report needs judgments")
    seen: set[tuple[str, str]] = set()
    judgments = []
    atomic_reused = 0
    for item in raw_judgments:
        judgment, atom_count = _migrate_judgment(item, rubrics)
        key = (judgment["cell_id"], judgment["rubric_id"])
        if key in seen:
            raise ValueError(f"duplicate legacy judgment {key}")
        seen.add(key)
        judgments.append(judgment)
        atomic_reused += atom_count
    passed_count = sum(1 for item in judgments if item["passed"])
    atomic_count = sum(item["atomic_count"] for item in judgments)
    atomic_passed_count = sum(item["atomic_passed_count"] for item in judgments)
    judgment_count = len(judgments)
    return {
        **report,
        "policy_version": MIGRATED_ANSWER_JUDGE_POLICY_VERSION,
        "judgment_count": judgment_count,
        "passed_count": passed_count,
        "pass_rate": passed_count / judgment_count if judgment_count else 1.0,
        "atomic_count": atomic_count,
        "atomic_passed_count": atomic_passed_count,
        "atomic_pass_rate": atomic_passed_count / atomic_count if atomic_count else 1.0,
        "judgments": judgments,
        "policy_migration": {
            "kind": "legacy_atomic_reuse",
            "source_policy_version": report.get("policy_version"),
            "atomic_reused": atomic_reused,
        },
    }


def _migrate_judgment(
    raw: Any,
    rubrics: Mapping[str, dict[str, Any]],
) -> tuple[dict[str, Any], int]:
    if not isinstance(raw, Mapping):
        raise ValueError("legacy judgment must be an object")
    cell_id = raw.get("cell_id")
    rubric_id = raw.get("rubric_id")
    atoms = raw.get("atomic_results")
    rubric = rubrics.get(rubric_id) if isinstance(rubric_id, str) else None
    canonical = _canonical_atoms(rubric)
    if (
        not isinstance(cell_id, str)
        or not isinstance(rubric_id, str)
        or not isinstance(atoms, list)
        or len(atoms) != len(canonical)
    ):
        raise ValueError("legacy judgment does not align with its rubric")
    migrated_atoms = [_migrate_atom(atom, spec) for atom, spec in zip(atoms, canonical)]
    atomic_count = len(migrated_atoms)
    atomic_passed_count = sum(1 for atom in migrated_atoms if atom.get("passed") is True)
    migrated = {
        **raw,
        "rationale": "migrated legacy atomic decisions reused verbatim",
        "passed": atomic_passed_count == atomic_count,
        "atomic_count": atomic_count,
        "atomic_passed_count": atomic_passed_count,
        "atomic_pass_rate": atomic_passed_count / atomic_count if atomic_count else 1.0,
        "atomic_results": migrated_atoms,
    }
    return migrated, atomic_count


def _canonical_atoms(rubric: Any) -> list[dict[str, Any]]:
    if not isinstance(rubric, Mapping) or not isinstance(rubric.get("atomic_criteria"), list):
        raise ValueError("legacy migration needs the source rubric")
    specs = []
    for index, item in enumerate(rubric["atomic_criteria"]):
        if not isinstance(item, Mapping):
            raise ValueError("rubric atom must be an object")
        criterion_index = item.get("criterion_index", index)
        dimension = item.get("dimension", "legacy.unspecified")
        criterion = item.get("criterion")
        if (
            not isinstance(criterion_index, int)
            or not isinstance(dimension, str)
            or not isinstance(criterion, str)
        ):
            raise ValueError("rubric atom fields are invalid")
        specs.append(
            {
                "criterion_index": criterion_index,
                "dimension": dimension,
                "criterion": criterion,
            }
        )
    return specs


def _migrate_atom(atom: Any, spec: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(atom, Mapping) or not isinstance(atom.get("passed"), bool):
        raise ValueError("legacy atomic result must be an object")
    if atom.get("criterion_index") != spec["criterion_index"]:
        raise ValueError("legacy atom criterion_index changed")
    if atom.get("criterion") != spec["criterion"]:
        raise ValueError("legacy atom criterion changed")
    if atom.get("dimension") not in {None, spec["dimension"]}:
        raise ValueError("legacy atom dimension changed")
    rationale = atom.get("rationale", "")
    if not isinstance(rationale, str):
        raise ValueError("legacy atom rationale must be a string")
    return {**spec, "passed": atom["passed"], "rationale": rationale}
