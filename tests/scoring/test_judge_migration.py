"""Legacy judge migration reuses every atomic decision, including tool.call_args."""

from __future__ import annotations

from copy import deepcopy

from ib.scoring.judge import ANSWER_JUDGE_POLICY_VERSION
from ib.scoring.judge_migration import (
    MIGRATED_ANSWER_JUDGE_POLICY_VERSION,
    effective_answer_judge_report,
)


def _report(*, tool_passed: bool, semantic_passed: bool = True) -> dict:
    return {
        "schema_version": "ib.answer_judge.v1",
        "judgments": [
            {
                "cell_id": "c1",
                "rubric_id": "r1",
                "passed": tool_passed and semantic_passed,
                "atomic_results": [
                    {
                        "criterion_index": 0,
                        "dimension": "tool.call_args",
                        "criterion": "save holder Alice",
                        "passed": tool_passed,
                        "rationale": "legacy model tool decision",
                    },
                    {
                        "criterion_index": 1,
                        "dimension": "response.required_behavior",
                        "criterion": "confirm the save",
                        "passed": semantic_passed,
                        "rationale": "semantic model decision",
                    },
                ],
            }
        ],
        "errors": [],
    }


def _rubrics() -> dict[str, dict]:
    return {
        "r1": {
            "rubric_id": "r1",
            "atomic_criteria": [
                {
                    "criterion_index": 0,
                    "dimension": "tool.call_args",
                    "criterion": "save holder Alice",
                },
                {
                    "criterion_index": 1,
                    "dimension": "response.required_behavior",
                    "criterion": "confirm the save",
                },
            ],
        }
    }


def test_legacy_tool_decision_is_reused_not_revalidated() -> None:
    legacy = _report(tool_passed=False)
    original = deepcopy(legacy)

    migrated, source = effective_answer_judge_report(legacy, [], {}, _rubrics())

    assert source == "migrated_legacy"
    assert migrated is not None
    assert migrated["policy_version"] == MIGRATED_ANSWER_JUDGE_POLICY_VERSION
    # The LLM judged the tool atom as failing; migration keeps that verbatim
    # rather than substituting the deterministic validator.
    assert migrated["judgments"][0]["passed"] is False
    atoms = migrated["judgments"][0]["atomic_results"]
    assert [atom["passed"] for atom in atoms] == [False, True]
    assert atoms[0]["rationale"] == "legacy model tool decision"
    assert atoms[1]["rationale"] == "semantic model decision"
    assert migrated["policy_migration"]["atomic_reused"] == 2
    assert legacy == original  # inputs untouched


def test_legacy_tool_pass_reused_without_manifest_or_predictions() -> None:
    # Reuse-only migration no longer needs manifest rows or predictions.
    migrated, source = effective_answer_judge_report(_report(tool_passed=True), [], {}, _rubrics())

    assert source == "migrated_legacy"
    assert migrated is not None
    assert migrated["pass_rate"] == 1.0
    assert migrated["judgments"][0]["atomic_results"][0]["passed"] is True


def test_current_and_missing_reports_keep_their_sources() -> None:
    current = {**_report(tool_passed=True), "policy_version": ANSWER_JUDGE_POLICY_VERSION}

    reused, current_source = effective_answer_judge_report(current, [], {})
    missing, missing_source = effective_answer_judge_report(None, [], {})

    assert reused == current
    assert current_source == "current_report"
    assert missing is None
    assert missing_source == "not_requested"


def test_previous_hybrid_report_remains_compatible() -> None:
    previous = {
        **_report(tool_passed=True),
        "policy_version": "ib.answer_judge.hybrid_tool_validation.v1",
    }

    reused, source = effective_answer_judge_report(previous, [], {})

    assert reused == previous
    assert source == "compatible_previous"


def test_missing_source_rubric_keeps_the_legacy_report_stale() -> None:
    migrated, source = effective_answer_judge_report(_report(tool_passed=True), [], {}, {})

    assert migrated is None
    assert source == "stale"


def test_changed_rubric_keeps_the_legacy_report_stale() -> None:
    rubrics = _rubrics()
    rubrics["r1"]["atomic_criteria"][1]["criterion"] = "changed criterion"

    migrated, source = effective_answer_judge_report(_report(tool_passed=True), [], {}, rubrics)

    assert migrated is None
    assert source == "stale"
