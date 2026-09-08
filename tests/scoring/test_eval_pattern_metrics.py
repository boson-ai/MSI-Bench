"""Tests for judge-gated eval leaderboard APR/ARS metrics."""

from __future__ import annotations

from ib.eval_pattern_metrics import apr_ars_metrics, judge_by_cell


def test_apr_is_direct_testcase_pass_rate() -> None:
    answer_judge = {
        "judgments": [
            {"cell_id": "a1", "passed": False, "atomic_count": 1, "atomic_passed_count": 0},
            {"cell_id": "a2", "passed": False, "atomic_count": 1, "atomic_passed_count": 0},
            {"cell_id": "e1", "passed": True, "atomic_count": 1, "atomic_passed_count": 1},
        ]
    }

    metrics = apr_ars_metrics(
        answer_judge,
        cell_patterns={
            "a1": "authority_gated_override",
            "a2": "authority_gated_override",
            "e1": "eavesdropping",
        },
    )

    assert metrics["apr_case_passed"] == 1
    assert metrics["apr_case_total"] == 3
    assert metrics["apr_raw"] == 1 / 3
    assert metrics["apr_pattern_total"] == 2
    assert metrics["apr"] == 1 / 3
    assert metrics["apr_by_pattern"]["authority_gated_override"]["apr"] == 0.0
    assert metrics["apr_by_pattern"]["eavesdropping"]["apr"] == 1.0


def test_validator_tool_atom_scores_while_llm_tool_atom_is_calibration_only() -> None:
    answer_judge = {
        "judgments": [
            {
                "cell_id": "llm-disagrees",
                "atomic_results": [
                    {
                        "dimension": "tool.call_args",
                        "passed": False,
                        "rationale": "LLM thinks the function arguments are wrong",
                    },
                    {"dimension": "response.required_behavior", "passed": True},
                ],
            },
            {
                "cell_id": "validator-fails",
                "atomic_results": [
                    {
                        "dimension": "tool.call_args",
                        "passed": True,
                        "rationale": "LLM accepts the function arguments",
                    },
                    {"dimension": "response.required_behavior", "passed": True},
                ],
            },
        ]
    }
    validator_passes = {
        "llm-disagrees": True,
        "validator-fails": False,
        "not-judged": True,
    }

    metrics = apr_ars_metrics(answer_judge, validator_pass_by_cell=validator_passes)
    by_cell = judge_by_cell(answer_judge, validator_pass_by_cell=validator_passes)

    # Two semantic atoms + one validator atom per judged applicable testcase.
    # The LLM tool.call_args atoms are retained only as calibration observations.
    assert metrics["ars_rubric_total"] == 4
    assert metrics["ars_rubric_passed"] == 3
    assert metrics["apr_case_total"] == 2
    assert metrics["apr_case_passed"] == 1

    assert by_cell["llm-disagrees"]["passed"] is True
    assert by_cell["llm-disagrees"]["atomic_count"] == 2
    assert by_cell["llm-disagrees"]["ignored_atomic_count"] == 1
    assert by_cell["llm-disagrees"]["llm_tool_call_args_pass"] is False
    assert by_cell["llm-disagrees"]["dimensions"] == {
        "response.required_behavior": (1, 1),
        "tool.validator": (1, 1),
    }
    assert by_cell["validator-fails"]["passed"] is False
    assert by_cell["validator-fails"]["llm_tool_call_args_pass"] is True
    assert "not-judged" not in by_cell


def test_validator_without_answer_judge_does_not_create_apr_or_ars() -> None:
    metrics = apr_ars_metrics({}, validator_pass_by_cell={"c1": True})

    assert metrics["apr_case_total"] == 0
    assert metrics["apr"] is None
    assert metrics["ars_rubric_total"] == 0
    assert metrics["ars"] is None
    assert judge_by_cell({}, validator_pass_by_cell={"c1": True}) == {}
