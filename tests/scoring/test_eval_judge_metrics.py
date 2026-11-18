"""judge_by_cell: per-case answer-judge scores with the deterministic tool validator folded in."""

from __future__ import annotations

from ib.evaluation.judge_metrics import judge_by_cell


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

    by_cell = judge_by_cell(answer_judge, validator_pass_by_cell=validator_passes)

    # One semantic atom + one validator atom per judged case; the LLM tool.call_args atom is
    # kept only as a calibration observation.
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


def test_a_validator_pass_without_a_judgment_scores_nothing() -> None:
    assert judge_by_cell({}, validator_pass_by_cell={"c1": True}) == {}
