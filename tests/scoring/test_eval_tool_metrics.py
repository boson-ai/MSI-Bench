"""Leaderboard tool aggregation separates strict and diagnostic denominators."""

from ib.eval_tool_metrics import leaderboard_tool_metrics


def test_strict_pass_does_not_require_a_fuzzy_diagnostic() -> None:
    metrics = leaderboard_tool_metrics(
        [
            {"tool_pass": True, "tool_score": None, "tool_argument_score": None},
            {"tool_pass": False, "tool_score": 0.9, "tool_argument_score": 0.8},
            {"tool_pass": None, "tool_score": 0.7, "tool_argument_score": 0.6},
        ]
    )

    assert metrics["tool_total"] == 2
    assert metrics["tool_passed"] == 1
    assert metrics["tool_pass_rate"] == 0.5
    assert metrics["tool_score"] == 0.8
    assert metrics["tool_argument_score"] == 0.7


def test_tool_execution_metrics_exclude_no_call_cases() -> None:
    metrics = leaderboard_tool_metrics(
        [
            {
                "tool_pass": True,
                "tool_score": 1.0,
                "tool_name_f1": 1.0,
                "tool_argument_score": 1.0,
                "tool_validation": {"required_call_count": 0},
            },
            {
                "tool_pass": True,
                "tool_score": 1.0,
                "tool_name_f1": 1.0,
                "tool_argument_score": 1.0,
                "tool_validation": {"required_call_count": 1},
            },
            {
                "tool_pass": False,
                "tool_score": 0.4,
                "tool_name_f1": 0.5,
                "tool_argument_score": 0.3,
                "tool_validation": {"required_call_count": 2},
            },
        ]
    )

    assert metrics["tool_total"] == 3
    assert metrics["tool_execution_total"] == 2
    assert metrics["tool_execution_passed"] == 1
    assert metrics["tool_execution_pass_rate"] == 0.5
    assert metrics["tool_execution_score"] == 0.7
    assert metrics["tool_execution_name_f1"] == 0.75
    assert metrics["tool_execution_argument_score"] == 0.65
