"""Eval reuse invalidates reports from older scoring policies."""

from __future__ import annotations

import json

from ib.cli import eval as eval_module
from ib.scoring.eval import MANIFEST_EVAL_POLICY_VERSION
from ib.scoring.judge import ANSWER_JUDGE_POLICY_VERSION
from ib.scoring.report_policy import summary_report_has_policy


def test_matching_answer_judge_requires_current_policy_version() -> None:
    summary = {
        "answer_judge_enabled": True,
        "judge_provider": "openai",
        "judge_model": "gpt-5.5",
    }

    assert (
        eval_module._summary_has_matching_answer_judge(
            summary,
            judge_provider="openai",
            judge_model="gpt-5.5",
        )
        is False
    )

    summary["answer_judge_policy_version"] = ANSWER_JUDGE_POLICY_VERSION

    assert eval_module._summary_has_matching_answer_judge(
        summary,
        judge_provider="openai",
        judge_model="gpt-5.5",
    )


def test_replacing_old_judge_recomputes_base_pass_state(tmp_path, monkeypatch) -> None:
    root = tmp_path / "eval"
    root.mkdir()
    (root / "manifest_eval.json").write_text(
        json.dumps({"policy_version": MANIFEST_EVAL_POLICY_VERSION, "errors": []}) + "\n",
        encoding="utf-8",
    )
    (root / "rubric_eval.json").write_text('{"errors": []}\n', encoding="utf-8")
    summary = {
        "schema_version": "ib.eval.v1",
        "passed": False,
        "predictions": "predictions.jsonl",
        "eval_manifest": "manifest.jsonl",
        "rubrics": "rubrics.jsonl",
        "reports": {
            "manifest_eval": "manifest_eval.json",
            "rubric_eval": "rubric_eval.json",
            "answer_judge": "old_answer_judge.json",
        },
        "answer_judge_enabled": True,
    }

    def passing_judge(*args, **kwargs):
        del args, kwargs
        return {"errors": [], "pass_rate": 1.0}

    monkeypatch.setattr(eval_module, "run_answer_judge", passing_judge)

    updated = eval_module._add_answer_judge_to_summary(
        root,
        summary,
        judge_provider="openai",
        judge_model="gpt-5.5",
    )

    assert updated["passed"] is True
    assert updated["manifest_eval_policy_version"] == MANIFEST_EVAL_POLICY_VERSION
    assert updated["answer_judge_policy_version"] == ANSWER_JUDGE_POLICY_VERSION
    persisted = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert persisted["passed"] is True


def test_base_eval_reports_rejects_stale_manifest_policy(tmp_path) -> None:
    reports = {
        "manifest_eval": "manifest_eval.json",
        "rubric_eval": "rubric_eval.json",
    }
    (tmp_path / "rubric_eval.json").write_text('{"errors": []}\n', encoding="utf-8")
    (tmp_path / "manifest_eval.json").write_text('{"errors": []}\n', encoding="utf-8")

    assert eval_module._base_eval_reports_pass(tmp_path, reports) is False
    assert (
        summary_report_has_policy(
            tmp_path,
            reports,
            "manifest_eval",
            MANIFEST_EVAL_POLICY_VERSION,
        )
        is False
    )

    (tmp_path / "manifest_eval.json").write_text(
        json.dumps({"policy_version": MANIFEST_EVAL_POLICY_VERSION, "errors": []}) + "\n",
        encoding="utf-8",
    )

    assert eval_module._base_eval_reports_pass(tmp_path, reports) is True
    assert summary_report_has_policy(
        tmp_path,
        reports,
        "manifest_eval",
        MANIFEST_EVAL_POLICY_VERSION,
    )


def test_failed_judge_calls_block_target_reuse(tmp_path) -> None:
    root = tmp_path / "eval"
    root.mkdir()
    summary = {"reports": {"answer_judge": "answer_judge.json"}}

    (root / "answer_judge.json").write_text(
        json.dumps({"errors": []}), encoding="utf-8"
    )
    assert eval_module._answer_judge_report_has_failed_calls(root, summary) is False

    (root / "answer_judge.json").write_text(
        json.dumps(
            {
                "errors": [
                    "answer judge failed for c1 c1:logic: JudgeProviderError: "
                    '{"error":{"message":"Key limit exceeded (weekly limit)","code":403}}'
                ]
            }
        ),
        encoding="utf-8",
    )
    assert eval_module._answer_judge_report_has_failed_calls(root, summary) is True

    (root / "answer_judge.json").write_text("not json", encoding="utf-8")
    assert eval_module._answer_judge_report_has_failed_calls(root, summary) is True

    assert (
        eval_module._answer_judge_report_has_failed_calls(root, {"reports": {}}) is False
    )
