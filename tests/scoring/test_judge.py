"""S8.A3 — optional answer-layer judge path."""

from __future__ import annotations

import asyncio
import json

import pytest

from ib.evaluation.reports import run_answer_judge
from ib.scoring import judge as judge_module
from ib.scoring.judge import (
    JudgeProviderError,
    ajudge_answers,
)


def judge_answers(*args, **kwargs) -> dict:
    return asyncio.run(ajudge_answers(*args, service=None, **kwargs))


def _write_jsonl(path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def _artifacts(tmp_path, *, answer_text="scene=work_professional"):
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    rubrics = tmp_path / "rubrics.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "cell_id": "c1",
                "expected_action": "respond",
                "rubric_ids": ["c1:o1"],
                "answer_judge_inputs": [{"layer": "O1", "rubric_id": "c1:o1"}],
            }
        ],
    )
    _write_jsonl(
        predictions, [{"cell_id": "c1", "predicted_action": "respond", "answer_text": answer_text}]
    )
    _write_jsonl(
        rubrics,
        [
            {
                "cell_id": "c1",
                "rubric_id": "c1:o1",
                "layer": "O1",
                "atomic_criteria": [
                    {
                        "criterion_index": 0,
                        "polarity": "satisfy",
                        "criterion": "scene=work_professional",
                    },
                    {
                        "criterion_index": 1,
                        "polarity": "avoid",
                        "criterion": "forbidden",
                    },
                ],
            }
        ],
    )
    return manifest, predictions, rubrics


def _write_single_atom_rubric(rubrics, *, polarity: str, criterion: str) -> None:
    _write_jsonl(
        rubrics,
        [
            {
                "cell_id": "c1",
                "rubric_id": "c1:o1",
                "layer": "O1",
                "atomic_criteria": [
                    {"criterion_index": 0, "polarity": polarity, "criterion": criterion}
                ],
            }
        ],
    )


class _StubJudge:
    """Judge provider that returns one fixed payload for every rubric."""

    name = "stub"

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def judge(self, *, manifest_row, prediction, rubric):
        del manifest_row, prediction, rubric
        return self.payload

    async def ajudge(self, *, manifest_row, prediction, rubric, service):
        del service
        return self.judge(manifest_row=manifest_row, prediction=prediction, rubric=rubric)


def test_answer_judge_consumes_planner_rubrics(tmp_path, det_judge) -> None:
    manifest, predictions, rubrics = _artifacts(tmp_path)

    result = judge_answers(manifest, predictions, rubrics, provider=det_judge)

    assert result["schema_version"] == "ib.answer_judge.v1"
    assert result["provider"] == "deterministic"
    assert result["judgment_count"] == 1
    assert result["pass_rate"] == 1.0
    assert result["atomic_count"] == 2
    assert result["atomic_passed_count"] == 2
    assert result["atomic_pass_rate"] == 1.0
    assert result["judgments"][0]["rubric_id"] == "c1:o1"
    assert result["judgments"][0]["atomic_results"] == [
        {
            "criterion_index": 0,
            "dimension": "legacy.unspecified",
            "criterion": "scene=work_professional",
            "passed": True,
            "rationale": "deterministic exact atomic check",
        },
        {
            "criterion_index": 1,
            "dimension": "legacy.unspecified",
            "criterion": "forbidden",
            "passed": True,
            "rationale": "deterministic exact atomic check",
        },
    ]


def test_answer_judge_reports_missing_required_content(tmp_path, det_judge) -> None:
    manifest, predictions, rubrics = _artifacts(tmp_path, answer_text="wrong")

    result = judge_answers(manifest, predictions, rubrics, provider=det_judge)

    assert result["pass_rate"] == 0.0
    assert result["judgments"][0]["passed"] is False
    assert result["judgments"][0]["atomic_pass_rate"] == 0.5


def test_answer_judge_normalizes_paraphrased_atomic_criterion_text(tmp_path) -> None:
    judge = _StubJudge(
        {
            "rationale": "ok",
            "atomic_results": [
                {"passed": True, "rationale": "matched first atom"},
                {"passed": True, "rationale": "matched second atom"},
            ],
        }
    )
    manifest, predictions, rubrics = _artifacts(tmp_path)

    result = judge_answers(manifest, predictions, rubrics, provider=judge)

    atoms = result["judgments"][0]["atomic_results"]
    assert [atom["criterion"] for atom in atoms] == ["scene=work_professional", "forbidden"]


def test_answer_judge_marks_omitted_atomic_criteria_failed(tmp_path) -> None:
    judge = _StubJudge(
        {"rationale": "ok", "atomic_results": [{"passed": True, "rationale": "matched first atom"}]}
    )
    manifest, predictions, rubrics = _artifacts(tmp_path)

    result = judge_answers(manifest, predictions, rubrics, provider=judge)

    atoms = result["judgments"][0]["atomic_results"]
    assert result["atomic_pass_rate"] == 0.5
    assert atoms[1] == {
        "criterion_index": 1,
        "dimension": "legacy.unspecified",
        "criterion": "forbidden",
        "passed": False,
        "rationale": "judge omitted this atomic criterion",
    }


def test_answer_judge_marks_missing_atomic_results_failed(tmp_path) -> None:
    manifest, predictions, rubrics = _artifacts(tmp_path)

    result = judge_answers(manifest, predictions, rubrics, provider=_StubJudge({"rationale": "ok"}))

    assert result["atomic_count"] == 2
    assert result["atomic_passed_count"] == 0
    assert result["pass_rate"] == 0.0


def test_answer_judge_fails_empty_answer_even_for_avoid_only_rubric(tmp_path) -> None:
    judge = _StubJudge(
        {
            "score": 1,
            "passed": True,
            "rationale": "avoid atom passed",
            "atomic_results": [
                {
                    "polarity": "avoid",
                    "criterion_index": 0,
                    "criterion": "forbidden",
                    "passed": True,
                    "score": 1,
                }
            ],
        }
    )
    manifest, predictions, rubrics = _artifacts(tmp_path, answer_text="")
    _write_single_atom_rubric(rubrics, polarity="avoid", criterion="forbidden")

    result = judge_answers(manifest, predictions, rubrics, provider=judge)

    atom = result["judgments"][0]["atomic_results"][0]
    assert result["pass_rate"] == 0.0
    assert atom["passed"] is False
    assert "empty answer cannot satisfy" in atom["rationale"]


def test_answer_judge_reports_forbidden_content_atom(tmp_path, det_judge) -> None:
    manifest, predictions, rubrics = _artifacts(
        tmp_path, answer_text="scene=work_professional forbidden"
    )

    result = judge_answers(manifest, predictions, rubrics, provider=det_judge)

    assert result["pass_rate"] == 0.0
    assert result["atomic_passed_count"] == 1
    assert result["judgments"][0]["atomic_results"][1]["passed"] is False


@pytest.mark.parametrize(
    "answer_text",
    [
        "I'll play the jazz mix from your phone on the rear speakers.",
        (
            "I'll start the route and keep the rear speakers low, but I need to confirm "
            "whether the rear source is Marcus's podcast or Sonia's jazz mix."
        ),
    ],
    ids=["answer_commits", "answer_clarifies"],
)
def test_answer_judge_does_not_override_clarification_atom_decisions(tmp_path, answer_text) -> None:
    judge = _StubJudge(
        {
            "score": 1,
            "passed": True,
            "rationale": "ok",
            "atomic_results": [
                {
                    "polarity": "satisfy",
                    "criterion_index": 0,
                    "criterion": "clarify the rear audio source",
                    "passed": True,
                    "score": 1,
                }
            ],
        }
    )
    manifest, predictions, rubrics = _artifacts(tmp_path, answer_text=answer_text)
    _write_single_atom_rubric(
        rubrics, polarity="satisfy", criterion="clarify the rear audio source"
    )

    result = judge_answers(manifest, predictions, rubrics, provider=judge)

    atom = result["judgments"][0]["atomic_results"][0]
    assert result["atomic_pass_rate"] == 1.0
    assert atom["passed"] is True
    assert atom["rationale"] == ""


def test_answer_judge_rejects_legacy_bucket_rubrics(tmp_path, det_judge) -> None:
    manifest, predictions, rubrics = _artifacts(tmp_path)
    _write_jsonl(
        rubrics,
        [
            {
                "cell_id": "c1",
                "rubric_id": "c1:o1",
                "layer": "O1",
                "must_include": ["scene=work_professional"],
                "must_not_include": ["forbidden"],
            }
        ],
    )

    with pytest.raises(JudgeProviderError, match="legacy must_include/must_not_include"):
        judge_answers(manifest, predictions, rubrics, provider=det_judge)


def test_answer_judge_cli_writes_report_with_selected_model(
    tmp_path, monkeypatch, det_judge
) -> None:
    selected = {}

    def _provider_for(name, *, model=None):
        selected.update(provider=name, model=model)
        return det_judge

    monkeypatch.setattr(judge_module, "provider_for", _provider_for)
    manifest, predictions, rubrics = _artifacts(tmp_path)
    output = tmp_path / "judge.json"

    report = run_answer_judge(
        str(manifest), str(predictions), str(rubrics), str(output), "openai", "judge-model"
    )

    assert selected == {"provider": "openai", "model": "judge-model"}
    assert report["passed_count"] == 1
    assert report["llm_call_count"] == 0
    assert report["llm_usage"].endswith("judge.usage.json")
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == "ib.answer_judge.v1"


def test_format_error_predictions_skip_judge_calls(tmp_path) -> None:
    manifest, predictions, rubrics = _artifacts(tmp_path)
    _write_jsonl(
        predictions,
        [
            {
                "cell_id": "c1",
                "predicted_action": "silent",
                "answer_text": None,
                "tool_calls": [],
                "confidence": 0.0,
                "metadata": {
                    "provider_error": {
                        "type": "prediction_format_error",
                        "message": "invalid JSON prediction for c1",
                    }
                },
            }
        ],
    )

    class _ExplodingJudge:
        name = "exploding"

        async def ajudge(self, **_kwargs):
            raise AssertionError("judge must not be called for format-error predictions")

    report = judge_answers(manifest, predictions, rubrics, provider=_ExplodingJudge())

    assert report["judgment_count"] == 1
    judgment = report["judgments"][0]
    assert judgment["passed"] is False
    assert judgment["atomic_count"] > 0
    assert judgment["atomic_passed_count"] == 0
    assert "format error" in judgment["rationale"]
    assert report["errors"] == []


def test_judge_prompt_hides_ignored_read_only_lookups() -> None:
    row = {
        "cell_id": "c1",
        "expected_action": "tool_call",
        "logic_tool_validation": {
            "call_policy": "exact_multiset",
            "required_calls": [
                {"name": "set_alarm", "arguments": {"time": {"op": "exact", "value": "07:00"}}}
            ],
            "read_only_call_names": ["get_current_time"],
        },
    }
    prediction = {
        "cell_id": "c1",
        "answer_text": "Alarm set.",
        "tool_calls": [
            {"name": "get_current_time", "arguments": {}},
            {"name": "set_alarm", "arguments": {"time": "07:00"}},
        ],
    }
    rubric = {"atomic_criteria": [{"polarity": "satisfy", "criterion": "sets the alarm"}]}

    request = judge_module.OpenAiJudgeProvider()._llm_request(
        manifest_row=row, prediction=prediction, rubric=rubric
    )
    user_prompt = request.messages[1]["content"]

    assert "set_alarm" in user_prompt
    assert "get_current_time" not in user_prompt
