"""S8.A3 — optional answer-layer judge path."""

from __future__ import annotations

import json

import pytest

from ib.cli.eval import run_answer_judge
from ib.llm.qwen import QWEN35_BASE_URL, QWEN35_MAX_OUTPUT_TOKENS, QWEN35_MODEL
from ib.scoring import judge as judge_module
from ib.scoring.judge import (
    DEFAULT_OPENAI_JUDGE_MODEL,
    JudgeProviderError,
    OpenAiJudgeProvider,
    judge_answers,
    judge_model_for_provider,
    provider_for,
)


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
    class _ParaphrasingJudge:
        name = "paraphrase"

        def judge(self, *, manifest_row, prediction, rubric):
            del manifest_row, prediction, rubric
            return {
                "rationale": "ok",
                "atomic_results": [
                    {
                        "passed": True,
                        "rationale": "matched first atom",
                    },
                    {
                        "passed": True,
                        "rationale": "matched second atom",
                    },
                ],
            }

        async def ajudge(self, *, manifest_row, prediction, rubric, service):
            del service
            return self.judge(manifest_row=manifest_row, prediction=prediction, rubric=rubric)

    manifest, predictions, rubrics = _artifacts(tmp_path)

    result = judge_answers(manifest, predictions, rubrics, provider=_ParaphrasingJudge())

    atoms = result["judgments"][0]["atomic_results"]
    assert [atom["criterion"] for atom in atoms] == ["scene=work_professional", "forbidden"]


def test_answer_judge_marks_omitted_atomic_criteria_failed(tmp_path) -> None:
    class _OmittingJudge:
        name = "omit"

        def judge(self, *, manifest_row, prediction, rubric):
            del manifest_row, prediction, rubric
            return {
                "rationale": "ok",
                "atomic_results": [
                    {
                        "passed": True,
                        "rationale": "matched first atom",
                    }
                ],
            }

        async def ajudge(self, *, manifest_row, prediction, rubric, service):
            del service
            return self.judge(manifest_row=manifest_row, prediction=prediction, rubric=rubric)

    manifest, predictions, rubrics = _artifacts(tmp_path)

    result = judge_answers(manifest, predictions, rubrics, provider=_OmittingJudge())

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
    class _NoAtomicJudge:
        name = "no-atomic"

        def judge(self, *, manifest_row, prediction, rubric):
            del manifest_row, prediction, rubric
            return {"rationale": "ok"}

        async def ajudge(self, *, manifest_row, prediction, rubric, service):
            del service
            return self.judge(manifest_row=manifest_row, prediction=prediction, rubric=rubric)

    manifest, predictions, rubrics = _artifacts(tmp_path)

    result = judge_answers(manifest, predictions, rubrics, provider=_NoAtomicJudge())

    assert result["atomic_count"] == 2
    assert result["atomic_passed_count"] == 0
    assert result["pass_rate"] == 0.0


def test_answer_judge_fails_empty_answer_even_for_avoid_only_rubric(tmp_path) -> None:
    class _OvercreditingJudge:
        name = "overcrediting"

        def judge(self, *, manifest_row, prediction, rubric):
            del manifest_row, prediction, rubric
            return {
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

        async def ajudge(self, *, manifest_row, prediction, rubric, service):
            del service
            return self.judge(manifest_row=manifest_row, prediction=prediction, rubric=rubric)

    manifest, predictions, rubrics = _artifacts(tmp_path, answer_text="")
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
                        "polarity": "avoid",
                        "criterion": "forbidden",
                    }
                ],
            }
        ],
    )

    result = judge_answers(manifest, predictions, rubrics, provider=_OvercreditingJudge())

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


def test_answer_judge_does_not_override_clarification_atoms_when_answer_commits(
    tmp_path,
) -> None:
    class _PassingJudge:
        name = "passing"

        def judge(self, *, manifest_row, prediction, rubric):
            del manifest_row, prediction, rubric
            return {
                "score": 1,
                "passed": True,
                "rationale": "over-credited",
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

        async def ajudge(self, *, manifest_row, prediction, rubric, service):
            del service
            return self.judge(manifest_row=manifest_row, prediction=prediction, rubric=rubric)

    manifest, predictions, rubrics = _artifacts(
        tmp_path,
        answer_text="I'll play the jazz mix from your phone on the rear speakers.",
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
                        "criterion": "clarify the rear audio source",
                    }
                ],
            }
        ],
    )

    result = judge_answers(manifest, predictions, rubrics, provider=_PassingJudge())

    atom = result["judgments"][0]["atomic_results"][0]
    assert result["atomic_pass_rate"] == 1.0
    assert atom["passed"] is True
    assert atom["rationale"] == ""


def test_answer_judge_allows_explicit_clarification_atoms(tmp_path) -> None:
    class _PassingJudge:
        name = "passing"

        def judge(self, *, manifest_row, prediction, rubric):
            del manifest_row, prediction, rubric
            return {
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

        async def ajudge(self, *, manifest_row, prediction, rubric, service):
            del service
            return self.judge(manifest_row=manifest_row, prediction=prediction, rubric=rubric)

    manifest, predictions, rubrics = _artifacts(
        tmp_path,
        answer_text=(
            "I'll start the route and keep the rear speakers low, but I need to confirm "
            "whether the rear source is Marcus's podcast or Sonia's jazz mix."
        ),
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
                        "criterion": "clarify the rear audio source",
                    }
                ],
            }
        ],
    )

    result = judge_answers(manifest, predictions, rubrics, provider=_PassingJudge())

    assert result["atomic_pass_rate"] == 1.0
    assert result["judgments"][0]["atomic_results"][0]["passed"] is True


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


def test_answer_judge_cli_writes_report(tmp_path, monkeypatch, det_judge) -> None:
    from ib.scoring import judge as judge_module

    monkeypatch.setattr(judge_module, "provider_for", lambda name, *, client=None: det_judge)
    manifest, predictions, rubrics = _artifacts(tmp_path)
    output = tmp_path / "judge.json"

    report = run_answer_judge(str(manifest), str(predictions), str(rubrics), str(output), "openai")

    assert report["passed_count"] == 1
    assert report["llm_call_count"] == 0
    assert report["llm_usage"].endswith("judge.cost.json")
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == "ib.answer_judge.v1"


def test_openai_judge_requires_api_key(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    manifest, predictions, rubrics = _artifacts(tmp_path)

    with pytest.raises(JudgeProviderError, match="OPENAI_API_KEY"):
        judge_answers(manifest, predictions, rubrics, provider="openai")


def test_openai_judge_uses_injected_client(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    class _Choice:
        message = type(
            "Message", (), {"content": '{"rationale": "ok", "atomic_results": []}'}
        )()

    class _Completions:
        def create(self, **kwargs):
            return type("Response", (), {"choices": [_Choice()]})()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    provider = OpenAiJudgeProvider(client=_Client())
    payload = provider.judge(
        manifest_row={"cell_id": "c1"}, prediction={"answer_text": "x"}, rubric={"rubric_id": "r1"}
    )

    assert payload == {
        "passed": True,
        "rationale": "ok",
        "atomic_count": 0,
        "atomic_passed_count": 0,
        "atomic_pass_rate": 1.0,
        "atomic_results": [],
    }


def test_openai_judge_prompt_includes_predicted_tool_calls_without_gold_calls(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    captured = {}

    class _Choice:
        message = type(
            "Message",
            (),
            {
                "content": json.dumps(
                    {
                        "rationale": "ok",
                        "atomic_results": [
                            {
                                "passed": True,
                                "rationale": "matched",
                            }
                        ],
                    }
                )
            },
        )()

    class _Completions:
        def create(self, **kwargs):
            captured["messages"] = kwargs["messages"]
            return type("Response", (), {"choices": [_Choice()]})()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    provider = OpenAiJudgeProvider(client=_Client())
    provider.judge(
        manifest_row={
            "cell_id": "c1",
            "scene": "commerce_service",
            "logic_expected_tool_calls": [
                {"name": "expected_secret_tool", "arguments": {"order_id": "SECRET-R2846"}}
            ],
            "logic_expected_state_updates": {"secret_state": "SECRET-STATE"},
        },
        prediction={
            "cell_id": "c1",
            "predicted_action": "respond",
            "answer_text": "The return is started.",
            "tool_calls": [{"name": "start_return", "arguments": {"order_id": "VISIBLE-R2846"}}],
            "confidence": 0.9,
        },
        rubric={
            "rubric_id": "c1:o1",
            "atomic_criteria": [
                {
                    "polarity": "satisfy",
                    "criterion_index": 0,
                    "criterion": "say return is started",
                }
            ],
        },
    )

    assert captured["messages"][0]["role"] == "system"
    assert "Evaluate prediction.answer_text and prediction.tool_calls" in captured["messages"][0]["content"]
    assert '"score"' not in captured["messages"][0]["content"]
    assert '"criterion_index"' not in captured["messages"][0]["content"]
    assert '"polarity"' not in captured["messages"][0]["content"]
    prompt = captured["messages"][1]["content"]
    assert "The return is started." in prompt
    assert "tool_calls" in prompt
    assert "start_return" in prompt
    assert "VISIBLE-R2846" in prompt
    assert "commerce_service" in prompt
    assert "expected_secret_tool" not in prompt
    assert "SECRET-R2846" not in prompt
    assert "SECRET-STATE" not in prompt


def test_judge_user_prompt_preserves_non_ascii_text() -> None:
    prompt = judge_module._user_prompt(
        {
            "scene": "domestic_household",
            "expected_action": "respond",
            "visible_context_lines": [
                {"line_id": "L1", "speaker": "妈", "text": "补一箱蓝盖纯奶。"}
            ],
        },
        {
            "predicted_action": "respond",
            "answer_text": "好的，我会记录。",
            "tool_calls": [],
        },
        {
            "judge_instruction": "判断中文答复。",
            "atomic_criteria": [
                {
                    "dimension": "response.required_behavior",
                    "criterion": "必须记录蓝盖纯奶。",
                }
            ],
        },
    )

    assert '"speaker": "妈"' in prompt
    assert '"answer_text": "好的，我会记录。"' in prompt
    assert '"criterion": "必须记录蓝盖纯奶。"' in prompt
    assert r"\u5988" not in prompt
    assert r"\u597d\u7684" not in prompt
    assert r"\u5fc5\u987b" not in prompt


def test_judge_manifest_view_adds_clean_visible_context_only() -> None:
    view = judge_module._judge_manifest_view(
        {
            "cell_id": "c1",
            "scene": "commerce_service",
            "expected_action": "respond",
            "logic_pattern": "authority_gated_override",
            "visible_context_lines": [
                {
                    "line_id": "L1",
                    "speaker": "林浩",
                    "addressed_to": "周敏",
                    "text": "<|emotion:enthusiasm|>周敏，我想升级。[keyboard_typing]",
                },
                {
                    "line_id": "L2",
                    "speaker": "周敏",
                    "addressed_to": "林浩",
                    "text": "<|sfx:sigh|>不同意。",
                },
            ],
            "participant_context": {"participants": [{"speaker_id": "S1"}]},
            "logic_available_functions": [{"name": "modify_room_type"}],
            "logic_standard_answer": "oracle answer",
        }
    )

    assert view == {
        "scene": "commerce_service",
        "expected_action": "respond",
        "logic_pattern": "authority_gated_override",
        "visible_context_lines": [
            {
                "line_id": "L1",
                "speaker": "林浩",
                "addressed_to": "周敏",
                "text": "周敏，我想升级。",
            },
            {
                "line_id": "L2",
                "speaker": "周敏",
                "addressed_to": "林浩",
                "text": "不同意。",
            },
        ],
    }
    assert "cell_id" not in view


def test_judge_rubric_view_omits_ids_and_layers() -> None:
    rubric = {
        "cell_id": "c1",
        "rubric_id": "c1:o1",
        "layer": "O1",
        "provider": "logic_contract",
        "target_turn_index": 6,
        "judge_instruction": "Assess the answer.",
        "atomic_criteria": [
            {
                "criterion_id": "c1:o1:0",
                "criterion_index": 0,
                "dimension": "response.required_behavior",
                "polarity": "satisfy",
                "criterion": "refuse the unauthorized upgrade",
                "source_index": 0,
            }
        ],
    }

    assert judge_module._judge_rubric_view(rubric) == {
        "judge_instruction": "Assess the answer.",
        "atomic_criteria": [
            {
                "polarity": "satisfy",
                "dimension": "response.required_behavior",
                "criterion": "refuse the unauthorized upgrade",
            }
        ],
    }


def test_unknown_judge_provider_fails_loudly() -> None:
    with pytest.raises(JudgeProviderError, match="unknown"):
        provider_for("anthropic")


def test_openai_judge_defaults_to_gpt55() -> None:
    provider = provider_for("openai")

    assert isinstance(provider, OpenAiJudgeProvider)
    assert provider.model == DEFAULT_OPENAI_JUDGE_MODEL
    assert judge_model_for_provider("openai") == DEFAULT_OPENAI_JUDGE_MODEL


def test_openai_judge_model_can_be_selected() -> None:
    provider = provider_for("openai", model="gpt-test")

    assert isinstance(provider, OpenAiJudgeProvider)
    assert provider.model == "gpt-test"


def test_qwen_judge_provider_uses_openai_compatible_endpoint() -> None:
    provider = provider_for("qwen3_5")

    assert isinstance(provider, OpenAiJudgeProvider)
    assert provider.name == "qwen3_5"
    assert provider.model == QWEN35_MODEL
    assert provider.base_url == QWEN35_BASE_URL
    assert provider.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    assert provider.max_tokens == QWEN35_MAX_OUTPUT_TOKENS


def test_qwen_judge_model_override_preserves_thinking_mode() -> None:
    provider = provider_for("qwen3_5_thinking", model="custom-qwen")

    assert isinstance(provider, OpenAiJudgeProvider)
    assert provider.model == "custom-qwen"
    assert provider.extra_body == {"chat_template_kwargs": {"enable_thinking": True}}
    assert provider.max_tokens == QWEN35_MAX_OUTPUT_TOKENS


def test_answer_judge_cli_forwards_selected_model(tmp_path, monkeypatch, det_judge) -> None:
    from ib.scoring import judge as judge_module

    selected = {}

    def _provider_for(name, *, model=None):
        selected.update(provider=name, model=model)
        return det_judge

    monkeypatch.setattr(judge_module, "provider_for", _provider_for)
    manifest, predictions, rubrics = _artifacts(tmp_path)

    run_answer_judge(
        str(manifest),
        str(predictions),
        str(rubrics),
        str(tmp_path / "judge.json"),
        "openai",
        "judge-model",
    )

    assert selected == {"provider": "openai", "model": "judge-model"}


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

        def judge(self, **_kwargs):
            raise AssertionError("judge must not be called for format-error predictions")

    report = judge_answers(manifest, predictions, rubrics, provider=_ExplodingJudge())

    assert report["judgment_count"] == 1
    judgment = report["judgments"][0]
    assert judgment["passed"] is False
    assert judgment["atomic_count"] > 0
    assert judgment["atomic_passed_count"] == 0
    assert "format error" in judgment["rationale"]
    assert report["errors"] == []
