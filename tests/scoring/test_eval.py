"""S8.A2 — manifest/rubric evaluation metrics consume planner artifacts."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ib.cli import main as cli_main
from ib.cli.eval import default_eval_output_dir, run_eval, run_manifest_eval, run_rubric_eval
from ib.eval_targets import EvalTarget
from ib.scoring.eval import manifest_eval, rubric_eval


def _write_jsonl(path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def _manifest_rows() -> list[dict]:
    labels = {
        "perception": {
            "scene": "work_professional",
            "addressee_cue": "wake_word",
            "acoustic_conditions": {
                "reverb": "dry",
                "competing_speech": "none",
                "channel": "clean_close",
            },
        },
        "interaction": {
            "expected_action": "respond",
            "expected_addressed_speaker": "colleague_a",
            "participation_frame": "user_addressed",
            "contract_required": False,
        },
        "answer": {"answerable": True, "rubric_ids": ["c1:o1"], "contract_text": None},
    }
    return [
        {
            "cell_id": "c1",
            "sibling_group": "g1",
            "expected_action": "respond",
            "rubric_ids": ["c1:o1"],
            "labels": labels,
        },
        {
            "cell_id": "c2",
            "sibling_group": "g1",
            "expected_action": "ignore",
            "rubric_ids": [],
            "labels": {
                **copy.deepcopy(labels),
                "answer": {"answerable": False, "rubric_ids": [], "contract_text": None},
            },
        },
    ]


def test_generated_eval_target_requires_explicit_model() -> None:
    with pytest.raises(ValueError, match="model is required"):
        EvalTarget(provider="openai")


def test_manifest_eval_reports_accuracy_labels_and_sibling_consistency(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(manifest, _manifest_rows())
    _write_jsonl(
        predictions,
        [
            {
                "cell_id": "c1",
                "predicted_action": "respond",
                "answer_text": "scene=work_professional",
                "tool_calls": [{"name": "note", "arguments": {"value": "ok"}}],
            },
            {"cell_id": "c2", "predicted_action": "silent"},
        ],
    )

    result = manifest_eval(manifest, predictions)

    assert result["schema_version"] == "ib.manifest_eval.v1"
    assert result["action_total"] == 0
    assert result["accuracy"] is None
    assert result["interaction_summary"] == {}
    assert result["counterfactual_sibling_consistency"]["evaluated_group_count"] == 0


def test_manifest_eval_reports_missing_predictions(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(manifest, _manifest_rows())
    _write_jsonl(predictions, [{"cell_id": "c1", "predicted_action": "respond"}])

    result = manifest_eval(manifest, predictions)

    assert result["missing_prediction_cell_ids"] == ["c2"]
    assert result["errors"]


def test_rubric_eval_validates_references_and_answerable_coverage(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    rubrics = tmp_path / "rubrics.jsonl"
    _write_jsonl(manifest, _manifest_rows())
    _write_jsonl(rubrics, [{"cell_id": "c1", "rubric_id": "c1:o1", "layer": "O1"}])

    result = rubric_eval(manifest, rubrics)

    assert result["schema_version"] == "ib.rubric_eval.v1"
    assert result["missing_rubric_ids"] == []
    assert result["answerable_rubric_coverage"] == 1.0
    assert result["errors"] == []


def test_rubric_eval_counts_refusals_as_answerable(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    rubrics = tmp_path / "rubrics.jsonl"
    rows = _manifest_rows()
    rows[1]["expected_action"] = "refuse"
    rows[1]["rubric_ids"] = ["c2:o1"]
    rows[1]["labels"]["interaction"]["expected_action"] = "refuse"
    rows[1]["labels"]["answer"] = {
        "answerable": True,
        "rubric_ids": ["c2:o1"],
        "contract_text": None,
    }
    _write_jsonl(manifest, rows)
    _write_jsonl(
        rubrics,
        [
            {"cell_id": "c1", "rubric_id": "c1:o1", "layer": "O1"},
            {"cell_id": "c2", "rubric_id": "c2:o1", "layer": "O1"},
        ],
    )

    result = rubric_eval(manifest, rubrics)

    assert result["answerable_row_count"] == 2
    assert result["answerable_rubric_coverage"] == 1.0
    assert result["errors"] == []


def test_rubric_eval_reports_missing_rubrics(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    rubrics = tmp_path / "rubrics.jsonl"
    _write_jsonl(manifest, _manifest_rows())
    _write_jsonl(rubrics, [])

    result = rubric_eval(manifest, rubrics)

    assert result["missing_rubric_ids"] == ["c1:o1"]
    assert result["errors"] == ["unknown rubric_id references in manifest: ['c1:o1']"]


def test_eval_cli_wrappers_write_reports(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    rubrics = tmp_path / "rubrics.jsonl"
    manifest_out = tmp_path / "manifest-eval.json"
    rubric_out = tmp_path / "rubric-eval.json"
    _write_jsonl(manifest, _manifest_rows())
    _write_jsonl(
        predictions,
        [
            {
                "cell_id": "c1",
                "predicted_action": "respond",
                "answer_text": "scene=work_professional",
                "tool_calls": [{"name": "note", "arguments": {"value": "ok"}}],
            },
            {"cell_id": "c2", "predicted_action": "silent"},
        ],
    )
    _write_jsonl(rubrics, [{"cell_id": "c1", "rubric_id": "c1:o1", "layer": "O1"}])

    manifest_report = run_manifest_eval(str(manifest), str(predictions), str(manifest_out))
    rubric_report = run_rubric_eval(str(manifest), str(rubrics), str(rubric_out))

    assert manifest_report["accuracy"] is None
    assert rubric_report["answerable_rubric_coverage"] == 1.0
    assert (
        json.loads(manifest_out.read_text(encoding="utf-8"))["schema_version"]
        == "ib.manifest_eval.v1"
    )
    assert (
        json.loads(rubric_out.read_text(encoding="utf-8"))["schema_version"] == "ib.rubric_eval.v1"
    )


def test_eval_recipe_cli_writes_model_evaluation_summary(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    rubrics = tmp_path / "rubrics.jsonl"
    output = tmp_path / "eval"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    rubrics.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(manifest, _manifest_rows())
    _write_jsonl(
        predictions,
        [
            {
                "cell_id": "c1",
                "predicted_action": "respond",
                "answer_text": "scene=work_professional",
                "tool_calls": [{"name": "note", "arguments": {"value": "ok"}}],
            },
            {"cell_id": "c2", "predicted_action": "silent"},
        ],
    )
    _write_jsonl(rubrics, [{"cell_id": "c1", "rubric_id": "c1:o1", "layer": "O1"}])

    code = cli_main.main(
        [
            "eval",
            "--manifest",
            str(manifest),
            "--predictions",
            str(predictions),
            "--rubrics",
            str(rubrics),
            "--output",
            str(output),
        ]
    )

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert code == 0
    assert summary["schema_version"] == "ib.eval.v1"
    assert summary["passed"] is True
    assert summary["answer_judge_enabled"] is False
    assert (output / summary["reports"]["manifest_eval"]).exists()
    assert (output / summary["reports"]["rubric_eval"]).exists()


def test_default_eval_output_mirrors_repo_relative_manifest_path(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    eval_root = tmp_path / "eval-root"
    monkeypatch.chdir(repo_root)
    monkeypatch.setenv("IB_EVAL_ROOT", str(eval_root))

    output = default_eval_output_dir(
        "artifacts/validation/build_domestic20260625T165052Z/mobility_transport/manifest/manifest.jsonl",
        understanding_provider="gemini",
        understanding_model="gemini-test",
    )

    assert output == (
        eval_root
        / "artifacts/validation/build_domestic20260625T165052Z/mobility_transport"
        / "generated/turn_based/native/gemini/gemini-test"
    )


def test_run_eval_infers_rubrics_for_canonical_scene_manifest(tmp_path) -> None:
    scene_root = tmp_path / "run" / "commerce_service"
    manifest = scene_root / "manifest" / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    rubrics = scene_root / "plans" / "rubrics.jsonl"
    output = tmp_path / "eval"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    rubrics.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(manifest, _manifest_rows())
    _write_jsonl(
        predictions,
        [
            {"cell_id": "c1", "predicted_action": "respond"},
            {"cell_id": "c2", "predicted_action": "silent"},
        ],
    )
    _write_jsonl(rubrics, [{"cell_id": "c1", "rubric_id": "c1:o1", "layer": "O1"}])

    summary = run_eval(str(manifest), str(predictions), None, str(output))

    assert summary["rubrics"] == str(rubrics)
    assert (output / "rubric_eval.json").exists()


def test_eval_leaderboard_filters_nonsemantic_judge_dimensions() -> None:
    from ib.eval_pattern_metrics import apr_ars_metrics, judge_by_cell

    answer_judge = {
        "judgments": [
            {
                "cell_id": "c1",
                "passed": False,
                "atomic_count": 4,
                "atomic_passed_count": 3,
                "atomic_results": [
                    {
                        "dimension": "tool.call_args",
                        "passed": True,
                        "rationale": "tool call matches the intended action",
                    },
                    {"dimension": "response.required_behavior", "passed": True},
                    {"dimension": "authority.uncertain_stance_blocks_execution", "passed": True},
                    {"dimension": "authority.request_authorized_confirmation", "passed": False},
                ],
            }
        ]
    }

    by_cell = judge_by_cell(answer_judge)
    metrics = apr_ars_metrics(answer_judge, included_cell_ids={"c1"})

    # The LLM tool.call_args atom is calibration-only; the authority confirmation
    # dimension is also excluded. No deterministic validator applies in this fixture.
    assert by_cell["c1"]["passed"] is True
    assert by_cell["c1"]["atomic_passed_count"] == 2
    assert by_cell["c1"]["atomic_count"] == 2
    assert by_cell["c1"]["ignored_atomic_count"] == 2
    assert by_cell["c1"]["llm_tool_call_args_pass"] is True
    assert metrics["apr_case_passed"] == 1
    assert metrics["apr_case_total"] == 1
    assert metrics["ars_rubric_passed"] == 2
    assert metrics["ars_rubric_total"] == 2


def test_eval_recipe_limit_applies_with_existing_predictions(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    rubrics = tmp_path / "rubrics.jsonl"
    output = tmp_path / "eval"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    rubrics.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(manifest, _manifest_rows())
    _write_jsonl(predictions, [{"cell_id": "c1", "predicted_action": "respond"}])
    _write_jsonl(rubrics, [{"cell_id": "c1", "rubric_id": "c1:o1", "layer": "O1"}])

    summary = run_eval(
        str(manifest),
        str(predictions),
        str(rubrics),
        str(output),
        limit=1,
    )

    manifest_report = json.loads((output / "manifest_eval.json").read_text(encoding="utf-8"))
    assert summary["eval_manifest"].endswith("manifest_subset.jsonl")
    assert manifest_report["total"] == 1
    assert manifest_report["missing_prediction_cell_ids"] == []


def test_eval_recipe_runs_mixed_mode_targets(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    turn_predictions = tmp_path / "turn_predictions.jsonl"
    live_predictions = tmp_path / "live_predictions.jsonl"
    rubrics = tmp_path / "rubrics.jsonl"
    output = tmp_path / "eval"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    rubrics.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(manifest, _manifest_rows())
    predictions_rows = [
        {"cell_id": "c1", "predicted_action": "respond"},
        {"cell_id": "c2", "predicted_action": "silent"},
    ]
    _write_jsonl(turn_predictions, predictions_rows)
    _write_jsonl(live_predictions, predictions_rows)
    _write_jsonl(rubrics, [{"cell_id": "c1", "rubric_id": "c1:o1", "layer": "O1"}])

    summary = run_eval(
        str(manifest),
        None,
        str(rubrics),
        str(output),
        target_specs=[
            f"turn_based:openai:gpt-audio-1.5:{turn_predictions}",
            f"fullduplex:openai_realtime:gpt-realtime:{live_predictions}",
        ],
    )

    assert summary["schema_version"] == "ib.eval_batch.v1"
    assert summary["target_count"] == 2
    assert {target["mode"] for target in summary["targets"]} == {"turn_based", "fullduplex"}
    for target in summary["targets"]:
        assert (output / target["summary"]).exists()


def test_eval_recipe_generates_gemini_live_fullduplex_target(tmp_path, monkeypatch) -> None:
    from ib.cli import eval as eval_module

    manifest = tmp_path / "manifest.jsonl"
    rubrics = tmp_path / "rubrics.jsonl"
    output = tmp_path / "eval"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    rubrics.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(manifest, _manifest_rows())
    _write_jsonl(rubrics, [{"cell_id": "c1", "rubric_id": "c1:o1", "layer": "O1"}])

    def fake_gemini_live(manifest_path, predictions_path, *, model, limit=None, live_probe=False, thinking_mode=None):
        del manifest_path, model, limit, live_probe, thinking_mode
        _write_jsonl(
            Path(predictions_path),
            [
                {"cell_id": "c1", "predicted_action": "respond"},
                {"cell_id": "c2", "predicted_action": "silent"},
            ],
        )
        return {
            "schema_version": "ib.gemini_fullduplex.v1",
            "provider": "gemini_live",
            "model": "gemini-live-test",
            "predictions": str(predictions_path),
        }

    monkeypatch.setattr(eval_module, "run_gemini_fullduplex_predictions", fake_gemini_live)

    summary = run_eval(
        str(manifest),
        None,
        str(rubrics),
        str(output),
        target_specs=["fullduplex:gemini_live:gemini-live-test"],
    )

    target_summary = json.loads(
        (output / summary["targets"][0]["summary"]).read_text(encoding="utf-8")
    )
    assert summary["passed"] is True
    assert target_summary["understanding_provider"] == "gemini_live"
    assert target_summary["understanding_model"] == "gemini-live-test"


def test_eval_recipe_accepts_external_live_predictions_for_any_provider(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "vendor_live.jsonl"
    rubrics = tmp_path / "rubrics.jsonl"
    output = tmp_path / "eval"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    rubrics.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(manifest, _manifest_rows())
    _write_jsonl(
        predictions,
        [
            {"cell_id": "c1", "predicted_action": "respond"},
            {"cell_id": "c2", "predicted_action": "silent"},
        ],
    )
    _write_jsonl(rubrics, [{"cell_id": "c1", "rubric_id": "c1:o1", "layer": "O1"}])

    summary = run_eval(
        str(manifest),
        None,
        str(rubrics),
        str(output),
        target_specs=[f"fullduplex:any_live_vendor:vendor-model:{predictions}"],
    )

    assert summary["passed"] is True
    assert summary["targets"][0]["provider"] == "any_live_vendor"


def test_eval_recipe_can_include_answer_judge(tmp_path, monkeypatch, det_judge) -> None:
    from ib.scoring import judge as judge_module

    monkeypatch.setattr(judge_module, "provider_for", lambda name, *, model=None: det_judge)
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    rubrics = tmp_path / "rubrics.jsonl"
    output = tmp_path / "eval"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    rubrics.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(manifest, _manifest_rows())
    _write_jsonl(
        predictions,
        [
            {
                "cell_id": "c1",
                "predicted_action": "respond",
                "answer_text": "scene=work_professional",
            },
            {"cell_id": "c2", "predicted_action": "silent"},
        ],
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
                    }
                ],
            }
        ],
    )

    summary = run_eval(
        str(manifest),
        str(predictions),
        str(rubrics),
        str(output),
        answer_judge=True,
        judge_provider="openai",
    )

    assert summary["passed"] is True
    assert summary["answer_judge_enabled"] is True
    assert (output / summary["reports"]["answer_judge"]).exists()


def test_manifest_eval_excludes_base_actions_from_sibling_consistency(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(manifest, _manifest_rows())
    _write_jsonl(
        predictions,
        [
            {"cell_id": "c1", "predicted_action": "silent"},
            {"cell_id": "c2", "predicted_action": "respond"},
        ],
    )

    result = manifest_eval(manifest, predictions)

    sibling = result["counterfactual_sibling_consistency"]
    assert sibling["evaluated_group_count"] == 0
    assert sibling["consistent_group_count"] == 0
    assert sibling["consistency_rate"] == 1.0


def test_manifest_eval_reports_logic_tool_call_match(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    rows = _manifest_rows()
    rows[0]["logic_expected_tool_calls"] = [
        {"name": "set_navigation", "arguments": {"route_name": "Richmond"}}
    ]
    rows[1]["logic_expected_tool_calls"] = []
    _write_jsonl(manifest, rows)
    _write_jsonl(
        predictions,
        [
            {
                "cell_id": "c1",
                "predicted_action": "respond",
                "tool_calls": [{"name": "set_navigation", "arguments": {"route_name": "Richmond"}}],
            },
            {"cell_id": "c2", "predicted_action": "silent"},
        ],
    )

    result = manifest_eval(manifest, predictions)

    summary = result["logic_tool_call_summary"]
    assert summary["total"] == 2
    assert summary["tool_call_score"] == 1.0
    assert summary["tool_call_passed"] == 2
    assert summary["tool_call_pass_rate"] == 1.0
    assert summary["name_precision"] == 1.0
    assert summary["name_recall"] == 1.0
    assert summary["name_f1"] == 1.0
    assert summary["argument_score"] == 1.0
    assert summary["name_any_overlap_matched"] == 1
    assert summary["name_multiset_matched"] == 2
    assert summary["cell_scores"] == {"c1": 1.0, "c2": 1.0}
    assert "matched" not in summary
    assert "match_rate" not in summary
    assert "exact_match" not in summary["details"]["c1"]
    assert summary["details"]["c1"]["matched_pairs"] == [
        {
            "expected_index": 0,
            "predicted_index": 0,
            "name": "set_navigation",
            "argument_score": 1.0,
        }
    ]
    assert summary["details"]["c2"]["expected_tool_count"] == 0
    assert summary["details"]["c2"]["predicted_tool_count"] == 0


def test_manifest_eval_reports_hear_time_probe_slice(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    rows = _manifest_rows()[:1]
    rows[0].update(
        {
            "logic_case_mode": "hear_time_probe",
            "expected_action": "ignore",
            "logic_expected_tool_calls": [],
        }
    )
    _write_jsonl(manifest, rows)
    _write_jsonl(
        predictions,
        [
            {
                "cell_id": "c1",
                "predicted_action": "silent",
                "tool_calls": [{"name": "set_navigation", "arguments": {"destination": "home"}}],
            }
        ],
    )

    summary = manifest_eval(manifest, predictions)["probe_summary"]["hear_time"]

    assert summary["total"] == 1
    assert summary["correct_ignore"] == 0
    assert summary["details"]["c1"]["action_correct"] is True
    assert summary["details"]["c1"]["predicted_action_class"] == "silent"
    assert summary["unexpected_tool_call"] == 1
    assert summary["details"]["c1"]["passed"] is False


def test_manifest_eval_maps_internal_ignore_to_silent_prediction(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    rows = _manifest_rows()[:1]
    rows[0].update({"logic_case_mode": "hear_time_probe", "expected_action": "ignore"})
    _write_jsonl(manifest, rows)
    _write_jsonl(predictions, [{"cell_id": "c1", "predicted_action": "silent"}])

    result = manifest_eval(manifest, predictions)

    assert result["action_total"] == 1
    assert result["correct"] == 1
    assert result["accuracy"] == 1.0
    assert result["probe_summary"]["hear_time"]["details"]["c1"]["action_correct"] is True


def test_manifest_eval_skips_speak_time_for_turn_based_mode(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    rows = _manifest_rows()[:1]
    rows[0].update(
        {
            "logic_live_expansion_type": "bystander_interference_suppression",
            "logic_live_expansion_types": ["bystander_interference_suppression"],
            "logic_expected_tool_calls": [],
        }
    )
    _write_jsonl(manifest, rows)
    _write_jsonl(predictions, [{"cell_id": "c1", "predicted_action": "respond"}])

    summary = manifest_eval(manifest, predictions)["probe_summary"]["speak_time"][
        "bystander_interference_suppression"
    ]

    assert summary["candidate_count"] == 1
    assert summary["applicable"] == 0
    assert summary["not_applicable"] == 1
    assert summary["not_applicable_reason"] == "eval_mode_not_fullduplex"


def _speak_probe_manifest_row() -> dict:
    row = copy.deepcopy(_manifest_rows()[0])
    row.update(
        {
            "cell_id": "c1s",
            "logic_case_mode": "speak_time_probe",
            "expected_action": "respond",
            "logic_expected_tool_calls": [
                {"name": "set_navigation", "arguments": {"destination": "恒隆"}}
            ],
        }
    )
    return row


def test_manifest_eval_scores_barge_in_resume_for_turn_based_approx(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(manifest, [_speak_probe_manifest_row()])
    _write_jsonl(
        predictions,
        [
            {
                "cell_id": "c1s",
                "predicted_action": "respond",
                "answer_text": "继续导航去恒隆。",
                "tool_calls": [{"name": "set_navigation", "arguments": {"destination": "恒隆"}}],
            }
        ],
    )

    summary = manifest_eval(manifest, predictions, speak_probe_approx=True)["probe_summary"][
        "speak_time"
    ]["bystander_interference_suppression"]

    assert summary["probe_variant"] == "barge_in_resume"
    assert summary["applicable"] == 1
    assert summary["executed"] == 1
    assert summary["passed"] == 1
    assert summary["pass_rate"] == 1.0
    detail = summary["details"]["c1s"]
    assert detail["probe_variant"] == "barge_in_resume"
    assert detail["action_preserved"] is True
    assert detail["base_tool_calls_preserved"] is True
    assert detail["followed_bystander"] is False
    assert detail["interrupted"] is None
    assert detail["passed"] is True


def test_barge_in_resume_fails_when_model_follows_bystander(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(manifest, [_speak_probe_manifest_row()])
    _write_jsonl(
        predictions,
        [
            {
                "cell_id": "c1s",
                "predicted_action": "respond",
                "tool_calls": [{"name": "set_temperature", "arguments": {"value": 18}}],
            }
        ],
    )

    summary = manifest_eval(manifest, predictions, speak_probe_approx=True)["probe_summary"][
        "speak_time"
    ]["bystander_interference_suppression"]

    detail = summary["details"]["c1s"]
    assert detail["followed_bystander"] is True
    assert detail["base_tool_calls_preserved"] is False
    assert detail["passed"] is False
    assert summary["passed"] == 0


def test_barge_in_resume_fails_when_model_abandons_task(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(manifest, [_speak_probe_manifest_row()])
    _write_jsonl(predictions, [{"cell_id": "c1s", "predicted_action": "silent"}])

    summary = manifest_eval(manifest, predictions, speak_probe_approx=True)["probe_summary"][
        "speak_time"
    ]["bystander_interference_suppression"]

    detail = summary["details"]["c1s"]
    assert detail["action_preserved"] is False
    assert detail["followed_bystander"] is False
    assert detail["passed"] is False


def test_case_mode_filter_drops_or_transforms_speak_rows(tmp_path) -> None:
    from ib.cli.eval import _case_mode_filtered_rows

    manifest = tmp_path / "manifest.jsonl"
    probe = _speak_probe_manifest_row()
    probe.update(
        {
            "logic_standard_answer": "Sure, I am setting navigation to the Hang Lung tower now.",
            "logic_live_event_audio_paths": ["audio/live_event.wav"],
            "audio_paths": ["audio/turn1.wav"],
            "user_turn_1_audio": "audio/turn1.wav",
        }
    )
    _write_jsonl(manifest, [_manifest_rows()[0], probe])

    dropped = _case_mode_filtered_rows(str(manifest), None, "turn_based")
    assert [row["cell_id"] for row in dropped] == ["c1"]

    kept = _case_mode_filtered_rows(
        str(manifest), None, "turn_based", speak_probe_approx=True
    )
    assert [row["cell_id"] for row in kept] == ["c1", "c1s"]
    transformed = kept[1]
    assert transformed["logic_speak_probe_render"]["variant"] == "barge_in_resume"
    assert transformed["assistant_turn_1_transcript"].endswith(" <STOPPED>")
    assert transformed["user_turn_2_audio"] == "audio/live_event.wav"
    assert transformed["audio_paths"][-1].endswith("audio/live_event.wav")

    assert _case_mode_filtered_rows(str(manifest), None, "fullduplex") is None


def test_barge_in_resume_counts_missing_prediction_as_not_executed(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(manifest, [_speak_probe_manifest_row()])
    _write_jsonl(predictions, [])

    summary = manifest_eval(manifest, predictions, speak_probe_approx=True)["probe_summary"][
        "speak_time"
    ]["bystander_interference_suppression"]

    assert summary["executed"] == 0
    assert summary["probe_not_executed"] == 1
    assert summary["passed"] == 0


def test_manifest_eval_scores_speak_time_for_fullduplex_mode(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    rows = _manifest_rows()[:1]
    rows[0].update(
        {
            "logic_live_expansion_type": "bystander_interference_suppression",
            "logic_live_expansion_types": ["bystander_interference_suppression"],
            "logic_expected_tool_calls": [
                {"name": "set_navigation", "arguments": {"destination": "恒隆"}}
            ],
        }
    )
    _write_jsonl(manifest, rows)
    _write_jsonl(
        predictions,
        [
            {
                "cell_id": "c1",
                "predicted_action": "respond",
                "tool_calls": [{"name": "set_navigation", "arguments": {"destination": "恒隆"}}],
                "metadata": {
                    "eval_mode": "fullduplex",
                    "live_probe": {"executed": True, "predicted_action": "respond"},
                },
            }
        ],
    )

    summary = manifest_eval(manifest, predictions, eval_mode="fullduplex")["probe_summary"][
        "speak_time"
    ]["bystander_interference_suppression"]

    assert summary["candidate_count"] == 1
    assert summary["applicable"] == 1
    assert summary["executed"] == 1
    assert summary["passed"] == 1
    assert summary["pass_rate"] == 1.0
    assert summary["details"]["c1"]["predicted_action_class"] == "respond"


def test_manifest_eval_scores_chinese_tool_arguments_by_unicode_text(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    rows = _manifest_rows()[:1]
    rows[0]["logic_expected_tool_calls"] = [
        {"name": "set_navigation", "arguments": {"destination": "恒隆"}}
    ]
    _write_jsonl(manifest, rows)
    _write_jsonl(
        predictions,
        [
            {
                "cell_id": "c1",
                "predicted_action": "respond",
                "tool_calls": [{"name": "set_navigation", "arguments": {"destination": "回家"}}],
            }
        ],
    )

    summary = manifest_eval(manifest, predictions)["logic_tool_call_summary"]

    assert summary["total"] == 1
    assert summary["argument_score"] < 0.5
    assert summary["tool_call_score"] < 0.8
    assert summary["tool_call_passed"] == 0


def test_manifest_eval_penalizes_extra_tool_calls_when_none_expected(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    rows = _manifest_rows()[:1]
    rows[0]["logic_expected_tool_calls"] = []
    _write_jsonl(manifest, rows)
    _write_jsonl(
        predictions,
        [
            {
                "cell_id": "c1",
                "predicted_action": "respond",
                "tool_calls": [
                    {"name": "set_navigation", "arguments": {"destination": "不该执行"}}
                ],
            }
        ],
    )

    summary = manifest_eval(manifest, predictions)["logic_tool_call_summary"]

    assert summary["total"] == 1
    assert summary["tool_call_score"] == 0.0
    assert summary["argument_score"] == 0.0
    assert summary["tool_call_passed"] == 0
    assert summary["details"]["c1"]["expected_tool_count"] == 0
    assert summary["details"]["c1"]["predicted_tool_count"] == 1


def test_manifest_eval_penalizes_extra_tool_calls_with_matching_required_call(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    rows = _manifest_rows()[:1]
    rows[0]["logic_expected_tool_calls"] = [
        {"name": "set_navigation", "arguments": {"destination": "恒隆"}}
    ]
    _write_jsonl(manifest, rows)
    _write_jsonl(
        predictions,
        [
            {
                "cell_id": "c1",
                "predicted_action": "respond",
                "tool_calls": [
                    {"name": "set_navigation", "arguments": {"destination": "恒隆"}},
                    {"name": "play_media", "arguments": {"source": "radio"}},
                ],
            }
        ],
    )

    summary = manifest_eval(manifest, predictions)["logic_tool_call_summary"]

    assert summary["name_multiset_match_rate"] == 0.0
    assert summary["argument_score"] == 0.5
    assert summary["tool_call_score"] < 0.8
    assert summary["tool_call_passed"] == 0


def test_manifest_eval_scores_tool_calls_without_raw_exact_match(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    rows = _manifest_rows()
    rows[0]["logic_expected_tool_calls"] = [
        {"name": "set_navigation", "arguments": {"route": "Richmond"}}
    ]
    _write_jsonl(manifest, rows[:1])
    _write_jsonl(
        predictions,
        [
            {
                "cell_id": "c1",
                "predicted_action": "respond",
                "tool_calls": [
                    {"name": "set_navigation", "arguments": {"route": "richmond route"}}
                ],
            }
        ],
    )

    summary = manifest_eval(manifest, predictions)["logic_tool_call_summary"]

    assert "matched" not in summary
    assert "match_rate" not in summary
    assert summary["name_any_overlap_matched"] == 1
    assert summary["name_multiset_matched"] == 1
    assert summary["tool_call_score"] > 0.9
    assert summary["argument_score"] > 0.9
    assert summary["details"]["c1"]["name_multiset_match"] is True


def test_manifest_eval_tool_call_metric_ignores_order_and_penalizes_missing_args(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    rows = _manifest_rows()
    rows[0]["logic_expected_tool_calls"] = [
        {"name": "send_message", "arguments": {"recipient": "Maya", "message": "call first"}},
        {"name": "update_calendar", "arguments": {"time_window": "Friday 10 to noon"}},
    ]
    _write_jsonl(manifest, rows[:1])
    _write_jsonl(
        predictions,
        [
            {
                "cell_id": "c1",
                "predicted_action": "respond",
                "tool_calls": [
                    {"name": "update_calendar", "arguments": {"time_window": "Friday 10 to noon"}},
                    {"name": "send_message", "arguments": {"recipient": "Maya"}},
                ],
            }
        ],
    )

    summary = manifest_eval(manifest, predictions)["logic_tool_call_summary"]

    assert summary["name_multiset_match_rate"] == 1.0
    assert summary["name_f1"] == 1.0
    assert 0.7 < summary["argument_score"] < 1.0
    assert 0.8 < summary["tool_call_score"] < 1.0
    assert summary["details"]["c1"]["matched_pairs"] == [
        {
            "expected_index": 0,
            "predicted_index": 1,
            "name": "send_message",
            "argument_score": 0.5,
        },
        {
            "expected_index": 1,
            "predicted_index": 0,
            "name": "update_calendar",
            "argument_score": 1.0,
        },
    ]


def test_run_manifest_eval_fullduplex_without_live_probe_skips_speak_time(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "manifest_eval.json"
    _write_jsonl(
        manifest,
        [
            {
                "cell_id": "c1",
                "expected_action": "respond",
                "labels": copy.deepcopy(_manifest_rows()[0]["labels"]),
                "logic_live_expansion_type": "bystander_interference_suppression",
            }
        ],
    )
    _write_jsonl(
        predictions,
        [
            {
                "cell_id": "c1",
                "predicted_action": "respond",
                "answer_text": "ok",
                "tool_calls": [],
            }
        ],
    )

    result = run_manifest_eval(str(manifest), str(predictions), str(output), mode="fullduplex")

    speak_time = result["probe_summary"]["speak_time"]["bystander_interference_suppression"]
    assert result["requested_eval_mode"] == "fullduplex"
    assert result["eval_mode"] == "turn_based"
    assert result["live_probe_enabled"] is False
    assert speak_time["applicable"] == 0
    assert speak_time["not_applicable_reason"] == "eval_mode_not_fullduplex"


def test_format_error_predictions_earn_no_silent_credit() -> None:
    from ib.predictions import PredictionRow
    from ib.scoring.eval import _hear_time_summary, _probe_action_summary

    row = {
        "cell_id": "c1h",
        "logic_case_mode": "hear_time_probe",
        "expected_action": "silent",
    }
    degraded = PredictionRow(
        cell_id="c1h",
        predicted_action="silent",
        answer_text=None,
        tool_calls=[],
        confidence=0.0,
        metadata={"provider_error": {"type": "prediction_format_error", "message": "bad json"}},
    )
    clean = PredictionRow(
        cell_id="c1h",
        predicted_action="silent",
        answer_text=None,
        tool_calls=[],
        confidence=0.9,
        metadata={},
    )

    degraded_hear = _hear_time_summary([row], {"c1h": degraded})["details"]["c1h"]
    assert degraded_hear["format_error"] is True
    assert degraded_hear["action_correct"] is False
    assert degraded_hear["passed"] is False

    clean_hear = _hear_time_summary([row], {"c1h": clean})["details"]["c1h"]
    assert clean_hear["format_error"] is False
    assert clean_hear["action_correct"] is True
    assert clean_hear["passed"] is True

    degraded_action = _probe_action_summary([row], {"c1h": degraded}, eval_mode="turn_based")
    assert degraded_action["correct"] == 0
    assert degraded_action["details"]["c1h"]["format_error"] is True

    clean_action = _probe_action_summary([row], {"c1h": clean}, eval_mode="turn_based")
    assert clean_action["correct"] == 1
