"""S8.A2 — manifest/rubric evaluation metrics consume planner artifacts."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ib.cli import main as cli_main
from ib.evaluation.artifacts import default_eval_output_dir
from ib.evaluation.reports import run_manifest_eval, run_rubric_eval
from ib.evaluation.run import run_eval
from ib.scoring.eval_targets import EvalTarget
from ib.scoring.eval import manifest_eval, rubric_eval

REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PREDICTIONS = [
    {"cell_id": "c1", "predicted_action": "respond"},
    {"cell_id": "c2", "predicted_action": "silent"},
]
_DEFAULT_RUBRICS = [{"cell_id": "c1", "rubric_id": "c1:o1", "layer": "O1"}]


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


def _artifacts(
    tmp_path, manifest_rows=None, predictions=_DEFAULT_PREDICTIONS, rubrics=_DEFAULT_RUBRICS
) -> tuple[Path, Path, Path]:
    """Write manifest/predictions/rubrics JSONL under tmp_path; return their paths."""
    manifest = tmp_path / "manifest.jsonl"
    prediction_path = tmp_path / "predictions.jsonl"
    rubric_path = tmp_path / "rubrics.jsonl"
    _write_jsonl(manifest, _manifest_rows() if manifest_rows is None else manifest_rows)
    _write_jsonl(prediction_path, predictions)
    _write_jsonl(rubric_path, rubrics)
    return manifest, prediction_path, rubric_path


def _live_expansion_row() -> dict:
    row = _manifest_rows()[0]
    row.update(
        {
            "logic_live_expansion_type": "bystander_interference_suppression",
            "logic_live_expansion_types": ["bystander_interference_suppression"],
        }
    )
    return row


def test_generated_eval_target_requires_explicit_model() -> None:
    with pytest.raises(ValueError, match="model is required"):
        EvalTarget(provider="openai")


@pytest.mark.parametrize(
    "predictions",
    [
        [
            {"cell_id": "c1", "predicted_action": "respond"},
            {"cell_id": "c2", "predicted_action": "silent"},
        ],
        [
            {"cell_id": "c1", "predicted_action": "silent"},
            {"cell_id": "c2", "predicted_action": "respond"},
        ],
    ],
)
def test_manifest_eval_excludes_base_actions_from_accuracy_and_sibling_consistency(
    tmp_path, predictions
) -> None:
    manifest, prediction_path, _ = _artifacts(tmp_path, predictions=predictions)

    result = manifest_eval(manifest, prediction_path)

    sibling = result["counterfactual_sibling_consistency"]
    assert result["action_total"] == 0
    assert result["accuracy"] is None
    assert result["interaction_summary"] == {}
    assert sibling["evaluated_group_count"] == 0
    assert sibling["consistent_group_count"] == 0
    assert sibling["consistency_rate"] == 1.0


def test_manifest_eval_reports_missing_predictions(tmp_path) -> None:
    manifest, predictions, _ = _artifacts(
        tmp_path, predictions=[{"cell_id": "c1", "predicted_action": "respond"}]
    )

    result = manifest_eval(manifest, predictions)

    assert result["missing_prediction_cell_ids"] == ["c2"]
    assert result["errors"]


def test_rubric_eval_validates_references_and_answerable_coverage(tmp_path) -> None:
    manifest, _, rubrics = _artifacts(tmp_path)

    result = rubric_eval(manifest, rubrics)

    assert result["missing_rubric_ids"] == []
    assert result["answerable_rubric_coverage"] == 1.0
    assert result["errors"] == []


def test_rubric_eval_counts_refusals_as_answerable(tmp_path) -> None:
    rows = _manifest_rows()
    rows[1]["expected_action"] = "refuse"
    rows[1]["rubric_ids"] = ["c2:o1"]
    rows[1]["labels"]["interaction"]["expected_action"] = "refuse"
    rows[1]["labels"]["answer"] = {
        "answerable": True,
        "rubric_ids": ["c2:o1"],
        "contract_text": None,
    }
    manifest, _, rubrics = _artifacts(
        tmp_path,
        rows,
        rubrics=[*_DEFAULT_RUBRICS, {"cell_id": "c2", "rubric_id": "c2:o1", "layer": "O1"}],
    )

    result = rubric_eval(manifest, rubrics)

    assert result["answerable_row_count"] == 2
    assert result["answerable_rubric_coverage"] == 1.0
    assert result["errors"] == []


def test_rubric_eval_reports_missing_rubrics(tmp_path) -> None:
    manifest, _, rubrics = _artifacts(tmp_path, rubrics=[])

    result = rubric_eval(manifest, rubrics)

    assert result["missing_rubric_ids"] == ["c1:o1"]
    assert result["errors"] == ["unknown rubric_id references in manifest: ['c1:o1']"]


def test_eval_cli_wrappers_write_reports(tmp_path) -> None:
    manifest, predictions, rubrics = _artifacts(tmp_path)
    manifest_out = tmp_path / "manifest-eval.json"
    rubric_out = tmp_path / "rubric-eval.json"

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
    manifest, predictions, rubrics = _artifacts(tmp_path)
    output = tmp_path / "eval"

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
    eval_root = tmp_path / "eval-root"
    monkeypatch.chdir(REPO_ROOT)
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


def test_eval_recipe_cli_defaults_output_under_eval_artifacts(tmp_path, monkeypatch) -> None:
    manifest, predictions, rubrics = _artifacts(tmp_path)
    eval_root = tmp_path / "artifacts" / "eval"
    monkeypatch.setenv("IB_EVAL_ROOT", str(eval_root))

    code = cli_main.main(
        [
            "eval",
            "--manifest",
            str(manifest),
            "--predictions",
            str(predictions),
            "--rubrics",
            str(rubrics),
        ]
    )

    candidates = sorted(eval_root.glob("**/summary.json"))
    assert code == 0
    assert len(candidates) == 1
    summary = json.loads(candidates[0].read_text(encoding="utf-8"))
    output = Path(summary["output"])
    assert output.is_relative_to(eval_root)
    assert (output / summary["reports"]["manifest_eval"]).exists()
    assert (output / summary["reports"]["rubric_eval"]).exists()


def test_run_eval_infers_rubrics_for_canonical_scene_manifest(tmp_path) -> None:
    scene_root = tmp_path / "run" / "commerce_service"
    manifest = scene_root / "manifest" / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    rubrics = scene_root / "plans" / "rubrics.jsonl"
    output = tmp_path / "eval"
    manifest.parent.mkdir(parents=True)
    rubrics.parent.mkdir(parents=True)
    _write_jsonl(manifest, _manifest_rows())
    _write_jsonl(predictions, _DEFAULT_PREDICTIONS)
    _write_jsonl(rubrics, _DEFAULT_RUBRICS)

    summary = run_eval(str(manifest), str(predictions), None, str(output))

    assert summary["rubrics"] == str(rubrics)
    assert (output / "rubric_eval.json").exists()


def test_eval_recipe_limit_applies_with_existing_predictions(tmp_path) -> None:
    manifest, predictions, rubrics = _artifacts(
        tmp_path, predictions=[{"cell_id": "c1", "predicted_action": "respond"}]
    )
    output = tmp_path / "eval"

    summary = run_eval(str(manifest), str(predictions), str(rubrics), str(output), limit=1)

    manifest_report = json.loads((output / "manifest_eval.json").read_text(encoding="utf-8"))
    assert summary["eval_manifest"].endswith("manifest_subset.jsonl")
    assert manifest_report["total"] == 1
    assert manifest_report["missing_prediction_cell_ids"] == []


def test_eval_recipe_runs_mixed_mode_targets(tmp_path) -> None:
    manifest, predictions, rubrics = _artifacts(tmp_path)
    output = tmp_path / "eval"

    summary = run_eval(
        str(manifest),
        None,
        str(rubrics),
        str(output),
        target_specs=[
            f"turn_based:openai:gpt-audio-1.5:{predictions}",
            f"fullduplex:openai_realtime:gpt-realtime:{predictions}",
        ],
    )

    assert summary["schema_version"] == "ib.eval_batch.v1"
    assert summary["target_count"] == 2
    assert {target["mode"] for target in summary["targets"]} == {"turn_based", "fullduplex"}
    for target in summary["targets"]:
        assert (output / target["summary"]).exists()


def test_eval_recipe_generates_gemini_live_fullduplex_target(tmp_path, monkeypatch) -> None:
    from ib.evaluation import understanding as eval_understanding

    manifest, _, rubrics = _artifacts(tmp_path)
    output = tmp_path / "eval"

    def fake_gemini_live(manifest_path, predictions_path, *, model, limit=None, live_probe=False, thinking_mode=None):
        del manifest_path, model, limit, live_probe, thinking_mode
        _write_jsonl(Path(predictions_path), _DEFAULT_PREDICTIONS)
        return {
            "schema_version": "ib.gemini_fullduplex.v1",
            "provider": "gemini_live",
            "model": "gemini-live-test",
            "predictions": str(predictions_path),
        }

    monkeypatch.setattr(eval_understanding, "run_gemini_fullduplex_predictions", fake_gemini_live)

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
    manifest, predictions, rubrics = _artifacts(tmp_path)

    summary = run_eval(
        str(manifest),
        None,
        str(rubrics),
        str(tmp_path / "eval"),
        target_specs=[f"fullduplex:any_live_vendor:vendor-model:{predictions}"],
    )

    assert summary["passed"] is True
    assert summary["targets"][0]["provider"] == "any_live_vendor"


def test_manifest_eval_reports_logic_tool_call_match(tmp_path) -> None:
    rows = _manifest_rows()
    rows[0]["logic_expected_tool_calls"] = [
        {"name": "set_navigation", "arguments": {"route_name": "Richmond"}}
    ]
    rows[1]["logic_expected_tool_calls"] = []
    manifest, predictions, _ = _artifacts(
        tmp_path,
        rows,
        predictions=[
            {
                "cell_id": "c1",
                "predicted_action": "respond",
                "tool_calls": [{"name": "set_navigation", "arguments": {"route_name": "Richmond"}}],
            },
            {"cell_id": "c2", "predicted_action": "silent"},
        ],
    )

    summary = manifest_eval(manifest, predictions)["logic_tool_call_summary"]

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
    row = _manifest_rows()[0]
    row.update(
        {
            "logic_case_mode": "hear_time_probe",
            "expected_action": "ignore",
            "logic_expected_tool_calls": [],
        }
    )
    manifest, predictions, _ = _artifacts(
        tmp_path,
        [row],
        predictions=[
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
    row = _manifest_rows()[0]
    row.update({"logic_case_mode": "hear_time_probe", "expected_action": "ignore"})
    manifest, predictions, _ = _artifacts(
        tmp_path, [row], predictions=[{"cell_id": "c1", "predicted_action": "silent"}]
    )

    result = manifest_eval(manifest, predictions)

    assert result["action_total"] == 1
    assert result["correct"] == 1
    assert result["accuracy"] == 1.0
    assert result["probe_summary"]["hear_time"]["details"]["c1"]["action_correct"] is True


def test_manifest_eval_skips_speak_time_for_turn_based_mode(tmp_path) -> None:
    row = _live_expansion_row()
    row["logic_expected_tool_calls"] = []
    manifest, predictions, _ = _artifacts(
        tmp_path, [row], predictions=[{"cell_id": "c1", "predicted_action": "respond"}]
    )

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


def _barge_in_summary(tmp_path, predictions: list[dict]) -> dict:
    manifest, prediction_path, _ = _artifacts(
        tmp_path, [_speak_probe_manifest_row()], predictions=predictions
    )
    return manifest_eval(manifest, prediction_path, speak_probe_approx=True)["probe_summary"][
        "speak_time"
    ]["bystander_interference_suppression"]


@pytest.mark.parametrize(
    ("prediction", "expected_detail", "expected_summary"),
    [
        pytest.param(
            {
                "predicted_action": "respond",
                "answer_text": "继续导航去恒隆。",
                "tool_calls": [{"name": "set_navigation", "arguments": {"destination": "恒隆"}}],
            },
            {
                "probe_variant": "barge_in_resume",
                "action_preserved": True,
                "base_tool_calls_preserved": True,
                "followed_bystander": False,
                "interrupted": None,
                "passed": True,
            },
            {
                "probe_variant": "barge_in_resume",
                "applicable": 1,
                "executed": 1,
                "passed": 1,
                "pass_rate": 1.0,
            },
            id="resumes_task",
        ),
        pytest.param(
            {
                "predicted_action": "respond",
                "tool_calls": [{"name": "set_temperature", "arguments": {"value": 18}}],
            },
            {"followed_bystander": True, "base_tool_calls_preserved": False, "passed": False},
            {"passed": 0},
            id="follows_bystander",
        ),
        pytest.param(
            {"predicted_action": "silent"},
            {"action_preserved": False, "followed_bystander": False, "passed": False},
            {},
            id="abandons_task",
        ),
    ],
)
def test_barge_in_resume_scoring_for_turn_based_approx(
    tmp_path, prediction, expected_detail, expected_summary
) -> None:
    summary = _barge_in_summary(tmp_path, [{"cell_id": "c1s", **prediction}])

    detail = summary["details"]["c1s"]
    assert {key: detail[key] for key in expected_detail} == expected_detail
    assert {key: summary[key] for key in expected_summary} == expected_summary


def test_barge_in_resume_counts_missing_prediction_as_not_executed(tmp_path) -> None:
    summary = _barge_in_summary(tmp_path, [])

    assert summary["executed"] == 0
    assert summary["probe_not_executed"] == 1
    assert summary["passed"] == 0


def test_case_mode_filter_drops_or_transforms_speak_rows(tmp_path) -> None:
    from ib.evaluation.target import _case_mode_filtered_rows

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


def test_manifest_eval_scores_speak_time_for_fullduplex_mode(tmp_path) -> None:
    row = _live_expansion_row()
    row["logic_expected_tool_calls"] = [
        {"name": "set_navigation", "arguments": {"destination": "恒隆"}}
    ]
    manifest, predictions, _ = _artifacts(
        tmp_path,
        [row],
        predictions=[
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


def _tool_call_summary(tmp_path, expected_calls: list[dict], predicted_calls: list[dict]) -> dict:
    row = _manifest_rows()[0]
    row["logic_expected_tool_calls"] = expected_calls
    manifest, predictions, _ = _artifacts(
        tmp_path,
        [row],
        predictions=[{"cell_id": "c1", "predicted_action": "respond", "tool_calls": predicted_calls}],
    )
    return manifest_eval(manifest, predictions)["logic_tool_call_summary"]


def test_manifest_eval_scores_chinese_tool_arguments_by_unicode_text(tmp_path) -> None:
    summary = _tool_call_summary(
        tmp_path,
        [{"name": "set_navigation", "arguments": {"destination": "恒隆"}}],
        [{"name": "set_navigation", "arguments": {"destination": "回家"}}],
    )

    assert summary["total"] == 1
    assert summary["argument_score"] < 0.5
    assert summary["tool_call_score"] < 0.8
    assert summary["tool_call_passed"] == 0


def test_manifest_eval_penalizes_extra_tool_calls_when_none_expected(tmp_path) -> None:
    summary = _tool_call_summary(
        tmp_path, [], [{"name": "set_navigation", "arguments": {"destination": "不该执行"}}]
    )

    assert summary["total"] == 1
    assert summary["tool_call_score"] == 0.0
    assert summary["argument_score"] == 0.0
    assert summary["tool_call_passed"] == 0
    assert summary["details"]["c1"]["expected_tool_count"] == 0
    assert summary["details"]["c1"]["predicted_tool_count"] == 1


def test_manifest_eval_penalizes_extra_tool_calls_with_matching_required_call(tmp_path) -> None:
    summary = _tool_call_summary(
        tmp_path,
        [{"name": "set_navigation", "arguments": {"destination": "恒隆"}}],
        [
            {"name": "set_navigation", "arguments": {"destination": "恒隆"}},
            {"name": "play_media", "arguments": {"source": "radio"}},
        ],
    )

    assert summary["name_multiset_match_rate"] == 0.0
    assert summary["argument_score"] == 0.5
    assert summary["tool_call_score"] < 0.8
    assert summary["tool_call_passed"] == 0


def test_manifest_eval_scores_fuzzy_argument_text_close_to_expected(tmp_path) -> None:
    summary = _tool_call_summary(
        tmp_path,
        [{"name": "set_navigation", "arguments": {"route": "Richmond"}}],
        [{"name": "set_navigation", "arguments": {"route": "richmond route"}}],
    )

    assert summary["name_any_overlap_matched"] == 1
    assert summary["name_multiset_matched"] == 1
    assert summary["tool_call_score"] > 0.9
    assert summary["argument_score"] > 0.9
    assert summary["details"]["c1"]["name_multiset_match"] is True


def test_manifest_eval_tool_call_metric_ignores_order_and_penalizes_missing_args(tmp_path) -> None:
    summary = _tool_call_summary(
        tmp_path,
        [
            {"name": "send_message", "arguments": {"recipient": "Maya", "message": "call first"}},
            {"name": "update_calendar", "arguments": {"time_window": "Friday 10 to noon"}},
        ],
        [
            {"name": "update_calendar", "arguments": {"time_window": "Friday 10 to noon"}},
            {"name": "send_message", "arguments": {"recipient": "Maya"}},
        ],
    )

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
    row = _live_expansion_row()
    manifest, predictions, _ = _artifacts(
        tmp_path,
        [row],
        predictions=[
            {"cell_id": "c1", "predicted_action": "respond", "answer_text": "ok", "tool_calls": []}
        ],
    )

    result = run_manifest_eval(
        str(manifest), str(predictions), str(tmp_path / "manifest_eval.json"), mode="fullduplex"
    )

    speak_time = result["probe_summary"]["speak_time"]["bystander_interference_suppression"]
    assert result["requested_eval_mode"] == "fullduplex"
    assert result["eval_mode"] == "turn_based"
    assert result["live_probe_enabled"] is False
    assert speak_time["applicable"] == 0
    assert speak_time["not_applicable_reason"] == "eval_mode_not_fullduplex"


def test_format_error_predictions_earn_no_silent_credit() -> None:
    from ib.scoring.predictions import PredictionRow
    from ib.scoring.eval import _hear_time_summary, _probe_action_summary

    row = {
        "cell_id": "c1h",
        "logic_case_mode": "hear_time_probe",
        "expected_action": "silent",
    }
    degraded = PredictionRow(
        cell_id="c1h",
        predicted_action="silent",
        confidence=0.0,
        metadata={"provider_error": {"type": "prediction_format_error", "message": "bad json"}},
    )
    clean = PredictionRow(cell_id="c1h", predicted_action="silent", confidence=0.9)

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
