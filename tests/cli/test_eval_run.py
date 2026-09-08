"""CLI eval run convenience coverage."""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading

from ib.cli import main as cli_main
from ib.cli import eval as eval_module
from ib.scoring.judge import DEFAULT_OPENAI_JUDGE_MODEL


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def _write_models_yaml(tmp_path: Path, content: str, *, encoding: str = "utf-8") -> None:
    path = tmp_path / "configs" / "eval-models.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding=encoding)


def _write_scene(run_dir: Path, scene: str) -> None:
    cell_id = f"{scene}-c1"
    rubric_id = f"{cell_id}:o1"
    manifest_row = {
        "cell_id": cell_id,
        "expected_action": "respond",
        "rubric_ids": [rubric_id],
        "labels": {
            "perception": {"scene": scene},
            "interaction": {"expected_action": "respond"},
            "answer": {"answerable": True, "rubric_ids": [rubric_id], "contract_text": None},
        },
    }
    _write_jsonl(run_dir / scene / "manifest" / "manifest.jsonl", [manifest_row])
    _write_jsonl(
        run_dir / scene / "plans" / "rubrics.jsonl",
        [{"cell_id": cell_id, "rubric_id": rubric_id, "layer": "O1"}],
    )


def test_eval_run_uses_env_file_and_default_models_yaml(tmp_path, monkeypatch, capsys) -> None:
    run_dir = tmp_path / "V0_test"
    output = tmp_path / "eval-out"
    for scene in ("commerce_service", "domestic_household"):
        _write_scene(run_dir, scene)
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=from-env-openai\nGEMINI_API_KEY=from-env-gemini\n", encoding="utf-8"
    )
    _write_models_yaml(
        tmp_path,
        """
base: []
live:
  - name: gpt
    base:
      provider: openai
      model: gpt-audio-1.5
    live:
      provider: openai_realtime
      model: gpt-realtime-1.5
  - name: gemini
    base:
      provider: gemini
      model: gemini-3.5-flash
    live:
      provider: gemini_live
      model: gemini-3.1-flash-live-preview
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    def fake_predictions(provider, manifest, predictions, model, limit, **kwargs):
        assert os.environ["OPENAI_API_KEY"] == "from-env-openai"
        assert os.environ["GEMINI_API_KEY"] == "from-env-gemini"
        del kwargs
        rows = [
            json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines()
        ]
        if limit is not None:
            rows = rows[:limit]
        _write_jsonl(
            Path(predictions),
            [
                {
                    "cell_id": row["cell_id"],
                    "predicted_action": row["expected_action"],
                    "answer_text": f"{provider}:{model}",
                    "tool_calls": [],
                }
                for row in rows
            ],
        )
        return {"provider": provider, "model": model, "prediction_count": len(rows)}

    monkeypatch.setattr(eval_module, "_run_understanding_predictions", fake_predictions)

    code = cli_main.main(["eval", "--run", str(run_dir), "--output", str(output), "--limit", "1"])

    assert code == 0
    progress = capsys.readouterr().err
    assert "ib eval: discovered 2 scene(s), 4 target(s)" in progress
    assert "ib eval: scene 1/2 commerce_service target 1/4" in progress
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == "ib.eval_validation_run.v1"
    assert summary["scene_count"] == 2
    assert summary["target_count"] == 4
    assert {scene["scene"] for scene in summary["scenes"]} == {
        "commerce_service",
        "domestic_household",
    }
    for scene in ("commerce_service", "domestic_household"):
        assert (
            output
            / scene
            / "generated"
            / "fullduplex"
            / "base_only"
            / "gemini_live"
            / "gemini-3.1-flash-live-preview"
            / "summary.json"
        ).exists()
        assert (
            output
            / scene
            / "generated"
            / "turn_based"
            / "native"
            / "openai"
            / "gpt-audio-1.5"
            / "summary.json"
        ).exists()


def test_eval_run_parallelizes_independent_targets(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "V0_test"
    output = tmp_path / "eval-out"
    _write_scene(run_dir, "commerce_service")
    monkeypatch.chdir(tmp_path)
    barrier = threading.Barrier(2, timeout=5)
    active = {"count": 0, "max": 0}
    lock = threading.Lock()

    def fake_predictions(provider, manifest, predictions, model, limit, **kwargs):
        del provider, limit, kwargs
        with lock:
            active["count"] += 1
            active["max"] = max(active["max"], active["count"])
        try:
            barrier.wait()
            rows = [
                json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines()
            ]
            _write_jsonl(
                Path(predictions),
                [
                    {
                        "cell_id": row["cell_id"],
                        "predicted_action": row["expected_action"],
                        "tool_calls": [],
                    }
                    for row in rows
                ],
            )
            return {"provider": "openai_realtime", "model": model, "prediction_count": len(rows)}
        finally:
            with lock:
                active["count"] -= 1

    monkeypatch.setattr(eval_module, "_run_understanding_predictions", fake_predictions)
    monkeypatch.setattr(eval_module, "_refresh_leaderboard_if_eval_artifact", lambda summary: None)

    code = cli_main.main(
        [
            "eval",
            "--run",
            str(run_dir),
            "--output",
            str(output),
            "--target",
            "fullduplex:openai_realtime:gpt-realtime-1.5",
            "--target",
            "fullduplex:openai_realtime:gpt-realtime-2.0",
        ]
    )

    assert code == 0
    assert active["max"] == 2
    summary = json.loads((output / "commerce_service" / "summary.json").read_text())
    assert [target["model"] for target in summary["targets"]] == [
        "gpt-realtime-1.5",
        "gpt-realtime-2.0",
    ]


def test_eval_run_resumes_latest_incomplete_default_output(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "V0_test"
    eval_root = tmp_path / "eval-root"
    resume_output = eval_root / "V0_test_20260101T000000Z"
    _write_scene(run_dir, "commerce_service")
    resume_output.mkdir(parents=True)
    _write_models_yaml(
        tmp_path,
        """
base:
  - provider: openai
    model: gpt-audio-1.5
live: []
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("IB_EVAL_ROOT", str(eval_root))

    def fake_predictions(provider, manifest, predictions, model, limit, **kwargs):
        del provider, model, limit, kwargs
        rows = [
            json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines()
        ]
        _write_jsonl(
            Path(predictions),
            [
                {
                    "cell_id": row["cell_id"],
                    "predicted_action": row["expected_action"],
                    "tool_calls": [],
                }
                for row in rows
            ],
        )
        return {"provider": "openai", "model": "gpt-audio-1.5", "prediction_count": len(rows)}

    monkeypatch.setattr(eval_module, "_run_understanding_predictions", fake_predictions)

    code = cli_main.main(["eval", "--run", str(run_dir)])

    assert code == 0
    assert (resume_output / "summary.json").exists()


def test_eval_refreshes_leaderboard_after_each_target(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "V0_test"
    output = tmp_path / "eval-root" / "V0_test_20260101T000000Z"
    _write_scene(run_dir, "commerce_service")
    _write_models_yaml(
        tmp_path,
        """
base:
  - provider: openai
    model: gpt-audio-1.5
  - provider: gemini
    model: gemini-3.5-flash
live: []
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    def fake_predictions(provider, manifest, predictions, model, limit, **kwargs):
        del provider, model, limit, kwargs
        rows = [
            json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines()
        ]
        _write_jsonl(
            Path(predictions),
            [
                {
                    "cell_id": row["cell_id"],
                    "predicted_action": row["expected_action"],
                    "tool_calls": [],
                }
                for row in rows
            ],
        )
        return {"provider": "fake", "model": "fake-model", "prediction_count": len(rows)}

    refresh_outputs: list[str] = []
    monkeypatch.setattr(eval_module, "_run_understanding_predictions", fake_predictions)
    monkeypatch.setattr(
        eval_module,
        "_refresh_leaderboard_if_eval_artifact",
        lambda summary: refresh_outputs.append(summary["output"]),
    )

    code = cli_main.main(["eval", "--run", str(run_dir), "--output", str(output)])

    assert code == 0
    assert (
        str(
            output
            / "commerce_service"
            / "generated"
            / "turn_based"
            / "native"
            / "openai"
            / "gpt-audio-1.5"
        )
        in refresh_outputs
    )
    assert (
        str(
            output
            / "commerce_service"
            / "generated"
            / "turn_based"
            / "native"
            / "gemini"
            / "gemini-3.5-flash"
        )
        in refresh_outputs
    )
    assert str(output) in refresh_outputs


def test_eval_auto_leaderboard_env_can_disable_refresh(monkeypatch) -> None:
    monkeypatch.setenv("IB_EVAL_AUTO_LEADERBOARD", "0")

    assert eval_module._auto_leaderboard_enabled() is False

    monkeypatch.setenv("IB_EVAL_AUTO_LEADERBOARD", "false")

    assert eval_module._auto_leaderboard_enabled() is False

    monkeypatch.setenv("IB_EVAL_AUTO_LEADERBOARD", "1")

    assert eval_module._auto_leaderboard_enabled() is True


def test_eval_run_reuses_completed_target_summary(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "V0_test"
    output = tmp_path / "eval-out"
    _write_scene(run_dir, "commerce_service")
    _write_models_yaml(
        tmp_path,
        """
base:
  - provider: openai
    model: gpt-audio-1.5
live: []
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    calls = {"count": 0}

    def fake_predictions(provider, manifest, predictions, model, limit, **kwargs):
        del provider, model, limit, kwargs
        calls["count"] += 1
        rows = [
            json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines()
        ]
        _write_jsonl(
            Path(predictions),
            [
                {
                    "cell_id": row["cell_id"],
                    "predicted_action": row["expected_action"],
                    "tool_calls": [],
                }
                for row in rows
            ],
        )
        return {"provider": "openai", "model": "gpt-audio-1.5", "prediction_count": len(rows)}

    monkeypatch.setattr(eval_module, "_run_understanding_predictions", fake_predictions)
    assert cli_main.main(["eval", "--run", str(run_dir), "--output", str(output)]) == 0
    assert calls["count"] == 1

    def fail_predictions(*args, **kwargs):
        del args, kwargs
        raise AssertionError("completed target should be reused")

    monkeypatch.setattr(eval_module, "_run_understanding_predictions", fail_predictions)
    assert cli_main.main(["eval", "--run", str(run_dir), "--output", str(output)]) == 0
    assert calls["count"] == 1


def test_eval_run_adds_answer_judge_without_regenerating_predictions(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "V0_test"
    output = tmp_path / "eval-out"
    _write_scene(run_dir, "commerce_service")
    _write_models_yaml(
        tmp_path,
        """
base:
  - provider: openai
    model: gpt-audio-1.5
live: []
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    calls = {"predictions": 0, "judge": 0}

    def fake_predictions(provider, manifest, predictions, model, limit, **kwargs):
        del provider, model, limit, kwargs
        calls["predictions"] += 1
        rows = [
            json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines()
        ]
        _write_jsonl(
            Path(predictions),
            [
                {
                    "cell_id": row["cell_id"],
                    "predicted_action": row["expected_action"],
                    "tool_calls": [],
                }
                for row in rows
            ],
        )
        return {"provider": "openai", "model": "gpt-audio-1.5", "prediction_count": len(rows)}

    monkeypatch.setattr(eval_module, "_run_understanding_predictions", fake_predictions)
    assert cli_main.main(["eval", "--run", str(run_dir), "--output", str(output)]) == 0
    assert calls["predictions"] == 1

    def fail_predictions(*args, **kwargs):
        del args, kwargs
        raise AssertionError("predictions should not be regenerated when adding judge")

    def fake_judge(
        manifest, predictions, rubrics, judge_output, provider, model, *, progress_model=None
    ):
        del manifest, predictions, rubrics, progress_model
        calls["judge"] += 1
        payload = {
            "schema_version": "ib.answer_judge.v1",
            "provider": provider,
            "judgment_count": 0,
            "passed_count": 0,
            "pass_rate": 1.0,
            "atomic_count": 0,
            "atomic_passed_count": 0,
            "atomic_pass_rate": 1.0,
            "judgments": [],
            "errors": [],
            "llm_call_count": 0,
            "llm_usage": str(Path(judge_output).with_suffix(".cost.json")),
        }
        Path(judge_output).write_text(json.dumps(payload) + "\n", encoding="utf-8")
        Path(judge_output).with_suffix(".cost.json").write_text("{}\n", encoding="utf-8")
        assert model == "gpt-5.4"
        return payload

    monkeypatch.setattr(eval_module, "_run_understanding_predictions", fail_predictions)
    monkeypatch.setattr(eval_module, "run_answer_judge", fake_judge)
    assert (
        cli_main.main(
            [
                "eval",
                "--run",
                str(run_dir),
                "--output",
                str(output),
                "--answer-judge",
                "--judge-provider",
                "openai",
                "--judge-model",
                "gpt-5.4",
            ]
        )
        == 0
    )

    target_summary = json.loads(
        (
            output
            / "commerce_service"
            / "generated"
            / "turn_based"
            / "native"
            / "openai"
            / "gpt-audio-1.5"
            / "summary.json"
        ).read_text(encoding="utf-8")
    )
    assert calls == {"predictions": 1, "judge": 1}
    assert target_summary["answer_judge_enabled"] is True
    assert target_summary["judge_provider"] == "openai"
    assert target_summary["judge_model"] == "gpt-5.4"
    assert target_summary["reports"]["answer_judge"] == "answer_judge.json"


def test_eval_run_records_default_answer_judge_model(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "V0_test"
    output = tmp_path / "eval-out"
    _write_scene(run_dir, "commerce_service")
    _write_models_yaml(tmp_path, "base: []\nlive: []\n")
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions,
        [
            {
                "cell_id": "commerce_service-c1",
                "predicted_action": "respond",
                "tool_calls": [],
            }
        ],
    )
    monkeypatch.chdir(tmp_path)

    def fake_judge(
        manifest, predictions_path, rubrics, judge_output, provider, model, *, progress_model=None
    ):
        del manifest, predictions_path, rubrics, progress_model
        assert provider == "openai"
        assert model is None
        payload = {
            "schema_version": "ib.answer_judge.v1",
            "provider": provider,
            "judgment_count": 0,
            "passed_count": 0,
            "pass_rate": 1.0,
            "atomic_count": 0,
            "atomic_passed_count": 0,
            "atomic_pass_rate": 1.0,
            "judgments": [],
            "errors": [],
            "llm_call_count": 0,
            "llm_usage": str(Path(judge_output).with_suffix(".cost.json")),
        }
        Path(judge_output).write_text(json.dumps(payload) + "\n", encoding="utf-8")
        Path(judge_output).with_suffix(".cost.json").write_text("{}\n", encoding="utf-8")
        return payload

    monkeypatch.setattr(eval_module, "run_answer_judge", fake_judge)
    assert (
        cli_main.main(
            [
                "eval",
                "--manifest",
                str(run_dir / "commerce_service" / "manifest" / "manifest.jsonl"),
                "--rubrics",
                str(run_dir / "commerce_service" / "plans" / "rubrics.jsonl"),
                "--predictions",
                str(predictions),
                "--output",
                str(output),
                "--answer-judge",
            ]
        )
        == 0
    )

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["judge_provider"] == "openai"
    assert summary["judge_model"] == DEFAULT_OPENAI_JUDGE_MODEL


def test_eval_run_does_not_resume_older_incomplete_when_newer_output_is_complete(
    tmp_path, monkeypatch
) -> None:
    run_dir = tmp_path / "V0_test"
    eval_root = tmp_path / "eval-root"
    older_incomplete = eval_root / "V0_test_20260101T000000Z"
    newer_complete = eval_root / "V0_test_20260102T000000Z"
    _write_scene(run_dir, "commerce_service")
    older_incomplete.mkdir(parents=True)
    newer_complete.mkdir(parents=True)
    (newer_complete / "summary.json").write_text('{"schema_version":"done"}\n', encoding="utf-8")
    _write_models_yaml(
        tmp_path,
        """
base:
  - provider: openai
    model: gpt-audio-1.5
live: []
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("IB_EVAL_ROOT", str(eval_root))

    def fake_predictions(provider, manifest, predictions, model, limit, **kwargs):
        del provider, model, limit, kwargs
        rows = [
            json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines()
        ]
        _write_jsonl(
            Path(predictions),
            [
                {
                    "cell_id": row["cell_id"],
                    "predicted_action": row["expected_action"],
                    "tool_calls": [],
                }
                for row in rows
            ],
        )
        return {"provider": "openai", "model": "gpt-audio-1.5", "prediction_count": len(rows)}

    monkeypatch.setattr(eval_module, "_run_understanding_predictions", fake_predictions)
    assert cli_main.main(["eval", "--run", str(run_dir)]) == 0

    assert not (older_incomplete / "summary.json").exists()
    created = [path for path in eval_root.iterdir() if path.name.startswith("V0_test_")]
    assert len(created) == 3
    assert max(path.name for path in created) > newer_complete.name
