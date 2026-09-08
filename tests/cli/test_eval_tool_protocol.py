"""CLI wiring for isolated prompt-json and native-tools eval targets."""

from __future__ import annotations

from pathlib import Path

from ib.cli import eval as eval_module
from ib.eval_targets import EvalTarget
from ib.io import read_jsonl, write_jsonl
from ib.predictions import PredictionRow


def test_native_target_uses_a_separate_generated_output_path() -> None:
    repo_root = Path("/repo")
    prompt_target = EvalTarget(
        provider="openai",
        model="gpt-audio-1.5",
        tool_protocol="prompt_json",
    )
    native_target = EvalTarget(
        provider="openai",
        model="gpt-audio-1.5",
    )

    assert eval_module._target_source_path(prompt_target, repo_root) == Path(
        "generated/turn_based/openai/gpt-audio-1.5"
    )
    assert eval_module._target_source_path(native_target, repo_root) == Path(
        "generated/turn_based/native/openai/gpt-audio-1.5"
    )


def test_realtime_effort_variants_use_separate_generated_output_paths() -> None:
    repo_root = Path("/repo")
    medium = EvalTarget(
        provider="openai_realtime",
        model="gpt-realtime-2",
        mode="fullduplex",
        reasoning_effort="medium",
    )
    xhigh = medium.model_copy(update={"reasoning_effort": "xhigh"})
    medium_21 = medium.model_copy(update={"model": "gpt-realtime-2.1"})

    assert eval_module._target_source_path(medium, repo_root) == Path(
        "generated/fullduplex/base_only/openai_realtime/gpt-realtime-2/reasoning_medium"
    )
    assert eval_module._target_source_path(xhigh, repo_root) == Path(
        "generated/fullduplex/base_only/openai_realtime/gpt-realtime-2/reasoning_xhigh"
    )
    assert eval_module._target_source_path(medium_21, repo_root) == Path(
        "generated/fullduplex/base_only/openai_realtime/gpt-realtime-2.1/reasoning_medium"
    )


def test_gemini_thinking_variants_use_separate_generated_output_paths() -> None:
    repo_root = Path("/repo")
    thinking = EvalTarget(
        provider="gemini",
        model="gemini-3-flash-preview",
        thinking_mode="enabled",
    )
    plain = thinking.model_copy(update={"thinking_mode": "disabled"})

    assert eval_module._target_source_path(thinking, repo_root) == Path(
        "generated/turn_based/native/gemini/gemini-3-flash-preview/thinking_enabled"
    )
    assert eval_module._target_source_path(plain, repo_root) == Path(
        "generated/turn_based/native/gemini/gemini-3-flash-preview/thinking_disabled"
    )


def test_prediction_dispatch_forwards_and_tags_native_protocol(tmp_path, monkeypatch) -> None:
    predictions = tmp_path / "predictions.jsonl"

    def fake_run(manifest, output, *, model, limit, tool_protocol):
        assert manifest == "manifest.jsonl"
        assert model == "gpt-audio-1.5"
        assert limit == 1
        assert tool_protocol == "native"
        write_jsonl(
            output,
            [
                PredictionRow(
                    cell_id="c1",
                    predicted_action="respond",
                    tool_calls=[{"name": "set_navigation", "arguments": {}}],
                )
            ],
        )
        return {"provider": "openai", "model": model}

    monkeypatch.setattr(eval_module, "run_openai_understanding_predictions", fake_run)

    report = eval_module._run_understanding_predictions(
        "openai",
        "manifest.jsonl",
        str(predictions),
        "gpt-audio-1.5",
        1,
        target_name="openai-native",
        tool_protocol="native",
    )

    prediction = read_jsonl(predictions, PredictionRow)[0]
    assert report["tool_protocol"] == "native"
    assert prediction.metadata == {
        "eval_mode": "turn_based",
        "tool_protocol": "native",
        "eval_target": "openai-native",
    }


def test_realtime_prediction_dispatch_forwards_reasoning_effort(tmp_path, monkeypatch) -> None:
    predictions = tmp_path / "predictions.jsonl"

    def fake_run(manifest, output, *, model, limit, live_probe, reasoning_effort):
        assert manifest == "manifest.jsonl"
        assert model == "gpt-realtime-2"
        assert limit == 1
        assert live_probe is False
        assert reasoning_effort == "xhigh"
        write_jsonl(
            output,
            [PredictionRow(cell_id="c1", predicted_action="respond", tool_calls=[])],
        )
        return {
            "provider": "openai_realtime",
            "model": model,
            "reasoning_effort": reasoning_effort,
        }

    monkeypatch.setattr(eval_module, "run_openai_fullduplex_predictions", fake_run)

    report = eval_module._run_understanding_predictions(
        "openai_realtime",
        "manifest.jsonl",
        str(predictions),
        "gpt-realtime-2",
        1,
        eval_mode="fullduplex",
        reasoning_effort="xhigh",
    )

    assert report["reasoning_effort"] == "xhigh"


def test_gemini_prediction_dispatch_forwards_thinking_mode(tmp_path, monkeypatch) -> None:
    predictions = tmp_path / "predictions.jsonl"

    def fake_run(manifest, output, *, model, limit, tool_protocol, thinking_mode):
        assert manifest == "manifest.jsonl"
        assert model == "gemini-3-flash-preview"
        assert limit == 1
        assert tool_protocol == "native"
        assert thinking_mode == "enabled"
        write_jsonl(
            output,
            [PredictionRow(cell_id="c1", predicted_action="respond", tool_calls=[])],
        )
        return {
            "provider": "gemini",
            "model": model,
            "thinking_mode": thinking_mode,
        }

    monkeypatch.setattr(eval_module, "run_gemini_understanding_predictions", fake_run)

    report = eval_module._run_understanding_predictions(
        "gemini",
        "manifest.jsonl",
        str(predictions),
        "gemini-3-flash-preview",
        1,
        target_name="gemini-thinking",
        tool_protocol="native",
        thinking_mode="enabled",
    )

    prediction = read_jsonl(predictions, PredictionRow)[0]
    assert report["thinking_mode"] == "enabled"
    assert prediction.metadata == {
        "eval_mode": "turn_based",
        "tool_protocol": "native",
        "thinking_mode": "enabled",
        "eval_target": "gemini-thinking",
    }


def test_local_prediction_dispatch_forwards_thinking_mode(tmp_path, monkeypatch) -> None:
    predictions = tmp_path / "predictions.jsonl"

    def fake_run(manifest, output, *, model, limit, thinking_mode):
        assert manifest == "manifest.jsonl"
        assert model == "MiMo-Audio-7B-Instruct"
        assert limit == 1
        assert thinking_mode == "disabled"
        write_jsonl(
            output,
            [PredictionRow(cell_id="c1", predicted_action="respond", tool_calls=[])],
        )
        return {"provider": "local", "model": model, "thinking_mode": thinking_mode}

    monkeypatch.setattr(eval_module, "run_local_understanding_predictions", fake_run)

    report = eval_module._run_understanding_predictions(
        "local",
        "manifest.jsonl",
        str(predictions),
        "MiMo-Audio-7B-Instruct",
        1,
        target_name="mimo-plain",
        thinking_mode="disabled",
    )

    prediction = read_jsonl(predictions, PredictionRow)[0]
    assert report["provider"] == "local"
    assert prediction.metadata == {
        "eval_mode": "turn_based",
        "tool_protocol": "prompt_json",
        "thinking_mode": "disabled",
        "eval_target": "mimo-plain",
    }
