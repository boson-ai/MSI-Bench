"""Eval target protocol validation and prompt/native source separation."""

from __future__ import annotations

from pathlib import Path

import pytest

from ib.eval_targets import EvalTarget, eval_targets_from_models_file


def test_models_file_keeps_prompt_and_native_targets_for_same_model(tmp_path) -> None:
    models = tmp_path / "models.yaml"
    models.write_text(
        """
base:
  - name: gemini-prompt
    provider: gemini
    model: gemini-3.5-flash
    tool_protocol: prompt_json
  - name: gemini-native
    provider: gemini
    model: gemini-3.5-flash
""".lstrip(),
        encoding="utf-8",
    )

    targets = eval_targets_from_models_file(models)

    assert [(target.name, target.tool_protocol) for target in targets] == [
        ("gemini-prompt", "prompt_json"),
        ("gemini-native", "native"),
    ]


def test_models_file_keeps_realtime_reasoning_effort_variants(tmp_path) -> None:
    models = tmp_path / "models.yaml"
    models.write_text(
        """
live:
  - name: realtime-2-medium
    base:
      provider: openai
      model: gpt-audio-1.5
    live:
      provider: openai_realtime
      model: gpt-realtime-2
      reasoning_effort: medium
  - name: realtime-2-xhigh
    base:
      provider: openai
      model: gpt-audio-1.5
    live:
      provider: openai_realtime
      model: gpt-realtime-2
      reasoning_effort: xhigh
  - name: realtime-2.1-medium
    base:
      provider: openai
      model: gpt-audio-1.5
    live:
      provider: openai_realtime
      model: gpt-realtime-2.1
      reasoning_effort: medium
  - name: realtime-2.1-xhigh
    base:
      provider: openai
      model: gpt-audio-1.5
    live:
      provider: openai_realtime
      model: gpt-realtime-2.1
      reasoning_effort: xhigh
""".lstrip(),
        encoding="utf-8",
    )

    targets = eval_targets_from_models_file(models)

    live_targets = [target for target in targets if target.mode == "fullduplex"]
    assert [(target.name, target.model, target.reasoning_effort) for target in live_targets] == [
        ("realtime-2-medium:live", "gpt-realtime-2", "medium"),
        ("realtime-2-xhigh:live", "gpt-realtime-2", "xhigh"),
        ("realtime-2.1-medium:live", "gpt-realtime-2.1", "medium"),
        ("realtime-2.1-xhigh:live", "gpt-realtime-2.1", "xhigh"),
    ]


def test_reasoning_effort_rejects_non_realtime_2_targets() -> None:
    with pytest.raises(ValueError, match="requires gpt-realtime-2 or gpt-realtime-2.1"):
        EvalTarget(
            provider="openai_realtime",
            model="gpt-realtime-1.5",
            mode="fullduplex",
            reasoning_effort="medium",
        )


def test_default_models_file_registers_requested_audio_model_variants() -> None:
    models = Path(__file__).parents[1] / "configs" / "eval-models.yaml"

    targets = eval_targets_from_models_file(models)
    actual = {
        (target.model, target.mode, target.reasoning_effort, target.thinking_mode)
        for target in targets
    }
    requested = {
        ("gpt-realtime-2", "fullduplex", "medium", None),
        ("gpt-realtime-2", "fullduplex", "xhigh", None),
        ("gpt-realtime-2.1", "fullduplex", "medium", None),
        ("gpt-realtime-2.1", "fullduplex", "xhigh", None),
        ("gpt-realtime-1.5", "fullduplex", None, None),
        ("gpt-audio", "turn_based", None, None),
        ("gemini-3-flash-preview", "turn_based", None, "enabled"),
        ("gemini-3-flash-preview", "turn_based", None, "disabled"),
        ("gpt-realtime", "fullduplex", None, None),
        ("gemini-2.5-flash", "turn_based", None, "enabled"),
        ("gemini-3.1-pro-preview", "turn_based", None, "enabled"),
        ("gemini-2.5-pro", "turn_based", None, "enabled"),
        ("gpt-4o-audio-preview", "turn_based", None, None),
        ("gemini-2.5-flash", "turn_based", None, "disabled"),
        ("gpt-audio-mini", "turn_based", None, None),
        ("gemini-3.1-flash-live-preview", "fullduplex", None, "disabled"),
        ("gpt-realtime-mini", "fullduplex", None, None),
        ("gemini-3.1-flash-live-preview", "fullduplex", None, "enabled"),
        ("gpt-4o-mini-audio-preview", "turn_based", None, None),
    }

    assert requested <= actual


def test_thinking_mode_rejects_unregistered_gemini_target() -> None:
    with pytest.raises(ValueError, match="registered Gemini thinking target"):
        EvalTarget(
            provider="gemini",
            model="gemini-3.5-pro",
            mode="turn_based",
            thinking_mode="enabled",
        )


def test_local_models_file_registers_requested_open_source_variants(monkeypatch) -> None:
    models = Path(__file__).parents[1] / "configs" / "eval-models-local.yaml"
    real_exists = Path.exists

    def _checkpoint_exists(path: Path) -> bool:
        if str(path).startswith("/ceph/models/"):
            return True
        return real_exists(path)

    monkeypatch.setattr(Path, "exists", _checkpoint_exists)

    targets = eval_targets_from_models_file(models)

    assert [(target.model, target.thinking_mode) for target in targets] == [
        ("gemma-4-12B-it", "enabled"),
        ("gemma-4-12B-it", "disabled"),
        ("MiMo-Audio-7B-Instruct", "enabled"),
        ("MiMo-Audio-7B-Instruct", "disabled"),
        ("Voxtral-Small-24B-2507", None),
        ("Qwen3-Omni-30B-A3B-Instruct", None),
        ("Kimi-Audio-7B-Instruct", None),
        ("Qwen2.5-Omni-7B", None),
        ("Phi-4-multimodal-instruct", None),
        ("Qwen2-Audio-7B-Instruct", None),
    ]
    assert {target.provider for target in targets} == {"local"}


def test_models_file_skips_missing_checkpoint_paths(tmp_path) -> None:
    present = tmp_path / "present-model"
    present.mkdir()
    models = tmp_path / "models.yaml"
    models.write_text(
        f"""
base:
  - name: present
    provider: local
    model: Qwen2-Audio-7B-Instruct
    checkpoint_path: {present}
  - name: missing
    provider: local
    model: Phi-4-multimodal-instruct
    checkpoint_path: {tmp_path / "missing-model"}
""".lstrip(),
        encoding="utf-8",
    )

    targets = eval_targets_from_models_file(models)

    assert [target.name for target in targets] == ["present"]


def test_generated_provider_targets_default_native_without_affecting_other_sources() -> None:
    openai = EvalTarget(provider="openai", model="gpt-audio-1.5")
    gemini = EvalTarget(provider="gemini", model="gemini-3.5-flash")
    live = EvalTarget(
        provider="openai_realtime",
        model="gpt-realtime-1.5",
        mode="fullduplex",
    )
    existing = EvalTarget(provider="openai", predictions="predictions.jsonl")

    assert (openai.tool_protocol, gemini.tool_protocol) == ("native", "native")
    assert (live.tool_protocol, existing.tool_protocol) == ("prompt_json", "prompt_json")


def test_native_target_rejects_unsupported_generation_paths() -> None:
    with pytest.raises(ValueError, match="turn_based openai or gemini"):
        EvalTarget(
            provider="openai_realtime",
            model="gpt-realtime-1.5",
            mode="fullduplex",
            tool_protocol="native",
        )
    with pytest.raises(ValueError, match="generated predictions"):
        EvalTarget(
            provider="openai",
            model="gpt-audio-1.5",
            predictions="predictions.jsonl",
            tool_protocol="native",
        )


def test_native_target_default_name_marks_protocol() -> None:
    target = EvalTarget(
        provider="openai",
        model="gpt-audio-1.5",
    )

    assert target.name == "turn_based:openai:gpt-audio-1.5:native"
