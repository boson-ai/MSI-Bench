"""eval_targets — validated model targets for MSI-Bench eval runs.

Calling spec:
    target = EvalTarget(...)
    targets = eval_targets_from_specs(["turn_based:openai:gpt-audio-1.5"])
    targets = eval_targets_from_models_file("configs/eval-models.yaml")
    targets = eval_targets_from_preset("gpt-gemini-live")
    targets = eval_targets_from_file("targets.yaml")

Each target binds a model/provider to one evaluation mode. Turn-based and
full-duplex targets can be evaluated in the same batch without sharing metric
applicability. Omitted tool_protocol defaults to provider-native tools for
generated turn-based OpenAI/Gemini targets, and prompt_json otherwise.
reasoning_effort is accepted only for registered GPT Realtime 2-family targets.
thinking_mode is accepted only for registered Gemini/Gemma/MiMo variants.
Model blocks with checkpoint_path are skipped when that local path is absent.

Side effects: eval_targets_from_file reads one YAML/JSON file only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import model_validator

from ib.models.base import IbBaseModel

EvalMode = Literal["turn_based", "fullduplex"]
ToolProtocol = Literal["prompt_json", "native"]
RealtimeReasoningEffort = Literal["minimal", "low", "medium", "high", "xhigh"]
ThinkingMode = Literal["disabled", "enabled"]
GeminiThinkingMode = ThinkingMode
EVAL_MODES = ("turn_based", "fullduplex")
EVAL_TARGET_PRESETS = {
    "gpt-gemini-live": (
        "turn_based:openai:gpt-audio-1.5",
        "fullduplex:openai_realtime:gpt-realtime-1.5",
        "turn_based:gemini:gemini-3.5-flash",
        "fullduplex:gemini_live:gemini-3.1-flash-live-preview",
    ),
}


_NATIVE_TOOL_PROVIDERS = {"openai", "gemini", "gemini_text"}
_REASONING_REALTIME_MODELS = {"gpt-realtime-2", "gpt-realtime-2.1"}
_THINKING_TARGETS = {
    ("turn_based", "gemini", "gemini-2.5-flash"),
    ("turn_based", "gemini", "gemini-2.5-pro"),
    ("turn_based", "gemini", "gemini-3-flash-preview"),
    ("turn_based", "gemini", "gemini-3.1-pro-preview"),
    ("turn_based", "gemini", "gemini-3.5-flash"),
    ("turn_based", "gemini_text", "gemini-3.1-pro-preview"),
    ("turn_based", "gemini_text", "gemini-3.5-flash"),
    ("fullduplex", "gemini_live", "gemini-3.1-flash-live-preview"),
    ("turn_based", "local", "gemma-4-12B-it"),
    ("turn_based", "local", "MiMo-Audio-7B-Instruct"),
    ("turn_based", "local_text", "gemma-4-12B-it"),
    ("turn_based", "local_text", "MiMo-Audio-7B-Instruct"),
}


def default_tool_protocol(*, mode: str, provider: str, predictions: object | None) -> ToolProtocol:
    """Return the deterministic transport default for one eval target."""
    if predictions is None and mode == "turn_based" and provider in _NATIVE_TOOL_PROVIDERS:
        return "native"
    return "prompt_json"


class EvalTarget(IbBaseModel):
    """One model or prediction artifact to evaluate."""

    name: str | None = None
    provider: str = "openai"
    model: str | None = None
    mode: EvalMode = "turn_based"
    predictions: str | None = None
    tool_protocol: ToolProtocol = "native"
    reasoning_effort: RealtimeReasoningEffort | None = None
    thinking_mode: ThinkingMode | None = None

    @model_validator(mode="after")
    def _fill_name_and_validate(self) -> "EvalTarget":
        if "tool_protocol" not in self.model_fields_set:
            self.tool_protocol = default_tool_protocol(
                mode=self.mode,
                provider=self.provider,
                predictions=self.predictions,
            )
        if self.predictions is None and not self.provider:
            raise ValueError("eval target provider is required when predictions is omitted")
        if self.predictions is None and not self.model:
            raise ValueError("eval target model is required when predictions is omitted")
        if self.tool_protocol == "native":
            if self.predictions is not None:
                raise ValueError("native tool protocol applies only to generated predictions")
            if self.mode != "turn_based" or self.provider not in _NATIVE_TOOL_PROVIDERS:
                raise ValueError(
                    "native tool protocol currently supports turn_based openai or gemini targets"
                )
        if self.reasoning_effort is not None:
            if self.predictions is not None:
                raise ValueError("reasoning_effort applies only to generated predictions")
            if self.mode != "fullduplex" or self.provider != "openai_realtime":
                raise ValueError(
                    "reasoning_effort currently supports fullduplex openai_realtime targets"
                )
            if self.model not in _REASONING_REALTIME_MODELS:
                raise ValueError("reasoning_effort requires gpt-realtime-2 or gpt-realtime-2.1")
        if self.thinking_mode is not None:
            if self.predictions is not None:
                raise ValueError("thinking_mode applies only to generated predictions")
            if (self.mode, self.provider, self.model) not in _THINKING_TARGETS:
                raise ValueError(
                    "thinking_mode requires a registered Gemini thinking target or "
                    "local Gemma/MiMo target"
                )
        if self.name is None:
            source = self.model or (
                Path(self.predictions).stem if self.predictions else self.provider
            )
            self.name = f"{self.mode}:{self.provider}:{source}"
            if self.tool_protocol == "native":
                self.name = f"{self.name}:native"
            if self.reasoning_effort is not None:
                self.name = f"{self.name}:reasoning-{self.reasoning_effort}"
            if self.thinking_mode is not None:
                self.name = f"{self.name}:thinking-{self.thinking_mode}"
        return self


def eval_targets_from_specs(specs: list[str] | tuple[str, ...] | None) -> list[EvalTarget]:
    """Parse repeated ``--target`` specs.

    Supported forms:
    - ``mode:provider:model`` for generated predictions.
    - ``mode:provider:model:predictions.jsonl`` for existing predictions.
    """
    targets: list[EvalTarget] = []
    for spec in specs or []:
        targets.append(eval_target_from_spec(spec))
    return targets


def eval_targets_from_preset(name: str | None) -> list[EvalTarget]:
    """Return built-in target groups for common eval comparisons."""
    if name is None:
        return []
    specs = EVAL_TARGET_PRESETS.get(name)
    if specs is None:
        raise ValueError(
            f"unknown eval target preset {name!r}; choose one of {sorted(EVAL_TARGET_PRESETS)}"
        )
    return eval_targets_from_specs(specs)


def eval_targets_from_models_file(path: str | Path | None) -> list[EvalTarget]:
    """Read declarative base/live model groups from YAML.

    Schema:
      base entries produce one turn_based target.
      live entries produce one turn_based base target plus one fullduplex target.
    """
    if path is None:
        return []
    source = Path(path)
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read eval models file {source}: {exc}") from exc
    if raw is None:
        return []
    if not isinstance(raw, dict):
        raise ValueError("eval models file must be a mapping with base: and/or live:")
    targets: list[EvalTarget] = []
    for index, item in enumerate(_list_section(raw, "base"), start=1):
        target = _target_from_model_block(
            item,
            mode="turn_based",
            section="base",
            index=index,
            source_dir=source.parent,
        )
        if target is not None:
            targets.append(target)
    for index, item in enumerate(_list_section(raw, "live"), start=1):
        if not isinstance(item, dict):
            raise ValueError(f"eval models live entry #{index} must be an object")
        name = item.get("name")
        name_prefix = str(name) if isinstance(name, str) and name else f"live-{index}"
        # A live entry may omit base: to run only the fullduplex half (e.g. when
        # the shared turn_based counterpart already ran under another output root).
        base_target = None
        if item.get("base") is not None:
            base_target = _target_from_model_block(
                item.get("base"),
                mode="turn_based",
                section="live.base",
                index=index,
                name=f"{name_prefix}:base",
                source_dir=source.parent,
            )
        live_target = _target_from_model_block(
            item.get("live"),
            mode="fullduplex",
            section="live.live",
            index=index,
            name=f"{name_prefix}:live",
            source_dir=source.parent,
        )
        if base_target is not None:
            targets.append(base_target)
        if live_target is not None:
            targets.append(live_target)
    return _dedupe_targets_by_source(targets)


def _list_section(raw: dict, key: str) -> list:
    value = raw.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"eval models {key}: must be a list")
    return value


def _target_from_model_block(
    item: object,
    *,
    mode: EvalMode,
    section: str,
    index: int,
    source_dir: Path,
    name: str | None = None,
) -> EvalTarget | None:
    if not isinstance(item, dict):
        raise ValueError(f"eval models {section} entry #{index} must be an object")
    provider = item.get("provider")
    model = item.get("model")
    target_name = name or item.get("name")
    if not isinstance(provider, str) or not provider:
        raise ValueError(f"eval models {section} entry #{index} missing provider")
    if not isinstance(model, str) or not model:
        raise ValueError(f"eval models {section} entry #{index} missing model")
    if target_name is not None and not isinstance(target_name, str):
        raise ValueError(f"eval models {section} entry #{index} name must be a string")
    checkpoint_path = item.get("checkpoint_path")
    if checkpoint_path is not None:
        if not isinstance(checkpoint_path, str) or not checkpoint_path.strip():
            raise ValueError(
                f"eval models {section} entry #{index} checkpoint_path must be a string"
            )
        candidate = Path(checkpoint_path).expanduser()
        if not candidate.is_absolute():
            candidate = source_dir / candidate
        if not candidate.exists():
            return None
    target_data: dict[str, object] = {
        "name": target_name,
        "provider": provider,
        "model": model,
        "mode": mode,
    }
    if "tool_protocol" in item:
        target_data["tool_protocol"] = item["tool_protocol"]
    if "reasoning_effort" in item:
        target_data["reasoning_effort"] = item["reasoning_effort"]
    if "thinking_mode" in item:
        target_data["thinking_mode"] = item["thinking_mode"]
    return EvalTarget.model_validate(target_data)


def _dedupe_targets_by_source(targets: list[EvalTarget]) -> list[EvalTarget]:
    deduped: list[EvalTarget] = []
    seen: set[
        tuple[
            str,
            str,
            str | None,
            str | None,
            ToolProtocol,
            RealtimeReasoningEffort | None,
            GeminiThinkingMode | None,
        ]
    ] = set()
    for target in targets:
        key = (
            target.mode,
            target.provider,
            target.model,
            target.predictions,
            target.tool_protocol,
            target.reasoning_effort,
            target.thinking_mode,
        )
        if key in seen:
            continue
        deduped.append(target)
        seen.add(key)
    return deduped


def eval_target_from_spec(spec: str) -> EvalTarget:
    """Parse one compact CLI target spec into an EvalTarget."""
    parts = spec.split(":", 3)
    if len(parts) < 3:
        raise ValueError("--target must be MODE:PROVIDER:MODEL or MODE:PROVIDER:MODEL:PREDICTIONS")
    mode, provider, model = (part.strip() for part in parts[:3])
    predictions = parts[3].strip() if len(parts) == 4 and parts[3].strip() else None
    if mode not in EVAL_MODES:
        raise ValueError(f"target mode must be one of {EVAL_MODES}; got {mode!r}")
    return EvalTarget(mode=mode, provider=provider, model=model or None, predictions=predictions)


def eval_targets_from_file(path: str | Path | None) -> list[EvalTarget]:
    """Read a YAML/JSON target file containing ``targets: [...]`` or a list."""
    if path is None:
        return []
    source = Path(path)
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read eval targets file {source}: {exc}") from exc
    if raw is None:
        return []
    items = raw.get("targets") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise ValueError("eval targets file must contain a list or a mapping with targets: [...]")
    targets: list[EvalTarget] = []
    for index, item in enumerate(items):
        if isinstance(item, str):
            targets.append(eval_target_from_spec(item))
            continue
        if not isinstance(item, dict):
            raise ValueError(f"eval target #{index + 1} must be a string or object")
        targets.append(EvalTarget.model_validate(item))
    return targets


def merge_eval_targets(*groups: list[EvalTarget]) -> list[EvalTarget]:
    """Return targets from all groups, rejecting duplicate target names."""
    targets = [target for group in groups for target in group]
    names: set[str] = set()
    duplicates: set[str] = set()
    for target in targets:
        name = target.name or ""
        if name in names:
            duplicates.add(name)
        names.add(name)
    if duplicates:
        raise ValueError(f"duplicate eval target name(s): {sorted(duplicates)}")
    return targets
