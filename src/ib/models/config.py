"""config — build-corpus and scene-logic selection schema.

Calling spec:
    BuildConfig.model_validate(<dict from build.yaml>)
    SceneSpec.model_validate(<dict from schema/scenes/<name>.yaml>)

BuildConfig owns the deterministic seed and the scene-logic selection matrix.
SceneSpec declares a scene's hard-logic profile; extra='forbid' rejects unknown
keys, with retired legacy sections stripped by explicit migration validators.

Side effects: none.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from ib.models.base import IbBaseModel
from ib.models.enums import (
    ReverbLevel,
    Scene,
)

class CorpusConfig(IbBaseModel):
    """The single deterministic corpus seed."""

    seed: int = Field(ge=0)


class LogicPatternQuantity(IbBaseModel):
    """One pattern's quota weight plus its subpattern quota weights."""

    weight: float = 1.0
    subpatterns: dict[str, float] = Field(default_factory=dict)

    @field_validator("weight")
    @classmethod
    def _valid_weight(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError("logic_selection.quantity.patterns.*.weight must be >= 0")
        return float(value)

    @field_validator("subpatterns")
    @classmethod
    def _valid_subpatterns(cls, value: dict[str, float]) -> dict[str, float]:
        return _valid_quantity_weights(
            value,
            "logic_selection.quantity.patterns.*.subpatterns",
        )


class LogicQuantityConfig(IbBaseModel):
    """Single scene-logic testcase count scheduler config."""

    total_testcases: int = Field(default=56, ge=1)
    scenes: dict[str, float] = Field(
        default_factory=lambda: {scene.value: 1.0 for scene in Scene}
    )
    patterns: dict[str, LogicPatternQuantity] = Field(default_factory=dict)

    @field_validator("scenes")
    @classmethod
    def _valid_scenes(cls, value: dict[str, float]) -> dict[str, float]:
        cleaned = _valid_quantity_weights(value, "logic_selection.quantity.scenes")
        allowed_scenes = {scene.value for scene in Scene}
        unknown_scenes = sorted(set(cleaned) - allowed_scenes)
        if unknown_scenes:
            raise ValueError(
                "logic_selection.quantity.scenes has unsupported "
                f"scene(s) {unknown_scenes}"
            )
        if sum(cleaned.values()) <= 0.0:
            raise ValueError("logic_selection.quantity.scenes must have positive total weight")
        return cleaned

    @field_validator("patterns")
    @classmethod
    def _valid_patterns(
        cls, value: dict[str, LogicPatternQuantity]
    ) -> dict[str, LogicPatternQuantity]:
        cleaned: dict[str, LogicPatternQuantity] = {}
        for key, quantity in value.items():
            stripped = key.strip()
            if not stripped:
                raise ValueError("logic_selection.quantity.patterns keys must be non-empty")
            cleaned[stripped] = quantity
        return cleaned


class LogicSelection(IbBaseModel):
    """Scene-logic target matrix used by the scene_logic_constraints plane."""

    patterns: list[str] = Field(default_factory=lambda: ["*"], min_length=1)
    subpatterns: list[str] = Field(default_factory=lambda: ["*"], min_length=1)
    live_expansions: list[str] = Field(default_factory=list)
    speaker_counts: list[int] = Field(default_factory=lambda: [2, 3], min_length=1)
    languages: list[str] = Field(default_factory=lambda: ["en"], min_length=1)
    include_tool_call_args_rubric: bool = True
    use_topic_examples: bool = True
    # Prior build roots whose already-used topic examples must not be reused.
    # Each root contains <scene>/generation topic artifacts (delta builds).
    topic_exclude_from: list[str] = Field(default_factory=list)
    polish: bool = False
    context_assistant_probability: float = Field(default=0.2, ge=0.0, le=1.0)
    inline_acoustic_cue_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    identity_mode_distribution: dict[str, float] = Field(
        default_factory=lambda: {"in_text": 0.7, "in_audio": 0.3}
    )
    quantity: LogicQuantityConfig = Field(default_factory=LogicQuantityConfig)

    @field_validator("patterns", "subpatterns", "languages", "topic_exclude_from")
    @classmethod
    def _non_empty_strings(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        if any(not item for item in cleaned):
            raise ValueError("logic selection values must be non-empty strings")
        return cleaned

    @field_validator("speaker_counts")
    @classmethod
    def _supported_speaker_counts(cls, value: list[int]) -> list[int]:
        if any(count not in {2, 3} for count in value):
            raise ValueError("logic_selection.speaker_counts values must be 2 or 3")
        return value

    @field_validator("languages")
    @classmethod
    def _supported_languages(cls, value: list[str]) -> list[str]:
        allowed = {"en", "zh-CN"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unsupported logic_selection.languages {unknown}")
        return value

    @field_validator("identity_mode_distribution")
    @classmethod
    def _valid_identity_mode_distribution(cls, value: dict[str, float]) -> dict[str, float]:
        allowed = {"in_text", "in_audio"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unsupported logic_selection.identity_mode_distribution {unknown}")
        if not value:
            raise ValueError("logic_selection.identity_mode_distribution must not be empty")
        if any(weight < 0.0 for weight in value.values()):
            raise ValueError("identity mode weights must be non-negative")
        if sum(value.values()) <= 0.0:
            raise ValueError("identity mode weights must have positive total mass")
        return {key: float(value.get(key, 0.0)) for key in ("in_text", "in_audio")}

    @field_validator("live_expansions")
    @classmethod
    def _supported_live_expansions(cls, value: list[str]) -> list[str]:
        allowed = {"bystander_interference_suppression"}
        cleaned = [item.strip() for item in value]
        if any(not item for item in cleaned):
            raise ValueError("logic_selection.live_expansions values must be non-empty")
        unknown = sorted(set(cleaned) - allowed)
        if unknown:
            raise ValueError(f"unsupported logic_selection.live_expansions {unknown}")
        return cleaned


def _valid_quantity_weights(value: dict[str, float], field: str) -> dict[str, float]:
    if not value:
        raise ValueError(f"{field} must not be empty")
    cleaned: dict[str, float] = {}
    for key, weight in value.items():
        stripped = key.strip()
        if not stripped:
            raise ValueError(f"{field} keys must be non-empty")
        if weight < 0.0:
            raise ValueError(f"{field}.{stripped} must be >= 0")
        cleaned[stripped] = float(weight)
    return cleaned


class ProviderModelChoice(IbBaseModel):
    """One concrete provider/model pair, with an optional reasoning effort."""

    provider: str
    model: str | None = None
    reasoning_effort: str | None = None


class ProviderModelConfig(ProviderModelChoice):
    """One configurable provider/model pair used as a CLI default."""

    provider: str = "openai"
    by_language: dict[str, ProviderModelChoice] = Field(default_factory=dict)

    @field_validator("by_language")
    @classmethod
    def _supported_language_routes(
        cls, value: dict[str, ProviderModelChoice]
    ) -> dict[str, ProviderModelChoice]:
        allowed = {"en", "zh-CN"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unsupported provider language routes {unknown}")
        return value


class LlmProviderConfig(IbBaseModel):
    """Provider defaults for each text-LLM role in planning/building."""

    director: ProviderModelConfig = Field(default_factory=ProviderModelConfig)
    actor: ProviderModelConfig = Field(default_factory=ProviderModelConfig)
    harness: ProviderModelConfig = Field(default_factory=ProviderModelConfig)
    rubric: ProviderModelConfig = Field(default_factory=ProviderModelConfig)
    logic_semantic_verifier: ProviderModelConfig = Field(
        default_factory=lambda: ProviderModelConfig(provider="openai", model="gpt-5.4")
    )


class BuildProviderConfig(IbBaseModel):
    """Provider defaults for `ib plan` and `ib build`.

    CLI flags override these values when supplied.
    """

    tts: ProviderModelConfig = Field(
        default_factory=lambda: ProviderModelConfig(provider="boson", model="higgs-tts-3")
    )
    llm: LlmProviderConfig = Field(default_factory=LlmProviderConfig)


_RETIRED_BUILD_KEYS = ("selection", "distributions", "level_subset")


class BuildConfig(IbBaseModel):
    """Top-level build config: corpus, scene list, providers, and logic selection."""

    corpus: CorpusConfig
    # benchmark: director -> actor -> harness (tools, oracle summary, rubric).
    # traindata: director -> actor only; the actor's expected_response is the
    # gold answer and no tool schema or rubric is generated.
    mode: Literal["benchmark", "traindata"] = "benchmark"
    scenes: list[Scene] = Field(default_factory=list)
    providers: BuildProviderConfig = Field(default_factory=BuildProviderConfig)
    logic_selection: LogicSelection = Field(default_factory=LogicSelection)

    @model_validator(mode="before")
    @classmethod
    def _strip_retired_sections(cls, data):
        """Ignore build.yaml sections retired with the participation-frame plane."""
        if not isinstance(data, dict):
            return data
        if not any(key in data for key in _RETIRED_BUILD_KEYS):
            return data
        migrated = dict(data)
        for key in _RETIRED_BUILD_KEYS:
            migrated.pop(key, None)
        return migrated

    @field_validator("scenes")
    @classmethod
    def _unique_scenes(cls, value: list[Scene]) -> list[Scene]:
        seen: set[Scene] = set()
        duplicates: list[str] = []
        for scene in value:
            if scene in seen:
                duplicates.append(scene.value)
            seen.add(scene)
        if duplicates:
            raise ValueError(f"duplicate build scenes {duplicates}")
        return value


class LogicSceneProfileSpec(IbBaseModel):
    """Hard-logic prompt/profile fields embedded in one scene YAML."""

    domain_label: str
    setting_phrase: str
    assistant_role: str
    definition: str
    setting_registry: list[str] | None = None
    setting_examples: list[str] | None = None
    boundary_rule: str
    reverb: ReverbLevel = ReverbLevel.dry
    topic_examples: list[str] | dict[str, Any] | None = None
    pattern_example_rules: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _has_one_example_source(self) -> "LogicSceneProfileSpec":
        has_rules = self.pattern_example_rules is not None
        has_scene_pools = self.setting_examples is not None or self.topic_examples is not None
        if has_rules and has_scene_pools:
            raise ValueError("pattern_example_rules must not be mixed with setting/topic pools")
        if not has_rules and (self.setting_examples is None or self.topic_examples is None):
            raise ValueError("logic_profile must define pattern_example_rules or setting/topic pools")
        return self


_RETIRED_SCENE_KEYS = ("admissible", "planes")


class SceneSpec(IbBaseModel):
    """One scene file: scene class + hard-logic profile."""

    scene: Scene
    logic_profile: LogicSceneProfileSpec

    @model_validator(mode="before")
    @classmethod
    def _strip_retired_sections(cls, data):
        """Ignore scene YAML sections retired with the participation-frame plane."""
        if not isinstance(data, dict):
            return data
        if not any(key in data for key in _RETIRED_SCENE_KEYS):
            return data
        migrated = dict(data)
        for key in _RETIRED_SCENE_KEYS:
            migrated.pop(key, None)
        return migrated
