"""local_audio_models — registry for local audio endpoint contracts.

Calling spec:
    spec = local_audio_model_spec(model)
    url = local_audio_base_url(spec)

Inputs are canonical served model names. Outputs describe checkpoint paths,
OpenAI-compatible endpoint defaults, prompt constraints, and thinking support.
Side effects: local_audio_base_url reads optional environment overrides only.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal

LocalAudioTransport = Literal["audio_url", "transcription_only"]
SystemPromptMode = Literal["system", "first_user"]


@dataclass(frozen=True)
class LocalAudioModelSpec:
    """One validated local serving contract."""

    model: str
    checkpoint_path: Path
    default_base_url: str
    base_url_env: str
    transport: LocalAudioTransport = "audio_url"
    system_prompt_mode: SystemPromptMode = "system"
    supports_thinking: bool = False
    max_audio_clips: int | None = None


LOCAL_AUDIO_MODELS: dict[str, LocalAudioModelSpec] = {
    "gemma-4-12B-it": LocalAudioModelSpec(
        model="gemma-4-12B-it",
        checkpoint_path=Path("/ceph/models/gemma-4-12B-it"),
        default_base_url="http://localhost:8014/v1",
        base_url_env="GEMMA4_UNDERSTANDING_BASE_URL",
        supports_thinking=True,
        # 13 = the deepest speak-probe rendering (12 base clips + 1 live event);
        # must match limit-mm-per-prompt audio in serve/configs/gemma4-12b.
        max_audio_clips=13,
    ),
    "MiMo-Audio-7B-Instruct": LocalAudioModelSpec(
        model="MiMo-Audio-7B-Instruct",
        checkpoint_path=Path("/ceph/models/MiMo-Audio-7B-Instruct"),
        default_base_url="http://localhost:8015/v1",
        base_url_env="MIMO_AUDIO_UNDERSTANDING_BASE_URL",
        supports_thinking=True,
    ),
    "Voxtral-Small-24B-2507": LocalAudioModelSpec(
        model="Voxtral-Small-24B-2507",
        checkpoint_path=Path("/ceph/models/Voxtral-Small-24B-2507"),
        default_base_url="http://localhost:8016/v1",
        base_url_env="VOXTRAL_UNDERSTANDING_BASE_URL",
        system_prompt_mode="first_user",
        max_audio_clips=5,
    ),
    "Qwen3-Omni-30B-A3B-Instruct": LocalAudioModelSpec(
        model="Qwen3-Omni-30B-A3B-Instruct",
        checkpoint_path=Path("/ceph/models/Qwen3-Omni-30B-A3B-Instruct"),
        default_base_url="http://localhost:8013/v1",
        base_url_env="QWEN_UNDERSTANDING_BASE_URL",
    ),
    "Kimi-Audio-7B-Instruct": LocalAudioModelSpec(
        model="Kimi-Audio-7B-Instruct",
        checkpoint_path=Path("/ceph/models/Kimi-Audio-7B-Instruct"),
        default_base_url="http://localhost:8017/v1",
        base_url_env="KIMI_AUDIO_UNDERSTANDING_BASE_URL",
        transport="transcription_only",
    ),
    "Qwen2.5-Omni-7B": LocalAudioModelSpec(
        model="Qwen2.5-Omni-7B",
        checkpoint_path=Path("/ceph/models/Qwen2.5-Omni-7B"),
        default_base_url="http://localhost:8018/v1",
        base_url_env="QWEN25_OMNI_UNDERSTANDING_BASE_URL",
    ),
    "Phi-4-multimodal-instruct": LocalAudioModelSpec(
        model="Phi-4-multimodal-instruct",
        checkpoint_path=Path("/ceph/models/Phi-4-multimodal-instruct"),
        default_base_url="http://localhost:8019/v1",
        base_url_env="PHI4_MULTIMODAL_UNDERSTANDING_BASE_URL",
    ),
    "Qwen2-Audio-7B-Instruct": LocalAudioModelSpec(
        model="Qwen2-Audio-7B-Instruct",
        checkpoint_path=Path("/ceph/models/Qwen2-Audio-7B-Instruct"),
        default_base_url="http://localhost:8020/v1",
        base_url_env="QWEN2_AUDIO_UNDERSTANDING_BASE_URL",
    ),
}


def local_audio_model_spec(model: str) -> LocalAudioModelSpec:
    """Return the exact registered contract for a served model name."""
    try:
        return LOCAL_AUDIO_MODELS[model]
    except KeyError as exc:
        raise ValueError(
            f"unregistered local audio model {model!r}; choose one of {sorted(LOCAL_AUDIO_MODELS)}"
        ) from exc


def local_audio_base_url(spec: LocalAudioModelSpec) -> str:
    """Resolve one endpoint without changing its registered model identity."""
    return (
        os.getenv(spec.base_url_env)
        or os.getenv("LOCAL_AUDIO_UNDERSTANDING_BASE_URL")
        or spec.default_base_url
    )
