"""scene_prompt_profiles — static per-scene prompt fields for eval prompts.

Calling spec:
    profile_for_scene(scene: str) -> SceneProfile   # .assistant_role, .setting_phrase
    scene_date_weekday(value: str) -> str           # ISO date -> lowercase weekday

Deterministic; values are baked from the benchmark's scene definitions.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class SceneProfile:
    assistant_role: str
    setting_phrase: str


SCENE_PROFILES: dict[str, SceneProfile] = {
    "commerce_service": SceneProfile(
        assistant_role='service assistant',
        setting_phrase='service setting',
    ),
    "domestic_household": SceneProfile(
        assistant_role='home assistant',
        setting_phrase='home',
    ),
    "education_learning": SceneProfile(
        assistant_role='classroom assistant',
        setting_phrase='learning setting',
    ),
    "healthcare_caregiving_accessibility": SceneProfile(
        assistant_role='care assistant',
        setting_phrase='caregiving setting',
    ),
    "leisure_media_social": SceneProfile(
        assistant_role='media assistant',
        setting_phrase='social media setting',
    ),
    "mobility_transport": SceneProfile(
        assistant_role='car voice assistant',
        setting_phrase='in-car',
    ),
    "public_civic": SceneProfile(
        assistant_role='civic kiosk assistant',
        setting_phrase='public-service setting',
    ),
    "work_professional": SceneProfile(
        assistant_role='work assistant',
        setting_phrase='workplace',
    ),
}


def profile_for_scene(scene: str) -> SceneProfile:
    return SCENE_PROFILES[scene]


def scene_date_weekday(value: str) -> str:
    return date.fromisoformat(value).strftime("%A").lower()
