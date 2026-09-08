"""enums — current planner-visible axis vocabulary.

Calling spec:
    from ib.models.enums import Scene, ParticipationFrame, ...

Every planner/admissible axis level is a member here. Pipeline-owned acoustic
mix ingredients live in sound-plan/acoustic modules rather than in this enum
vocabulary.

Side effects: none.
"""

from __future__ import annotations

from enum import Enum


class _StrEnum(str, Enum):
    """str-valued enum whose str() is the bare value (stable for JSON/YAML)."""

    def __str__(self) -> str:
        return self.value


class Scene(_StrEnum):
    """Axis A — primary behavior-setting scene class (conditioner). 8 classes."""

    domestic_household = "domestic_household"
    work_professional = "work_professional"
    mobility_transport = "mobility_transport"
    public_civic = "public_civic"
    education_learning = "education_learning"
    commerce_service = "commerce_service"
    healthcare_caregiving_accessibility = "healthcare_caregiving_accessibility"
    leisure_media_social = "leisure_media_social"


class ParticipationFrame(_StrEnum):
    """Axis B — final-turn participation framework. 5 levels."""

    user_addressed = "user_addressed"
    not_addressed = "not_addressed"
    ambiguous_address = "ambiguous_address"
    new_participant_joins = "new_participant_joins"
    user_interrupted = "user_interrupted"


class AddresseeCue(_StrEnum):
    """Axis C — addressee cue (signal carrier, shortcut probe). 5 levels."""

    wake_word = "wake_word"
    pronoun_address = "pronoun_address"
    imperative_only = "imperative_only"
    declarative_actionable = "declarative_actionable"
    none = "none"


class ExpectedAction(_StrEnum):
    """Axis D — expected assistant action. 6 levels."""

    respond = "respond"
    ignore = "ignore"
    wait = "wait"
    clarify = "clarify"
    incorporate = "incorporate"
    refuse = "refuse"


class ReverbLevel(_StrEnum):
    """Axis E.1 — convolutional room impulse response."""

    dry = "dry"
    mild_room = "mild_room"
    large_hall = "large_hall"
    car_cabin = "car_cabin"
    bathroom = "bathroom"


class CompetingSpeechLevel(_StrEnum):
    """Axis E.2 — informational masking by intelligible non-target speech."""

    none = "none"
    bystander_voice = "bystander_voice"
    public_speech = "public_speech"
    tv_speech = "tv_speech"
    party_babble = "party_babble"


class ChannelLevel(_StrEnum):
    """Axis E.3 — capture and transmission path."""

    clean_close = "clean_close"
    far_field_static = "far_field_static"
    far_field_moving = "far_field_moving"
    phone_codec = "phone_codec"
    BT_codec = "BT_codec"
    compressed_low = "compressed_low"
