"""scenario — the atomic benchmark cell schema.

Calling spec:
    Cell(cell_id=..., scene=..., competing_speech=..., ...)

One Cell is one audio cell + one assistant decision target. The scene-logic
contract is the source of truth for interaction semantics; the Cell keeps the
two acoustic fields the renderer consumes (`reverb`, `competing_speech`) plus
pairing metadata (sibling_group, plane_id, intervened_axis). Retired legacy
fields from the participation-frame plane are stripped on load so historical
frozen artifacts keep validating.

Side effects: none.
"""

from __future__ import annotations

from pydantic import model_validator

from ib.models.logic import LogicContract
from ib.models.base import IbBaseModel
from ib.models.enums import (
    CompetingSpeechLevel,
    ExpectedAction,
    ReverbLevel,
    Scene,
)

_RETIRED_CELL_KEYS = (
    "participation_frame",
    "addressee_cue",
    "channel",
    "authorization",
    "is_off_prior",
)


class Cell(IbBaseModel):
    """One atomic benchmark row."""

    cell_id: str
    scene: Scene
    reverb: ReverbLevel
    competing_speech: CompetingSpeechLevel
    expected_action: ExpectedAction
    expected_addressed_speaker: str | None = None
    contract: str | None = None
    logic_contract: LogicContract | None = None
    # pairing / plane metadata (relational structure over cells)
    sibling_group: str
    plane_id: str
    intervened_axis: str

    @model_validator(mode="before")
    @classmethod
    def _strip_retired_fields(cls, data):
        """Ignore cell fields retired with the participation-frame plane."""
        if not isinstance(data, dict):
            return data
        if not any(key in data for key in _RETIRED_CELL_KEYS):
            return data
        migrated = dict(data)
        for key in _RETIRED_CELL_KEYS:
            migrated.pop(key, None)
        return migrated
