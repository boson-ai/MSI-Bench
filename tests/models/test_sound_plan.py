"""Acoustic realism SoundPlan schema tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ib.models.sound_plan import SoundPlan, SoundPlanEvent, SoundPlanLine


def test_sound_plan_accepts_ambient_and_event_kinds() -> None:
    plan = SoundPlan(
        mode="tta",
        scene_ambient=SoundPlanEvent(
            event_id="scene_car",
            kind="ambient",
            description="inside a moving car with road noise",
            anchor_policy="scene",
            loudness_class="medium",
        ),
        line_plans=[
            SoundPlanLine(
                line_index=0,
                clean_text="I need to go",
                events=[
                    SoundPlanEvent(
                        event_id="honk_1",
                        kind="event",
                        description="brief exterior car horn",
                        anchor_policy="infix",
                    )
                ],
            )
        ],
    )

    assert plan.schema_version == "ib.sound_plan.v1"
    assert plan.scene_ambient is not None
    assert plan.scene_ambient.kind == "ambient"
    assert plan.line_plans[0].events[0].kind == "event"


def test_sound_plan_rejects_bed_kind_and_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        SoundPlanEvent(event_id="old_name", kind="bed", description="engine")

    with pytest.raises(ValidationError):
        SoundPlanEvent(
            event_id="honk_1",
            kind="event",
            description="honk",
            hidden_prompt="do not allow typo fields",
        )
