"""labels — richer planner-owned target labels for manifest/evaluation layers."""

from __future__ import annotations

from ib.models.base import IbBaseModel
from ib.models.enums import ExpectedAction
from ib.planner.scenario import FrozenScenario


class PerceptionLabel(IbBaseModel):
    """What a model should perceive before deciding to act."""

    scene: str
    acoustic_conditions: dict[str, str]


class InteractionLabel(IbBaseModel):
    """Interaction target expected from the assistant."""

    expected_action: ExpectedAction
    expected_addressed_speaker: str | None
    participation_frame: str
    contract_required: bool = False


class AnswerLabel(IbBaseModel):
    """Answer-layer target fields consumed by rubric/judge evaluation."""

    answerable: bool
    rubric_ids: list[str]
    contract_text: str | None = None


class LabelBundle(IbBaseModel):
    """Planner-owned perception, interaction, and answer labels for one cell."""

    perception: PerceptionLabel
    interaction: InteractionLabel
    answer: AnswerLabel


def labels_for(frozen: FrozenScenario, rubric_ids: list[str]) -> LabelBundle:
    """Return the structured label bundle for one frozen scenario."""
    cell = frozen.cell
    answerable = cell.expected_action in {
        ExpectedAction.respond,
        ExpectedAction.incorporate,
        ExpectedAction.refuse,
    }
    return LabelBundle(
        perception=PerceptionLabel(
            scene=cell.scene.value,
            acoustic_conditions={
                "reverb": cell.reverb.value,
                "competing_speech": cell.competing_speech.value,
            },
        ),
        interaction=InteractionLabel(
            expected_action=cell.expected_action,
            expected_addressed_speaker=cell.expected_addressed_speaker,
            participation_frame=(
                cell.logic_contract.participation_frame.value
                if cell.logic_contract is not None
                else "user_addressed"
            ),
            contract_required=cell.contract is not None,
        ),
        answer=AnswerLabel(
            answerable=answerable,
            rubric_ids=sorted(rubric_ids),
            contract_text=cell.contract,
        ),
    )
