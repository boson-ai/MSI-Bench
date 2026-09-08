from __future__ import annotations

import wave
from io import BytesIO

import pytest


def canonical_wav_bytes(sample_rate: int = 24_000) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes((b"\x00\x00") * 16)
    return buffer.getvalue()


class DeterministicJudgeProvider:
    """Test stand-in: exact atomic rubric check over the predicted answer text."""

    name = "deterministic"

    def judge(self, *, manifest_row: dict, prediction: dict, rubric: dict) -> dict:
        answer_text = prediction.get("answer_text") or ""
        criteria = [item for item in rubric.get("atomic_criteria", []) if isinstance(item, dict)]
        atomic_results = [
            {
                "criterion_index": item["criterion_index"],
                "criterion": item["criterion"],
                "passed": _deterministic_atom_passed(item, answer_text),
                "rationale": "deterministic exact atomic check",
            }
            for item in criteria
        ]
        missing = [
            item["criterion"]
            for item in criteria
            if item.get("polarity", "satisfy") == "satisfy"
            and item["criterion"] not in answer_text
        ]
        violations = [
            item["criterion"]
            for item in criteria
            if item.get("polarity") == "avoid" and item["criterion"] in answer_text
        ]
        passed = not missing and not violations
        return {
            "passed": passed,
            "rationale": "deterministic exact rubric-atom check",
            "atomic_results": atomic_results,
            "missing": missing,
            "violations": violations,
        }

    async def ajudge(self, *, manifest_row: dict, prediction: dict, rubric: dict, service=None) -> dict:
        return self.judge(manifest_row=manifest_row, prediction=prediction, rubric=rubric)


def _deterministic_atom_passed(item: dict, answer_text: str) -> bool:
    criterion = item["criterion"]
    if item.get("polarity") == "avoid":
        return criterion not in answer_text
    return criterion in answer_text


@pytest.fixture
def det_judge() -> DeterministicJudgeProvider:
    return DeterministicJudgeProvider()
