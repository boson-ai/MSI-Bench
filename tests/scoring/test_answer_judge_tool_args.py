"""Answer judging sends tool.call_args and semantic atoms to the LLM."""

from __future__ import annotations

import asyncio
import json

from ib.llm import LlmConfig, LlmService
from ib.scoring.judge import (
    ANSWER_JUDGE_POLICY_VERSION,
    _system_prompt,
    ajudge_answers,
    judge_answers,
)


class RecordingJudge:
    """Pass every rubric atom and retain the rubric sent by the orchestrator."""

    name = "recording"

    def __init__(self) -> None:
        self.rubrics: list[dict] = []

    def judge(self, *, manifest_row: dict, prediction: dict, rubric: dict) -> dict:
        del manifest_row, prediction
        self.rubrics.append(rubric)
        return _passing_payload(rubric)

    async def ajudge(
        self, *, manifest_row: dict, prediction: dict, rubric: dict, service=None
    ) -> dict:
        del service
        return self.judge(
            manifest_row=manifest_row,
            prediction=prediction,
            rubric=rubric,
        )


class ToolFailingJudge(RecordingJudge):
    """Fail the tool atom while passing every semantic atom."""

    def judge(self, *, manifest_row: dict, prediction: dict, rubric: dict) -> dict:
        del manifest_row, prediction
        self.rubrics.append(rubric)
        return {
            "rationale": "tool arguments are semantically wrong",
            "atomic_results": [
                {
                    "passed": atom.get("dimension") != "tool.call_args",
                    "rationale": "LLM tool decision",
                }
                for atom in rubric.get("atomic_criteria", [])
            ],
        }


class TimeoutJudge(RecordingJudge):
    async def ajudge(
        self, *, manifest_row: dict, prediction: dict, rubric: dict, service=None
    ) -> dict:
        del manifest_row, prediction, service
        self.rubrics.append(rubric)
        raise asyncio.TimeoutError("judge timed out")


def _passing_payload(rubric: dict) -> dict:
    return {
        "rationale": "rubric atoms passed",
        "atomic_results": [
            {"passed": True, "rationale": "atom pass"}
            for _atom in rubric.get("atomic_criteria", [])
        ],
    }


def _write_case(
    tmp_path,
    *,
    expected_calls: list[dict],
    predicted_calls: list[dict],
    atoms: list[dict],
    answer_text: str | None = "handled",
    explicit_validation: dict | None = None,
):
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    rubrics = tmp_path / "rubrics.jsonl"
    manifest_row = {
        "cell_id": "c1",
        "expected_action": "respond",
        "rubric_ids": ["c1:logic"],
        "answer_judge_inputs": [{"layer": "O1", "rubric_id": "c1:logic"}],
        "logic_expected_tool_calls": expected_calls,
    }
    if explicit_validation is not None:
        manifest_row["logic_tool_validation"] = explicit_validation
    _write_jsonl(manifest, [manifest_row])
    _write_jsonl(
        predictions,
        [
            {
                "cell_id": "c1",
                "predicted_action": "respond",
                "answer_text": answer_text,
                "tool_calls": predicted_calls,
            }
        ],
    )
    _write_jsonl(
        rubrics,
        [
            {
                "cell_id": "c1",
                "rubric_id": "c1:logic",
                "layer": "O1",
                "atomic_criteria": atoms,
            }
        ],
    )
    return manifest, predictions, rubrics


def _write_jsonl(path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _tool_atom(index: int = 0) -> dict:
    return {
        "criterion_index": index,
        "dimension": "tool.call_args",
        "criterion": "use the required tool and arguments",
    }


def _semantic_atom(index: int = 1) -> dict:
    return {
        "criterion_index": index,
        "dimension": "response.required_behavior",
        "criterion": "explain the outcome",
    }


def test_tool_atom_is_sent_to_llm_independently_of_explicit_validator(tmp_path) -> None:
    explicit = {
        "call_policy": "exact_multiset",
        "required_calls": [
            {
                "name": "update_itinerary",
                "arguments": {
                    "status": {
                        "op": "one_of",
                        "values": ["pending_revision", "on_hold"],
                    }
                },
            }
        ],
    }
    artifacts = _write_case(
        tmp_path,
        expected_calls=[{"name": "update_itinerary", "arguments": {"status": "pending_revision"}}],
        predicted_calls=[{"name": "update_itinerary", "arguments": {"status": "on_hold"}}],
        atoms=[_tool_atom(), _semantic_atom()],
        explicit_validation=explicit,
    )
    provider = RecordingJudge()

    result = judge_answers(*artifacts, provider=provider)

    judgment = result["judgments"][0]
    assert result["policy_version"] == ANSWER_JUDGE_POLICY_VERSION
    assert judgment["passed"] is True
    assert [atom["dimension"] for atom in provider.rubrics[0]["atomic_criteria"]] == [
        "tool.call_args",
        "response.required_behavior",
    ]
    assert [atom["passed"] for atom in judgment["atomic_results"]] == [True, True]
    assert "tool_validation" not in judgment


def test_answer_judge_prompt_keeps_atom_decisions_and_validator_independent() -> None:
    prompt = _system_prompt()

    assert "Judge every item independently" in prompt
    assert "For `tool.call_args`" in prompt
    assert "Do not substitute or infer the separate deterministic validator result" in prompt


def test_llm_tool_atom_failure_controls_answer_judge_pass(tmp_path) -> None:
    artifacts = _write_case(
        tmp_path,
        expected_calls=[{"name": "save_order", "arguments": {"holder": "Alice"}}],
        predicted_calls=[{"name": "save_order", "arguments": {"holder": "Bob"}}],
        atoms=[_tool_atom(), _semantic_atom()],
    )

    result = judge_answers(*artifacts, provider=ToolFailingJudge())

    judgment = result["judgments"][0]
    assert judgment["passed"] is False
    assert judgment["atomic_passed_count"] == 1
    assert judgment["atomic_results"][0]["passed"] is False
    assert judgment["atomic_results"][1]["passed"] is True


def test_semantic_only_rubric_leaves_tool_checks_to_manifest_validation(tmp_path) -> None:
    artifacts = _write_case(
        tmp_path,
        expected_calls=[{"name": "save_order", "arguments": {"holder": "Alice"}}],
        predicted_calls=[{"name": "save_order", "arguments": {"holder": "Bob"}}],
        atoms=[_semantic_atom(index=0)],
    )
    provider = RecordingJudge()

    result = judge_answers(*artifacts, provider=provider)

    judgment = result["judgments"][0]
    assert judgment["passed"] is True
    assert "tool_validation" not in judgment
    assert provider.rubrics[0]["atomic_criteria"] == [_semantic_atom(index=0)]


def test_empty_answer_keeps_no_call_atom_independent_from_semantic_guard(tmp_path) -> None:
    artifacts = _write_case(
        tmp_path,
        expected_calls=[],
        predicted_calls=[],
        atoms=[_tool_atom(), _semantic_atom()],
        answer_text=None,
    )

    result = judge_answers(*artifacts, provider=RecordingJudge())

    atoms = result["judgments"][0]["atomic_results"]
    assert [atom["passed"] for atom in atoms] == [True, False]
    assert "empty answer cannot satisfy" in atoms[1]["rationale"]


def test_tool_only_rubric_calls_model_judge(tmp_path) -> None:
    artifacts = _write_case(
        tmp_path,
        expected_calls=[],
        predicted_calls=[],
        atoms=[_tool_atom()],
        answer_text=None,
    )

    provider = RecordingJudge()
    result = judge_answers(*artifacts, provider=provider)

    assert result["pass_rate"] == 1.0
    assert result["judgments"][0]["atomic_results"][0]["passed"] is True
    assert provider.rubrics[0]["atomic_criteria"] == [_tool_atom()]


def test_tool_atom_does_not_require_validator_contract(tmp_path) -> None:
    artifacts = _write_case(
        tmp_path,
        expected_calls=[],
        predicted_calls=[],
        atoms=[_tool_atom()],
    )
    manifest = artifacts[0]
    row = json.loads(manifest.read_text(encoding="utf-8"))
    row.pop("logic_expected_tool_calls")
    _write_jsonl(manifest, [row])
    provider = RecordingJudge()

    result = judge_answers(*artifacts, provider=provider)

    assert result["pass_rate"] == 1.0
    assert provider.rubrics[0]["atomic_criteria"] == [_tool_atom()]


def test_async_timeout_fails_tool_and_semantic_atoms(tmp_path) -> None:
    artifacts = _write_case(
        tmp_path,
        expected_calls=[],
        predicted_calls=[],
        atoms=[_semantic_atom(index=0), _tool_atom(index=1)],
        answer_text="I cannot perform that action.",
    )
    provider = TimeoutJudge()

    async def run() -> dict:
        service = LlmService(LlmConfig())
        try:
            return await ajudge_answers(
                *artifacts,
                provider=provider,
                service=service,
                max_concurrency=1,
            )
        finally:
            await service.aclose()

    result = asyncio.run(run())

    atoms = result["judgments"][0]["atomic_results"]
    assert result["errors"] and "TimeoutError" in result["errors"][0]
    assert [atom["dimension"] for atom in atoms] == [
        "response.required_behavior",
        "tool.call_args",
    ]
    assert [atom["passed"] for atom in atoms] == [False, False]
