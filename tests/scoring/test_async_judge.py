"""ajudge_answers: concurrent answer judging matches the sync path."""

from __future__ import annotations

import asyncio
import json
from operator import itemgetter

from ib.llm import LlmConfig, LlmService
from ib.scoring.judge import ajudge_answers, judge_answers


def _write_jsonl(path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def _artifacts(tmp_path, n: int = 5):
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    rubrics = tmp_path / "rubrics.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "cell_id": f"c{i}",
                "expected_action": "respond",
                "rubric_ids": [f"c{i}:o1"],
                "answer_judge_inputs": [{"layer": "O1", "rubric_id": f"c{i}:o1"}],
            }
            for i in range(n)
        ],
    )
    _write_jsonl(
        predictions,
        [{"cell_id": f"c{i}", "predicted_action": "respond", "answer_text": f"scene=s{i}"} for i in range(n)],
    )
    _write_jsonl(
        rubrics,
        [
            {
                "cell_id": f"c{i}",
                "rubric_id": f"c{i}:o1",
                "layer": "O1",
                "atomic_criteria": [
                    {
                        "criterion_index": 0,
                        "polarity": "satisfy",
                        "criterion": f"scene=s{i}",
                    }
                ],
            }
            for i in range(n)
        ],
    )
    return manifest, predictions, rubrics


def test_ajudge_answers_matches_judge_answers(tmp_path, det_judge) -> None:
    manifest, predictions, rubrics = _artifacts(tmp_path, n=5)
    sync_result = judge_answers(manifest, predictions, rubrics, provider=det_judge)

    async def run():
        svc = LlmService(LlmConfig())
        try:
            return await ajudge_answers(
                manifest, predictions, rubrics, provider=det_judge, service=svc
            )
        finally:
            await svc.aclose()

    async_result = asyncio.run(run())
    # Judgments may complete in any order; compare as a set keyed by (cell, rubric).
    assert async_result["judgment_count"] == sync_result["judgment_count"] == 5
    assert async_result["passed_count"] == sync_result["passed_count"] == 5
    key = itemgetter("cell_id", "rubric_id")
    assert sorted(async_result["judgments"], key=key) == sorted(sync_result["judgments"], key=key)


class _FlakyJudge:
    name = "flaky"

    def judge(self, *, manifest_row: dict, prediction: dict, rubric: dict) -> dict:
        del manifest_row, prediction
        return _passing_payload(rubric)

    async def ajudge(self, *, manifest_row: dict, prediction: dict, rubric: dict, service=None) -> dict:
        del prediction, service
        if manifest_row["cell_id"] == "c1":
            raise asyncio.TimeoutError("boom")
        return _passing_payload(rubric)


def _passing_payload(rubric: dict) -> dict:
    atomic_results = [
        {
            "passed": True,
            "rationale": "ok",
        }
        for item in rubric["atomic_criteria"]
    ]
    return {
        "rationale": "ok",
        "atomic_results": atomic_results,
    }


def test_ajudge_answers_records_failed_judgment_for_timeout(tmp_path) -> None:
    manifest, predictions, rubrics = _artifacts(tmp_path, n=2)

    async def run():
        svc = LlmService(LlmConfig())
        try:
            return await ajudge_answers(
                manifest,
                predictions,
                rubrics,
                provider=_FlakyJudge(),
                service=svc,
                max_concurrency=1,
            )
        finally:
            await svc.aclose()

    result = asyncio.run(run())
    assert result["judgment_count"] == 2
    assert result["passed_count"] == 1
    assert result["errors"]
    assert "TimeoutError: boom" in result["errors"][0]
    failed = next(item for item in result["judgments"] if item["cell_id"] == "c1")
    assert failed["passed"] is False
    assert failed["atomic_passed_count"] == 0


def test_ajudge_answers_prints_progress(tmp_path, det_judge, capsys) -> None:
    manifest, predictions, rubrics = _artifacts(tmp_path, n=2)

    async def run():
        svc = LlmService(LlmConfig())
        try:
            return await ajudge_answers(
                manifest,
                predictions,
                rubrics,
                provider=det_judge,
                service=svc,
                progress_provider="answer_judge",
                progress_model="target-a",
                max_concurrency=1,
            )
        finally:
            await svc.aclose()

    result = asyncio.run(run())
    captured = capsys.readouterr()
    assert result["judgment_count"] == 2
    assert "ib eval: answer_judge:target-a case 1/2" in captured.err
    assert "ib eval: answer_judge:target-a case 2/2" in captured.err
