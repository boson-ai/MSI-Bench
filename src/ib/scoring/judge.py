"""judge — optional answer-layer judge path over planner rubrics."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from ib.audio.acoustic_markup import clean_spoken_text
from ib.llm import LlmProviderError, LlmRequest, complete_json, render_prompt
from ib.llm.openrouter import (
    OPENROUTER_BASE_URL,
    OPENROUTER_DEFAULT_MODEL,
    OPENROUTER_PROVIDER,
)
from ib.llm.qwen import (
    DEFAULT_LLM_PROVIDER,
    QWEN35_BASE_URL,
    QWEN35_MODEL,
    QWEN35_MAX_OUTPUT_TOKENS,
    QWEN35_PROVIDER,
    QWEN35_THINKING_PROVIDER,
    qwen_request_options,
)
from ib.predictions import read_predictions
from ib.scoring.progress import print_eval_case_progress
from ib.scoring.smoke import _read_rows, _rubric_references
from ib.tool_call_rubric import TOOL_CALL_ARGS_DIMENSION

if TYPE_CHECKING:
    from ib.llm import LlmService

JUDGE_MANIFEST_FIELDS = (
    "scene",
    "expected_action",
    "logic_pattern",
    "contract",
)
JUDGE_PREDICTION_FIELDS = (
    "predicted_action",
    "answer_text",
    "tool_calls",
    "confidence",
)
JUDGE_RUBRIC_ATOMIC_FIELDS = ("polarity", "dimension", "criterion")
DEFAULT_OPENAI_JUDGE_MODEL = "gpt-5.5"
ANSWER_JUDGE_POLICY_VERSION = "ib.answer_judge.llm_tool_args.v1"
_DEFAULT_ASYNC_JUDGE_CONCURRENCY = 4
_MAX_CONTEXT_TURNS = 8


class JudgeProviderError(ValueError):
    """Raised when an optional judge provider is unavailable or invalid."""


class JudgeProvider(Protocol):
    name: str

    def judge(self, *, manifest_row: dict, prediction: dict, rubric: dict) -> dict:
        """Return one answer-layer judgment payload."""
        ...

    async def ajudge(
        self, *, manifest_row: dict, prediction: dict, rubric: dict, service: "LlmService"
    ) -> dict:
        """Async variant of ``judge`` routed through the shared LlmService."""
        ...


class OpenAiJudgeProvider:
    """Credential-gated OpenAI answer judge adapter."""

    name = "openai"

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_JUDGE_MODEL,
        api_key: str | None = None,
        client=None,
        *,
        name: str = "openai",
        base_url: str | None = None,
        extra_body: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.api_key = api_key
        self.client = client
        self.base_url = base_url
        self.extra_body = extra_body
        self.max_tokens = max_tokens

    def with_model(self, model: str) -> OpenAiJudgeProvider:
        """Return the same provider endpoint/options with a different model."""
        return OpenAiJudgeProvider(
            model=model,
            api_key=self.api_key,
            client=self.client,
            name=self.name,
            base_url=self.base_url,
            extra_body=self.extra_body,
            max_tokens=self.max_tokens,
        )

    def _llm_request(self, *, manifest_row: dict, prediction: dict, rubric: dict) -> LlmRequest:
        return LlmRequest(
            provider=self.name,
            model=self.model,
            messages=[
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": _user_prompt(manifest_row, prediction, rubric)},
            ],
            response_format=_response_format_for(self.name),
            max_tokens=self.max_tokens,
            api_key=self.api_key,
            base_url=self.base_url,
            extra_body=self.extra_body,
            client=self.client,
        )

    def judge(self, *, manifest_row: dict, prediction: dict, rubric: dict) -> dict:
        try:
            payload = complete_json(
                self._llm_request(
                    manifest_row=manifest_row,
                    prediction=prediction,
                    rubric=rubric,
                )
            )
        except LlmProviderError as exc:
            raise JudgeProviderError(str(exc)) from exc
        return _validate_judgment(payload, rubric, prediction=prediction)

    async def ajudge(
        self, *, manifest_row: dict, prediction: dict, rubric: dict, service: "LlmService"
    ) -> dict:
        try:
            payload = await service.acomplete_json(
                self._llm_request(
                    manifest_row=manifest_row,
                    prediction=prediction,
                    rubric=rubric,
                ),
                module="scoring.judge",
            )
        except LlmProviderError as exc:
            raise JudgeProviderError(str(exc)) from exc
        return _validate_judgment(payload, rubric, prediction=prediction)


def judge_model_for_provider(name: str, model: str | None = None) -> str:
    """Return the effective model used by a judge provider."""
    if model is not None:
        return model
    if name == "openai":
        return DEFAULT_OPENAI_JUDGE_MODEL
    if name in {QWEN35_PROVIDER, QWEN35_THINKING_PROVIDER}:
        return QWEN35_MODEL
    if name == OPENROUTER_PROVIDER:
        return OPENROUTER_DEFAULT_MODEL
    raise JudgeProviderError(f"unknown judge provider {name!r}")


def provider_for(name: str, *, model: str | None = None, client=None) -> JudgeProvider:
    if name == "openai":
        return OpenAiJudgeProvider(model=judge_model_for_provider(name, model), client=client)
    if name == QWEN35_PROVIDER:
        provider = OpenAiJudgeProvider(
            model=QWEN35_MODEL,
            client=client,
            name=QWEN35_PROVIDER,
            base_url=QWEN35_BASE_URL,
            extra_body=qwen_request_options(enable_thinking=False),
            max_tokens=QWEN35_MAX_OUTPUT_TOKENS,
        )
        return provider.with_model(model) if model is not None else provider
    if name == QWEN35_THINKING_PROVIDER:
        provider = OpenAiJudgeProvider(
            model=QWEN35_MODEL,
            client=client,
            name=QWEN35_THINKING_PROVIDER,
            base_url=QWEN35_BASE_URL,
            extra_body=qwen_request_options(enable_thinking=True),
            max_tokens=QWEN35_MAX_OUTPUT_TOKENS,
        )
        return provider.with_model(model) if model is not None else provider
    if name == OPENROUTER_PROVIDER:
        return OpenAiJudgeProvider(
            model=judge_model_for_provider(name, model),
            client=client,
            name=OPENROUTER_PROVIDER,
            base_url=OPENROUTER_BASE_URL,
        )
    raise JudgeProviderError(f"unknown judge provider {name!r}")


def _response_format_for(provider_name: str) -> dict[str, str] | None:
    """Return JSON mode unless Qwen thinking would enter its slow path."""
    if provider_name == QWEN35_THINKING_PROVIDER:
        return None
    return {"type": "json_object"}


def judge_answers(
    manifest_path: str | Path,
    predictions_path: str | Path,
    rubrics_path: str | Path,
    *,
    provider: str | JudgeProvider = DEFAULT_LLM_PROVIDER,
    model: str | None = None,
) -> dict:
    """Judge answerable manifest rows using supplied predictions and planner rubrics."""
    manifest = _read_rows(manifest_path)
    predictions = {
        row.cell_id: row.model_dump(mode="json") for row in read_predictions(predictions_path)
    }
    rubrics = {row["rubric_id"]: row for row in _read_rows(rubrics_path)}
    if isinstance(provider, str):
        resolved = (
            provider_for(provider, model=model) if model is not None else provider_for(provider)
        )
    else:
        resolved = provider
    judgments = []
    errors = []
    for row in manifest:
        cell_id = row.get("cell_id")
        if not isinstance(cell_id, str):
            errors.append("manifest row has invalid cell_id")
            continue
        refs = sorted(set(_rubric_references(row)))
        if not refs:
            continue
        prediction = predictions.get(cell_id)
        if prediction is None:
            errors.append(f"missing prediction for answerable row {cell_id}")
            continue
        for rubric_id in refs:
            rubric = rubrics.get(rubric_id)
            if rubric is None:
                errors.append(f"missing rubric {rubric_id}")
                continue
            if _prediction_has_format_error(prediction):
                payload = _failed_judgment_payload(rubric, _FORMAT_ERROR_RATIONALE)
            else:
                payload = resolved.judge(
                    manifest_row=row,
                    prediction=prediction,
                    rubric=rubric,
                )
            judgment = _validate_judgment(
                payload,
                rubric,
                prediction=prediction,
            )
            judgments.append(
                {
                    "cell_id": cell_id,
                    "rubric_id": rubric_id,
                    "provider": resolved.name,
                    **judgment,
                }
            )
    passed = sum(1 for item in judgments if item["passed"])
    atomic_count = sum(item["atomic_count"] for item in judgments)
    atomic_passed_count = sum(item["atomic_passed_count"] for item in judgments)
    return {
        "schema_version": "ib.answer_judge.v1",
        "policy_version": ANSWER_JUDGE_POLICY_VERSION,
        "provider": resolved.name,
        "judgment_count": len(judgments),
        "passed_count": passed,
        "pass_rate": passed / len(judgments) if judgments else 1.0,
        "atomic_count": atomic_count,
        "atomic_passed_count": atomic_passed_count,
        "atomic_pass_rate": atomic_passed_count / atomic_count if atomic_count else 1.0,
        "judgments": judgments,
        "errors": errors,
    }


async def ajudge_answers(
    manifest_path: str | Path,
    predictions_path: str | Path,
    rubrics_path: str | Path,
    *,
    provider: str | JudgeProvider = DEFAULT_LLM_PROVIDER,
    model: str | None = None,
    service: "LlmService",
    progress_provider: str | None = None,
    progress_model: str | None = None,
    max_concurrency: int = _DEFAULT_ASYNC_JUDGE_CONCURRENCY,
) -> dict:
    """Async sibling of ``judge_answers`` — judges every (row, rubric) pair with bounded concurrency."""
    manifest = _read_rows(manifest_path)
    predictions = {
        row.cell_id: row.model_dump(mode="json") for row in read_predictions(predictions_path)
    }
    rubrics = {row["rubric_id"]: row for row in _read_rows(rubrics_path)}
    if isinstance(provider, str):
        resolved = (
            provider_for(provider, model=model) if model is not None else provider_for(provider)
        )
    else:
        resolved = provider

    work: list[tuple[str, str, dict, dict, dict]] = []
    errors: list[str] = []
    for row in manifest:
        cell_id = row.get("cell_id")
        if not isinstance(cell_id, str):
            errors.append("manifest row has invalid cell_id")
            continue
        refs = sorted(set(_rubric_references(row)))
        if not refs:
            continue
        prediction = predictions.get(cell_id)
        if prediction is None:
            errors.append(f"missing prediction for answerable row {cell_id}")
            continue
        for rubric_id in refs:
            rubric = rubrics.get(rubric_id)
            if rubric is None:
                errors.append(f"missing rubric {rubric_id}")
                continue
            work.append((cell_id, rubric_id, row, prediction, rubric))

    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def run_one(position: int, item: tuple[str, str, dict, dict, dict]):
        cell_id, rubric_id, row, prediction, rubric = item
        if _prediction_has_format_error(prediction):
            return position, item, _failed_judgment_payload(rubric, _FORMAT_ERROR_RATIONALE), None
        async with semaphore:
            try:
                payload = await resolved.ajudge(
                    manifest_row=row,
                    prediction=prediction,
                    rubric=rubric,
                    service=service,
                )
                return position, item, payload, None
            except Exception as exc:  # noqa: BLE001 — one judge failure should not abort the eval
                message = (
                    f"answer judge failed for {cell_id} {rubric_id}: {type(exc).__name__}: {exc}"
                )
                return (
                    position,
                    item,
                    _failed_judgment_payload(rubric, message),
                    message,
                )

    completed = []
    tasks = [asyncio.create_task(run_one(position, item)) for position, item in enumerate(work)]
    for done_count, task in enumerate(asyncio.as_completed(tasks), start=1):
        position, item, payload, error = await task
        cell_id = item[0]
        if progress_provider is not None:
            print_eval_case_progress(
                progress_provider,
                progress_model or resolved.name,
                cell_id,
                done_count,
                len(work),
            )
        if error is not None:
            errors.append(error)
        completed.append((position, item, payload))

    completed.sort(key=lambda item: item[0])
    judgments = [
        {
            "cell_id": cell_id,
            "rubric_id": rubric_id,
            "provider": resolved.name,
            **_validate_judgment(
                payload,
                _rubric,
                prediction=prediction,
            ),
        }
        for (
            _position,
            (cell_id, rubric_id, _row, prediction, _rubric),
            payload,
        ) in completed
    ]
    passed = sum(1 for item in judgments if item["passed"])
    atomic_count = sum(item["atomic_count"] for item in judgments)
    atomic_passed_count = sum(item["atomic_passed_count"] for item in judgments)
    return {
        "schema_version": "ib.answer_judge.v1",
        "policy_version": ANSWER_JUDGE_POLICY_VERSION,
        "provider": resolved.name,
        "judgment_count": len(judgments),
        "passed_count": passed,
        "pass_rate": passed / len(judgments) if judgments else 1.0,
        "atomic_count": atomic_count,
        "atomic_passed_count": atomic_passed_count,
        "atomic_pass_rate": atomic_passed_count / atomic_count if atomic_count else 1.0,
        "judgments": judgments,
        "errors": errors,
    }


_FORMAT_ERROR_RATIONALE = (
    "format error: prediction recorded a provider_error, so every atom fails without judging"
)


def _prediction_has_format_error(prediction: dict) -> bool:
    """Return whether a prediction row degraded to a provider format error."""
    metadata = prediction.get("metadata")
    return isinstance(metadata, dict) and isinstance(metadata.get("provider_error"), dict)


def _failed_judgment_payload(rubric: dict, message: str) -> dict:
    specs = _atomic_specs(rubric)
    return {
        "passed": False,
        "rationale": message,
        "atomic_results": [
            {
                "criterion_index": spec["criterion_index"],
                "dimension": spec["dimension"],
                "criterion": spec["criterion"],
                "passed": False,
                "rationale": message,
            }
            for spec in specs
        ],
    }


def _validate_judgment(
    payload: dict,
    rubric: dict | None = None,
    *,
    prediction: dict | None = None,
) -> dict:
    if not isinstance(payload, dict):
        raise JudgeProviderError("judge payload must be an object")
    atomic_results = _validate_atomic_results(payload.get("atomic_results"), rubric)
    if prediction is not None:
        atomic_results = _apply_empty_answer_guard(atomic_results, prediction)
    atomic_count = len(atomic_results)
    atomic_passed_count = sum(1 for item in atomic_results if item["passed"])
    atomic_pass_rate = atomic_passed_count / atomic_count if atomic_count else 1.0
    passed = atomic_passed_count == atomic_count
    rationale = payload.get("rationale", "")
    if not isinstance(rationale, str):
        raise JudgeProviderError("judge rationale must be a string when present")
    return {
        "passed": passed,
        "rationale": rationale,
        "atomic_count": atomic_count,
        "atomic_passed_count": atomic_passed_count,
        "atomic_pass_rate": atomic_pass_rate,
        "atomic_results": atomic_results,
    }


def _apply_empty_answer_guard(atomic_results: list[dict], prediction: dict) -> list[dict]:
    """Fail non-tool atoms when the model emitted no answer text or tool calls."""
    answer_text = prediction.get("answer_text")
    tool_calls = prediction.get("tool_calls")
    has_text = isinstance(answer_text, str) and bool(answer_text.strip())
    has_tool_calls = isinstance(tool_calls, list) and bool(tool_calls)
    if has_text or has_tool_calls:
        return atomic_results
    return [
        item
        if item.get("dimension") == TOOL_CALL_ARGS_DIMENSION
        else {
            **item,
            "passed": False,
            "rationale": "deterministic guard: empty answer cannot satisfy answer-layer rubric",
        }
        for item in atomic_results
    ]


def _validate_atomic_results(raw: Any, rubric: dict | None) -> list[dict]:
    expected = _atomic_specs(rubric or {})
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        raise JudgeProviderError("judge atomic_results must be a list when present")
    if not expected:
        return []
    aligned = []
    for index, spec in enumerate(expected):
        if index >= len(raw):
            aligned.append(
                {
                    "criterion_index": spec["criterion_index"],
                    "dimension": spec["dimension"],
                    "criterion": spec["criterion"],
                    "passed": False,
                    "rationale": "judge omitted this atomic criterion",
                }
            )
            continue
        aligned.append(_validate_atomic_decision(raw[index], spec))
    return aligned


def _validate_atomic_decision(item: Any, spec: dict) -> dict:
    if not isinstance(item, dict):
        raise JudgeProviderError("judge atomic_results entries must be objects")
    passed = item.get("passed")
    if not isinstance(passed, bool):
        raise JudgeProviderError("atomic passed must be boolean")
    rationale = item.get("rationale", "")
    if not isinstance(rationale, str):
        raise JudgeProviderError("atomic rationale must be a string when present")
    return {
        "criterion_index": spec["criterion_index"],
        "dimension": spec["dimension"],
        "criterion": spec["criterion"],
        "passed": passed,
        "rationale": rationale,
    }


def _atomic_specs(rubric: dict) -> list[dict]:
    raw_items = rubric.get("atomic_criteria")
    if raw_items is None and (
        rubric.get("must_include") is not None or rubric.get("must_not_include") is not None
    ):
        raise JudgeProviderError(
            "legacy must_include/must_not_include judge rubrics are no longer supported; "
            "use atomic_criteria"
        )
    if raw_items is None:
        raw_items = []
    if not isinstance(raw_items, list):
        raise JudgeProviderError("rubric atomic_criteria must be a list")
    specs = []
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            raise JudgeProviderError("rubric atomic_criteria entries must be objects")
        raw_index = item.get("criterion_index", index)
        if not isinstance(raw_index, int) or raw_index < 0:
            raise JudgeProviderError(
                "rubric atomic_criteria criterion_index must be a non-negative integer"
            )
        criterion = item.get("criterion")
        if not isinstance(criterion, str) or not criterion:
            raise JudgeProviderError("rubric atomic_criteria criterion must be a non-empty string")
        dimension = item.get("dimension", "legacy.unspecified")
        if not isinstance(dimension, str) or not dimension:
            raise JudgeProviderError("rubric atomic_criteria dimension must be a non-empty string")
        specs.append(
            {
                "criterion_index": raw_index,
                "dimension": dimension,
                "criterion": criterion,
            }
        )
    return specs


def _prompt(manifest_row: dict, prediction: dict, rubric: dict) -> str:
    return "\n\n".join([_system_prompt(), _user_prompt(manifest_row, prediction, rubric)])


def _system_prompt() -> str:
    return render_prompt("evaluation/judge_system.j2", {})


def _user_prompt(manifest_row: dict, prediction: dict, rubric: dict) -> str:
    return render_prompt(
        "evaluation/judge_user.j2",
        {
            "manifest_json": json.dumps(
                _judge_manifest_view(manifest_row), sort_keys=True, ensure_ascii=False
            ),
            "prediction_json": json.dumps(
                _judge_prediction_view(prediction), sort_keys=True, ensure_ascii=False
            ),
            "rubric_json": json.dumps(
                _judge_rubric_view(rubric), sort_keys=True, ensure_ascii=False
            ),
        },
    )


def _judge_manifest_view(manifest_row: dict) -> dict:
    """Return non-oracle manifest context for answer-only judging."""
    view = {key: manifest_row[key] for key in JUDGE_MANIFEST_FIELDS if key in manifest_row}
    visible_context_lines = _judge_visible_context_lines(manifest_row)
    if visible_context_lines:
        view["visible_context_lines"] = visible_context_lines
    return view


def _judge_visible_context_lines(manifest_row: dict) -> list[dict]:
    """Return visible dialogue lines without acoustic or speech-control markers."""
    explicit = manifest_row.get("visible_context_lines")
    if isinstance(explicit, list):
        return _clean_visible_context_entries(explicit)
    logic_contract = manifest_row.get("logic_contract")
    if isinstance(logic_contract, dict) and isinstance(logic_contract.get("lines"), list):
        return _clean_visible_context_entries(logic_contract["lines"])
    return _visible_context_from_turn_columns(manifest_row)


def _clean_visible_context_entries(entries: list) -> list[dict]:
    cleaned = []
    for index, entry in enumerate(entries, start=1):
        line = _clean_visible_context_entry(entry, default_line_id=f"L{index}")
        if line:
            cleaned.append(line)
    return cleaned


def _clean_visible_context_entry(entry: Any, *, default_line_id: str) -> dict:
    if isinstance(entry, str):
        text = clean_spoken_text(entry)
        return {"line_id": default_line_id, "text": text} if text else {}
    if not isinstance(entry, dict):
        return {}
    raw_text = entry.get("text", entry.get("transcript"))
    if not isinstance(raw_text, str):
        return {}
    text = clean_spoken_text(raw_text)
    if not text:
        return {}
    line: dict[str, Any] = {"line_id": entry.get("line_id", default_line_id)}
    for key in ("speaker", "addressed_to"):
        value = entry.get(key)
        if isinstance(value, str) and value:
            line[key] = value
    line["text"] = text
    return line


def _visible_context_from_turn_columns(manifest_row: dict) -> list[dict]:
    lines = []
    for index in range(1, _MAX_CONTEXT_TURNS + 1):
        for role in ("user", "assistant"):
            key = f"{role}_turn_{index}_transcript"
            text = manifest_row.get(key)
            if not isinstance(text, str):
                continue
            cleaned = clean_spoken_text(text)
            if cleaned:
                lines.append({"line_id": key.removesuffix("_transcript"), "text": cleaned})
    return lines


def _judge_rubric_view(rubric: dict) -> dict:
    """Return only semantic rubric fields needed by the answer judge."""
    view = {}
    instruction = rubric.get("judge_instruction")
    if isinstance(instruction, str) and instruction:
        view["judge_instruction"] = instruction
    atoms = []
    raw_atoms = rubric.get("atomic_criteria")
    if isinstance(raw_atoms, list):
        for item in raw_atoms:
            if not isinstance(item, dict):
                continue
            atom = {
                key: item[key]
                for key in JUDGE_RUBRIC_ATOMIC_FIELDS
                if isinstance(item.get(key), str) and item[key]
            }
            if atom:
                atoms.append(atom)
    view["atomic_criteria"] = atoms
    return view


def _judge_prediction_view(prediction: dict) -> dict:
    """Return only model-emitted fields the answer judge may score."""
    return {key: prediction[key] for key in JUDGE_PREDICTION_FIELDS if key in prediction}
