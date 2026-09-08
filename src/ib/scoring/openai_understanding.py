"""openai_understanding — generate prediction rows from OpenAI audio understanding.

Calling spec:
    run_openai_understanding_predictions(
        manifest, output, model, limit=None, client=None, tool_protocol="prompt_json"
    ) -> dict

Inputs are benchmark manifest rows whose audio paths are relative to the run root.
Outputs are strict PredictionRow JSONL rows consumable by ib eval / manifest_eval.

Side effects: may call the OpenAI API and writes prediction and sibling
raw-response JSONL artifacts.
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any

from ib.io import canonical_json, write_jsonl
from ib.llm import LlmProviderError, parse_json_object, render_prompt
from ib.predictions import PredictionAction, PredictionRow, read_predictions
from ib.scoring.multiturn_audio import (
    AudioHistoryMessage,
    history_messages,
    setup_prompt_for,
)
from ib.scoring.native_tool_schema import (
    NativeToolBundle,
    build_native_tool_bundle,
    openai_chat_tools,
)
from ib.scoring.progress import print_eval_case_progress
from ib.scoring.raw_response_io import append_raw_response, raw_response_path
from ib.scoring.smoke import _read_rows

OPENAI_UNDERSTANDING_PROVIDER = "openai"
_ACTION_VALUES = tuple(action.value for action in PredictionAction)
_OPENAI_MODEL_ALIASES = {
    "gptaudio1.5": "gpt-audio-1.5",
    "gptaudio-1.5": "gpt-audio-1.5",
    "gpt-audio1.5": "gpt-audio-1.5",
}


class UnderstandingProviderError(ValueError):
    """Raised when an understanding provider cannot generate valid predictions."""


class PredictionFormatError(UnderstandingProviderError):
    """Raised when a model response cannot be parsed into a valid prediction.

    Distinct from transport/setup failures: format errors are the model's own
    output failing the schema, so eval loops record them as a scored miss
    instead of aborting the run.
    """


def run_openai_understanding_predictions(
    manifest: str | Path,
    output: str | Path,
    *,
    model: str | None = None,
    limit: int | None = None,
    client: Any | None = None,
    tool_protocol: str = "prompt_json",
) -> dict[str, Any]:
    """Generate predictions for up to ``limit`` manifest rows using OpenAI audio input."""
    if not model:
        raise UnderstandingProviderError("OpenAI understanding eval requires explicit model")
    raw_output = raw_response_path(output)
    rows = _limited_rows(_read_rows(manifest), limit)
    provider = OpenAiUnderstandingProvider(
        model=_normalize_openai_audio_model(model),
        client=client,
        tool_protocol=tool_protocol,
        raw_response_output=raw_output,
    )
    predictions = _predict_with_resume(provider, rows, Path(manifest), Path(output))
    write_jsonl(output, predictions)
    return {
        "schema_version": "ib.openai_understanding.v1",
        "provider": OPENAI_UNDERSTANDING_PROVIDER,
        "model": provider.model,
        "manifest": str(manifest),
        "predictions": str(output),
        "limit": limit,
        "predicted_count": len(predictions),
        "llm_call_count": provider.call_count,
        "usage": provider.usage,
        "tool_protocol": provider.tool_protocol,
        "raw_responses": str(raw_output),
    }


class OpenAiUnderstandingProvider:
    """Chat Completions adapter for audio-in/text-out MSI-Bench predictions."""

    def __init__(
        self,
        *,
        model: str,
        client: Any | None = None,
        api_key: str | None = None,
        tool_protocol: str = "prompt_json",
        raw_response_output: Path | None = None,
    ) -> None:
        if tool_protocol not in {"prompt_json", "native"}:
            raise ValueError(f"unsupported tool protocol: {tool_protocol!r}")
        self.model = model
        self.client = client
        self.api_key = api_key
        self.tool_protocol = tool_protocol
        self.raw_response_output = raw_response_output
        self.call_count = 0
        self.usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        self._use_response_format = True

    def predict(self, manifest_row: dict[str, Any], *, manifest_path: Path) -> PredictionRow:
        """Return one strict PredictionRow for a manifest row."""
        cell_id = _required_str(manifest_row, "cell_id")
        audio_paths = _audio_paths_for(manifest_row, manifest_path)
        if not audio_paths:
            raise UnderstandingProviderError(f"manifest row {cell_id!r} has no audio paths")
        native_tools = self.tool_protocol == "native"
        native_bundle = (
            build_native_tool_bundle(manifest_row.get("logic_available_functions"))
            if native_tools
            else NativeToolBundle(declarations=[], name_map={})
        )
        system_prompt = _system_prompt_with_setup(manifest_row, native_tools=native_tools)
        user_prompt = _json_user_prompt(manifest_row, audio_paths)
        request = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}]
            + _openai_history_messages(manifest_row, audio_paths),
            "max_completion_tokens": 700,
            "temperature": 0,
        }
        if native_tools and native_bundle.declarations:
            request["tools"] = openai_chat_tools(native_bundle)
            request["tool_choice"] = "auto"
        started = time.perf_counter()
        response = self._create_completion(request)
        completion_time_s = round(time.perf_counter() - started, 3)
        self.call_count += 1
        self._record_usage(response)
        if self.raw_response_output is not None:
            append_raw_response(
                self.raw_response_output,
                cell_id=cell_id,
                provider=OPENAI_UNDERSTANDING_PROVIDER,
                model=self.model,
                eval_mode="turn_based",
                phase="prediction",
                response=response,
                metadata={"tool_protocol": self.tool_protocol},
            )
        payload = _prediction_payload_for_response(
            response,
            cell_id,
            native_bundle=native_bundle if native_tools else None,
        )
        return PredictionRow(
            cell_id=cell_id,
            predicted_action=payload["predicted_action"],
            answer_text=payload.get("answer_text"),
            tool_calls=payload.get("tool_calls", []),
            confidence=payload.get("confidence"),
            metadata={
                "provider": OPENAI_UNDERSTANDING_PROVIDER,
                "model": self.model,
                "response_id": getattr(response, "id", None),
                "completion_time_s": completion_time_s,
                "audio_clip_count": len(audio_paths),
                "tool_protocol": self.tool_protocol,
                "native_tool_count": len(native_bundle.declarations) if native_tools else 0,
                "prompts": _prompt_metadata(
                    provider=OPENAI_UNDERSTANDING_PROVIDER,
                    model=self.model,
                    mode="turn_based",
                    system=system_prompt,
                    request_rendering=user_prompt,
                ),
            },
        )

    def _client(self) -> Any:
        if self.client is not None:
            return self.client
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise UnderstandingProviderError(
                "OPENAI_API_KEY is required for OpenAI understanding evaluation"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise UnderstandingProviderError("openai package is required") from exc
        self.client = OpenAI(api_key=api_key, timeout=120.0)
        return self.client

    def _create_completion(self, request: dict[str, Any]) -> Any:
        if self._use_response_format:
            try:
                return self._client().chat.completions.create(
                    **request, response_format={"type": "json_object"}
                )
            except Exception as exc:
                if not _response_format_unsupported(exc):
                    raise
                self._use_response_format = False
        return self._client().chat.completions.create(**request)

    def _record_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        input_tokens = _usage_value(usage, "input_tokens")
        output_tokens = _usage_value(usage, "output_tokens")
        total_tokens = _usage_value(usage, "total_tokens") or input_tokens + output_tokens
        self.usage["input_tokens"] += input_tokens
        self.usage["output_tokens"] += output_tokens
        self.usage["total_tokens"] += total_tokens



def _predict_with_resume(
    provider: OpenAiUnderstandingProvider,
    rows: list[dict[str, Any]],
    manifest_path: Path,
    output: Path,
) -> list[PredictionRow]:
    """Generate OpenAI predictions incrementally so interrupted evals can resume."""
    existing = _existing_predictions(output, tool_protocol=provider.tool_protocol)
    predictions: list[PredictionRow] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            cell_id = _required_str(row, "cell_id")
            print_eval_case_progress(
                OPENAI_UNDERSTANDING_PROVIDER, provider.model, cell_id, index, len(rows)
            )
            if cell_id in existing:
                predictions.append(existing[cell_id])
                continue
            try:
                prediction = provider.predict(row, manifest_path=manifest_path)
            except PredictionFormatError as exc:
                prediction = _error_prediction(row, provider, exc)
            handle.write(f"{canonical_json(prediction)}\n")
            handle.flush()
            predictions.append(prediction)
    return predictions


def _error_prediction(
    row: dict[str, Any],
    provider: OpenAiUnderstandingProvider,
    error: PredictionFormatError,
) -> PredictionRow:
    """Return a valid JSONL prediction row that records an unrecoverable format error."""
    cell_id = _required_str(row, "cell_id")
    return PredictionRow(
        cell_id=cell_id,
        predicted_action="silent",
        answer_text=None,
        tool_calls=[],
        confidence=0.0,
        metadata={
            "provider": OPENAI_UNDERSTANDING_PROVIDER,
            "model": provider.model,
            "eval_mode": "turn_based",
            "tool_protocol": provider.tool_protocol,
            "provider_error": {
                "type": "prediction_format_error",
                "message": str(error),
            },
        },
    )


def _existing_predictions(output: Path, *, tool_protocol: str) -> dict[str, PredictionRow]:
    if not output.exists() or output.stat().st_size == 0:
        return {}
    return {
        row.cell_id: row
        for row in read_predictions(output)
        if _prediction_tool_protocol(row) == tool_protocol
    }


def _prediction_tool_protocol(row: PredictionRow) -> str:
    metadata = row.metadata if isinstance(row.metadata, dict) else {}
    value = metadata.get("tool_protocol")
    return value if value in {"prompt_json", "native"} else "prompt_json"

def _normalize_openai_audio_model(model: str) -> str:
    normalized = model.strip()
    return _OPENAI_MODEL_ALIASES.get(normalized.lower(), normalized)


def _limited_rows(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return rows
    if limit < 0:
        raise UnderstandingProviderError("limit must be non-negative")
    return rows[:limit]


def _audio_paths_for(row: dict[str, Any], manifest_path: Path) -> list[Path]:
    root = _run_root_for(manifest_path)
    raw_paths = row.get("mixed_testcase_audio_paths") or row.get("audio_paths") or []
    if not isinstance(raw_paths, list):
        raise UnderstandingProviderError("manifest audio path field must be a list")
    paths: list[Path] = []
    for raw in raw_paths:
        if not isinstance(raw, str) or not raw:
            raise UnderstandingProviderError("manifest audio paths must be non-empty strings")
        path = Path(raw)
        paths.append(path if path.is_absolute() else root / path)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise UnderstandingProviderError(f"missing audio file(s): {missing}")
    return paths


def _run_root_for(manifest_path: Path) -> Path:
    return manifest_path.parent.parent if manifest_path.parent.name == "manifest" else manifest_path.parent


def _audio_part(path: Path) -> dict[str, Any]:
    return {
        "type": "input_audio",
        "input_audio": {
            "data": base64.b64encode(path.read_bytes()).decode("ascii"),
            "format": _audio_format(path),
        },
    }


def _audio_format(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    if suffix not in {"wav", "mp3"}:
        raise UnderstandingProviderError(f"unsupported audio format for {path}: {suffix!r}")
    return suffix


def _prompt_for(row: dict[str, Any], audio_paths: list[Path]) -> str:
    return "\n\n".join(
        [_system_prompt_with_setup(row), _user_prompt_for(row, audio_paths)]
    )


def _system_prompt_for(row: dict[str, Any], *, native_tools: bool = False) -> str:
    _ = row
    kind = "core_native" if native_tools else "core"
    return render_prompt("understanding/system.j2", {"kind": kind})


def _system_prompt_with_setup(row: dict[str, Any], *, native_tools: bool = False) -> str:
    tool_protocol = "native" if native_tools else "prompt_json"
    return "\n\n".join(
        [
            setup_prompt_for(row, tool_protocol=tool_protocol),
            _system_prompt_for(row, native_tools=native_tools),
        ]
    )


def _user_prompt_for(row: dict[str, Any], audio_paths: list[Path]) -> str:
    """Return legacy text prompt content; structured history carries the audio turns."""
    del row, audio_paths
    return ""


def _json_user_prompt(row: dict[str, Any], audio_paths: list[Path]) -> str:
    del row, audio_paths
    return ""


def _openai_history_messages(row: dict[str, Any], audio_paths: list[Path]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for message in history_messages(row, audio_paths):
        if message.role == "user":
            messages.append({"role": "user", "content": [_audio_part(_required_audio_path(message))]})
        elif message.assistant_text is not None:
            messages.append({"role": "assistant", "content": message.assistant_text})
    return messages


def _required_audio_path(message: AudioHistoryMessage) -> Path:
    if message.audio_path is None:
        raise UnderstandingProviderError("user audio history message missing audio_path")
    return message.audio_path


def _prompt_metadata(
    *,
    provider: str,
    model: str,
    mode: str,
    system: str,
    request_rendering: str,
    source: str = "captured_at_generation",
    live_system: str | None = None,
    live_request_rendering: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source": source,
        "provider": provider,
        "model": model,
        "mode": mode,
        "system": system,
        "request_rendering": request_rendering,
    }
    if live_system is not None:
        metadata["live_system"] = live_system
    if live_request_rendering is not None:
        metadata["live_request_rendering"] = live_request_rendering
    return metadata


def _response_format_unsupported(exc: Exception) -> bool:
    text = str(exc).lower()
    return "response_format" in text and "not supported" in text


def _response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text
    choices = getattr(response, "choices", None)
    if isinstance(choices, list) and choices:
        content = getattr(getattr(choices[0], "message", None), "content", None)
        if isinstance(content, str) and content.strip():
            return content
    output = getattr(response, "output", None)
    if isinstance(output, list):
        chunks: list[str] = []
        for item in output:
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", None) == "output_text":
                    value = getattr(content, "text", "")
                    if isinstance(value, str):
                        chunks.append(value)
        if chunks:
            return "".join(chunks)
    raise UnderstandingProviderError("OpenAI understanding response did not contain text")


def _prediction_payload_for_response(
    response: Any,
    cell_id: str,
    *,
    native_bundle: NativeToolBundle | None,
) -> dict[str, Any]:
    """Return a prediction payload from prompt JSON or provider-native calls."""

    if native_bundle is None:
        return _parse_prediction_payload(_response_text(response), cell_id)
    calls = _openai_native_tool_calls(response, native_bundle.name_map, cell_id)
    return _merge_native_prediction(_optional_response_text(response), calls, cell_id)


def _merge_native_prediction(
    text: str | None,
    calls: list[dict[str, Any]],
    cell_id: str,
) -> dict[str, Any]:
    """Combine the model's explicit native decision with its native function calls.

    Native mode carries the respond/silent choice as a small JSON decision object
    (``predicted_action``) in text, kept separate from the action channel: a real
    function call means ``respond``; an explicit ``silent`` decision with no call
    abstains. This decouples the decision from the tool list.
    """

    if text:
        try:
            payload = _parse_prediction_payload(text, cell_id)
        except PredictionFormatError:
            payload = {
                "predicted_action": "respond",
                "answer_text": text.strip(),
                "confidence": None,
            }
    else:
        payload = {
            "predicted_action": "respond" if calls else "silent",
            "answer_text": None,
            "confidence": None,
        }
    payload["tool_calls"] = calls
    if calls:
        payload["predicted_action"] = "respond"
    return payload


def _optional_response_text(response: Any) -> str | None:
    try:
        return _response_text(response)
    except UnderstandingProviderError:
        return None


def _openai_native_tool_calls(
    response: Any,
    name_map: dict[str, str],
    cell_id: str,
) -> list[dict[str, Any]]:
    choices = _field(response, "choices")
    if not isinstance(choices, list) or not choices:
        return []
    message = _field(choices[0], "message")
    raw_calls = _field(message, "tool_calls") or []
    if not isinstance(raw_calls, list):
        raise PredictionFormatError(f"native tool_calls for {cell_id} must be a list")
    calls: list[dict[str, Any]] = []
    for index, raw_call in enumerate(raw_calls):
        function = _field(raw_call, "function")
        provider_name = _field(function, "name")
        if not isinstance(provider_name, str) or provider_name not in name_map:
            raise PredictionFormatError(
                f"native tool_calls[{index}] for {cell_id} used an unknown function"
            )
        raw_arguments = _field(function, "arguments")
        try:
            arguments = json.loads(raw_arguments or "{}") if isinstance(raw_arguments, str) else raw_arguments
        except json.JSONDecodeError as exc:
            raise PredictionFormatError(
                f"native tool_calls[{index}] arguments for {cell_id} were invalid JSON"
            ) from exc
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise PredictionFormatError(
                f"native tool_calls[{index}] arguments for {cell_id} must be an object"
            )
        calls.append({"name": name_map[provider_name], "arguments": dict(arguments)})
    return calls


def _field(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def _parse_prediction_payload(text: str, cell_id: str) -> dict[str, Any]:
    payload = _prediction_payload_from_text(text, cell_id)
    action = payload.get("predicted_action")
    if action not in _ACTION_VALUES:
        raise PredictionFormatError(
            f"invalid predicted_action for {cell_id}: {action!r}; expected one of {_ACTION_VALUES}"
        )
    answer_text = payload.get("answer_text")
    if answer_text is not None and not isinstance(answer_text, str):
        raise PredictionFormatError(f"answer_text for {cell_id} must be a string or null")
    payload["tool_calls"] = _normalize_tool_calls(payload.get("tool_calls", []), cell_id)
    confidence = payload.get("confidence")
    if confidence is not None:
        if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
            raise PredictionFormatError(f"confidence for {cell_id} must be between 0 and 1")
        payload["confidence"] = float(confidence)
    return payload


def _prediction_payload_from_text(text: str, cell_id: str) -> dict[str, Any]:
    """Return the first JSON object candidate that looks like a prediction."""
    first_payload: dict[str, Any] | None = None
    first_error: LlmProviderError | None = None
    for candidate in _prediction_json_candidates(text):
        try:
            payload = parse_json_object(candidate)
        except LlmProviderError as exc:
            if first_error is None:
                first_error = exc
            continue
        if first_payload is None:
            first_payload = payload
        if payload.get("predicted_action") in _ACTION_VALUES:
            return payload
    if first_payload is not None:
        return first_payload
    error = first_error or LlmProviderError("LLM provider returned invalid JSON")
    raise PredictionFormatError(f"invalid JSON prediction for {cell_id}: {error}") from error


def _prediction_json_candidates(text: str) -> list[str]:
    """Return raw text plus every balanced JSON-object-looking substring."""
    candidates = [text]
    repaired = _close_unbalanced_containers(text)
    if repaired != text:
        candidates.append(repaired)
    candidates.extend(_balanced_json_objects(text))
    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        value = candidate.strip()
        if value and value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def _close_unbalanced_containers(text: str) -> str:
    """Append the closers a truncated JSON payload is missing, respecting strings.

    Gemini JSON mode occasionally stops (finish_reason=STOP) with the trailing
    ``}``/``]`` dropped; appending the open container closers recovers it.
    """
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if stack and stack[-1] == char:
                stack.pop()
    if in_string or not stack:
        return text
    return text.rstrip() + "".join(reversed(stack))


def _balanced_json_objects(text: str) -> list[str]:
    """Return all balanced object substrings, respecting quoted strings."""
    objects: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(text[start : index + 1])
                start = None
    return objects


def _normalize_tool_calls(raw_tool_calls: Any, cell_id: str) -> list[dict[str, Any]]:
    """Normalize model-emitted tool calls into {name, arguments} objects."""
    if raw_tool_calls is None:
        return []
    if isinstance(raw_tool_calls, dict):
        raw_tool_calls = [raw_tool_calls]
    if not isinstance(raw_tool_calls, list):
        raise PredictionFormatError("tool_calls must be a list when present")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(raw_tool_calls):
        normalized.append(_normalize_tool_call_item(item, cell_id, index))
    return normalized


def _normalize_tool_call_item(item: Any, cell_id: str, index: int) -> dict[str, Any]:
    if isinstance(item, str):
        name = item.strip()
        if not name:
            raise PredictionFormatError(f"tool_calls[{index}] for {cell_id} missing string name")
        return {"name": name, "arguments": {}}
    if not isinstance(item, dict):
        raise PredictionFormatError(f"tool_calls[{index}] for {cell_id} must be an object")
    if isinstance(item.get("function"), dict):
        merged = dict(item["function"])
        merged.update({key: value for key, value in item.items() if key != "function"})
        item = merged
    name = item.get("name")
    if not isinstance(name, str) or not name.strip():
        raise PredictionFormatError(f"tool_calls[{index}] for {cell_id} missing string name")
    raw_arguments = item.get("arguments", {})
    if raw_arguments is None:
        raw_arguments = {}
    if isinstance(raw_arguments, str):
        raw_arguments = {"value": raw_arguments}
    if not isinstance(raw_arguments, dict):
        raise PredictionFormatError(
            f"tool_calls[{index}].arguments for {cell_id} must be an object"
        )
    arguments = dict(raw_arguments)
    for key, value in item.items():
        if key not in {"name", "arguments"}:
            arguments[key] = value
    return {"name": name.strip(), "arguments": arguments}

def _required_str(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise UnderstandingProviderError(f"manifest row missing non-empty {key!r}")
    return value


def _usage_value(usage: Any, key: str) -> int:
    aliases = {
        "input_tokens": ("input_tokens", "prompt_tokens"),
        "output_tokens": ("output_tokens", "completion_tokens"),
        "total_tokens": ("total_tokens",),
    }[key]
    if isinstance(usage, dict):
        value = next((usage.get(alias) for alias in aliases if usage.get(alias) is not None), 0)
    else:
        value = next((getattr(usage, alias, None) for alias in aliases if getattr(usage, alias, None) is not None), 0)
    return int(value or 0)
