"""gemini_understanding — generate prediction rows from Gemini audio understanding.

Calling spec:
    run_gemini_understanding_predictions(
        manifest, output, model, limit=None, client=None,
        tool_protocol="prompt_json", thinking_mode=None
    ) -> dict

Inputs are benchmark manifest rows whose audio paths are relative to the run root.
Outputs are strict PredictionRow JSONL rows consumable by ib eval / manifest_eval.

Side effects: may call the Google Gemini API and writes prediction and sibling
raw-response JSONL artifacts.
"""

from __future__ import annotations

import importlib
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ib.eval_targets import GeminiThinkingMode
from ib.io import canonical_json, write_jsonl
from ib.predictions import PredictionRow, read_predictions
from ib.scoring.openai_understanding import (
    PredictionFormatError,
    UnderstandingProviderError,
    _audio_format,
    _audio_paths_for,
    _limited_rows,
    _merge_native_prediction,
    _parse_prediction_payload,
    _prediction_tool_protocol,
    _prompt_metadata,
    _required_str,
    _system_prompt_with_setup,
    _user_prompt_for,
)
from ib.scoring.multiturn_audio import history_messages
from ib.scoring.native_tool_schema import (
    NativeToolBundle,
    build_native_tool_bundle,
    gemini_function_declarations,
)
from ib.scoring.progress import print_eval_case_progress
from ib.scoring.raw_response_io import append_raw_response, raw_response_path
from ib.scoring.smoke import _read_rows
from ib.scoring.transcript_render import render_visible_transcript

GEMINI_UNDERSTANDING_PROVIDER = "gemini"
GEMINI_TEXT_UNDERSTANDING_PROVIDER = "gemini_text"
GEMINI_THINKING_BUDGET = 8_192
# Client-side cap per generate_content call (SDK default is no timeout, so a dead
# connection hangs the eval loop forever). Milliseconds, per google-genai HttpOptions.
GEMINI_REQUEST_TIMEOUT_MS = 300_000
_INPUT_MODES = ("audio", "transcript")
# One neutral descriptor so the labeled block reads as dialogue, not instructions.
# The system prompt (the actual task prompt) is byte-identical to the audio path.
_TRANSCRIPT_HEADER = (
    "Conversation transcript (each line is 'Ln Speaker -> Addressee: spoken text'):"
)
_GEMINI_MODEL_ALIASES = {
    "gemini3.5flash": "gemini-3.5-flash",
    "gemini-3.5flash": "gemini-3.5-flash",
    "gemini3.5-flash": "gemini-3.5-flash",
    "models/gemini-3.5-flash": "gemini-3.5-flash",
}


def run_gemini_understanding_predictions(
    manifest: str | Path,
    output: str | Path,
    *,
    model: str | None = None,
    limit: int | None = None,
    client: Any | None = None,
    tool_protocol: str = "prompt_json",
    input_mode: str = "audio",
    thinking_mode: GeminiThinkingMode | None = None,
) -> dict[str, Any]:
    """Generate predictions for up to ``limit`` manifest rows using Gemini.

    ``input_mode="audio"`` sends the interleaved testcase audio (default).
    ``input_mode="transcript"`` sends the speaker/addressee labeled transcript
    instead, keeping the system prompt, tool protocol, and output schema identical
    so the two runs differ only in input modality.
    """
    if not model:
        raise UnderstandingProviderError("Gemini understanding eval requires explicit model")
    raw_output = raw_response_path(output)
    rows = _limited_rows(_read_rows(manifest), limit)
    provider = GeminiUnderstandingProvider(
        model=_normalize_gemini_model(model),
        client=client,
        tool_protocol=tool_protocol,
        input_mode=input_mode,
        thinking_mode=thinking_mode,
        raw_response_output=raw_output,
    )
    predictions = _predict_with_resume(provider, rows, Path(manifest), Path(output))
    write_jsonl(output, predictions)
    return {
        "schema_version": "ib.gemini_understanding.v1",
        "provider": provider.provider_name,
        "model": provider.model,
        "manifest": str(manifest),
        "predictions": str(output),
        "limit": limit,
        "predicted_count": len(predictions),
        "llm_call_count": provider.call_count,
        "usage": provider.usage,
        "tool_protocol": provider.tool_protocol,
        "input_mode": provider.input_mode,
        "thinking_mode": provider.thinking_mode,
        "raw_responses": str(raw_output),
    }


class GeminiUnderstandingProvider:
    """Gemini adapter for audio-in/text-out MSI-Bench predictions."""

    def __init__(
        self,
        *,
        model: str,
        client: Any | None = None,
        api_key: str | None = None,
        tool_protocol: str = "prompt_json",
        input_mode: str = "audio",
        thinking_mode: GeminiThinkingMode | None = None,
        raw_response_output: Path | None = None,
    ) -> None:
        if tool_protocol not in {"prompt_json", "native"}:
            raise ValueError(f"unsupported tool protocol: {tool_protocol!r}")
        if input_mode not in _INPUT_MODES:
            raise ValueError(f"unsupported input mode: {input_mode!r}")
        self.model = model
        self.client = client
        self.api_key = api_key
        self.tool_protocol = tool_protocol
        self.input_mode = input_mode
        self.thinking_mode = thinking_mode
        self.raw_response_output = raw_response_output
        self.provider_name = (
            GEMINI_TEXT_UNDERSTANDING_PROVIDER
            if input_mode == "transcript"
            else GEMINI_UNDERSTANDING_PROVIDER
        )
        self.call_count = 0
        self.usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        self.max_attempts = 5

    def predict(self, manifest_row: dict[str, Any], *, manifest_path: Path) -> PredictionRow:
        """Return one strict PredictionRow for a manifest row."""
        cell_id = _required_str(manifest_row, "cell_id")
        transcript = (
            render_visible_transcript(manifest_row) if self.input_mode == "transcript" else None
        )
        if self.input_mode == "audio":
            audio_paths = _audio_paths_for(manifest_row, manifest_path)
            if not audio_paths:
                raise UnderstandingProviderError(f"manifest row {cell_id!r} has no audio paths")
        else:
            audio_paths = []
        native_tools = self.tool_protocol == "native"
        native_bundle = (
            build_native_tool_bundle(manifest_row.get("logic_available_functions"))
            if native_tools
            else NativeToolBundle(declarations=[], name_map={})
        )
        system_prompt = _system_prompt_with_setup(manifest_row, native_tools=native_tools)
        user_prompt = (
            transcript if transcript is not None else _user_prompt_for(manifest_row, audio_paths)
        )
        # Gemini occasionally returns truncated/invalid JSON even at temperature 0,
        # so parse failures re-generate instead of aborting the whole eval.
        last_parse_error: UnderstandingProviderError | None = None
        payload: dict[str, Any] | None = None
        started = time.perf_counter()
        for _attempt in range(self.max_attempts):
            response = self._generate_content_with_retries(
                cell_id,
                manifest_row,
                audio_paths,
                system_prompt=system_prompt,
                native_bundle=native_bundle if native_tools else None,
                transcript=transcript,
            )
            self.call_count += 1
            self._record_usage(response)
            if self.raw_response_output is not None:
                append_raw_response(
                    self.raw_response_output,
                    cell_id=cell_id,
                    provider=self.provider_name,
                    model=self.model,
                    eval_mode="turn_based",
                    phase="prediction",
                    attempt=_attempt + 1,
                    response=response,
                    metadata={
                        "input_mode": self.input_mode,
                        "thinking_mode": self.thinking_mode,
                        "tool_protocol": self.tool_protocol,
                    },
                )
            try:
                if native_tools:
                    calls = _gemini_native_tool_calls(response, native_bundle.name_map, cell_id)
                    payload = _merge_native_prediction(
                        _optional_response_text(response), calls, cell_id
                    )
                else:
                    payload = _parse_prediction_payload(_response_text(response), cell_id)
                break
            except UnderstandingProviderError as exc:
                last_parse_error = exc
        completion_time_s = round(time.perf_counter() - started, 3)
        if payload is None:
            raise last_parse_error or PredictionFormatError(
                f"Gemini returned no parsable prediction for {cell_id}"
            )
        return PredictionRow(
            cell_id=cell_id,
            predicted_action=payload["predicted_action"],
            answer_text=payload.get("answer_text"),
            tool_calls=payload.get("tool_calls", []),
            confidence=payload.get("confidence"),
            metadata={
                "provider": self.provider_name,
                "model": self.model,
                "input_mode": self.input_mode,
                "response_id": getattr(response, "response_id", None),
                "completion_time_s": completion_time_s,
                "audio_clip_count": len(audio_paths),
                "tool_protocol": self.tool_protocol,
                "thinking_mode": self.thinking_mode,
                "native_tool_count": len(native_bundle.declarations) if native_tools else 0,
                "prompts": _prompt_metadata(
                    provider=self.provider_name,
                    model=self.model,
                    mode="turn_based",
                    system=system_prompt,
                    request_rendering=user_prompt,
                ),
            },
        )

    def _generate_content_with_retries(
        self,
        cell_id: str,
        manifest_row: dict[str, Any],
        audio_paths: list[Path],
        *,
        system_prompt: str,
        native_bundle: NativeToolBundle | None,
        transcript: str | None = None,
    ) -> Any:
        contents = (
            _transcript_contents(transcript)
            if transcript is not None
            else _history_contents(manifest_row, audio_paths)
        )
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self._client().models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=self._generation_config(system_prompt, native_bundle=native_bundle),
                )
            except Exception as exc:
                if attempt == self.max_attempts or not _is_retryable_gemini_error(exc):
                    raise UnderstandingProviderError(
                        f"Gemini request failed for {cell_id} after {attempt} attempt(s): {exc}"
                    ) from exc
                time.sleep(min(2 ** (attempt - 1), 20))
        raise UnderstandingProviderError(f"Gemini request failed for {cell_id}")

    def _client(self) -> Any:
        if self.client is not None:
            return self.client
        api_key = self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise UnderstandingProviderError(
                "GEMINI_API_KEY or GOOGLE_API_KEY is required for Gemini understanding evaluation"
            )
        try:
            genai: Any = importlib.import_module("google.genai")
        except ImportError as exc:
            raise UnderstandingProviderError("google-genai package is required") from exc
        self.client = genai.Client(
            api_key=api_key,
            http_options={"timeout": GEMINI_REQUEST_TIMEOUT_MS},
        )
        return self.client

    def _generation_config(
        self,
        system_prompt: str,
        *,
        native_bundle: NativeToolBundle | None,
    ) -> Any:
        try:
            types: Any = importlib.import_module("google.genai.types")
        except ImportError as exc:
            raise UnderstandingProviderError("google-genai package is required") from exc
        thinking_enabled = self.thinking_mode == "enabled"
        config: dict[str, Any] = {
            "system_instruction": system_prompt,
            "temperature": 0,
            "max_output_tokens": 4096 if thinking_enabled else 1400,
        }
        if self.thinking_mode is not None:
            if self.model and self.model.startswith("gemini-3.5"):
                # Gemini 3.5 models take discrete thinking levels, not token budgets.
                config["thinking_config"] = types.ThinkingConfig(
                    include_thoughts=thinking_enabled,
                    thinking_level=(
                        types.ThinkingLevel.HIGH if thinking_enabled else types.ThinkingLevel.MINIMAL
                    ),
                )
            else:
                config["thinking_config"] = types.ThinkingConfig(
                    include_thoughts=thinking_enabled,
                    thinking_budget=GEMINI_THINKING_BUDGET if thinking_enabled else 0,
                )
        if native_bundle is not None and native_bundle.declarations:
            declarations = [
                types.FunctionDeclaration(**declaration)
                for declaration in gemini_function_declarations(native_bundle)
            ]
            config.update(
                {
                    "tools": [types.Tool(function_declarations=declarations)],
                    "tool_config": types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(mode="AUTO")
                    ),
                    "automatic_function_calling": types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                }
            )
        else:
            config["response_mime_type"] = "application/json"
        return types.GenerateContentConfig(**config)

    def _record_usage(self, response: Any) -> None:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return
        input_tokens = _gemini_usage_value(usage, "prompt_token_count")
        output_tokens = _gemini_usage_value(usage, "candidates_token_count")
        total_tokens = (
            _gemini_usage_value(usage, "total_token_count") or input_tokens + output_tokens
        )
        self.usage["input_tokens"] += input_tokens
        self.usage["output_tokens"] += output_tokens
        self.usage["total_tokens"] += total_tokens


def _transcript_contents(transcript: str) -> list[Any]:
    """Return a single user turn carrying the labeled transcript text."""
    try:
        types: Any = importlib.import_module("google.genai.types")
    except ImportError as exc:
        raise UnderstandingProviderError("google-genai package is required") from exc
    return [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=f"{_TRANSCRIPT_HEADER}\n{transcript}")],
        )
    ]


def _history_contents(row: dict[str, Any], audio_paths: list[Path]) -> list[Any]:
    try:
        types: Any = importlib.import_module("google.genai.types")
    except ImportError as exc:
        raise UnderstandingProviderError("google-genai package is required") from exc
    contents: list[Any] = []
    for message in history_messages(row, audio_paths):
        if message.role == "user":
            if message.audio_path is None:
                raise UnderstandingProviderError("user audio history message missing audio_path")
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(
                            data=message.audio_path.read_bytes(),
                            mime_type=_mime_type(message.audio_path),
                        )
                    ],
                )
            )
        elif message.assistant_text is not None:
            contents.append(
                types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=message.assistant_text)],
                )
            )
    return contents


def _predict_with_resume(
    provider: GeminiUnderstandingProvider,
    rows: list[dict[str, Any]],
    manifest_path: Path,
    output: Path,
) -> list[PredictionRow]:
    existing = _existing_predictions(output, tool_protocol=provider.tool_protocol)
    predictions: list[PredictionRow] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            cell_id = _required_str(row, "cell_id")
            print_eval_case_progress(
                GEMINI_UNDERSTANDING_PROVIDER, provider.model, cell_id, index, len(rows)
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
    provider: GeminiUnderstandingProvider,
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
            "provider": provider.provider_name,
            "model": provider.model,
            "eval_mode": "turn_based",
            "input_mode": provider.input_mode,
            "thinking_mode": provider.thinking_mode,
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


def _mime_type(path: Path) -> str:
    return {"wav": "audio/wav", "mp3": "audio/mpeg"}[_audio_format(path)]


def _response_text(response: Any) -> str:
    candidate_text = _visible_candidate_text(response)
    if candidate_text is not None:
        return candidate_text
    try:
        text = getattr(response, "text", None)
    except ValueError:
        text = None
    if isinstance(text, str) and text.strip():
        return text
    raise UnderstandingProviderError("Gemini understanding response did not contain text")


def _visible_candidate_text(response: Any) -> str | None:
    """Return candidate text while excluding Gemini thought-summary parts."""
    candidates = getattr(response, "candidates", None)
    if isinstance(candidates, list):
        chunks: list[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", []) or []:
                if getattr(part, "thought", False) is True:
                    continue
                value = getattr(part, "text", None)
                if isinstance(value, str):
                    chunks.append(value)
        if chunks:
            return "".join(chunks)
    return None


def _optional_response_text(response: Any) -> str | None:
    candidate_text = _visible_candidate_text(response)
    if candidate_text is not None:
        return candidate_text
    stored = vars(response).get("text") if hasattr(response, "__dict__") else None
    return stored if isinstance(stored, str) and stored.strip() else None


def _gemini_native_tool_calls(
    response: Any,
    name_map: dict[str, str],
    cell_id: str,
) -> list[dict[str, Any]]:
    """Return normalized Gemini function calls, restoring harness names."""

    try:
        raw_calls = getattr(response, "function_calls", None)
    except ValueError:
        raw_calls = None
    if not isinstance(raw_calls, list) or not raw_calls:
        raw_calls = _candidate_function_calls(response)
    calls: list[dict[str, Any]] = []
    for index, raw_call in enumerate(raw_calls):
        provider_name = _field(raw_call, "name")
        if not isinstance(provider_name, str) or provider_name not in name_map:
            raise PredictionFormatError(
                f"native function call #{index + 1} for {cell_id} used an unknown function"
            )
        arguments = _field(raw_call, "args")
        if arguments is None:
            arguments = _field(raw_call, "arguments")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, Mapping):
            raise PredictionFormatError(
                f"native function call #{index + 1} arguments for {cell_id} must be an object"
            )
        calls.append({"name": name_map[provider_name], "arguments": dict(arguments)})
    return calls


def _candidate_function_calls(response: Any) -> list[Any]:
    calls: list[Any] = []
    candidates = _field(response, "candidates")
    if not isinstance(candidates, list):
        return calls
    for candidate in candidates:
        content = _field(candidate, "content")
        for part in _field(content, "parts") or []:
            function_call = _field(part, "function_call")
            if function_call is not None:
                calls.append(function_call)
    return calls


def _field(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)


def _normalize_gemini_model(model: str) -> str:
    normalized = model.strip()
    return _GEMINI_MODEL_ALIASES.get(normalized.lower(), normalized.removeprefix("models/"))


def _gemini_usage_value(usage: Any, key: str) -> int:
    if isinstance(usage, dict):
        value = usage.get(key)
    else:
        value = getattr(usage, key, None)
    return int(value or 0)


def _is_retryable_gemini_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status in {429, 500, 502, 503, 504}:
        return True
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return (
        "servererror" in name
        or "timeout" in name
        or "ssl" in name
        or "connecterror" in name
        or "protocolerror" in name
        or "unexpected eof" in text
        or "unavailable" in text
        or "resource_exhausted" in text
        or "rate limit" in text
    )
