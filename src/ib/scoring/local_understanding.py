"""local_understanding — generate predictions from local OpenAI-compatible audio models.

Calling spec:
    run_local_understanding_predictions(manifest, output, model, limit=None, client=None) -> dict
    run_qwen_understanding_predictions(manifest, output, model, limit=None, client=None) -> dict

Inputs are benchmark manifest rows whose audio paths are relative to the run root.
Outputs are strict PredictionRow JSONL rows consumable by ib eval / manifest_eval.

Side effects: may call local OpenAI-compatible endpoints and writes prediction
and sibling raw-response JSONL artifacts.
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any

from ib.io import canonical_json, write_jsonl
from ib.predictions import PredictionRow, read_predictions
from ib.scoring.local_audio_models import (
    LocalAudioModelSpec,
    local_audio_base_url,
    local_audio_model_spec,
)
from ib.scoring.openai_understanding import (
    PredictionFormatError,
    UnderstandingProviderError,
    _audio_format,
    _audio_paths_for,
    _limited_rows,
    _parse_prediction_payload,
    _prompt_metadata,
    _required_str,
    _response_text,
    _system_prompt_with_setup,
    _user_prompt_for,
    _usage_value,
)
from ib.scoring.multiturn_audio import history_messages
from ib.scoring.realtime_prediction_tool import emit_prediction_tool
from ib.scoring.transcript_render import render_visible_transcript
from ib.scoring.progress import print_eval_case_progress


def guided_json_enabled() -> bool:
    """Return whether IB_LOCAL_GUIDED_JSON opts this run into constrained decoding."""
    enabled = os.getenv("IB_LOCAL_GUIDED_JSON", "").strip() == "1"
    if not _GUIDED_JSON_ANNOUNCED:
        _GUIDED_JSON_ANNOUNCED.append(enabled)
        print(f"[ib-eval] local guided JSON decoding: {'ON' if enabled else 'off'}", flush=True)
    return enabled


_GUIDED_JSON_ANNOUNCED: list[bool] = []
from ib.scoring.raw_response_io import append_raw_response, raw_response_path
from ib.scoring.smoke import _read_rows

LOCAL_UNDERSTANDING_PROVIDER = "local"
LOCAL_TEXT_UNDERSTANDING_PROVIDER = "local_text"
QWEN_UNDERSTANDING_PROVIDER = "qwen"
DEFAULT_QWEN_BASE_URL = "http://localhost:8013/v1"
_INPUT_MODES = {"audio", "transcript"}


def run_local_understanding_predictions(
    manifest: str | Path,
    output: str | Path,
    *,
    model: str | None = None,
    limit: int | None = None,
    client: Any | None = None,
    thinking_mode: str | None = None,
    input_mode: str = "audio",
) -> dict[str, Any]:
    """Generate predictions using one registered local audio endpoint.

    ``input_mode="audio"`` sends the interleaved testcase audio (default).
    ``input_mode="transcript"`` sends the speaker/addressee labeled transcript
    instead — the text-modality control twin, mirroring ``gemini_text``.
    """
    if not model:
        raise UnderstandingProviderError("local understanding eval requires explicit model")
    if input_mode not in _INPUT_MODES:
        raise UnderstandingProviderError(f"unsupported input mode: {input_mode!r}")
    try:
        spec = local_audio_model_spec(model)
    except ValueError as exc:
        raise UnderstandingProviderError(str(exc)) from exc
    _validate_local_transport(spec, thinking_mode)
    provider_name = (
        LOCAL_TEXT_UNDERSTANDING_PROVIDER
        if input_mode == "transcript"
        else LOCAL_UNDERSTANDING_PROVIDER
    )
    rows = _limited_rows(_read_rows(manifest), limit)
    raw_output = raw_response_path(output)
    provider = LocalUnderstandingProvider(
        model=model,
        provider_name=provider_name,
        base_url=local_audio_base_url(spec),
        model_spec=spec,
        thinking_mode=thinking_mode,
        input_mode=input_mode,
        client=client,
        raw_response_output=raw_output,
    )
    predictions = _predict_with_resume(provider, rows, Path(manifest), Path(output))
    write_jsonl(output, predictions)
    return _report(
        schema_version="ib.local_understanding.v1",
        provider=provider_name,
        model=provider.model,
        manifest=manifest,
        output=output,
        raw_responses=raw_output,
        limit=limit,
        predicted_count=len(predictions),
        llm_call_count=provider.call_count,
        usage=provider.usage,
        metadata={
            "base_url": provider.base_url,
            "checkpoint_path": str(spec.checkpoint_path),
            "thinking_mode": thinking_mode,
            "input_mode": input_mode,
            "transport": spec.transport,
        },
    )


def _validate_local_transport(spec: LocalAudioModelSpec, thinking_mode: str | None) -> None:
    if spec.transport != "audio_url":
        raise UnderstandingProviderError(
            f"{spec.model} currently exposes transcription only; its vLLM chat template "
            "cannot reliably generate MSI-Bench decisions"
        )
    if thinking_mode not in {None, "disabled", "enabled"}:
        raise UnderstandingProviderError("local thinking_mode must be 'disabled' or 'enabled'")
    if thinking_mode is not None and not spec.supports_thinking:
        raise UnderstandingProviderError(
            f"thinking_mode is not registered for local model {spec.model}"
        )


def run_qwen_understanding_predictions(
    manifest: str | Path,
    output: str | Path,
    *,
    model: str | None = None,
    limit: int | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """Generate predictions using a local Qwen3-Omni vLLM audio endpoint."""
    if not model:
        raise UnderstandingProviderError("Qwen understanding eval requires explicit model")
    rows = _limited_rows(_read_rows(manifest), limit)
    raw_output = raw_response_path(output)
    provider = LocalUnderstandingProvider(
        model=model,
        provider_name=QWEN_UNDERSTANDING_PROVIDER,
        base_url=_env("QWEN_UNDERSTANDING_BASE_URL", "QWEN_BASE_URL", DEFAULT_QWEN_BASE_URL),
        api_key=_env("QWEN_UNDERSTANDING_API_KEY", "QWEN_API_KEY", "EMPTY"),
        client=client,
        raw_response_output=raw_output,
    )
    predictions = _predict_with_resume(provider, rows, Path(manifest), Path(output))
    write_jsonl(output, predictions)
    return _report(
        schema_version="ib.local_understanding.qwen.v1",
        provider=QWEN_UNDERSTANDING_PROVIDER,
        model=provider.model,
        manifest=manifest,
        output=output,
        raw_responses=raw_output,
        limit=limit,
        predicted_count=len(predictions),
        llm_call_count=provider.call_count,
        usage=provider.usage,
        metadata={"base_url": provider.base_url},
    )


class LocalUnderstandingProvider:
    """OpenAI-compatible audio_url adapter for registered local models."""

    def __init__(
        self,
        *,
        model: str,
        provider_name: str,
        base_url: str,
        model_spec: LocalAudioModelSpec | None = None,
        thinking_mode: str | None = None,
        input_mode: str = "audio",
        client: Any | None = None,
        api_key: str | None = None,
        raw_response_output: Path | None = None,
    ) -> None:
        if input_mode not in _INPUT_MODES:
            raise UnderstandingProviderError(f"unsupported input mode: {input_mode!r}")
        self.model = model
        self.provider_name = provider_name
        self.model_spec = model_spec
        self.thinking_mode = thinking_mode
        self.input_mode = input_mode
        self.client = client
        self.api_key = api_key
        self.base_url = base_url
        self.raw_response_output = raw_response_output
        self.call_count = 0
        self.max_attempts = 3
        self.usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

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
            if (
                self.model_spec is not None
                and self.model_spec.max_audio_clips is not None
                and len(audio_paths) > self.model_spec.max_audio_clips
            ):
                raise UnderstandingProviderError(
                    f"{self.model} supports at most {self.model_spec.max_audio_clips} audio "
                    f"clip(s) per request; manifest row {cell_id!r} has {len(audio_paths)}"
                )
        else:
            audio_paths = []
        system_prompt = _system_prompt_with_setup(manifest_row)
        user_prompt = (
            transcript if transcript is not None else _vllm_user_text(manifest_row, audio_paths)
        )
        started = time.perf_counter()
        response = self._chat_completion_with_retries(
            cell_id, manifest_row, audio_paths, transcript=transcript
        )
        completion_time_s = round(time.perf_counter() - started, 3)
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
                response=response,
                metadata={
                    "base_url": self.base_url,
                    "thinking_mode": self.thinking_mode,
                    "input_mode": self.input_mode,
                },
            )
        raw_text = _response_text(response)
        payload = _parse_prediction_payload(raw_text, cell_id)
        return PredictionRow(
            cell_id=cell_id,
            predicted_action=payload["predicted_action"],
            answer_text=payload.get("answer_text"),
            tool_calls=payload.get("tool_calls", []),
            confidence=payload.get("confidence"),
            metadata={
                "provider": self.provider_name,
                "model": self.model,
                "base_url": self.base_url,
                "response_id": getattr(response, "id", None),
                "completion_time_s": completion_time_s,
                "audio_clip_count": len(audio_paths),
                "input_mode": self.input_mode,
                "guided_json": guided_json_enabled(),
                "prompts": _prompt_metadata(
                    provider=self.provider_name,
                    model=self.model,
                    mode="turn_based",
                    system=system_prompt,
                    request_rendering=user_prompt,
                ),
            },
        )

    def _chat_completion_with_retries(
        self,
        cell_id: str,
        row: dict[str, Any],
        audio_paths: list[Path],
        *,
        transcript: str | None = None,
    ) -> Any:
        request = self._chat_request(row, audio_paths, transcript=transcript)
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self._client().chat.completions.create(**request)
            except Exception as exc:
                if attempt == self.max_attempts:
                    raise UnderstandingProviderError(
                        f"{self.provider_name} request failed for {cell_id} "
                        f"after {attempt} attempt(s): {exc}"
                    ) from exc
                time.sleep(min(2 ** (attempt - 1), 10))
        raise UnderstandingProviderError(f"{self.provider_name} request failed for {cell_id}")

    def _chat_request(
        self, row: dict[str, Any], audio_paths: list[Path], *, transcript: str | None = None
    ) -> dict[str, Any]:
        system_prompt = _system_prompt_with_setup(row)
        system_prompt_mode = (
            self.model_spec.system_prompt_mode if self.model_spec is not None else "system"
        )
        if transcript is not None:
            # Transcript control: the whole labeled dialogue replaces the
            # interleaved audio history as one text-only user turn.
            history = [{"role": "user", "content": [{"type": "text", "text": transcript}]}]
            if system_prompt_mode == "first_user":
                history[0]["content"].insert(0, {"type": "text", "text": system_prompt})
        else:
            history = _vllm_history_messages(
                row,
                audio_paths,
                inline_system=system_prompt if system_prompt_mode == "first_user" else None,
            )
        messages = history
        if system_prompt_mode == "system":
            messages = [{"role": "system", "content": system_prompt}, *history]
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 2048,
            "timeout": 300,
        }
        if guided_json_enabled():
            # vLLM >=0.19 dropped extra_body guided_json for the OpenAI-compatible
            # json_schema structured-output API.
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "ib_prediction",
                    "schema": emit_prediction_tool()["parameters"],
                    "strict": True,
                },
            }
        if self.thinking_mode is not None:
            request["extra_body"] = {
                "chat_template_kwargs": {
                    "enable_thinking": self.thinking_mode == "enabled",
                }
            }
        return request

    def _client(self) -> Any:
        if self.client is not None:
            return self.client
        self.client = _openai_client(
            api_key=self.api_key
            or _env("LOCAL_AUDIO_UNDERSTANDING_API_KEY", "LOCAL_AUDIO_API_KEY", "EMPTY"),
            base_url=self.base_url,
            timeout=300,
        )
        return self.client

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
    provider: LocalUnderstandingProvider,
    rows: list[dict[str, Any]],
    manifest_path: Path,
    output: Path,
) -> list[PredictionRow]:
    existing = _existing_predictions(output)
    predictions: list[PredictionRow] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            cell_id = _required_str(row, "cell_id")
            print_eval_case_progress(
                provider.provider_name, provider.model, cell_id, index, len(rows)
            )
            if cell_id in existing:
                predictions.append(existing[cell_id])
                continue
            try:
                prediction = provider.predict(row, manifest_path=manifest_path)
            except PredictionFormatError as exc:
                prediction = _error_prediction(row, provider, exc)
            except UnderstandingProviderError as exc:
                # A server that keeps rejecting THIS input (4xx after retries) is a
                # per-cell defect, not an outage: degrade the row and keep the sweep
                # alive. Transport/5xx failures still abort the run.
                if not _client_rejected_input(exc):
                    raise
                prediction = _error_prediction(row, provider, exc)
            handle.write(f"{canonical_json(prediction)}\n")
            handle.flush()
            predictions.append(prediction)
    return predictions


_CLIENT_REJECT_MARKERS = ("Error code: 400", "Error code: 422", "BadRequestError")


def _client_rejected_input(error: Exception) -> bool:
    """Return whether the server rejected this specific request payload."""
    text = str(error)
    return any(marker in text for marker in _CLIENT_REJECT_MARKERS)


def _error_prediction(
    row: dict[str, Any],
    provider: LocalUnderstandingProvider,
    error: Exception,
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
            "base_url": provider.base_url,
            "provider_error": {
                "type": "prediction_format_error",
                "message": str(error),
            },
        },
    )


def _existing_predictions(output: Path) -> dict[str, PredictionRow]:
    if not output.exists() or output.stat().st_size == 0:
        return {}
    return {row.cell_id: row for row in read_predictions(output)}


def _vllm_history_messages(
    row: dict[str, Any],
    audio_paths: list[Path],
    *,
    inline_system: str | None = None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for message in history_messages(row, audio_paths):
        if message.role == "user":
            if message.audio_path is None:
                raise UnderstandingProviderError("user audio history message missing audio_path")
            messages.append({"role": "user", "content": [_audio_url_part(message.audio_path)]})
        elif message.assistant_text is not None:
            messages.append({"role": "assistant", "content": message.assistant_text})
    if inline_system is not None:
        first_user_index = next(
            (index for index, message in enumerate(messages) if message["role"] == "user"),
            None,
        )
        if first_user_index is None:
            raise UnderstandingProviderError("local audio history has no user message")
        system_part = {"type": "text", "text": inline_system}
        if first_user_index == 0:
            messages[0]["content"].insert(0, system_part)
        else:
            messages.insert(0, {"role": "user", "content": [system_part]})
        messages = _merge_adjacent_user_messages(messages)
    return messages


def _merge_adjacent_user_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return an alternating history accepted by Mistral/Voxtral templates."""
    merged: list[dict[str, Any]] = []
    for message in messages:
        if merged and message["role"] == "user" and merged[-1]["role"] == "user":
            merged[-1]["content"].extend(message["content"])
        else:
            merged.append(message)
    return merged


def _vllm_user_text(row: dict[str, Any], audio_paths: list[Path]) -> str:
    return _user_prompt_for(row, audio_paths)


def _audio_url_part(path: Path) -> dict[str, Any]:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "audio_url", "audio_url": {"url": f"{_mime_type(path)};base64,{data}"}}


def _mime_type(path: Path) -> str:
    return {"wav": "data:audio/wav", "mp3": "data:audio/mpeg"}[_audio_format(path)]


def _report(
    *,
    schema_version: str,
    provider: str,
    model: str,
    manifest: str | Path,
    output: str | Path,
    raw_responses: str | Path,
    limit: int | None,
    predicted_count: int,
    llm_call_count: int,
    usage: dict[str, int],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "provider": provider,
        "model": model,
        "manifest": str(manifest),
        "predictions": str(output),
        "raw_responses": str(raw_responses),
        "limit": limit,
        "predicted_count": predicted_count,
        "llm_call_count": llm_call_count,
        "usage": usage,
        "metadata": metadata,
    }


def _openai_client(*, api_key: str, base_url: str, timeout: int) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise UnderstandingProviderError("openai package is required") from exc
    return OpenAI(api_key=api_key, base_url=base_url.rstrip("/"), timeout=timeout)


def _env(primary: str, fallback: str, default: str) -> str:
    return os.getenv(primary) or os.getenv(fallback) or default
