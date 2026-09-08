"""openai_fullduplex — generate predictions from OpenAI Realtime full-duplex runs.

Calling spec:
    run_openai_fullduplex_predictions(
        manifest, output, model, limit=None, client=None, reasoning_effort=None
    ) -> dict

For each manifest row this runner obtains the ordinary base prediction using
text output. When live_probe=True and the row has a speak-time live expansion, it
then opens an audio-output response, waits until generated assistant audio reaches
the manifest anchor offset, injects the live event audio, and stores probe
observations under prediction.metadata.live_probe.

Side effects: may call the OpenAI Realtime API and writes prediction and sibling
raw-response JSONL artifacts.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from ib.eval_targets import RealtimeReasoningEffort
from ib.scoring.probe_sampling import inject_offset_s_for_row
from ib.io import canonical_json, write_jsonl
from ib.llm import render_prompt
from ib.models.raw_response import RawResponsePhase
from ib.predictions import PredictionRow, read_predictions
from ib.scoring.openai_understanding import (
    UnderstandingProviderError,
    _audio_paths_for,
    _limited_rows,
    _parse_prediction_payload,
    _prompt_metadata,
    _required_str,
    _run_root_for,
    _system_prompt_with_setup,
)
from ib.scoring.openai_realtime_capture import RealtimeCapture
from ib.scoring.openai_realtime_audio import (
    DEFAULT_REALTIME_SAMPLE_RATE,
    pcm16_delta_ms as _pcm16_delta_ms,
    wav_as_pcm16_base64 as _wav_as_pcm16_base64,
)
from ib.scoring.openai_realtime_error_io import (
    error_artifacts_for,
    write_prediction_format_failure,
)
from ib.scoring.multiturn_audio import history_messages, setup_prompt_for
from ib.scoring.native_tool_schema import (
    build_native_tool_bundle,
    openai_realtime_tools,
)
from ib.scoring.progress import print_eval_case_progress
from ib.scoring.raw_response_io import (
    append_raw_response,
    json_safe_response,
    raw_response_path,
)
from ib.scoring.realtime_prediction_tool import (
    EMIT_PREDICTION_TOOL_NAME,
    emit_prediction_tool,
    emit_prediction_tool_choice,
    raw_arguments_text,
)
from ib.scoring.smoke import _read_rows

OPENAI_FULLDUPLEX_PROVIDER = "openai_realtime"
DEFAULT_REALTIME_VOICE = "alloy"
DEFAULT_REALTIME_MAX_OUTPUT_TOKENS = 700
XHIGH_REALTIME_MAX_OUTPUT_TOKENS = 4_096
_REALTIME_PCM16 = {"type": "audio/pcm", "rate": DEFAULT_REALTIME_SAMPLE_RATE}


def _max_output_tokens(reasoning_effort: RealtimeReasoningEffort | None) -> int:
    """Give xhigh reasoning enough room without changing existing targets."""
    if reasoning_effort == "xhigh":
        return XHIGH_REALTIME_MAX_OUTPUT_TOKENS
    return DEFAULT_REALTIME_MAX_OUTPUT_TOKENS


class RealtimePredictionFormatError(UnderstandingProviderError):
    """Raised when Realtime returned a response that is not a valid prediction JSON."""


class OpenAiFullDuplexProvider:
    """OpenAI Realtime adapter for full-duplex MSI-Bench predictions."""

    def __init__(
        self,
        *,
        model: str,
        client: Any | None = None,
        api_key: str | None = None,
        voice: str = DEFAULT_REALTIME_VOICE,
        recv_timeout_s: float = 120.0,
        error_dir: Path | None = None,
        live_probe: bool = False,
        reasoning_effort: RealtimeReasoningEffort | None = None,
        raw_response_output: Path | None = None,
    ) -> None:
        self.model = model
        self.client = client
        self.api_key = api_key
        self.voice = voice
        self.recv_timeout_s = recv_timeout_s
        self.error_dir = error_dir
        self.call_count = 0
        self.live_probe = live_probe
        self.reasoning_effort = reasoning_effort
        self.raw_response_output = raw_response_output
        self.live_probe_count = 0
        self.usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def predict(self, manifest_row: dict[str, Any], *, manifest_path: Path) -> PredictionRow:
        """Return one full-duplex prediction row."""
        cell_id = _required_str(manifest_row, "cell_id")
        if manifest_row.get("logic_case_mode") == "speak_time_probe":
            return self._speak_probe_prediction(cell_id, manifest_row, manifest_path)
        audio_paths = _audio_paths_for(manifest_row, manifest_path)
        prompts = _prompt_metadata(
            provider=OPENAI_FULLDUPLEX_PROVIDER,
            model=self.model,
            mode="fullduplex",
            system=_json_instructions(_system_prompt_with_setup(manifest_row), attempt=1),
            request_rendering=_json_user_prompt(manifest_row, audio_paths, attempt=1),
        )
        base_started = time.perf_counter()
        base_payload, base_capture = self._base_prediction(manifest_row, manifest_path)
        metadata: dict[str, Any] = {
            "provider": OPENAI_FULLDUPLEX_PROVIDER,
            "model": self.model,
            "eval_mode": "fullduplex",
            "base_response_id": base_capture.response_id,
            "base_response_status": base_capture.response_status,
            "completion_time_s": round(time.perf_counter() - base_started, 3),
            "reasoning_effort": self.reasoning_effort,
            "prompts": prompts,
        }
        if self.live_probe and _has_live_bystander_probe(manifest_row):
            prompts["live_system"] = _live_instructions(manifest_row, audio_paths)
            prompts["live_request_rendering"] = _live_user_prompt(manifest_row, audio_paths)
            metadata["live_probe"] = self._live_probe(manifest_row, manifest_path)
        return PredictionRow(
            cell_id=cell_id,
            predicted_action=base_payload["predicted_action"],
            answer_text=base_payload.get("answer_text"),
            tool_calls=base_payload.get("tool_calls", []),
            confidence=base_payload.get("confidence"),
            metadata=metadata,
        )

    def _base_prediction(
        self, manifest_row: dict[str, Any], manifest_path: Path
    ) -> tuple[dict[str, Any], "RealtimeCapture"]:
        cell_id = _required_str(manifest_row, "cell_id")
        last_error: UnderstandingProviderError | None = None
        for attempt in range(1, 4):
            capture = self._base_capture(manifest_row, manifest_path, attempt=attempt)
            self._record_raw_capture(cell_id, "base_prediction", capture, attempt=attempt)
            try:
                return _parse_prediction_payload(capture.prediction_text(), cell_id), capture
            except UnderstandingProviderError as exc:
                write_prediction_format_failure(
                    self.error_dir,
                    provider=OPENAI_FULLDUPLEX_PROVIDER,
                    model=self.model,
                    cell_id=cell_id,
                    attempt=attempt,
                    capture=capture,
                    error=exc,
                )
                last_error = exc
        raise RealtimePredictionFormatError(
            f"invalid Realtime prediction for {cell_id} after 3 attempts: {last_error}"
        )

    def _base_capture(
        self, manifest_row: dict[str, Any], manifest_path: Path, *, attempt: int
    ) -> "RealtimeCapture":
        cell_id = _required_str(manifest_row, "cell_id")
        audio_paths = _audio_paths_for(manifest_row, manifest_path)
        if not audio_paths:
            raise UnderstandingProviderError(f"manifest row {cell_id!r} has no audio paths")
        instructions = _json_instructions(_system_prompt_with_setup(manifest_row), attempt=attempt)
        with self._client().realtime.connect(model=self.model) as connection:
            emit_tool = emit_prediction_tool()
            connection.session.update(
                session=_text_session(
                    instructions,
                    tools=[emit_tool],
                    reasoning_effort=self.reasoning_effort,
                )
            )
            for item in _realtime_history_items(manifest_row, audio_paths):
                connection.conversation.item.create(item=item)
            connection.response.create(
                response={
                    "output_modalities": ["text"],
                    "instructions": instructions,
                    "tools": [emit_tool],
                    "tool_choice": emit_prediction_tool_choice(),
                    "max_output_tokens": _max_output_tokens(self.reasoning_effort),
                }
            )
            capture = _collect_response(connection, timeout_s=self.recv_timeout_s)
        self.call_count += 1
        self._record_usage(capture)
        return capture

    def _live_probe(self, manifest_row: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
        cell_id = _required_str(manifest_row, "cell_id")
        audio_paths = _audio_paths_for(manifest_row, manifest_path)
        live_paths = _live_event_audio_paths(manifest_row, manifest_path)
        # Injection timing is a benchmark parameter sampled per case; the built
        # anchor's uniform start_offset_seconds never encoded per-case intent.
        offset_s = inject_offset_s_for_row(manifest_row)
        live_probe: dict[str, Any] = {
            "type": "bystander_interference_suppression",
            "executed": False,
            "assistant_speech_started": False,
            "injection_started": False,
            "injection_scheduled_offset_s": offset_s,
            "live_event_audio_paths": [str(path) for path in live_paths],
            "interrupted": False,
            "followed_bystander": False,
        }
        if not live_paths:
            live_probe["reason"] = "missing_live_event_audio"
            return live_probe
        with self._client().realtime.connect(model=self.model) as connection:
            tools, tool_name_map = _realtime_tools(manifest_row.get("logic_available_functions"))
            connection.session.update(
                session=_audio_session(
                    _live_instructions(manifest_row, audio_paths),
                    voice=self.voice,
                    tools=tools,
                    reasoning_effort=self.reasoning_effort,
                )
            )
            for item in _realtime_history_items(manifest_row, audio_paths):
                connection.conversation.item.create(item=item)
            connection.response.create(
                response={
                    "output_modalities": ["audio"],
                    "tools": tools,
                    "tool_choice": "auto",
                    "audio": {
                        "output": {
                            "format": _REALTIME_PCM16,
                            "voice": self.voice,
                        }
                    },
                }
            )
            capture = _collect_response(
                connection,
                timeout_s=self.recv_timeout_s,
                inject_after_audio_ms=round(offset_s * 1000),
                injection_audio_base64=[_wav_as_pcm16_base64(path) for path in live_paths],
                tool_name_map=tool_name_map,
            )
        self.call_count += 1
        self.live_probe_count += 1
        self._record_usage(capture)
        self._record_raw_capture(cell_id, "live_probe", capture)
        live_tool_calls = capture.tool_calls
        expected_tool_calls = manifest_row.get("logic_expected_tool_calls") or []
        followed_bystander = bool(
            live_tool_calls and not _tool_names_match(expected_tool_calls, live_tool_calls)
        )
        live_probe.update(
            {
                "executed": capture.injection_started,
                "assistant_speech_started": capture.assistant_audio_ms > 0,
                "injection_started": capture.injection_started,
                "assistant_audio_ms_before_injection": capture.injected_at_audio_ms,
                "assistant_audio_ms_total": capture.assistant_audio_ms,
                "predicted_action": "respond" if capture.has_output() else "silent",
                "answer_text": capture.answer_text(),
                "tool_calls": live_tool_calls,
                "response_id": capture.response_id,
                "response_status": capture.response_status,
                "interrupted": capture.response_status in {"cancelled", "failed", "incomplete"},
                "followed_bystander": followed_bystander,
            }
        )
        if not capture.injection_started:
            live_probe["reason"] = "assistant_finished_before_injection"
        return live_probe

    def _speak_probe_prediction(
        self, cell_id: str, manifest_row: dict[str, Any], manifest_path: Path
    ) -> PredictionRow:
        """Probe-only row for logic_case_mode=speak_time_probe: no base session."""
        audio_paths = _audio_paths_for(manifest_row, manifest_path)
        live_probe = self._live_probe(manifest_row, manifest_path)
        prompts = _prompt_metadata(
            provider=OPENAI_FULLDUPLEX_PROVIDER,
            model=self.model,
            mode="fullduplex",
            system=_live_instructions(manifest_row, audio_paths),
            request_rendering=_live_user_prompt(manifest_row, audio_paths),
        )
        return PredictionRow(
            cell_id=cell_id,
            predicted_action=live_probe.get("predicted_action") or "silent",
            answer_text=live_probe.get("answer_text") or None,
            tool_calls=live_probe.get("tool_calls") or [],
            metadata={
                "provider": OPENAI_FULLDUPLEX_PROVIDER,
                "model": self.model,
                "eval_mode": "fullduplex",
                "reasoning_effort": self.reasoning_effort,
                "prompts": prompts,
                "live_probe": live_probe,
            },
        )

    def _client(self) -> Any:
        if self.client is not None:
            return self.client
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise UnderstandingProviderError(
                "OPENAI_API_KEY is required for OpenAI full-duplex evaluation"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise UnderstandingProviderError("openai package is required") from exc
        self.client = OpenAI(api_key=api_key)
        return self.client

    def _record_usage(self, capture: "RealtimeCapture") -> None:
        for key in self.usage:
            self.usage[key] += int(capture.usage.get(key, 0) or 0)

    def _record_raw_capture(
        self,
        cell_id: str,
        phase: RawResponsePhase,
        capture: "RealtimeCapture",
        *,
        attempt: int = 1,
    ) -> None:
        if self.raw_response_output is None:
            return
        append_raw_response(
            self.raw_response_output,
            cell_id=cell_id,
            provider=OPENAI_FULLDUPLEX_PROVIDER,
            model=self.model,
            eval_mode="fullduplex",
            phase=phase,
            attempt=attempt,
            response_id=capture.response_id,
            response={"events": capture.raw_events},
            metadata={"response_status": capture.response_status},
        )


def run_openai_fullduplex_predictions(
    manifest: str | Path,
    output: str | Path,
    *,
    model: str | None = None,
    limit: int | None = None,
    client: Any | None = None,
    live_probe: bool = False,
    reasoning_effort: RealtimeReasoningEffort | None = None,
) -> dict[str, Any]:
    """Generate full-duplex predictions for up to ``limit`` manifest rows."""
    if not model:
        raise UnderstandingProviderError("full-duplex OpenAI Realtime eval requires --model")
    rows = _limited_rows(_read_rows(manifest), limit)
    raw_output = raw_response_path(output)
    provider = OpenAiFullDuplexProvider(
        model=model,
        client=client,
        error_dir=Path(output).parent / "prediction_errors",
        live_probe=live_probe,
        reasoning_effort=reasoning_effort,
        raw_response_output=raw_output,
    )
    predictions = _predict_with_resume(provider, rows, Path(manifest), Path(output))
    write_jsonl(output, predictions)
    return {
        "schema_version": "ib.openai_fullduplex.v1",
        "provider": OPENAI_FULLDUPLEX_PROVIDER,
        "model": provider.model,
        "reasoning_effort": provider.reasoning_effort,
        "manifest": str(manifest),
        "predictions": str(output),
        "raw_responses": str(raw_output),
        "limit": limit,
        "predicted_count": len(predictions),
        "llm_call_count": provider.call_count,
        "live_probe_enabled": live_probe,
        "live_probe_count": provider.live_probe_count,
        "usage": provider.usage,
        "metadata": {
            "eval_mode": "fullduplex",
            "live_probe_enabled": live_probe,
            "reasoning_effort": provider.reasoning_effort,
        },
    }

def _collect_response(
    connection: Any,
    *,
    timeout_s: float,
    inject_after_audio_ms: int | None = None,
    injection_audio_base64: list[str] | None = None,
    tool_name_map: dict[str, str] | None = None,
) -> RealtimeCapture:
    capture = RealtimeCapture()
    while True:
        event = _recv_event(connection, timeout_s=timeout_s)
        capture.raw_events.append(json_safe_response(event))
        event_type = _event_type(event)
        # GA realtime API renamed the beta audio events to output_audio.*
        if event_type in ("response.audio.delta", "response.output_audio.delta"):
            capture.assistant_audio_ms += _pcm16_delta_ms(getattr(event, "delta", ""))
            if _should_inject(capture, inject_after_audio_ms, injection_audio_base64):
                for audio in injection_audio_base64 or []:
                    connection.input_audio_buffer.append(audio=audio)
                connection.input_audio_buffer.commit()
                capture.injection_started = True
                capture.injected_at_audio_ms = capture.assistant_audio_ms
        elif event_type == "response.text.done":
            capture.text_parts.append(str(getattr(event, "text", "") or ""))
        elif event_type in (
            "response.audio_transcript.done",
            "response.output_audio_transcript.done",
        ):
            capture.audio_transcripts.append(str(getattr(event, "transcript", "") or ""))
        elif event_type == "response.function_call_arguments.done":
            _append_tool_call_arguments(capture, event, tool_name_map=tool_name_map)
        elif event_type == "response.output_item.done":
            _append_tool_call_from_item(
                capture, getattr(event, "item", None), tool_name_map=tool_name_map
            )
        elif event_type == "response.done":
            response = getattr(event, "response", None)
            capture.response_id = getattr(response, "id", None)
            capture.response_status = getattr(response, "status", None)
            _append_response_outputs(capture, response, tool_name_map=tool_name_map)
            capture.usage = _usage_dict(getattr(response, "usage", None))
            return capture
        elif event_type == "error":
            error = getattr(event, "error", None)
            raise UnderstandingProviderError(f"OpenAI Realtime error: {error}")


def _text_session(
    instructions: str,
    *,
    tools: list[dict[str, Any]] | None = None,
    reasoning_effort: RealtimeReasoningEffort | None = None,
) -> dict[str, Any]:
    """Return current Realtime session schema for text output over audio input."""
    session = {
        "type": "realtime",
        "instructions": instructions,
        "output_modalities": ["text"],
        "audio": {"input": {"format": _REALTIME_PCM16, "turn_detection": None}},
    }
    if tools:
        session["tools"] = tools
        session["tool_choice"] = emit_prediction_tool_choice()
    if reasoning_effort is not None:
        session["reasoning"] = {"effort": reasoning_effort}
    return session


def _audio_session(
    instructions: str,
    *,
    voice: str,
    tools: list[dict[str, Any]],
    reasoning_effort: RealtimeReasoningEffort | None = None,
) -> dict[str, Any]:
    """Return current Realtime session schema for audio output and live probes."""
    session = {
        "type": "realtime",
        "instructions": instructions,
        "output_modalities": ["audio"],
        "audio": {
            "input": {"format": _REALTIME_PCM16, "turn_detection": None},
            "output": {"format": _REALTIME_PCM16, "voice": voice},
        },
        "tools": tools,
        "tool_choice": "auto",
    }
    if reasoning_effort is not None:
        session["reasoning"] = {"effort": reasoning_effort}
    return session


def _json_instructions(system_prompt: str, *, attempt: int = 1) -> str:
    del attempt
    return system_prompt


def _json_user_prompt(row: dict[str, Any], audio_paths: list[Path], *, attempt: int = 1) -> str:
    del row, audio_paths, attempt
    return ""


def _recv_event(connection: Any, *, timeout_s: float) -> Any:
    try:
        raw = connection._connection.recv(timeout=timeout_s, decode=False)
    except TypeError:
        raw = connection.recv_bytes()
    return connection.parse_event(raw)


def _event_type(event: Any) -> str:
    value = getattr(event, "type", None)
    return value if isinstance(value, str) else ""


def _should_inject(
    capture: RealtimeCapture, offset_ms: int | None, injection_audio: list[str] | None
) -> bool:
    return bool(
        offset_ms is not None
        and injection_audio
        and not capture.injection_started
        and capture.assistant_audio_ms >= offset_ms
    )


def _append_response_outputs(
    capture: RealtimeCapture, response: Any, *, tool_name_map: dict[str, str] | None = None
) -> None:
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) == "function_call":
            _append_tool_call_from_item(capture, item, tool_name_map=tool_name_map)
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None) or getattr(content, "transcript", None)
            if isinstance(text, str) and text.strip():
                capture.text_parts.append(text.strip())


def _append_tool_call_arguments(
    capture: RealtimeCapture, event: Any, *, tool_name_map: dict[str, str] | None = None
) -> None:
    name = getattr(event, "name", None)
    if not isinstance(name, str) or not name:
        return
    arguments = getattr(event, "arguments", None)
    _append_function_call(capture, name, arguments, tool_name_map=tool_name_map)


def _append_tool_call_from_item(
    capture: RealtimeCapture, item: Any, *, tool_name_map: dict[str, str] | None = None
) -> None:
    if item is None or getattr(item, "type", None) != "function_call":
        return
    name = getattr(item, "name", None)
    if not isinstance(name, str) or not name:
        return
    _append_function_call(
        capture, name, getattr(item, "arguments", None), tool_name_map=tool_name_map
    )


def _append_function_call(
    capture: RealtimeCapture,
    name: str,
    raw_arguments: Any,
    *,
    tool_name_map: dict[str, str] | None = None,
) -> None:
    if name == EMIT_PREDICTION_TOOL_NAME:
        capture.emit_prediction_arguments = raw_arguments_text(raw_arguments)
        return
    name = (tool_name_map or {}).get(name, name)
    arguments = _parse_arguments(raw_arguments)
    call = {"name": name, "arguments": arguments}
    if call not in capture.tool_calls:
        capture.tool_calls.append(call)


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"value": raw}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return {}


def _usage_dict(usage: Any) -> dict[str, int]:
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump(mode="json")
    if not isinstance(usage, dict):
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    input_tokens = int(usage.get("input_tokens") or usage.get("input_token_count") or 0)
    output_tokens = int(usage.get("output_tokens") or usage.get("output_token_count") or 0)
    total_tokens = int(usage.get("total_tokens") or input_tokens + output_tokens)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _predict_with_resume(
    provider: OpenAiFullDuplexProvider,
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
                OPENAI_FULLDUPLEX_PROVIDER, provider.model, cell_id, index, len(rows)
            )
            if cell_id in existing:
                predictions.append(existing[cell_id])
                continue
            try:
                prediction = provider.predict(row, manifest_path=manifest_path)
            except RealtimePredictionFormatError as exc:
                prediction = _error_prediction(row, provider, exc)
            handle.write(f"{canonical_json(prediction)}\n")
            handle.flush()
            predictions.append(prediction)
    return predictions


def _existing_predictions(output: Path) -> dict[str, PredictionRow]:
    if not output.exists() or output.stat().st_size == 0:
        return {}
    return {row.cell_id: row for row in read_predictions(output)}


def _error_prediction(
    row: dict[str, Any],
    provider: OpenAiFullDuplexProvider,
    error: RealtimePredictionFormatError,
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
            "provider": OPENAI_FULLDUPLEX_PROVIDER,
            "model": provider.model,
            "eval_mode": "fullduplex",
            "provider_error": {
                "type": "prediction_format_error",
                "message": str(error),
                "error_artifacts": error_artifacts_for(provider.error_dir, cell_id),
            },
        },
    )


def _live_event_audio_paths(row: dict[str, Any], manifest_path: Path) -> list[Path]:
    root = _run_root_for(manifest_path)
    raw_paths = row.get("logic_live_event_audio_paths") or []
    if not isinstance(raw_paths, list):
        return []
    return [Path(path) if Path(path).is_absolute() else root / str(path) for path in raw_paths]


def _has_live_bystander_probe(row: dict[str, Any]) -> bool:
    if row.get("logic_live_expansion_type") == "bystander_interference_suppression":
        return True
    types = row.get("logic_live_expansion_types")
    return isinstance(types, list) and "bystander_interference_suppression" in types


def _live_system_prompt(row: dict[str, Any]) -> str:
    return render_prompt(
        "understanding/system.j2",
        {"kind": "live", "scene": row.get("scene") or "ambient interaction"},
    )


def _live_user_prompt(row: dict[str, Any], audio_paths: list[Path]) -> str:
    functions = row.get("logic_available_functions")
    return render_prompt(
        "understanding/user.j2",
        {
            "kind": "live",
            "audio_count": len(audio_paths),
            "available_functions": json.dumps(functions or [], ensure_ascii=False),
        },
    )


def _live_instructions(row: dict[str, Any], audio_paths: list[Path]) -> str:
    return "\n\n".join(
        [setup_prompt_for(row), _live_system_prompt(row), _live_user_prompt(row, audio_paths)]
    )


def _realtime_history_items(row: dict[str, Any], audio_paths: list[Path]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in history_messages(row, audio_paths):
        if message.role == "user":
            if message.audio_path is None:
                raise UnderstandingProviderError("user audio history message missing audio_path")
            items.append(
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_audio", "audio": _wav_as_pcm16_base64(message.audio_path)}
                    ],
                }
            )
        elif message.assistant_text is not None:
            items.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": message.assistant_text}],
                }
            )
    return items


def _realtime_tools(functions: Any) -> tuple[list[dict[str, Any]], dict[str, str]]:
    bundle = build_native_tool_bundle(functions)
    return openai_realtime_tools(bundle), bundle.name_map


def _tool_names_match(expected: Any, predicted: Any) -> bool:
    return sorted(_tool_names(expected)) == sorted(_tool_names(predicted))


def _tool_names(calls: Any) -> list[str]:
    if not isinstance(calls, list):
        return []
    return [
        call.get("name")
        for call in calls
        if isinstance(call, dict) and isinstance(call.get("name"), str)
    ]
