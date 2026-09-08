"""gemini_fullduplex — generate full-duplex predictions with Gemini Live.

Calling spec:
    run_gemini_fullduplex_predictions(
        manifest, output, model, limit=None, client=None, thinking_mode=None
    ) -> dict

Inputs are built manifest rows with testcase audio paths. Outputs are strict
PredictionRow JSONL rows aligned to the manifest. When live_probe=True, rows
with speak-time live expansions receive an additional audio-output probe call.
Side effects: may call Gemini Live API and writes/resumes prediction and sibling
raw-response JSONL artifacts.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import importlib
import os
from pathlib import Path
from typing import Any

from ib.eval_targets import GeminiThinkingMode
from ib.scoring.probe_sampling import inject_offset_s_for_row
from ib.io import canonical_json, write_jsonl
from ib.llm import render_prompt
from ib.models.raw_response import RawResponsePhase
from ib.predictions import PredictionRow, read_predictions
from ib.scoring.gemini_understanding import _mime_type, _normalize_gemini_model
from ib.scoring.openai_realtime_audio import (
    DEFAULT_REALTIME_SAMPLE_RATE,
    float_samples_to_pcm16 as _float_samples_to_pcm16,
    resample as _resample,
)
from ib.scoring.openai_fullduplex import (
    _live_event_audio_paths,
)
from ib.scoring.openai_understanding import (
    UnderstandingProviderError,
    _audio_paths_for,
    _limited_rows,
    _parse_prediction_payload,
    _prompt_metadata,
    _required_str,
    _system_prompt_with_setup,
)
from ib.scoring.multiturn_audio import history_messages, setup_prompt_for
from ib.scoring.native_tool_schema import build_native_tool_bundle, gemini_live_tools
from ib.scoring.progress import print_eval_case_progress
from ib.scoring.raw_response_io import (
    append_raw_response,
    json_safe_response,
    raw_response_path,
)
from ib.scoring.smoke import _read_rows
from ib.audio.wav_io import read_mono_pcm16_wav

GEMINI_FULLDUPLEX_PROVIDER = "gemini_live"
DEFAULT_GEMINI_LIVE_VOICE = "Zephyr"


class GeminiLiveTimeoutError(UnderstandingProviderError):
    """Raised when a Gemini Live turn does not produce a complete response in time."""


class GeminiLiveProvider:
    """Gemini Live adapter for full-duplex MSI-Bench predictions."""

    def __init__(
        self,
        *,
        model: str,
        client: Any | None = None,
        api_key: str | None = None,
        voice: str = DEFAULT_GEMINI_LIVE_VOICE,
        recv_timeout_s: float = 120.0,
        live_probe: bool = False,
        thinking_mode: GeminiThinkingMode | None = None,
        raw_response_output: Path | None = None,
    ) -> None:
        self.model = _normalize_gemini_model(model)
        self.client = client
        self.api_key = api_key
        self.voice = voice
        self.recv_timeout_s = recv_timeout_s
        self.call_count = 0
        self.live_probe = live_probe
        self.thinking_mode = thinking_mode
        self.raw_response_output = raw_response_output
        self.live_probe_count = 0
        self.usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    async def predict(self, manifest_row: dict[str, Any], *, manifest_path: Path) -> PredictionRow:
        """Return one Gemini Live full-duplex prediction row."""
        cell_id = _required_str(manifest_row, "cell_id")
        if manifest_row.get("logic_case_mode") == "speak_time_probe":
            raise UnderstandingProviderError(
                "speak_time_probe rows are not supported for gemini_live targets yet; "
                "filter them out with --case-modes"
            )
        audio_paths = _audio_paths_for(manifest_row, manifest_path)
        prompts = _prompt_metadata(
            provider=GEMINI_FULLDUPLEX_PROVIDER,
            model=self.model,
            mode="fullduplex",
            system=_json_instructions(_system_prompt_with_setup(manifest_row)),
            request_rendering=_json_user_prompt(manifest_row, audio_paths),
        )
        base_payload, base_capture = await self._base_prediction(manifest_row, manifest_path)
        metadata: dict[str, Any] = {
            "provider": GEMINI_FULLDUPLEX_PROVIDER,
            "model": self.model,
            "eval_mode": "fullduplex",
            "thinking_mode": self.thinking_mode,
            "base_turn_complete": base_capture.turn_complete,
            "prompts": prompts,
        }
        if self.live_probe and _has_live_bystander_probe(manifest_row):
            prompts["live_system"] = _live_system_prompt(manifest_row)
            prompts["live_request_rendering"] = _live_user_prompt(manifest_row, audio_paths)
            metadata["live_probe"] = await self._live_probe(manifest_row, manifest_path)
        return PredictionRow(
            cell_id=cell_id,
            predicted_action=base_payload["predicted_action"],
            answer_text=base_payload.get("answer_text"),
            tool_calls=base_payload.get("tool_calls", []),
            confidence=base_payload.get("confidence"),
            metadata=metadata,
        )

    async def _base_prediction(
        self, manifest_row: dict[str, Any], manifest_path: Path
    ) -> tuple[dict[str, Any], "GeminiLiveCapture"]:
        cell_id = _required_str(manifest_row, "cell_id")
        audio_paths = _audio_paths_for(manifest_row, manifest_path)
        instructions = _json_instructions(_system_prompt_with_setup(manifest_row))
        async with self._client().aio.live.connect(
            model=self.model,
            config=_with_thinking_config(
                {
                    "response_modalities": ["TEXT"],
                    "system_instruction": instructions,
                    "temperature": 0,
                    "max_output_tokens": 4096 if self.thinking_mode == "enabled" else 1400,
                },
                self.thinking_mode,
            ),
        ) as session:
            await session.send_client_content(
                turns=_gemini_contents(manifest_row, audio_paths), turn_complete=True
            )
            capture = await _collect_gemini_response(session, timeout_s=self.recv_timeout_s)
        self.call_count += 1
        self._record_usage(capture)
        self._record_raw_capture(cell_id, "base_prediction", capture)
        return _parse_prediction_payload(capture.text(), cell_id), capture

    async def _live_probe(
        self, manifest_row: dict[str, Any], manifest_path: Path
    ) -> dict[str, Any]:
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
        native_bundle = build_native_tool_bundle(manifest_row.get("logic_available_functions"))
        async with self._client().aio.live.connect(
            model=self.model,
            config=_with_thinking_config(
                {
                    "response_modalities": ["AUDIO"],
                    "system_instruction": _live_instructions(manifest_row, audio_paths),
                    "temperature": 0,
                    "tools": gemini_live_tools(native_bundle),
                    "output_audio_transcription": {},
                    "speech_config": {
                        "voice_config": {"prebuilt_voice_config": {"voice_name": self.voice}}
                    },
                },
                self.thinking_mode,
            ),
        ) as session:
            await session.send_client_content(
                turns=_gemini_contents(manifest_row, audio_paths), turn_complete=True
            )
            capture = await _collect_gemini_response(
                session,
                timeout_s=self.recv_timeout_s,
                inject_after_audio_ms=round(offset_s * 1000),
                injection_audio=[_wav_as_pcm16_bytes(path) for path in live_paths],
                tool_name_map=native_bundle.name_map,
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
                "answer_text": capture.text() or None,
                "tool_calls": live_tool_calls,
                "interrupted": capture.interrupted,
                "followed_bystander": followed_bystander,
            }
        )
        if not capture.injection_started:
            live_probe["reason"] = "assistant_finished_before_injection"
        return live_probe

    def _client(self) -> Any:
        if self.client is not None:
            return self.client
        api_key = self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise UnderstandingProviderError(
                "GEMINI_API_KEY or GOOGLE_API_KEY is required for Gemini Live evaluation"
            )
        try:
            genai: Any = importlib.import_module("google.genai")
        except ImportError as exc:
            raise UnderstandingProviderError("google-genai package is required") from exc
        self.client = genai.Client(api_key=api_key)
        return self.client

    def _record_usage(self, capture: "GeminiLiveCapture") -> None:
        for key in self.usage:
            self.usage[key] += int(capture.usage.get(key, 0) or 0)

    def _record_raw_capture(
        self, cell_id: str, phase: RawResponsePhase, capture: "GeminiLiveCapture"
    ) -> None:
        if self.raw_response_output is None:
            return
        append_raw_response(
            self.raw_response_output,
            cell_id=cell_id,
            provider=GEMINI_FULLDUPLEX_PROVIDER,
            model=self.model,
            eval_mode="fullduplex",
            phase=phase,
            response={"messages": capture.raw_messages},
            metadata={
                "thinking_mode": self.thinking_mode,
                "turn_complete": capture.turn_complete,
                "interrupted": capture.interrupted,
            },
        )


def run_gemini_fullduplex_predictions(
    manifest: str | Path,
    output: str | Path,
    *,
    model: str | None = None,
    limit: int | None = None,
    client: Any | None = None,
    recv_timeout_s: float = 120.0,
    live_probe: bool = False,
    thinking_mode: GeminiThinkingMode | None = None,
) -> dict[str, Any]:
    """Generate Gemini Live full-duplex predictions for up to ``limit`` rows."""
    if not model:
        raise UnderstandingProviderError("Gemini Live full-duplex eval requires --model")
    rows = _limited_rows(_read_rows(manifest), limit)
    raw_output = raw_response_path(output)
    provider = GeminiLiveProvider(
        model=model,
        client=client,
        recv_timeout_s=recv_timeout_s,
        live_probe=live_probe,
        thinking_mode=thinking_mode,
        raw_response_output=raw_output,
    )
    predictions = asyncio.run(_apredict_with_resume(provider, rows, Path(manifest), Path(output)))
    write_jsonl(output, predictions)
    return {
        "schema_version": "ib.gemini_fullduplex.v1",
        "provider": GEMINI_FULLDUPLEX_PROVIDER,
        "model": provider.model,
        "manifest": str(manifest),
        "predictions": str(output),
        "raw_responses": str(raw_output),
        "limit": limit,
        "predicted_count": len(predictions),
        "llm_call_count": provider.call_count,
        "live_probe_enabled": live_probe,
        "live_probe_count": provider.live_probe_count,
        "thinking_mode": provider.thinking_mode,
        "usage": provider.usage,
        "metadata": {
            "eval_mode": "fullduplex",
            "live_probe_enabled": live_probe,
            "thinking_mode": provider.thinking_mode,
        },
    }


async def _apredict_with_resume(
    provider: GeminiLiveProvider,
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
                GEMINI_FULLDUPLEX_PROVIDER, provider.model, cell_id, index, len(rows)
            )
            if cell_id in existing:
                predictions.append(existing[cell_id])
                continue
            try:
                prediction = await provider.predict(row, manifest_path=manifest_path)
            except GeminiLiveTimeoutError as exc:
                prediction = _timeout_prediction(row, provider, exc)
            handle.write(f"{canonical_json(prediction)}\n")
            handle.flush()
            predictions.append(prediction)
    return predictions


def _timeout_prediction(
    row: dict[str, Any],
    provider: GeminiLiveProvider,
    error: GeminiLiveTimeoutError,
) -> PredictionRow:
    """Return a valid prediction row for one timed-out Gemini Live testcase."""
    cell_id = _required_str(row, "cell_id")
    return PredictionRow(
        cell_id=cell_id,
        predicted_action="silent",
        answer_text=None,
        tool_calls=[],
        confidence=0.0,
        metadata={
            "provider": GEMINI_FULLDUPLEX_PROVIDER,
            "model": provider.model,
            "eval_mode": "fullduplex",
            "thinking_mode": provider.thinking_mode,
            "provider_error": {
                "type": "gemini_live_timeout",
                "message": str(error),
            },
        },
    )


def _with_thinking_config(
    config: dict[str, Any], thinking_mode: GeminiThinkingMode | None
) -> dict[str, Any]:
    """Return a copied Live config with an explicit Gemini thinking variant."""
    configured = dict(config)
    if thinking_mode is not None:
        configured["thinking_config"] = {"include_thoughts": thinking_mode == "enabled"}
    return configured


@dataclass
class GeminiLiveCapture:
    """Collected fields from one Gemini Live turn."""

    text_parts: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    raw_messages: list[Any] = field(default_factory=list)
    assistant_audio_ms: int = 0
    injection_started: bool = False
    injected_at_audio_ms: int | None = None
    interrupted: bool = False
    turn_complete: bool = False

    def text(self) -> str:
        return "\n".join(part for part in self.text_parts if part).strip()

    def has_output(self) -> bool:
        return bool(self.text() or self.tool_calls or self.assistant_audio_ms > 0)


async def _collect_gemini_response(
    session: Any,
    *,
    timeout_s: float,
    inject_after_audio_ms: int | None = None,
    injection_audio: list[bytes] | None = None,
    tool_name_map: dict[str, str] | None = None,
) -> GeminiLiveCapture:
    capture = GeminiLiveCapture()
    async for message in _timeout_iter(session.receive(), timeout_s=timeout_s):
        capture.raw_messages.append(json_safe_response(message))
        _capture_message(capture, message, tool_name_map=tool_name_map)
        if _should_inject(capture, inject_after_audio_ms, injection_audio):
            for audio in injection_audio or []:
                await session.send_realtime_input(
                    audio={
                        "data": audio,
                        "mime_type": f"audio/pcm;rate={DEFAULT_REALTIME_SAMPLE_RATE}",
                    }
                )
            await session.send_realtime_input(audio_stream_end=True)
            capture.injection_started = True
            capture.injected_at_audio_ms = capture.assistant_audio_ms
        if capture.turn_complete:
            break
    return capture


async def _timeout_iter(source: Any, *, timeout_s: float):
    iterator = source.__aiter__()
    while True:
        try:
            yield await asyncio.wait_for(iterator.__anext__(), timeout=timeout_s)
        except StopAsyncIteration:
            return
        except asyncio.TimeoutError as exc:
            raise GeminiLiveTimeoutError("Gemini Live response timed out") from exc


def _capture_message(
    capture: GeminiLiveCapture,
    message: Any,
    *,
    tool_name_map: dict[str, str] | None = None,
) -> None:
    usage = getattr(message, "usage_metadata", None)
    if usage is not None:
        capture.usage = _usage_dict(usage)
    tool_call = getattr(message, "tool_call", None)
    for call in getattr(tool_call, "function_calls", []) or []:
        _append_function_call(capture, call, tool_name_map=tool_name_map)
    content = getattr(message, "server_content", None)
    if content is None:
        _append_direct_text(capture, message)
        return
    capture.interrupted = capture.interrupted or bool(getattr(content, "interrupted", False))
    capture.turn_complete = capture.turn_complete or bool(getattr(content, "turn_complete", False))
    transcription = getattr(content, "output_transcription", None)
    transcript_text = getattr(transcription, "text", None)
    has_structured_text = False
    if isinstance(transcript_text, str) and transcript_text.strip():
        capture.text_parts.append(transcript_text.strip())
        has_structured_text = True
    model_turn = getattr(content, "model_turn", None)
    for part in getattr(model_turn, "parts", []) or []:
        text = getattr(part, "text", None)
        if isinstance(text, str) and text.strip():
            has_structured_text = True
            if getattr(part, "thought", False) is not True:
                capture.text_parts.append(text.strip())
        inline = getattr(part, "inline_data", None)
        data = getattr(inline, "data", None)
        mime_type = str(getattr(inline, "mime_type", "") or "")
        if data and mime_type.startswith("audio/"):
            capture.assistant_audio_ms += _pcm_audio_ms(data, mime_type)
        function_call = getattr(part, "function_call", None)
        if function_call is not None:
            _append_function_call(capture, function_call, tool_name_map=tool_name_map)
    if not has_structured_text:
        _append_direct_text(capture, message)


def _append_direct_text(capture: GeminiLiveCapture, message: Any) -> None:
    direct_text = getattr(message, "text", None)
    if isinstance(direct_text, str) and direct_text.strip():
        capture.text_parts.append(direct_text.strip())


def _append_function_call(
    capture: GeminiLiveCapture,
    call: Any,
    *,
    tool_name_map: dict[str, str] | None = None,
) -> None:
    name = getattr(call, "name", None)
    if not isinstance(name, str) or not name:
        return
    name = (tool_name_map or {}).get(name, name)
    args = getattr(call, "args", None)
    tool_call = {"name": name, "arguments": args if isinstance(args, dict) else {}}
    if tool_call not in capture.tool_calls:
        capture.tool_calls.append(tool_call)


def _should_inject(
    capture: GeminiLiveCapture, offset_ms: int | None, injection_audio: list[bytes] | None
) -> bool:
    return bool(
        offset_ms is not None
        and injection_audio
        and not capture.injection_started
        and capture.assistant_audio_ms >= offset_ms
    )


def _gemini_contents(row: dict[str, Any], audio_paths: list[Path]) -> list[dict[str, Any]]:
    contents: list[dict[str, Any]] = []
    for message in history_messages(row, audio_paths):
        if message.role == "user":
            if message.audio_path is None:
                raise UnderstandingProviderError("user audio history message missing audio_path")
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "inline_data": {
                                "data": message.audio_path.read_bytes(),
                                "mime_type": _mime_type(message.audio_path),
                            }
                        }
                    ],
                }
            )
        elif message.assistant_text is not None:
            contents.append({"role": "model", "parts": [{"text": message.assistant_text}]})
    return contents


def _json_instructions(system_prompt: str) -> str:
    return system_prompt


def _json_user_prompt(row: dict[str, Any], audio_paths: list[Path]) -> str:
    del row, audio_paths
    return ""


def _gemini_tools(functions: Any) -> list[dict[str, Any]]:
    return gemini_live_tools(build_native_tool_bundle(functions))


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
            "available_functions": str(functions or []),
        },
    )


def _live_instructions(row: dict[str, Any], audio_paths: list[Path]) -> str:
    return "\n\n".join(
        [setup_prompt_for(row), _live_system_prompt(row), _live_user_prompt(row, audio_paths)]
    )


def _wav_as_pcm16_bytes(path: Path) -> bytes:
    sample_rate, samples = read_mono_pcm16_wav(path)
    if sample_rate != DEFAULT_REALTIME_SAMPLE_RATE:
        samples = _resample(samples, sample_rate, DEFAULT_REALTIME_SAMPLE_RATE)
    return _float_samples_to_pcm16(samples)


def _pcm_audio_ms(data: bytes, mime_type: str) -> int:
    rate = DEFAULT_REALTIME_SAMPLE_RATE
    if "rate=" in mime_type:
        try:
            rate = int(mime_type.split("rate=", 1)[1].split(";", 1)[0])
        except ValueError:
            rate = DEFAULT_REALTIME_SAMPLE_RATE
    return round((len(data) / 2) * 1000 / rate)


def _usage_dict(usage: Any) -> dict[str, int]:
    return {
        "input_tokens": _usage_value(usage, "prompt_token_count"),
        "output_tokens": _usage_value(usage, "response_token_count"),
        "total_tokens": _usage_value(usage, "total_token_count"),
    }


def _usage_value(usage: Any, key: str) -> int:
    if isinstance(usage, dict):
        value = usage.get(key)
    else:
        value = getattr(usage, key, None)
    return int(value or 0)


def _existing_predictions(output: Path) -> dict[str, PredictionRow]:
    if not output.exists() or output.stat().st_size == 0:
        return {}
    return {row.cell_id: row for row in read_predictions(output)}


def _has_live_bystander_probe(row: dict[str, Any]) -> bool:
    if row.get("logic_live_expansion_type") == "bystander_interference_suppression":
        return True
    types = row.get("logic_live_expansion_types")
    return isinstance(types, list) and "bystander_interference_suppression" in types


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
