"""boson_fullduplex — generate full-duplex predictions with Boson Higgs Realtime.

Calling spec:
    run_boson_fullduplex_predictions(
        manifest, output, model, limit=None, recv_timeout_s=120.0, live_probe=False
    ) -> dict

Inputs are built manifest rows with testcase audio paths. Outputs are strict
PredictionRow JSONL rows aligned to the manifest, with append/resume semantics.
With live_probe=True, dedicated speak_time_probe rows open an audio-output
session with the scene's native tools, wait until generated assistant audio
reaches the sampled injection offset, then inject the live event audio and
store observations under prediction.metadata.live_probe (base rows are never
paired with an attached probe session; use the dedicated ``-s`` rows).
Side effects: calls wss://api.boson.ai/v1/realtime and writes prediction and
sibling raw-response JSONL artifacts.

Protocol constraints probed 2026-08-04 against the live endpoint:
- ``conversation.item.create`` with ``input_audio`` content is silently skipped
  by the server, so user history turns are fed via ``input_audio_buffer.append``
  + ``commit`` (one commit per turn, ack-gated to preserve ordering).
- Text-only sessions (``output_modalities: ["text"]``) do not consume audio
  input at all, so predictions run audio-output sessions.
- A forced ``tool_choice`` hangs the response; ``"auto"`` plus instructions that
  require the call is reliable, with parse-retry as backstop.
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any

from ib.io import canonical_json
from ib.predictions import PredictionRow, read_predictions
from ib.scoring.multiturn_audio import history_messages
from ib.scoring.native_tool_schema import build_native_tool_bundle, openai_realtime_tools
from ib.scoring.openai_fullduplex import (
    _live_event_audio_paths,
    _live_instructions,
    _live_user_prompt,
    _tool_names_match,
)
from ib.scoring.openai_realtime_audio import pcm16_delta_ms, wav_as_pcm16_base64
from ib.scoring.openai_realtime_capture import RealtimeCapture
from ib.scoring.probe_sampling import inject_offset_s_for_row
from ib.scoring.openai_understanding import (
    UnderstandingProviderError,
    _audio_paths_for,
    _limited_rows,
    _parse_prediction_payload,
    _prompt_metadata,
    _required_str,
    _system_prompt_with_setup,
)
from ib.scoring.progress import print_eval_case_progress
from ib.scoring.raw_response_io import append_raw_response, raw_response_path
from ib.predictions import PredictionAction
from ib.scoring.realtime_prediction_tool import (
    EMIT_PREDICTION_TOOL_NAME,
    raw_arguments_text,
)
from ib.scoring.smoke import _read_rows

BOSON_FULLDUPLEX_PROVIDER = "boson_realtime"
BOSON_REALTIME_URL = "wss://api.boson.ai/v1/realtime"
BOSON_API_KEY_ENVS = ("BOSONAI_API_KEY", "BOSON_API_KEY")
DEFAULT_BOSON_VOICE = "default"
_APPEND_CHUNK_BYTES = 96_000  # 2 s of 24 kHz PCM16 per input_audio_buffer.append
_ACK_TIMEOUT_S = 30.0
_MAX_ATTEMPTS = 3
_MAX_CONNECTION_FAILURES = 6  # backoff 2s..60s; Boson 502 blips last a few minutes

# tool_choice "auto" needs the instruction to carry the obligation the forced
# choice would otherwise enforce (forcing hangs the Boson endpoint).
_EMIT_TOOL_REMINDER = (
    "You MUST answer by calling the emit_prediction function exactly once with "
    "the final prediction. Never speak or write the JSON aloud; never answer "
    "without calling the function. tool_calls_json must be a JSON array encoded "
    'as a string, e.g. "[]" or "[{\\"name\\": \\"fn\\", \\"arguments\\": {}}]".'
)


def _boson_emit_prediction_tool() -> dict[str, Any]:
    """emit_prediction variant for Boson, whose function-call argument handling
    rejects array-typed parameters — tool_calls travels as a JSON string."""
    return {
        "type": "function",
        "name": EMIT_PREDICTION_TOOL_NAME,
        "description": "Return the final MSI-Bench prediction. Do not speak JSON.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "predicted_action": {
                    "type": "string",
                    "enum": [action.value for action in PredictionAction],
                },
                "answer_text": {"type": "string"},
                "tool_calls_json": {
                    "type": "string",
                    "description": 'JSON array of {"name", "arguments"} objects, "[]" if none.',
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["predicted_action", "answer_text", "tool_calls_json"],
        },
    }


def _canonical_payload_text(prediction_text: str) -> str:
    """Rewrite Boson tool arguments into the canonical prediction payload shape.

    Passes non-JSON or already-canonical payloads through untouched so
    ``_parse_prediction_payload`` stays the single validation gate.
    """
    try:
        payload = json.loads(prediction_text)
    except json.JSONDecodeError:
        return prediction_text
    if not isinstance(payload, dict) or "tool_calls_json" not in payload:
        return prediction_text
    raw_calls = payload.pop("tool_calls_json")
    try:
        calls = json.loads(raw_calls) if isinstance(raw_calls, str) and raw_calls.strip() else []
    except json.JSONDecodeError:
        calls = []
    payload["tool_calls"] = calls if isinstance(calls, list) else []
    if payload.get("answer_text") == "":
        payload["answer_text"] = None
    return json.dumps(payload, ensure_ascii=False)


class BosonRealtimeError(UnderstandingProviderError):
    """Raised when the Boson Realtime endpoint returns an error event."""


class BosonRealtimeConnectionError(UnderstandingProviderError):
    """Raised on transport failures (handshake rejects, mid-session drops)."""


class BosonRealtimeTimeoutError(UnderstandingProviderError):
    """Raised when a Boson Realtime response does not complete in time."""


class BosonPredictionFormatError(UnderstandingProviderError):
    """Raised when Boson returned a response that is not a valid prediction JSON."""


class BosonFullDuplexProvider:
    """Boson Higgs Realtime adapter for full-duplex MSI-Bench predictions."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        voice: str = DEFAULT_BOSON_VOICE,
        recv_timeout_s: float = 120.0,
        live_probe: bool = False,
        raw_response_output: Path | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.voice = voice
        self.recv_timeout_s = recv_timeout_s
        self.live_probe = live_probe
        self.raw_response_output = raw_response_output
        self.call_count = 0
        self.live_probe_count = 0
        self.usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def predict(self, manifest_row: dict[str, Any], *, manifest_path: Path) -> PredictionRow:
        """Return one Boson Realtime full-duplex prediction row."""
        cell_id = _required_str(manifest_row, "cell_id")
        if manifest_row.get("logic_case_mode") == "speak_time_probe":
            if not self.live_probe:
                raise UnderstandingProviderError(
                    "speak_time_probe rows require --live-probe for boson_realtime "
                    "targets; filter them out with --case-modes"
                )
            return self._speak_probe_prediction(cell_id, manifest_row, manifest_path)
        instructions = self._instructions(manifest_row)
        prompts = _prompt_metadata(
            provider=BOSON_FULLDUPLEX_PROVIDER,
            model=self.model,
            mode="fullduplex",
            system=instructions,
            request_rendering="",
        )
        base_started = time.perf_counter()
        base_payload, base_capture = self._base_prediction(manifest_row, manifest_path)
        metadata: dict[str, Any] = {
            "provider": BOSON_FULLDUPLEX_PROVIDER,
            "model": self.model,
            "eval_mode": "fullduplex",
            "base_response_id": base_capture.response_id,
            "base_response_status": base_capture.response_status,
            "completion_time_s": round(time.perf_counter() - base_started, 3),
            "prompts": prompts,
        }
        return PredictionRow(
            cell_id=cell_id,
            predicted_action=base_payload["predicted_action"],
            answer_text=base_payload.get("answer_text"),
            tool_calls=base_payload.get("tool_calls", []),
            confidence=base_payload.get("confidence"),
            metadata=metadata,
        )

    def _instructions(self, manifest_row: dict[str, Any]) -> str:
        return f"{_system_prompt_with_setup(manifest_row)}\n\n{_EMIT_TOOL_REMINDER}"

    def _base_prediction(
        self, manifest_row: dict[str, Any], manifest_path: Path
    ) -> tuple[dict[str, Any], RealtimeCapture]:
        cell_id = _required_str(manifest_row, "cell_id")
        last_error: UnderstandingProviderError | None = None
        attempt = 0
        connection_failures = 0
        while attempt < _MAX_ATTEMPTS:
            try:
                capture = self._base_capture(manifest_row, manifest_path)
            except BosonRealtimeTimeoutError:
                raise
            except BosonRealtimeConnectionError as exc:
                # Transport blips (502 handshakes, dropped sockets) are transient
                # server-side issues; back off without consuming parse attempts.
                connection_failures += 1
                if connection_failures > _MAX_CONNECTION_FAILURES:
                    raise
                time.sleep(min(60.0, 2.0**connection_failures))
                continue
            except UnderstandingProviderError as exc:
                last_error = exc
                attempt += 1
                continue
            attempt += 1
            self.call_count += 1
            self._record_usage(capture)
            self._record_raw_capture(cell_id, capture, attempt=attempt)
            try:
                payload_text = _canonical_payload_text(capture.prediction_text())
                return _parse_prediction_payload(payload_text, cell_id), capture
            except UnderstandingProviderError as exc:
                last_error = exc
        raise BosonPredictionFormatError(
            f"invalid Boson Realtime prediction for {cell_id} "
            f"after {_MAX_ATTEMPTS} attempts: {last_error}"
        )

    def _base_capture(
        self, manifest_row: dict[str, Any], manifest_path: Path
    ) -> RealtimeCapture:
        from websockets.exceptions import WebSocketException
        from websockets.sync.client import connect

        audio_paths = _audio_paths_for(manifest_row, manifest_path)
        cell_id = _required_str(manifest_row, "cell_id")
        if not audio_paths:
            raise UnderstandingProviderError(f"manifest row {cell_id!r} has no audio paths")
        capture = RealtimeCapture()
        try:
            connection = connect(
                f"{BOSON_REALTIME_URL}?model={self.model}",
                additional_headers={"Authorization": f"Bearer {self._api_key()}"},
                max_size=64 * 1024 * 1024,
            )
        except (WebSocketException, OSError) as exc:
            raise BosonRealtimeConnectionError(
                f"Boson Realtime connect failed: {exc}"
            ) from exc
        with connection as ws:
            self._send(ws, {"type": "session.update", "session": self._audio_session(manifest_row)})
            self._feed_history(ws, capture, manifest_row, audio_paths)
            self._send(ws, {"type": "response.create"})
            self._collect_response(ws, capture)
        return capture

    def _audio_session(self, manifest_row: dict[str, Any]) -> dict[str, Any]:
        return self._session_config(
            self._instructions(manifest_row), tools=[_boson_emit_prediction_tool()]
        )

    def _session_config(self, instructions: str, *, tools: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "type": "realtime",
            "instructions": instructions,
            "output_modalities": ["audio"],
            "audio": {
                "input": {"turn_detection": None},
                "output": {"voice": self.voice},
            },
            "tools": tools,
            "tool_choice": "auto",
        }

    def _speak_probe_prediction(
        self, cell_id: str, manifest_row: dict[str, Any], manifest_path: Path
    ) -> PredictionRow:
        """Probe-only row for logic_case_mode=speak_time_probe: no base session."""
        audio_paths = _audio_paths_for(manifest_row, manifest_path)
        live_probe = self._live_probe(manifest_row, manifest_path)
        prompts = _prompt_metadata(
            provider=BOSON_FULLDUPLEX_PROVIDER,
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
                "provider": BOSON_FULLDUPLEX_PROVIDER,
                "model": self.model,
                "eval_mode": "fullduplex",
                "prompts": prompts,
                "live_probe": live_probe,
            },
        )

    def _live_probe(self, manifest_row: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
        cell_id = _required_str(manifest_row, "cell_id")
        live_paths = _live_event_audio_paths(manifest_row, manifest_path)
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
        capture = self._live_capture_with_retries(
            manifest_row,
            manifest_path,
            inject_after_audio_ms=round(offset_s * 1000),
            injection_audio_base64=[wav_as_pcm16_base64(path) for path in live_paths],
        )
        self.call_count += 1
        self.live_probe_count += 1
        self._record_usage(capture)
        self._record_raw_capture(cell_id, capture, attempt=1, phase="live_probe")
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

    def _live_capture_with_retries(
        self,
        manifest_row: dict[str, Any],
        manifest_path: Path,
        *,
        inject_after_audio_ms: int,
        injection_audio_base64: list[str],
    ) -> RealtimeCapture:
        attempt = 0
        connection_failures = 0
        while True:
            try:
                return self._live_capture(
                    manifest_row,
                    manifest_path,
                    inject_after_audio_ms=inject_after_audio_ms,
                    injection_audio_base64=injection_audio_base64,
                )
            except BosonRealtimeTimeoutError:
                raise
            except BosonRealtimeConnectionError:
                connection_failures += 1
                if connection_failures > _MAX_CONNECTION_FAILURES:
                    raise
                time.sleep(min(60.0, 2.0**connection_failures))
            except UnderstandingProviderError:
                attempt += 1
                if attempt >= _MAX_ATTEMPTS:
                    raise

    def _live_capture(
        self,
        manifest_row: dict[str, Any],
        manifest_path: Path,
        *,
        inject_after_audio_ms: int,
        injection_audio_base64: list[str],
    ) -> RealtimeCapture:
        from websockets.exceptions import WebSocketException
        from websockets.sync.client import connect

        audio_paths = _audio_paths_for(manifest_row, manifest_path)
        cell_id = _required_str(manifest_row, "cell_id")
        if not audio_paths:
            raise UnderstandingProviderError(f"manifest row {cell_id!r} has no audio paths")
        bundle = build_native_tool_bundle(manifest_row.get("logic_available_functions"))
        capture = RealtimeCapture()
        try:
            connection = connect(
                f"{BOSON_REALTIME_URL}?model={self.model}",
                additional_headers={"Authorization": f"Bearer {self._api_key()}"},
                max_size=64 * 1024 * 1024,
            )
        except (WebSocketException, OSError) as exc:
            raise BosonRealtimeConnectionError(
                f"Boson Realtime connect failed: {exc}"
            ) from exc
        with connection as ws:
            self._send(
                ws,
                {
                    "type": "session.update",
                    "session": self._session_config(
                        _live_instructions(manifest_row, audio_paths),
                        tools=openai_realtime_tools(bundle),
                    ),
                },
            )
            self._feed_history(ws, capture, manifest_row, audio_paths)
            self._send(ws, {"type": "response.create"})
            self._collect_response(
                ws,
                capture,
                inject_after_audio_ms=inject_after_audio_ms,
                injection_audio_base64=injection_audio_base64,
                tool_name_map=bundle.name_map,
            )
        return capture

    def _feed_history(
        self,
        ws: Any,
        capture: RealtimeCapture,
        manifest_row: dict[str, Any],
        audio_paths: list[Path],
    ) -> None:
        for message in history_messages(manifest_row, audio_paths):
            if message.role == "user":
                if message.audio_path is None:
                    raise UnderstandingProviderError(
                        "user audio history message missing audio_path"
                    )
                audio = base64.b64decode(wav_as_pcm16_base64(message.audio_path))
                for start in range(0, len(audio), _APPEND_CHUNK_BYTES):
                    chunk = audio[start : start + _APPEND_CHUNK_BYTES]
                    self._send(
                        ws,
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(chunk).decode("ascii"),
                        },
                    )
                self._send(ws, {"type": "input_audio_buffer.commit"})
                self._recv_until(ws, capture, {"input_audio_buffer.committed"}, _ACK_TIMEOUT_S)
            elif message.assistant_text is not None:
                self._send(
                    ws,
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "text", "text": message.assistant_text}],
                        },
                    },
                )
                self._recv_until(ws, capture, {"conversation.item.added"}, _ACK_TIMEOUT_S)

    def _collect_response(
        self,
        ws: Any,
        capture: RealtimeCapture,
        *,
        inject_after_audio_ms: int | None = None,
        injection_audio_base64: list[str] | None = None,
        tool_name_map: dict[str, str] | None = None,
    ) -> None:
        deadline = time.monotonic() + self.recv_timeout_s
        while True:
            event = self._recv_event(ws, deadline)
            event_type = event.get("type", "")
            self._record_event(capture, event)
            if event_type == "error":
                raise BosonRealtimeError(
                    f"Boson Realtime error event: {json.dumps(event.get('error') or event)}"
                )
            if event_type == "response.output_audio.delta":
                capture.assistant_audio_ms += pcm16_delta_ms(event.get("delta") or "")
                if _should_inject(capture, inject_after_audio_ms, injection_audio_base64):
                    for audio in injection_audio_base64 or []:
                        for start in range(0, len(audio), _APPEND_CHUNK_BYTES):
                            self._send(
                                ws,
                                {
                                    "type": "input_audio_buffer.append",
                                    "audio": audio[start : start + _APPEND_CHUNK_BYTES],
                                },
                            )
                    self._send(ws, {"type": "input_audio_buffer.commit"})
                    capture.injection_started = True
                    capture.injected_at_audio_ms = capture.assistant_audio_ms
            elif event_type == "response.function_call_arguments.done":
                if tool_name_map is None:
                    capture.emit_prediction_arguments = raw_arguments_text(event.get("arguments"))
                else:
                    _append_live_tool_call(
                        capture, event.get("name"), event.get("arguments"), tool_name_map
                    )
            elif event_type == "response.output_audio_transcript.done":
                transcript = event.get("transcript")
                if isinstance(transcript, str) and transcript.strip():
                    capture.audio_transcripts.append(transcript.strip())
                    # Higgs often speaks the JSON instead of calling the tool;
                    # surface the transcript to prediction_text() as fallback.
                    capture.text_parts.append(transcript.strip())
            elif event_type == "response.created":
                response = event.get("response") or {}
                capture.response_id = response.get("id")
            elif event_type == "response.done":
                response = event.get("response") or {}
                capture.response_status = response.get("status")
                self._capture_done_outputs(capture, response, tool_name_map=tool_name_map)
                usage = response.get("usage") or {}
                capture.usage = {
                    "input_tokens": int(usage.get("input_tokens") or 0),
                    "output_tokens": int(usage.get("output_tokens") or 0),
                    "total_tokens": int(usage.get("total_tokens") or 0),
                }
                return

    def _capture_done_outputs(
        self,
        capture: RealtimeCapture,
        response: dict[str, Any],
        *,
        tool_name_map: dict[str, str] | None = None,
    ) -> None:
        for item in response.get("output") or []:
            if not isinstance(item, dict) or item.get("type") != "function_call":
                continue
            if item.get("name") == EMIT_PREDICTION_TOOL_NAME:
                if not capture.emit_prediction_arguments:
                    capture.emit_prediction_arguments = raw_arguments_text(item.get("arguments"))
            elif tool_name_map is not None:
                _append_live_tool_call(
                    capture, item.get("name"), item.get("arguments"), tool_name_map
                )

    def _recv_until(
        self, ws: Any, capture: RealtimeCapture, stop_types: set[str], timeout_s: float
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while True:
            event = self._recv_event(ws, deadline)
            self._record_event(capture, event)
            if event.get("type") == "error":
                raise BosonRealtimeError(
                    f"Boson Realtime error event: {json.dumps(event.get('error') or event)}"
                )
            if event.get("type") in stop_types:
                return event

    def _recv_event(self, ws: Any, deadline: float) -> dict[str, Any]:
        from websockets.exceptions import ConnectionClosedOK, WebSocketException

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise BosonRealtimeTimeoutError("Boson Realtime response timed out")
        try:
            raw = ws.recv(timeout=remaining)
        except TimeoutError as exc:
            raise BosonRealtimeTimeoutError("Boson Realtime response timed out") from exc
        except ConnectionClosedOK as exc:
            # Boson closes idle sessions server-side ("Idle timeout: no user
            # speech for 300s"); that cutoff is a timeout, not a transport failure.
            raise BosonRealtimeTimeoutError(
                f"Boson Realtime server closed the session: {exc}"
            ) from exc
        except (WebSocketException, OSError) as exc:
            raise BosonRealtimeConnectionError(
                f"Boson Realtime connection failed: {exc}"
            ) from exc
        try:
            event = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise UnderstandingProviderError(f"Boson Realtime sent non-JSON frame: {exc}") from exc
        return event if isinstance(event, dict) else {}

    def _send(self, ws: Any, payload: dict[str, Any]) -> None:
        from websockets.exceptions import WebSocketException

        try:
            ws.send(json.dumps(payload))
        except (WebSocketException, OSError) as exc:
            raise BosonRealtimeConnectionError(f"Boson Realtime send failed: {exc}") from exc

    def _record_event(self, capture: RealtimeCapture, event: dict[str, Any]) -> None:
        capture.raw_events.append(_redact_event(event))

    def _record_usage(self, capture: RealtimeCapture) -> None:
        for key in self.usage:
            self.usage[key] += int(capture.usage.get(key, 0) or 0)

    def _record_raw_capture(
        self,
        cell_id: str,
        capture: RealtimeCapture,
        *,
        attempt: int,
        phase: str = "base_prediction",
    ) -> None:
        if self.raw_response_output is None:
            return
        append_raw_response(
            self.raw_response_output,
            cell_id=cell_id,
            provider=BOSON_FULLDUPLEX_PROVIDER,
            model=self.model,
            eval_mode="fullduplex",
            phase=phase,
            response={"events": capture.raw_events},
            metadata={"attempt": attempt, "response_status": capture.response_status},
        )
        capture.raw_events = []

    def _api_key(self) -> str:
        if self.api_key:
            return self.api_key
        for env in BOSON_API_KEY_ENVS:
            value = os.getenv(env)
            if value:
                return value
        raise UnderstandingProviderError(
            "BOSONAI_API_KEY is required for Boson Realtime evaluation"
        )


def run_boson_fullduplex_predictions(
    manifest: str | Path,
    output: str | Path,
    *,
    model: str | None = None,
    limit: int | None = None,
    recv_timeout_s: float = 120.0,
    live_probe: bool = False,
) -> dict[str, Any]:
    """Generate Boson Realtime full-duplex predictions for up to ``limit`` rows."""
    if not model:
        raise UnderstandingProviderError("Boson Realtime full-duplex eval requires --model")
    rows = _limited_rows(_read_rows(manifest), limit)
    raw_output = raw_response_path(output)
    provider = BosonFullDuplexProvider(
        model=model,
        recv_timeout_s=recv_timeout_s,
        live_probe=live_probe,
        raw_response_output=raw_output,
    )
    predictions = _predict_with_resume(provider, rows, Path(manifest), Path(output))
    return {
        "schema_version": "ib.boson_fullduplex.v1",
        "provider": BOSON_FULLDUPLEX_PROVIDER,
        "model": provider.model,
        "manifest": str(manifest),
        "predictions": str(output),
        "raw_responses": str(raw_output),
        "limit": limit,
        "predicted_count": len(predictions),
        "llm_call_count": provider.call_count,
        "live_probe_enabled": live_probe,
        "live_probe_count": provider.live_probe_count,
        "usage": provider.usage,
        "metadata": {"eval_mode": "fullduplex", "live_probe_enabled": live_probe},
    }


def _predict_with_resume(
    provider: BosonFullDuplexProvider,
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
                BOSON_FULLDUPLEX_PROVIDER, provider.model, cell_id, index, len(rows)
            )
            if cell_id in existing:
                predictions.append(existing[cell_id])
                continue
            try:
                prediction = provider.predict(row, manifest_path=manifest_path)
            except BosonRealtimeTimeoutError as exc:
                prediction = _timeout_prediction(row, provider, exc)
            handle.write(f"{canonical_json(prediction)}\n")
            handle.flush()
            predictions.append(prediction)
    return predictions


def _timeout_prediction(
    row: dict[str, Any],
    provider: BosonFullDuplexProvider,
    error: BosonRealtimeTimeoutError,
) -> PredictionRow:
    """Return a valid prediction row for one timed-out Boson Realtime testcase."""
    return PredictionRow(
        cell_id=_required_str(row, "cell_id"),
        predicted_action="silent",
        answer_text=None,
        tool_calls=[],
        confidence=0.0,
        metadata={
            "provider": BOSON_FULLDUPLEX_PROVIDER,
            "model": provider.model,
            "eval_mode": "fullduplex",
            "provider_error": {"type": "boson_realtime_timeout", "message": str(error)},
        },
    )


def _should_inject(
    capture: RealtimeCapture, offset_ms: int | None, injection_audio: list[str] | None
) -> bool:
    return bool(
        offset_ms is not None
        and injection_audio
        and not capture.injection_started
        and capture.assistant_audio_ms >= offset_ms
    )


def _append_live_tool_call(
    capture: RealtimeCapture,
    name: Any,
    raw_arguments: Any,
    tool_name_map: dict[str, str],
) -> None:
    """Record one scene tool call from a live probe response, mapping provider names."""
    if not isinstance(name, str) or not name or name == EMIT_PREDICTION_TOOL_NAME:
        return
    arguments: Any = raw_arguments
    if isinstance(raw_arguments, str):
        try:
            arguments = json.loads(raw_arguments) if raw_arguments.strip() else {}
        except json.JSONDecodeError:
            arguments = {"value": raw_arguments}
    if not isinstance(arguments, dict):
        arguments = {"value": arguments}
    call = {"name": tool_name_map.get(name, name), "arguments": arguments}
    if call not in capture.tool_calls:
        capture.tool_calls.append(call)


def _redact_event(event: dict[str, Any]) -> dict[str, Any]:
    """Return the event with bulky base64 audio payloads replaced by length stubs."""
    redacted = dict(event)
    for key in ("audio", "delta"):
        value = redacted.get(key)
        if isinstance(value, str) and len(value) > 256:
            redacted[key] = f"<base64:{len(value)} chars>"
    return redacted


def _existing_predictions(output: Path) -> dict[str, PredictionRow]:
    if not output.exists() or output.stat().st_size == 0:
        return {}
    return {row.cell_id: row for row in read_predictions(output)}
