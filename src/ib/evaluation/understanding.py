"""understanding — generate model predictions for an eval target (understanding / full-duplex).

Calling spec:
    path = run_understanding_predictions(target, manifest, output_dir, ...) -> Path
    supports_generated_fullduplex(provider) -> bool
Side effects: calls the configured model provider; writes the prediction JSONL.
"""

from __future__ import annotations

from pathlib import Path

from ib.scoring.eval_targets import (
    GeminiThinkingMode,
)
from ib.io import write_jsonl
from ib.scoring.predictions import read_predictions
from ib.scoring.openai_fullduplex import (
    OPENAI_FULLDUPLEX_PROVIDER,
    run_openai_fullduplex_predictions,
)
from ib.scoring.gemini_understanding import (
    GEMINI_TEXT_UNDERSTANDING_PROVIDER,
    GEMINI_UNDERSTANDING_PROVIDER,
    run_gemini_understanding_predictions,
)
from ib.scoring.gemini_fullduplex import (
    GEMINI_FULLDUPLEX_PROVIDER,
    run_gemini_fullduplex_predictions,
)
from ib.scoring.local_understanding import (
    LOCAL_TEXT_UNDERSTANDING_PROVIDER,
    LOCAL_UNDERSTANDING_PROVIDER,
    QWEN_UNDERSTANDING_PROVIDER,
    run_local_understanding_predictions,
    run_qwen_understanding_predictions,
)
from ib.scoring.openai_understanding import (
    OPENAI_UNDERSTANDING_PROVIDER,
    run_openai_understanding_predictions,
)


GENERATED_FULLDUPLEX_PROVIDERS = {
    OPENAI_FULLDUPLEX_PROVIDER,
    GEMINI_FULLDUPLEX_PROVIDER,
}


def supports_generated_fullduplex(provider: str) -> bool:
    """Return whether the CLI can generate live full-duplex predictions directly."""
    return provider in GENERATED_FULLDUPLEX_PROVIDERS


def run_understanding_predictions(
    provider: str,
    manifest: str,
    predictions: str,
    model: str | None,
    limit: int | None,
    *,
    eval_mode: str = "turn_based",
    target_name: str | None = None,
    live_probe: bool = False,
    tool_protocol: str = "prompt_json",
    reasoning_effort: str | None = None,
    thinking_mode: GeminiThinkingMode | None = None,
) -> dict:
    protocol_kwargs = {"tool_protocol": tool_protocol} if tool_protocol == "native" else {}
    thinking_kwargs = {"thinking_mode": thinking_mode} if thinking_mode is not None else {}
    if provider == OPENAI_UNDERSTANDING_PROVIDER:
        report = run_openai_understanding_predictions(
            manifest,
            predictions,
            model=model,
            limit=limit,
            **protocol_kwargs,
        )
    elif provider == OPENAI_FULLDUPLEX_PROVIDER:
        report = run_openai_fullduplex_predictions(
            manifest,
            predictions,
            model=model,
            limit=limit,
            live_probe=live_probe,
            reasoning_effort=reasoning_effort,
        )
    elif provider == GEMINI_FULLDUPLEX_PROVIDER:
        report = run_gemini_fullduplex_predictions(
            manifest,
            predictions,
            model=model,
            limit=limit,
            live_probe=live_probe,
            **thinking_kwargs,
        )
    elif provider == GEMINI_UNDERSTANDING_PROVIDER:
        report = run_gemini_understanding_predictions(
            manifest,
            predictions,
            model=model,
            limit=limit,
            **protocol_kwargs,
            **thinking_kwargs,
        )
    elif provider == GEMINI_TEXT_UNDERSTANDING_PROVIDER:
        report = run_gemini_understanding_predictions(
            manifest,
            predictions,
            model=model,
            limit=limit,
            input_mode="transcript",
            **protocol_kwargs,
            **thinking_kwargs,
        )
    elif provider == QWEN_UNDERSTANDING_PROVIDER:
        report = run_qwen_understanding_predictions(
            manifest,
            predictions,
            model=model,
            limit=limit,
        )
    elif provider == LOCAL_UNDERSTANDING_PROVIDER:
        report = run_local_understanding_predictions(
            manifest,
            predictions,
            model=model,
            limit=limit,
            **thinking_kwargs,
        )
    elif provider == LOCAL_TEXT_UNDERSTANDING_PROVIDER:
        report = run_local_understanding_predictions(
            manifest,
            predictions,
            model=model,
            limit=limit,
            input_mode="transcript",
            **thinking_kwargs,
        )
    else:
        raise ValueError(f"unknown understanding provider {provider!r}")
    _tag_prediction_file(
        predictions,
        eval_mode=eval_mode,
        target_name=target_name,
        tool_protocol=tool_protocol,
        thinking_mode=thinking_mode,
    )
    report["eval_mode"] = eval_mode
    report["target"] = target_name
    report["live_probe_enabled"] = live_probe
    report["tool_protocol"] = tool_protocol
    report["thinking_mode"] = thinking_mode
    return report


def _tag_prediction_file(
    predictions: str | Path,
    *,
    eval_mode: str,
    target_name: str | None,
    tool_protocol: str = "prompt_json",
    thinking_mode: GeminiThinkingMode | None = None,
) -> None:
    """Persist generated target metadata on prediction rows."""
    rows = []
    for row in read_predictions(predictions):
        metadata = dict(row.metadata or {})
        metadata.setdefault("eval_mode", eval_mode)
        metadata.setdefault("tool_protocol", tool_protocol)
        if thinking_mode is not None:
            metadata.setdefault("thinking_mode", thinking_mode)
        if target_name is not None:
            metadata.setdefault("eval_target", target_name)
        rows.append(row.model_copy(update={"metadata": metadata}))
    write_jsonl(predictions, rows)
