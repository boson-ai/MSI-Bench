"""eval_reports — run and persist the three low-level eval reports.

Calling spec:
    run_manifest_eval(manifest, predictions, output, mode=..., live_probe=False) -> dict
    run_rubric_eval(manifest, rubrics, output) -> dict
    run_answer_judge(manifest, predictions, rubrics, output, provider, model=None) -> dict

Each function writes only its explicit report path; answer judging also writes
the sibling ``.cost.json`` usage report.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from ib.eval_targets import EvalMode
from ib.io import write_json
from ib.llm import LlmConfig, LlmService
from ib.scoring.eval import manifest_eval, rubric_eval
from ib.scoring.judge import ajudge_answers


def run_manifest_eval(
    manifest: str,
    predictions: str,
    output: str,
    *,
    mode: EvalMode = "turn_based",
    live_probe: bool = False,
    speak_probe_approx: bool = False,
) -> dict:
    # Dedicated speak_time_probe rows carry their own probe predictions, so they
    # score in fullduplex mode even without the legacy --live-probe injection flag.
    manifest_eval_mode = mode if (live_probe or _has_speak_probe_rows(manifest)) else "turn_based"
    result = manifest_eval(
        manifest,
        predictions,
        eval_mode=manifest_eval_mode,
        speak_probe_approx=speak_probe_approx,
    )
    result["requested_eval_mode"] = mode
    result["live_probe_enabled"] = live_probe
    result["speak_probe_approx"] = speak_probe_approx
    write_json(output, result)
    return result


def _has_speak_probe_rows(manifest: str) -> bool:
    """Return whether the manifest declares dedicated speak_time_probe rows."""
    try:
        with open(manifest, encoding="utf-8") as handle:
            return any('"speak_time_probe"' in line for line in handle)
    except OSError:
        return False


def run_rubric_eval(manifest: str, rubrics: str, output: str) -> dict:
    result = rubric_eval(manifest, rubrics)
    write_json(output, result)
    return result


def run_answer_judge(
    manifest: str,
    predictions: str,
    rubrics: str,
    output: str,
    provider: str,
    model: str | None = None,
    *,
    progress_model: str | None = None,
) -> dict:
    """Judge answers concurrently via a cost-tracked service, then write reports."""
    result, cost_summary = asyncio.run(
        _ajudge(
            manifest,
            predictions,
            rubrics,
            provider=provider,
            model=model,
            progress_model=progress_model,
        )
    )
    result["llm_call_count"] = cost_summary["call_count"]
    result["llm_usage"] = str(Path(output).with_suffix(".cost.json"))
    write_json(output, result)
    write_json(Path(output).with_suffix(".cost.json"), cost_summary)
    return result


async def _ajudge(manifest, predictions, rubrics, *, provider, model, progress_model):
    service = LlmService(LlmConfig())
    try:
        result = await ajudge_answers(
            manifest,
            predictions,
            rubrics,
            provider=provider,
            model=model,
            service=service,
            progress_provider="answer_judge",
            progress_model=progress_model,
        )
        return result, service.cost.summary()
    finally:
        await service.aclose()
