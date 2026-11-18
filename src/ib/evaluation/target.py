"""target — run one eval target: predictions, the three reports, optional answer judge.

Calling spec:
    summary = run_eval_target(target, ...) -> dict      reuses a matching finished summary
    print_eval_progress(message)
Side effects: writes the target's prediction/report JSON under its output dir.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

from ib.evaluation.artifacts import (
    _same_existing_path,
    _summary_artifacts_exist,
    _summary_matches_target,
)
from ib.evaluation.reports import run_answer_judge, run_manifest_eval, run_rubric_eval
from ib.scoring.eval_targets import (
    EvalTarget,
)
from ib.io import write_json, write_jsonl
from ib.scoring.eval import MANIFEST_EVAL_POLICY_VERSION
from ib.scoring.judge import ANSWER_JUDGE_POLICY_VERSION, judge_model_for_provider
from ib.scoring.report_policy import summary_report_has_policy
from ib.scoring.smoke import _read_rows
from ib.scoring.turnbased_speak_probe import barge_in_probe_row, is_speak_probe_row

from ib.evaluation.understanding import (
    GENERATED_FULLDUPLEX_PROVIDERS,
    run_understanding_predictions,
    supports_generated_fullduplex,
)


def print_eval_progress(message: str) -> None:
    """Emit human-readable eval progress without changing JSON artifacts."""
    print(f"ib eval: {message}", file=sys.stderr, flush=True)


def _row_case_mode(row: dict) -> str:
    """Return the manifest row's case mode, defaulting legacy rows to base."""
    return str(row.get("logic_case_mode") or "base_semantic")


_AUDIO_PATH_FIELDS = ("mixed_testcase_audio_paths", "audio_paths", "logic_live_event_audio_paths")


def _row_with_absolute_audio(row: dict, run_root: Path) -> dict:
    """Copy a row, absolutizing audio fields so a filtered manifest can live anywhere."""
    updated = dict(row)
    for field in _AUDIO_PATH_FIELDS:
        values = row.get(field)
        if isinstance(values, list):
            updated[field] = [
                value if Path(str(value)).is_absolute() else str(run_root / str(value))
                for value in values
            ]
    return updated


def _case_mode_filtered_rows(
    manifest: str,
    case_modes: tuple[str, ...] | None,
    mode: str,
    *,
    speak_probe_approx: bool = False,
) -> list[dict] | None:
    """Return filtered manifest rows, or None when every row passes unchanged.

    speak_time_probe rows require a live session, so non-fullduplex targets drop
    them unless speak_probe_approx rewrites them as turn-based barge-in rows.
    Kept rows get absolute audio paths because the filtered manifest is written
    under the target output root, not the scene run root.
    """
    rows = _read_rows(manifest)
    kept = rows
    if case_modes:
        allowed = set(case_modes)
        kept = [row for row in kept if _row_case_mode(row) in allowed]
    transformed = False
    if mode != "fullduplex":
        if speak_probe_approx:
            rewritten = []
            for row in kept:
                if is_speak_probe_row(row):
                    rewritten.append(barge_in_probe_row(row))
                    transformed = True
                else:
                    rewritten.append(row)
            kept = rewritten
        else:
            kept = [row for row in kept if _row_case_mode(row) != "speak_time_probe"]
    if len(kept) == len(rows) and not transformed:
        return None
    run_root = Path(manifest).resolve().parent
    if run_root.name == "manifest":
        run_root = run_root.parent
    return [_row_with_absolute_audio(row, run_root) for row in kept]


def run_eval_target(
    manifest: str,
    rubrics: str,
    root: Path,
    *,
    target: EvalTarget,
    answer_judge: bool,
    judge_provider: str,
    judge_model: str | None,
    limit: int | None,
    live_probe: bool,
    case_modes: tuple[str, ...] | None = None,
    speak_probe_approx: bool = False,
) -> dict:
    """Run one target into one output directory."""
    root.mkdir(parents=True, exist_ok=True)
    target_live_probe = live_probe and target.mode == "fullduplex"
    target_speak_probe_approx = speak_probe_approx and target.mode != "fullduplex"
    existing_summary = _reusable_eval_target_summary(
        root,
        manifest=manifest,
        rubrics=rubrics,
        target=target,
        answer_judge=answer_judge,
        judge_provider=judge_provider,
        judge_model=judge_model,
        limit=limit,
        live_probe=target_live_probe,
        case_modes=case_modes,
        speak_probe_approx=target_speak_probe_approx,
    )
    if existing_summary is not None:
        print_eval_progress(f"reusing completed target {root}")
        return existing_summary
    existing_without_judge = _reusable_eval_target_summary(
        root,
        manifest=manifest,
        rubrics=rubrics,
        target=target,
        answer_judge=False,
        judge_provider=judge_provider,
        judge_model=judge_model,
        limit=limit,
        live_probe=target_live_probe,
        case_modes=case_modes,
        speak_probe_approx=target_speak_probe_approx,
    )
    if existing_without_judge is not None and answer_judge:
        print_eval_progress(f"adding answer judge to completed target {root}")
        return _add_answer_judge_to_summary(
            root,
            existing_without_judge,
            judge_provider=judge_provider,
            judge_model=judge_model,
        )
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    eval_manifest = manifest
    prediction_manifest = manifest
    filtered_rows = _case_mode_filtered_rows(
        manifest, case_modes, target.mode, speak_probe_approx=target_speak_probe_approx
    )
    if filtered_rows is not None:
        prediction_manifest = str(root / "manifest_case_modes.jsonl")
        write_jsonl(prediction_manifest, filtered_rows)
        eval_manifest = prediction_manifest
    if limit is not None:
        eval_manifest = str(root / "manifest_subset.jsonl")
        write_jsonl(eval_manifest, _read_rows(prediction_manifest)[:limit])
    prediction_report = None
    predictions = target.predictions
    if predictions is None:
        if target.mode == "fullduplex":
            if not supports_generated_fullduplex(target.provider):
                raise ValueError(
                    "fullduplex prediction generation currently supports "
                    f"{sorted(GENERATED_FULLDUPLEX_PROVIDERS)!r}; got {target.provider!r}. "
                    "Pass MODE:PROVIDER:MODEL:PREDICTIONS to evaluate another live provider."
                )
        predictions = str(root / "predictions.jsonl")
        prediction_report = run_understanding_predictions(
            target.provider,
            prediction_manifest,
            predictions,
            target.model,
            limit,
            eval_mode=target.mode,
            target_name=target.name,
            live_probe=target_live_probe,
            tool_protocol=target.tool_protocol,
            reasoning_effort=target.reasoning_effort,
            thinking_mode=target.thinking_mode,
        )
        write_json(root / "prediction_generation.json", prediction_report)
    manifest_report = run_manifest_eval(
        eval_manifest,
        predictions,
        str(root / "manifest_eval.json"),
        mode=target.mode,
        live_probe=target_live_probe,
        speak_probe_approx=target_speak_probe_approx,
    )
    rubric_report = run_rubric_eval(eval_manifest, rubrics, str(root / "rubric_eval.json"))
    reports = {
        "manifest_eval": "manifest_eval.json",
        "rubric_eval": "rubric_eval.json",
    }
    if prediction_report is not None:
        reports["prediction_generation"] = "prediction_generation.json"
    passed = not manifest_report["errors"] and not rubric_report["errors"]
    if answer_judge:
        judge_report = run_answer_judge(
            eval_manifest,
            predictions,
            rubrics,
            str(root / "answer_judge.json"),
            judge_provider,
            judge_model,
            progress_model=target.name,
        )
        reports["answer_judge"] = "answer_judge.json"
        passed = passed and not judge_report["errors"] and judge_report["pass_rate"] == 1.0
    summary = {
        "schema_version": "ib.eval.v1",
        "passed": bool(passed),
        "target": target.name,
        "mode": target.mode,
        "manifest": manifest,
        "eval_manifest": eval_manifest,
        "predictions": predictions,
        "rubrics": rubrics,
        "output": str(root),
        "reports": reports,
        "manifest_eval_policy_version": manifest_report["policy_version"],
        "answer_judge_enabled": answer_judge,
        "answer_judge_policy_version": ANSWER_JUDGE_POLICY_VERSION if answer_judge else None,
        "judge_provider": judge_provider if answer_judge else None,
        "judge_model": _effective_judge_model(judge_provider, judge_model)
        if answer_judge
        else None,
        "prediction_generation_enabled": prediction_report is not None,
        "understanding_provider": prediction_report.get("provider")
        if prediction_report is not None
        else None,
        "understanding_model": prediction_report.get("model")
        if prediction_report is not None
        else None,
        "reasoning_effort": target.reasoning_effort,
        "thinking_mode": target.thinking_mode,
        "tool_protocol": target.tool_protocol,
        "limit": limit,
        "case_modes": list(case_modes) if case_modes else None,
        "live_probe_enabled": target_live_probe,
        "live_probe_requested": live_probe,
        "speak_probe_approx": target_speak_probe_approx,
    }
    write_json(root / "summary.json", summary)
    return summary


def _reusable_eval_target_summary(
    root: Path,
    *,
    manifest: str,
    rubrics: str,
    target: EvalTarget,
    answer_judge: bool,
    judge_provider: str,
    judge_model: str | None,
    limit: int | None,
    live_probe: bool,
    case_modes: tuple[str, ...] | None = None,
    speak_probe_approx: bool = False,
) -> dict | None:
    """Return an existing target summary when it matches the requested target."""
    summary_path = root / "summary.json"
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if summary.get("schema_version") != "ib.eval.v1":
        return None
    if summary.get("mode") != target.mode or summary.get("limit") != limit:
        return None
    if summary.get("case_modes") != (list(case_modes) if case_modes else None):
        return None
    if summary.get("tool_protocol", "prompt_json") != target.tool_protocol:
        return None
    if summary.get("manifest_eval_policy_version") != MANIFEST_EVAL_POLICY_VERSION:
        return None
    if bool(summary.get("live_probe_enabled", False)) != live_probe:
        return None
    if bool(summary.get("speak_probe_approx", False)) != speak_probe_approx:
        return None
    if answer_judge and not _summary_has_matching_answer_judge(
        summary, judge_provider=judge_provider, judge_model=judge_model
    ):
        return None
    if answer_judge and _answer_judge_report_has_failed_calls(root, summary):
        print_eval_progress(
            f"existing answer judge for {root} recorded failed judge calls; re-judging"
        )
        return None
    if not _same_existing_path(summary.get("manifest"), manifest):
        return None
    if not _same_existing_path(summary.get("rubrics"), rubrics):
        return None
    if not _summary_matches_target(summary, target):
        return None
    if not _summary_artifacts_exist(root, summary, answer_judge=answer_judge):
        return None
    if not summary_report_has_policy(
        root,
        summary.get("reports") or {},
        "manifest_eval",
        MANIFEST_EVAL_POLICY_VERSION,
    ):
        return None
    return summary


def _answer_judge_report_has_failed_calls(root: Path, summary: dict) -> bool:
    """Return whether a stored answer-judge report recorded failed judge calls.

    A report whose errors include "answer judge failed ..." entries (e.g. from a
    provider outage or key limit) is retryable and must not be silently reused.
    """
    relative = (summary.get("reports") or {}).get("answer_judge")
    if not isinstance(relative, str) or not relative:
        return False
    try:
        report = json.loads((root / relative).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True
    errors = report.get("errors")
    if not isinstance(errors, list):
        return False
    return any(
        isinstance(item, str) and item.startswith("answer judge failed")
        for item in errors
    )


def _summary_has_matching_answer_judge(
    summary: dict,
    *,
    judge_provider: str,
    judge_model: str | None,
) -> bool:
    if not summary.get("answer_judge_enabled"):
        return False
    if summary.get("answer_judge_policy_version") != ANSWER_JUDGE_POLICY_VERSION:
        return False
    summary_provider = summary.get("judge_provider")
    summary_model = summary.get("judge_model")
    if summary_provider is not None and summary_provider != judge_provider:
        return False
    return summary_model == _effective_judge_model(judge_provider, judge_model)


def _effective_judge_model(judge_provider: str, judge_model: str | None) -> str:
    """Return the judge model stored in summaries for reuse checks."""
    return judge_model_for_provider(judge_provider, judge_model)


def _add_answer_judge_to_summary(
    root: Path,
    summary: dict,
    *,
    judge_provider: str,
    judge_model: str | None,
) -> dict:
    """Run answer judge against an existing target output and update summary.json."""
    reports = dict(summary.get("reports") or {})
    predictions = str(summary.get("predictions") or "")
    eval_manifest = str(summary.get("eval_manifest") or summary.get("manifest") or "")
    rubrics = str(summary.get("rubrics") or "")
    judge_report = run_answer_judge(
        eval_manifest,
        predictions,
        rubrics,
        str(root / "answer_judge.json"),
        judge_provider,
        judge_model,
        progress_model=root.name,
    )
    reports["answer_judge"] = "answer_judge.json"
    updated = {
        **summary,
        "reports": reports,
        "manifest_eval_policy_version": MANIFEST_EVAL_POLICY_VERSION,
        "answer_judge_enabled": True,
        "answer_judge_policy_version": ANSWER_JUDGE_POLICY_VERSION,
        "judge_provider": judge_provider,
        "judge_model": _effective_judge_model(judge_provider, judge_model),
        "passed": bool(
            _base_eval_reports_pass(root, reports)
            and not judge_report["errors"]
            and judge_report["pass_rate"] == 1.0
        ),
    }
    write_json(root / "summary.json", updated)
    return updated


def _base_eval_reports_pass(root: Path, reports: dict) -> bool:
    """Recompute non-judge pass state so replacing an old judge can recover."""
    for name in ("manifest_eval", "rubric_eval"):
        relative = reports.get(name)
        if not isinstance(relative, str):
            return False
        try:
            payload = json.loads((root / relative).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if (
            name == "manifest_eval"
            and payload.get("policy_version") != MANIFEST_EVAL_POLICY_VERSION
        ):
            return False
        if payload.get("errors"):
            return False
    return True
