"""eval — CLI wrappers for model-output evaluation reports.

Calling spec:
    run_eval(manifest, predictions, rubrics, output_dir|None, answer_judge=False, ...) -> dict
    default_eval_output_dir(manifest, predictions=None, understanding_provider="openai", understanding_model=None, tool_protocol=None) -> Path
    run_manifest_eval(manifest, predictions, output) -> dict
    run_rubric_eval(manifest, rubrics, output) -> dict
    run_answer_judge(manifest, predictions, rubrics, output, provider, model=None, *, progress_model=None) -> dict

Side effects: writes explicit JSON reports under the requested output dir, or
repo-local artifacts/eval/<manifest-relative-run>/<eval-source> when omitted.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys

from ib.cli.eval_artifacts import (
    DEFAULT_EVAL_ROOT as _DEFAULT_EVAL_ROOT,
    _auto_leaderboard_enabled as _artifact_auto_leaderboard_enabled,
    _default_validation_run_output_dir,
    _eval_root,
    _manifest_source_path,
    _relative_to,
    _repo_root,
    _resolved_targets,
    _rubrics_for_manifest,
    _same_existing_path,
    _summary_artifacts_exist,
    _summary_matches_target,
    _target_source_path,
    _validation_run_scenes,
    default_eval_output_dir,
    refresh_leaderboard_if_eval_artifact,
)
from ib.cli.eval_reports import run_answer_judge, run_manifest_eval, run_rubric_eval
from ib.cli.eval_target_batch import (
    DEFAULT_EVAL_MODELS_FILES as _DEFAULT_EVAL_MODELS_FILES,
    _default_models_file,
    _eval_target_worker_count,
    _preload_target_worker_imports,
    _resolved_run_targets,
    _target_batch_summary_row,
    _target_specs_with_preset,
)
from ib.eval_targets import (
    EvalMode,
    EvalTarget,
    GeminiThinkingMode,
)
from ib.io import write_json, write_jsonl
from ib.llm.qwen import DEFAULT_LLM_PROVIDER
from ib.predictions import read_predictions
from ib.scoring.boson_fullduplex import (
    BOSON_FULLDUPLEX_PROVIDER,
    run_boson_fullduplex_predictions,
)
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
from ib.scoring.eval import MANIFEST_EVAL_POLICY_VERSION
from ib.scoring.judge import ANSWER_JUDGE_POLICY_VERSION, judge_model_for_provider
from ib.scoring.report_policy import summary_report_has_policy
from ib.scoring.smoke import _read_rows
from ib.scoring.turnbased_speak_probe import barge_in_probe_row, is_speak_probe_row

DEFAULT_EVAL_ROOT = _DEFAULT_EVAL_ROOT
DEFAULT_EVAL_MODELS_FILES = _DEFAULT_EVAL_MODELS_FILES


def run_eval(
    manifest: str | None,
    predictions: str | None,
    rubrics: str | None,
    output: str | None,
    *,
    answer_judge: bool = False,
    judge_provider: str = DEFAULT_LLM_PROVIDER,
    judge_model: str | None = None,
    understanding_provider: str = OPENAI_UNDERSTANDING_PROVIDER,
    understanding_model: str | None = None,
    mode: EvalMode = "turn_based",
    limit: int | None = None,
    target_specs: list[str] | None = None,
    target_preset: str | None = None,
    models_file: str | None = None,
    targets_file: str | None = None,
    targets: list[EvalTarget] | None = None,
    validation_run: str | None = None,
    target_workers: int | None = None,
    live_probe: bool = False,
    case_modes: str | tuple[str, ...] | None = None,
    speak_probe_approx: bool = False,
) -> dict:
    """Run the model-evaluation recipe and write a summary plus component reports."""
    case_modes = _normalized_case_modes(case_modes)
    if validation_run is not None:
        if manifest is not None or rubrics is not None or predictions is not None:
            raise ValueError("use either --run or --manifest/--rubrics/--predictions, not both")
        resolved_run_targets = _resolved_run_targets(
            target_specs, target_preset, models_file, targets_file, targets
        )
        summary = _run_eval_validation_run(
            validation_run,
            output,
            targets=resolved_run_targets,
            answer_judge=answer_judge,
            judge_provider=judge_provider,
            judge_model=judge_model,
            limit=limit,
            target_workers=target_workers,
            live_probe=live_probe,
            case_modes=case_modes,
            speak_probe_approx=speak_probe_approx,
        )
        _refresh_leaderboard_if_eval_artifact(summary)
        return summary
    if manifest is None:
        raise ValueError("ib eval requires --manifest, or --run VALIDATION_RUN")
    if rubrics is None:
        rubrics = str(_rubrics_for_manifest(Path(manifest)))
    if (
        predictions is None
        and not target_specs
        and target_preset is None
        and models_file is None
        and targets_file is None
        and not targets
        and understanding_provider == OPENAI_UNDERSTANDING_PROVIDER
        and understanding_model is None
    ):
        models_file = _default_models_file()
    target_specs = _target_specs_with_preset(target_specs, target_preset)
    resolved_targets = _resolved_targets(
        predictions=predictions,
        understanding_provider=understanding_provider,
        understanding_model=understanding_model,
        mode=mode,
        target_specs=target_specs,
        models_file=models_file,
        targets_file=targets_file,
        targets=targets,
    )
    if resolved_targets is not None:
        summary = _run_eval_batch(
            manifest,
            rubrics,
            output,
            targets=resolved_targets,
            answer_judge=answer_judge,
            judge_provider=judge_provider,
            judge_model=judge_model,
            limit=limit,
            target_workers=target_workers,
            live_probe=live_probe,
            case_modes=case_modes,
            speak_probe_approx=speak_probe_approx,
        )
        _refresh_leaderboard_if_eval_artifact(summary)
        return summary
    root = (
        Path(output)
        if output is not None
        else default_eval_output_dir(
            manifest,
            predictions=predictions,
            understanding_provider=understanding_provider,
            understanding_model=understanding_model,
            mode=mode,
            live_probe=live_probe,
        )
    )
    target = EvalTarget(
        provider=understanding_provider,
        model=understanding_model,
        mode=mode,
        predictions=predictions,
    )
    summary = _run_eval_target(
        manifest,
        rubrics,
        root,
        target=target,
        answer_judge=answer_judge,
        judge_provider=judge_provider,
        judge_model=judge_model,
        limit=limit,
        live_probe=live_probe,
        case_modes=case_modes,
        speak_probe_approx=speak_probe_approx,
    )
    _refresh_leaderboard_if_eval_artifact(summary)
    return summary


def _run_eval_batch(
    manifest: str,
    rubrics: str,
    output: str | None,
    *,
    targets: list[EvalTarget],
    answer_judge: bool,
    judge_provider: str,
    judge_model: str | None,
    limit: int | None,
    progress_prefix: str | None = None,
    target_workers: int | None = None,
    live_probe: bool = False,
    case_modes: tuple[str, ...] | None = None,
    speak_probe_approx: bool = False,
) -> dict:
    """Run all eval targets under one aggregate output directory."""
    if not targets:
        raise ValueError("at least one eval target is required")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    repo_root = _repo_root()
    root = (
        Path(output)
        if output is not None
        else _eval_root(repo_root) / _manifest_source_path(manifest, repo_root)
    )
    root.mkdir(parents=True, exist_ok=True)
    worker_count = _eval_target_worker_count(target_workers, len(targets))
    _preload_target_worker_imports(targets, worker_count)
    target_summaries_by_index: dict[int, dict] = {}

    def run_one(index: int, target: EvalTarget) -> tuple[int, EvalTarget, Path, dict]:
        target_root = root / _target_source_path(target, repo_root, live_probe=live_probe)
        target_summary = _run_eval_target(
            manifest,
            rubrics,
            target_root,
            target=target,
            answer_judge=answer_judge,
            judge_provider=judge_provider,
            judge_model=judge_model,
            limit=limit,
            live_probe=live_probe,
            case_modes=case_modes,
            speak_probe_approx=speak_probe_approx,
        )
        return index, target, target_root, target_summary

    if worker_count == 1:
        completed = []
        for index, target in enumerate(targets, start=1):
            if progress_prefix:
                _print_eval_progress(
                    f"{progress_prefix} target {index}/{len(targets)} "
                    f"{target.mode}:{target.provider}:{target.model or 'predictions'}"
                )
            completed.append(run_one(index, target))
    else:
        completed = []
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = []
            for index, target in enumerate(targets, start=1):
                if progress_prefix:
                    _print_eval_progress(
                        f"{progress_prefix} target {index}/{len(targets)} "
                        f"{target.mode}:{target.provider}:{target.model or 'predictions'} started"
                    )
                futures.append(executor.submit(run_one, index, target))
            for future in as_completed(futures):
                completed.append(future.result())

    for index, target, target_root, target_summary in sorted(completed, key=lambda item: item[0]):
        _refresh_leaderboard_if_eval_artifact(target_summary)
        if progress_prefix and worker_count > 1:
            _print_eval_progress(f"{progress_prefix} target {index}/{len(targets)} completed")
        target_summaries_by_index[index] = _target_batch_summary_row(
            target, target_summary, target_root, root
        )
    target_summaries = [target_summaries_by_index[index] for index in range(1, len(targets) + 1)]
    passed = all(item["passed"] for item in target_summaries)
    summary = {
        "schema_version": "ib.eval_batch.v1",
        "passed": bool(passed),
        "manifest": manifest,
        "rubrics": rubrics,
        "output": str(root),
        "limit": limit,
        "target_count": len(target_summaries),
        "live_probe_enabled": live_probe,
        "speak_probe_approx": speak_probe_approx,
        "targets": target_summaries,
    }
    write_json(root / "summary.json", summary)
    return summary


def _run_eval_validation_run(
    validation_run: str,
    output: str | None,
    *,
    targets: list[EvalTarget],
    answer_judge: bool,
    judge_provider: str,
    judge_model: str | None,
    limit: int | None,
    target_workers: int | None,
    live_probe: bool,
    case_modes: tuple[str, ...] | None = None,
    speak_probe_approx: bool = False,
) -> dict:
    """Run the same target set across every scene under one validation run."""
    repo_root = _repo_root()
    run_root = Path(validation_run)
    if not run_root.exists():
        raise ValueError(f"validation run does not exist: {run_root}")
    scenes = _validation_run_scenes(run_root)
    if not scenes:
        raise ValueError(f"no scene manifests found under {run_root}")
    root = (
        Path(output)
        if output is not None
        else _default_validation_run_output_dir(run_root, repo_root)
    )
    root.mkdir(parents=True, exist_ok=True)
    _print_eval_progress(
        f"discovered {len(scenes)} scene(s), {len(targets)} target(s); output {root}"
    )
    scene_summaries = []
    for scene_index, scene in enumerate(scenes, start=1):
        scene_label = f"scene {scene_index}/{len(scenes)} {scene['scene']}"
        _print_eval_progress(f"{scene_label} started")
        scene_root = root / scene["scene"]
        scene_summary = _run_eval_batch(
            scene["manifest"],
            scene["rubrics"],
            str(scene_root),
            targets=targets,
            answer_judge=answer_judge,
            judge_provider=judge_provider,
            judge_model=judge_model,
            limit=limit,
            progress_prefix=scene_label,
            target_workers=target_workers,
            live_probe=live_probe,
            case_modes=case_modes,
            speak_probe_approx=speak_probe_approx,
        )
        _print_eval_progress(f"{scene_label} completed")
        scene_summaries.append(
            {
                "scene": scene["scene"],
                "manifest": scene["manifest"],
                "rubrics": scene["rubrics"],
                "output": str(scene_root),
                "summary": _relative_to(scene_root / "summary.json", root),
                "passed": scene_summary["passed"],
                "target_count": scene_summary["target_count"],
            }
        )
    summary = {
        "schema_version": "ib.eval_validation_run.v1",
        "passed": all(item["passed"] for item in scene_summaries),
        "validation_run": str(run_root),
        "output": str(root),
        "limit": limit,
        "scene_count": len(scene_summaries),
        "target_count": len(targets),
        "live_probe_enabled": live_probe,
        "speak_probe_approx": speak_probe_approx,
        "scenes": scene_summaries,
    }
    write_json(root / "summary.json", summary)
    return summary


def _print_eval_progress(message: str) -> None:
    """Emit human-readable eval progress without changing JSON artifacts."""
    print(f"ib eval: {message}", file=sys.stderr, flush=True)


_CASE_MODE_ALIASES = {"base": "base_semantic", "hear": "hear_time_probe", "speak": "speak_time_probe"}
_KNOWN_CASE_MODES = ("base_semantic", "hear_time_probe", "speak_time_probe")


def _normalized_case_modes(raw: str | tuple[str, ...] | None) -> tuple[str, ...] | None:
    """Return a sorted tuple of manifest case modes, or None for no filtering."""
    if raw is None:
        return None
    parts = raw.split(",") if isinstance(raw, str) else list(raw)
    modes: set[str] = set()
    for part in parts:
        token = str(part).strip()
        if not token:
            continue
        mode_name = _CASE_MODE_ALIASES.get(token, token)
        if mode_name not in _KNOWN_CASE_MODES:
            raise ValueError(
                f"unknown case mode {token!r}; choose from {sorted(_KNOWN_CASE_MODES)} "
                f"or aliases {sorted(_CASE_MODE_ALIASES)}"
            )
        modes.add(mode_name)
    return tuple(sorted(modes)) or None


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


def _run_eval_target(
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
        _print_eval_progress(f"reusing completed target {root}")
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
        _print_eval_progress(f"adding answer judge to completed target {root}")
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
            if not _supports_generated_fullduplex(target.provider):
                raise ValueError(
                    "fullduplex prediction generation currently supports "
                    f"{sorted(_GENERATED_FULLDUPLEX_PROVIDERS)!r}; got {target.provider!r}. "
                    "Pass MODE:PROVIDER:MODEL:PREDICTIONS to evaluate another live provider."
                )
        predictions = str(root / "predictions.jsonl")
        prediction_report = _run_understanding_predictions(
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
        _print_eval_progress(
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


def _refresh_leaderboard_if_eval_artifact(summary: dict) -> None:
    """Refresh one eligible summary while preserving this module's patch seam."""
    refresh_leaderboard_if_eval_artifact(summary, repo_root=_repo_root())


def _auto_leaderboard_enabled() -> bool:
    """Preserve the public CLI helper while delegating env parsing."""
    return _artifact_auto_leaderboard_enabled()


def _run_understanding_predictions(
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
    elif provider == BOSON_FULLDUPLEX_PROVIDER:
        report = run_boson_fullduplex_predictions(
            manifest,
            predictions,
            model=model,
            limit=limit,
            live_probe=live_probe,
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


_GENERATED_FULLDUPLEX_PROVIDERS = {
    OPENAI_FULLDUPLEX_PROVIDER,
    GEMINI_FULLDUPLEX_PROVIDER,
    BOSON_FULLDUPLEX_PROVIDER,
}


def _supports_generated_fullduplex(provider: str) -> bool:
    """Return whether the CLI can generate live full-duplex predictions directly."""
    return provider in _GENERATED_FULLDUPLEX_PROVIDERS


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
