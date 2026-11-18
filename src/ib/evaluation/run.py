"""run — the `ib eval` recipe: resolve targets, run each (optionally in a batch or validation run).

Calling spec:
    run_eval(manifest, predictions, rubrics, output_dir|None, answer_judge=False, ...) -> dict
    DEFAULT_EVAL_ROOT, DEFAULT_EVAL_MODELS_FILES
One target is run by ib.evaluation.target.run_eval_target; the three reports come from
ib.evaluation.reports and default output paths from ib.evaluation.artifacts.

Side effects: writes explicit JSON reports under the requested output dir, or
repo-local artifacts/eval/<manifest-relative-run>/<eval-source> when omitted.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ib.evaluation.artifacts import (
    DEFAULT_EVAL_ROOT as _DEFAULT_EVAL_ROOT,
    _default_validation_run_output_dir,
    _eval_root,
    _manifest_source_path,
    _relative_to,
    _repo_root,
    _resolved_targets,
    _rubrics_for_manifest,
    _target_source_path,
    _validation_run_scenes,
    default_eval_output_dir,
)
from ib.evaluation.target_batch import (
    DEFAULT_EVAL_MODELS_FILES as _DEFAULT_EVAL_MODELS_FILES,
    _default_models_file,
    _eval_target_worker_count,
    _preload_target_worker_imports,
    _resolved_run_targets,
    _target_batch_summary_row,
    _target_specs_with_preset,
)
from ib.scoring.eval_targets import (
    EvalMode,
    EvalTarget,
)
from ib.io import write_json
from ib.llm.qwen import DEFAULT_LLM_PROVIDER
from ib.scoring.openai_understanding import (
    OPENAI_UNDERSTANDING_PROVIDER,
)

from ib.evaluation.target import print_eval_progress, run_eval_target


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
    summary = run_eval_target(
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
        target_summary = run_eval_target(
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
                print_eval_progress(
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
                    print_eval_progress(
                        f"{progress_prefix} target {index}/{len(targets)} "
                        f"{target.mode}:{target.provider}:{target.model or 'predictions'} started"
                    )
                futures.append(executor.submit(run_one, index, target))
            for future in as_completed(futures):
                completed.append(future.result())

    for index, target, target_root, target_summary in sorted(completed, key=lambda item: item[0]):
        if progress_prefix and worker_count > 1:
            print_eval_progress(f"{progress_prefix} target {index}/{len(targets)} completed")
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
    print_eval_progress(
        f"discovered {len(scenes)} scene(s), {len(targets)} target(s); output {root}"
    )
    scene_summaries = []
    for scene_index, scene in enumerate(scenes, start=1):
        scene_label = f"scene {scene_index}/{len(scenes)} {scene['scene']}"
        print_eval_progress(f"{scene_label} started")
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
        print_eval_progress(f"{scene_label} completed")
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

