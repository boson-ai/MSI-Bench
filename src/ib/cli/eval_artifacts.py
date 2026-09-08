"""eval_artifacts — deterministic eval target identity and artifact paths.

Calling spec:
    default_eval_output_dir(...) -> Path
    _target_source_path(target, repo_root, live_probe=False) -> Path
    _summary_matches_target(summary, target) -> bool
    refresh_leaderboard_if_eval_artifact(summary, repo_root=...) -> None

Path and identity helpers are deterministic. Leaderboard refresh is the only
side effect and runs only for eligible eval artifacts when enabled by env.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import re
import sys

from ib.eval_targets import (
    EvalMode,
    EvalTarget,
    GeminiThinkingMode,
    ToolProtocol,
    default_tool_protocol,
    eval_targets_from_file,
    eval_targets_from_models_file,
    eval_targets_from_specs,
    merge_eval_targets,
)
from ib.scoring.openai_understanding import OPENAI_UNDERSTANDING_PROVIDER

DEFAULT_EVAL_ROOT = Path("artifacts") / "eval"


def _validation_run_scenes(run_root: Path) -> list[dict[str, str]]:
    """Discover scene manifests and rubrics under one validation run."""
    scenes = []
    for manifest_path in sorted(run_root.glob("*/manifest/manifest.jsonl")):
        scene_root = manifest_path.parents[1]
        scenes.append(
            {
                "scene": scene_root.name,
                "manifest": str(manifest_path),
                "rubrics": str(_scene_rubrics(scene_root)),
            }
        )
    return scenes


def _scene_rubrics(scene_root: Path) -> Path:
    for candidate in (
        scene_root / "plans" / "rubrics.jsonl",
        scene_root / "generation" / "atomic_rubrics.jsonl",
    ):
        if candidate.exists():
            return candidate
    raise ValueError(f"missing rubrics for scene {scene_root.name}: expected plans/rubrics.jsonl")


def _rubrics_for_manifest(manifest: Path) -> Path:
    """Infer a scene build's rubrics path from its canonical manifest path."""
    if manifest.name != "manifest.jsonl" or manifest.parent.name != "manifest":
        raise ValueError(
            "ib eval requires --rubrics unless --manifest points to "
            ".../<scene>/manifest/manifest.jsonl"
        )
    return _scene_rubrics(manifest.parents[1])


def _default_validation_run_output_dir(run_root: Path, repo_root: Path) -> Path:
    eval_root = _eval_root(repo_root)
    incomplete = _latest_incomplete_validation_run_output(eval_root, run_root.name)
    if incomplete is not None:
        print(f"ib eval: resuming incomplete output {incomplete}", file=sys.stderr, flush=True)
        return incomplete
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return eval_root / f"{_safe_path_segment(run_root.name)}_{stamp}"


def _latest_incomplete_validation_run_output(eval_root: Path, run_name: str) -> Path | None:
    """Return the newest timestamped run output missing its top-level summary."""
    prefix = f"{_safe_path_segment(run_name)}_"
    if not eval_root.exists():
        return None
    candidates = [
        path for path in eval_root.iterdir() if path.is_dir() and path.name.startswith(prefix)
    ]
    if not candidates:
        return None
    latest = max(candidates, key=lambda path: path.name)
    return latest if not (latest / "summary.json").exists() else None


def _summary_matches_target(summary: dict, target: EvalTarget) -> bool:
    if target.predictions is not None:
        return _same_existing_path(summary.get("predictions"), target.predictions)
    provider = summary.get("understanding_provider")
    model = summary.get("understanding_model")
    return (
        provider == target.provider
        and model == _resolved_understanding_model(target.provider, target.model)
        and summary.get("reasoning_effort") == target.reasoning_effort
        and summary.get("thinking_mode") == target.thinking_mode
        and summary.get("tool_protocol", "prompt_json") == target.tool_protocol
    )


def _summary_artifacts_exist(root: Path, summary: dict, *, answer_judge: bool) -> bool:
    predictions = summary.get("predictions")
    if not predictions or not _absolute_path(predictions).exists():
        return False
    reports = summary.get("reports")
    if not isinstance(reports, dict):
        return False
    required_reports = ["manifest_eval", "rubric_eval"]
    if summary.get("prediction_generation_enabled"):
        required_reports.append("prediction_generation")
    if answer_judge:
        required_reports.append("answer_judge")
    return all(
        isinstance(reports.get(name), str) and (root / reports[name]).exists()
        for name in required_reports
    )


def _same_existing_path(left: object, right: str | Path) -> bool:
    if not isinstance(left, str) or not left:
        return False
    return _absolute_path(left) == _absolute_path(right)


def _resolved_targets(
    *,
    predictions: str | None,
    understanding_provider: str,
    understanding_model: str | None,
    mode: EvalMode,
    target_specs: list[str] | None,
    models_file: str | None,
    targets_file: str | None,
    targets: list[EvalTarget] | None,
) -> list[EvalTarget] | None:
    """Return explicit batch targets, or None for legacy single-target eval."""
    explicit = merge_eval_targets(
        list(targets or []),
        eval_targets_from_specs(target_specs),
        eval_targets_from_models_file(models_file),
        eval_targets_from_file(targets_file),
    )
    if not explicit:
        return None
    if predictions is not None:
        raise ValueError("use either legacy --predictions or explicit --target/--targets, not both")
    _ = understanding_provider, understanding_model, mode
    return explicit


def default_eval_output_dir(
    manifest: str | Path,
    *,
    predictions: str | Path | None = None,
    understanding_provider: str = OPENAI_UNDERSTANDING_PROVIDER,
    understanding_model: str | None = None,
    mode: EvalMode = "turn_based",
    live_probe: bool = False,
    tool_protocol: ToolProtocol | None = None,
) -> Path:
    """Return the deterministic repo-local output path for an eval target."""
    repo_root = _repo_root()
    resolved_tool_protocol = tool_protocol or default_tool_protocol(
        mode=mode,
        provider=understanding_provider,
        predictions=predictions,
    )
    return (
        _eval_root(repo_root)
        / _manifest_source_path(manifest, repo_root)
        / _eval_source_path(
            predictions,
            repo_root,
            mode=mode,
            understanding_provider=understanding_provider,
            understanding_model=understanding_model,
            live_probe=live_probe,
            tool_protocol=resolved_tool_protocol,
        )
    )


def _eval_root(repo_root: Path) -> Path:
    override = os.getenv("IB_EVAL_ROOT")
    return Path(override) if override else repo_root / DEFAULT_EVAL_ROOT


def refresh_leaderboard_if_eval_artifact(summary: dict, *, repo_root: Path) -> None:
    """Update the static leaderboard from one eligible target summary."""
    if not _auto_leaderboard_enabled() or summary.get("schema_version") != "ib.eval.v1":
        return
    raw_output = summary.get("output")
    if not raw_output:
        return
    output = Path(str(raw_output))
    eval_root = _eval_root(repo_root).resolve()
    validation_root = (repo_root / "artifacts" / "validation").resolve()
    output_path = (Path.cwd() / output).resolve() if not output.is_absolute() else output.resolve()
    if not _is_relative_to_any(output_path, [eval_root, validation_root]):
        return
    summary_path = output_path / "summary.json"
    if not summary_path.exists():
        return
    # The leaderboard index builder is private tooling; the released
    # `ib eval` stops after writing summary.json.
    return


def _auto_leaderboard_enabled() -> bool:
    """Return whether eval should refresh the leaderboard automatically."""
    value = os.getenv("IB_EVAL_AUTO_LEADERBOARD")
    if value is None:
        return True
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _is_relative_to_any(path: Path, roots: list[Path]) -> bool:
    """Return true when path is contained by any root."""
    for root in roots:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd().resolve()


def _manifest_source_path(manifest: str | Path, repo_root: Path) -> Path:
    relative = _safe_relative_path(manifest, repo_root)
    if relative.name == "manifest.jsonl" and relative.parent.name == "manifest":
        return relative.parent.parent
    return _without_jsonl_suffix(relative)


def _eval_source_path(
    predictions: str | Path | None,
    repo_root: Path,
    *,
    mode: str,
    understanding_provider: str,
    understanding_model: str | None,
    live_probe: bool = False,
    tool_protocol: str = "prompt_json",
    reasoning_effort: str | None = None,
    thinking_mode: GeminiThinkingMode | None = None,
) -> Path:
    if predictions is not None:
        return (
            Path("predictions")
            / _safe_path_segment(mode)
            / _without_jsonl_suffix(_safe_relative_path(predictions, repo_root))
        )
    mode_path = Path("generated") / _safe_path_segment(mode)
    if mode == "turn_based" and tool_protocol == "native":
        mode_path = mode_path / "native"
    if mode == "fullduplex":
        mode_path = mode_path / ("with_probe" if live_probe else "base_only")
    source = (
        mode_path
        / _safe_path_segment(understanding_provider)
        / _safe_path_segment(
            _resolved_understanding_model(understanding_provider, understanding_model)
        )
    )
    if reasoning_effort is not None:
        source = source / f"reasoning_{_safe_path_segment(reasoning_effort)}"
    if thinking_mode is not None:
        source = source / f"thinking_{_safe_path_segment(thinking_mode)}"
    return source


def _target_source_path(target: EvalTarget, repo_root: Path, *, live_probe: bool = False) -> Path:
    return _eval_source_path(
        target.predictions,
        repo_root,
        mode=target.mode,
        understanding_provider=target.provider,
        understanding_model=target.model,
        live_probe=live_probe,
        tool_protocol=target.tool_protocol,
        reasoning_effort=target.reasoning_effort,
        thinking_mode=target.thinking_mode,
    )


def _resolved_understanding_model(provider: str, model: str | None) -> str:
    if model:
        return model
    raise ValueError(
        f"eval target for provider {provider!r} requires an explicit model; "
        "pass --models, --target, or --understanding-model"
    )


def _safe_relative_path(path: str | Path, repo_root: Path) -> Path:
    resolved = _absolute_path(path)
    try:
        relative = resolved.relative_to(repo_root)
    except ValueError:
        relative = Path("external") / Path(*resolved.parts[1:])
    return Path(*(_safe_path_segment(part) for part in relative.parts if part not in {"", "."}))


def _absolute_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path.cwd() / candidate).resolve()


def _without_jsonl_suffix(path: Path) -> Path:
    return path.with_suffix("") if path.suffix == ".jsonl" else path


def _relative_to(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _safe_path_segment(value: str) -> str:
    segment = re.sub(r"[^A-Za-z0-9._=-]+", "_", value).strip("._")
    return segment or "unnamed"
