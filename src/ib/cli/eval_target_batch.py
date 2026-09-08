"""eval_target_batch — resolve eval target sets and target-worker settings.

Calling spec:
    _resolved_run_targets(...) -> list[EvalTarget]
    _eval_target_worker_count(requested, target_count) -> int
    _target_batch_summary_row(target, summary, target_root, batch_root) -> dict

Inputs are target declarations and completed summaries. Outputs are validated,
deterministic values; model modules may be preloaded as the sole side effect.
"""

from __future__ import annotations

from pathlib import Path

from ib.cli.eval_artifacts import _relative_to
from ib.eval_targets import (
    EvalTarget,
    eval_targets_from_file,
    eval_targets_from_models_file,
    eval_targets_from_preset,
    eval_targets_from_specs,
    merge_eval_targets,
)
from ib.scoring.openai_fullduplex import OPENAI_FULLDUPLEX_PROVIDER
from ib.scoring.openai_understanding import OPENAI_UNDERSTANDING_PROVIDER

DEFAULT_EVAL_MODELS_FILES = ("configs/eval-models.yaml", "configs/eval-models.yml")


def _eval_target_worker_count(target_workers: int | None, target_count: int) -> int:
    """Return bounded target-level parallelism for independent model targets."""
    if target_count <= 0:
        return 1
    if target_workers is None:
        return target_count
    if target_workers <= 0:
        raise ValueError("--target-workers must be > 0")
    return min(target_workers, target_count)


def _preload_target_worker_imports(targets: list[EvalTarget], worker_count: int) -> None:
    """Preload provider modules that are unsafe to first-import in worker threads."""
    if worker_count <= 1:
        return
    needs_openai = any(
        target.predictions is None
        and target.provider in {OPENAI_UNDERSTANDING_PROVIDER, OPENAI_FULLDUPLEX_PROVIDER}
        for target in targets
    )
    if not needs_openai:
        return
    __import__("openai.resources.chat")
    __import__("openai.resources.realtime")


def _target_batch_summary_row(
    target: EvalTarget, target_summary: dict, target_root: Path, batch_root: Path
) -> dict:
    """Return one stable aggregate target-summary row."""
    return {
        "name": target.name,
        "mode": target.mode,
        "provider": target.provider,
        "model": target_summary.get("understanding_model") or target.model,
        "reasoning_effort": target.reasoning_effort,
        "thinking_mode": target.thinking_mode,
        "tool_protocol": target.tool_protocol,
        "predictions": target_summary["predictions"],
        "output": target_summary["output"],
        "summary": _relative_to(target_root / "summary.json", batch_root),
        "passed": target_summary["passed"],
    }


def _resolved_run_targets(
    target_specs: list[str] | None,
    target_preset: str | None,
    models_file: str | None,
    targets_file: str | None,
    targets: list[EvalTarget] | None,
) -> list[EvalTarget]:
    """Return explicit run targets, defaulting to the standard models file."""
    resolved_models_file = models_file
    if (
        resolved_models_file is None
        and not target_specs
        and target_preset is None
        and targets_file is None
        and not targets
    ):
        resolved_models_file = _default_models_file()
    resolved = merge_eval_targets(
        list(targets or []),
        eval_targets_from_specs(target_specs),
        eval_targets_from_preset(target_preset),
        eval_targets_from_models_file(resolved_models_file),
        eval_targets_from_file(targets_file),
    )
    if not resolved:
        raise ValueError(
            "at least one eval target is required; pass --models configs/eval-models.yaml, "
            "--target, --targets, or --preset"
        )
    return resolved


def _default_models_file() -> str | None:
    for filename in DEFAULT_EVAL_MODELS_FILES:
        candidate = Path(filename)
        if candidate.exists():
            return str(candidate)
    return None


def _target_specs_with_preset(
    target_specs: list[str] | None, target_preset: str | None
) -> list[str] | None:
    """Append preset targets to compact ``--target`` specs for normal eval."""
    if target_preset is None:
        return target_specs
    specs = list(target_specs or [])
    specs.extend(
        f"{target.mode}:{target.provider}:{target.model}"
        + (f":{target.predictions}" if target.predictions else "")
        for target in eval_targets_from_preset(target_preset)
    )
    return specs
