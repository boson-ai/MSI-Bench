"""main — argparse command router for the independently runnable benchmark phases.

Calling spec:
    main(argv=None) -> int

Each command delegates to one phase module. Expected input/config errors are
reported without tracebacks and result in a non-zero exit code.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from ib.eval_targets import EVAL_TARGET_PRESETS
from ib.llm.qwen import DEFAULT_LLM_PROVIDER

LLM_PROVIDER_CHOICES = ("openai", "gemini", "openrouter", "qwen3_5", "qwen3_5_thinking")
JUDGE_PROVIDER_CHOICES = ("openai", "openrouter", "qwen3_5", "qwen3_5_thinking")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ib", description="MSI-Bench staged builder")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-predictions", help="validate model prediction JSONL")
    validate.add_argument("--predictions", required=True)
    validate.add_argument("--output", required=True)

    manifest_eval = commands.add_parser(
        "manifest-eval", help="run manifest-level prediction metrics"
    )
    manifest_eval.add_argument("--manifest", required=True)
    manifest_eval.add_argument("--predictions", required=True)
    manifest_eval.add_argument("--output", required=True)
    manifest_eval.add_argument(
        "--mode",
        default="turn_based",
        choices=("turn_based", "fullduplex"),
        help="evaluation mode for probe applicability",
    )

    rubric_eval = commands.add_parser("rubric-eval", help="validate rubric references and coverage")
    rubric_eval.add_argument("--manifest", required=True)
    rubric_eval.add_argument("--rubrics", required=True)
    rubric_eval.add_argument("--output", required=True)

    eval_cmd = commands.add_parser("eval", help="run model-output evaluation recipe")
    eval_cmd.add_argument(
        "--run",
        dest="validation_run",
        help=(
            "validation run directory containing scene subdirectories; discovers all "
            "*/manifest/manifest.jsonl files and uses each scene's rubrics"
        ),
    )
    eval_cmd.add_argument("--manifest")
    eval_cmd.add_argument("--predictions")
    eval_cmd.add_argument("--rubrics")
    eval_cmd.add_argument(
        "--mode",
        default="turn_based",
        choices=("turn_based", "fullduplex"),
        help="evaluation mode for legacy single-target eval",
    )
    eval_cmd.add_argument(
        "--target",
        action="append",
        default=[],
        help=(
            "repeatable batch target: MODE:PROVIDER:MODEL or "
            "MODE:PROVIDER:MODEL:PREDICTIONS"
        ),
    )
    eval_cmd.add_argument(
        "--targets",
        help="YAML/JSON file containing a list or {targets: [...]} eval target objects",
    )
    eval_cmd.add_argument(
        "--models",
        dest="models_file",
        help=(
            "YAML file with base: and live: model declarations; --run reads "
            "configs/eval-models.yaml automatically when this is omitted"
        ),
    )
    eval_cmd.add_argument(
        "--preset",
        dest="target_preset",
        choices=tuple(EVAL_TARGET_PRESETS),
        help=(
            "built-in target set for explicit use; prefer --models configs/eval-models.yaml "
            "for declared benchmark runs"
        ),
    )
    eval_cmd.add_argument(
        "--output",
        help=(
            "directory for eval reports; single-manifest default mirrors the manifest path "
            "under artifacts/eval; --run default writes a timestamped run under artifacts/eval"
        ),
    )
    eval_cmd.add_argument(
        "--understanding-provider",
        default="openai",
        choices=("openai", "openai_realtime", "gemini", "gemini_live", "boson_realtime", "local", "qwen"),
        help="provider used to generate predictions when --predictions is omitted",
    )
    eval_cmd.add_argument(
        "--understanding-model",
        help="audio understanding model used when --predictions is omitted",
    )
    eval_cmd.add_argument(
        "--limit",
        type=int,
        help="limit live understanding prediction generation to the first N manifest rows",
    )
    eval_cmd.add_argument(
        "--target-workers",
        type=int,
        help=(
            "number of eval targets to run concurrently per scene; defaults to all targets. "
            "Use 1 for previous serial behavior."
        ),
    )
    eval_cmd.add_argument(
        "--live-probe",
        action="store_true",
        help=(
            "enable speak-time live probe injection for fullduplex targets; "
            "when omitted, fullduplex eval runs base/no-probe only"
        ),
    )
    eval_cmd.add_argument(
        "--case-modes",
        help=(
            "comma-separated logic case modes to evaluate "
            "(base, hear, speak or full logic_case_mode names); default all applicable"
        ),
    )
    eval_cmd.add_argument(
        "--speak-probe-approx",
        action="store_true",
        help=(
            "evaluate speak_time_probe rows on turn-based targets via the "
            "barge-in resume approximation (truncated assistant turn + live "
            "event as new final user turn); scored as probe_variant="
            "barge_in_resume, never mixed with fullduplex live-probe numbers"
        ),
    )
    eval_cmd.add_argument("--answer-judge", action="store_true")
    eval_cmd.add_argument(
        "--judge-provider",
        default=DEFAULT_LLM_PROVIDER,
        choices=JUDGE_PROVIDER_CHOICES,
    )
    eval_cmd.add_argument("--judge-model")

    answer_judge = commands.add_parser("answer-judge", help="run optional answer-layer judge")
    answer_judge.add_argument("--manifest", required=True)
    answer_judge.add_argument("--predictions", required=True)
    answer_judge.add_argument("--rubrics", required=True)
    answer_judge.add_argument("--output", required=True)
    answer_judge.add_argument(
        "--provider",
        default=DEFAULT_LLM_PROVIDER,
        choices=JUDGE_PROVIDER_CHOICES,
    )
    answer_judge.add_argument("--model")

    score = commands.add_parser("score-smoke", help="run exact-action scoring smoke")
    score.add_argument("--manifest", required=True)
    score.add_argument("--predictions")
    score.add_argument("--rubrics")
    score.add_argument("--output", required=True)

    return parser


def _split_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("comma-separated logic option must contain at least one value")
    return items


def _split_int_csv(value: str | None) -> list[int] | None:
    items = _split_csv(value)
    if items is None:
        return None
    try:
        return [int(item) for item in items]
    except ValueError as exc:
        raise ValueError("--logic-speaker-counts must be comma-separated integers") from exc


def _split_live_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    if value.strip().lower() in {"none", "off", "base"}:
        return []
    return _split_csv(value)


def _dispatch(args: argparse.Namespace) -> None:
    if args.command == "validate-predictions":
        from ib.cli.predictions import run

        run(args.predictions, args.output)
    elif args.command == "manifest-eval":
        from ib.cli.eval import run_manifest_eval

        run_manifest_eval(args.manifest, args.predictions, args.output, mode=args.mode)
    elif args.command == "rubric-eval":
        from ib.cli.eval import run_rubric_eval

        run_rubric_eval(args.manifest, args.rubrics, args.output)
    elif args.command == "eval":
        from ib.cli.eval import run_eval

        summary = run_eval(
            args.manifest,
            args.predictions,
            args.rubrics,
            args.output,
            answer_judge=args.answer_judge,
            judge_provider=args.judge_provider,
            judge_model=args.judge_model,
            understanding_provider=args.understanding_provider,
            understanding_model=args.understanding_model,
            mode=args.mode,
            limit=args.limit,
            target_specs=args.target,
            target_preset=args.target_preset,
            models_file=args.models_file,
            targets_file=args.targets,
            validation_run=args.validation_run,
            target_workers=args.target_workers,
            live_probe=args.live_probe,
            case_modes=args.case_modes,
            speak_probe_approx=args.speak_probe_approx,
        )
        output = summary.get("output") if isinstance(summary, dict) else None
        passed = summary.get("passed") if isinstance(summary, dict) else None
        print(f"eval complete passed={passed} output={output}")
    elif args.command == "answer-judge":
        from ib.cli.eval import run_answer_judge

        run_answer_judge(
            args.manifest, args.predictions, args.rubrics, args.output, args.provider, args.model
        )
    elif args.command == "score-smoke":
        from ib.cli.score_smoke import run

        run(args.manifest, args.output, args.predictions, args.rubrics)

def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run one phase, and translate actionable errors to exit status 2."""
    from ib.cli.environment import load_cli_environment
    from ib.registry.errors import UnsupportedHandlerError

    load_cli_environment()
    try:
        args = _parser().parse_args(argv)
        _dispatch(args)
    except (OSError, ValueError, UnsupportedHandlerError) as exc:
        print(f"ib: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
