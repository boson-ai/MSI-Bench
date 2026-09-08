#!/usr/bin/env bash
# Calling spec:
#   slurm/submit.sh <model-key>|all [--suite DIR] [--output-root DIR] [--limit N]
#                   [--time HH:MM:SS] [--partition NAME] [--dry-run]
# Inputs: slurm/models.d/<key>.env for GPUS/CPUS/MEM/TIME resource defaults.
# Outputs: one sbatch job per model key; prints submitted job ids.
# Side effects: submits Slurm jobs (or prints the commands with --dry-run).
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"

usage() {
    cat <<'USAGE'
Usage:
  slurm/submit.sh <model-key>|all [options]

Model keys: any slurm/models.d/<key>.env (e.g. qwen2-audio, gemma4-12b,
  mimo-audio, qwen3-omni-instruct, voxtral-small-24b, qwen2.5-omni, phi4-multimodal)

Options:
  --suite DIR        suite dir with <run>/manifest/manifest.jsonl entries
                     (default: slurm/suites/logic_preview_20260716)
  --output-root DIR  eval output root; job writes to <root>/<model-key>
                     (default: artifacts/eval/logic_preview_20260716)
  --limit N          pass --limit N to ib eval (smoke tests)
  --extra-args STR   extra ib eval flags appended verbatim (word-split),
                     e.g. "--case-modes speak --speak-probe-approx"
  --time HH:MM:SS    override the .env TIME value
  --partition NAME   Slurm partition (default: main)
  --dry-run          print sbatch commands without submitting
USAGE
}

[[ $# -ge 1 ]] || { usage >&2; exit 2; }
selector="$1"
shift

suite="$repo_root/slurm/suites/logic_preview_20260716"
output_root="$repo_root/artifacts/eval/logic_preview_20260716"
limit=""
extra_args=""
time_override=""
partition="main"
dry_run=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --suite) suite="${2:?missing --suite value}"; shift 2 ;;
        --output-root) output_root="${2:?missing --output-root value}"; shift 2 ;;
        --limit) limit="${2:?missing --limit value}"; shift 2 ;;
        --extra-args) extra_args="${2:?missing --extra-args value}"; shift 2 ;;
        --time) time_override="${2:?missing --time value}"; shift 2 ;;
        --partition) partition="${2:?missing --partition value}"; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

keys=()
if [[ "$selector" == "all" ]]; then
    for env_file in "$script_dir"/models.d/*.env; do
        keys+=("$(basename "$env_file" .env)")
    done
else
    keys=("$selector")
fi

for key in "${keys[@]}"; do
    env_file="$script_dir/models.d/$key.env"
    [[ -f "$env_file" ]] || { echo "missing $env_file" >&2; exit 2; }
    # Subshell so one model's env values never leak into the next.
    (
        source "$env_file"
        job_time="${time_override:-$TIME}"
        export IB_EVAL_EXTRA_ARGS="$extra_args"
        cmd=(sbatch
            --job-name "ib-eval-$key"
            --partition "$partition"
            --gpus-per-node "$GPUS"
            --cpus-per-task "$CPUS"
            --mem "$MEM"
            --time "$job_time"
            "$script_dir/ib_eval_local.sbatch" "$key" "$suite" "$output_root")
        [[ -n "$limit" ]] && cmd+=("$limit")
        if [[ "$dry_run" -eq 1 ]]; then
            printf '%q ' "${cmd[@]}"
            printf '\n'
        else
            "${cmd[@]}"
        fi
    )
done
