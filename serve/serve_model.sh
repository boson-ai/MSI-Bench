#!/usr/bin/env bash
# Calling spec:
#   serve_model.sh <model-key> [options]
# Inputs: local Docker image, local model weights under /ceph/models, NVIDIA GPUs.
# Outputs: starts an OpenAI-compatible vLLM server container, or prints the command with --dry-run.
# Side effects: creates/removes no files except Docker container state; uses repo-local configs under serve/configs.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  serve/serve_model.sh <model> [options]

Models:
  qwen3-omni-thinking   Qwen3-Omni-30B-A3B-Thinking checkpoint, served as Qwen3-Omni-30B-A3B-Instruct
  qwen3-omni-instruct   Qwen3-Omni-30B-A3B-Instruct checkpoint
  gemma4-12b            Gemma 4 12B Instruct (request-level thinking toggle)
  mimo-audio            MiMo-Audio-7B-Instruct (request-level thinking toggle)
  voxtral-small-24b     Voxtral-Small-24B-2507
  kimi-audio            Kimi-Audio-7B-Instruct ASR-capable endpoint
  qwen2.5-omni          Qwen2.5-Omni-7B
  phi4-multimodal       Phi-4-multimodal-instruct
  qwen2-audio           Qwen2-Audio-7B-Instruct

Options:
  --gpus <ids>          CUDA_VISIBLE_DEVICES list; model-specific default.
  --tp <n>              Tensor parallel size; model-specific default.
  --port <port>         Host port; defaults match configs/eval-models-local.yaml.
  --name <name>         Docker container name.
  --model-path <path>   Container model path. Defaults are under /models.
  --image <tag>         Override Docker image.
  --extra-arg <arg>     Append a raw vLLM arg after `serve --config ...`; repeatable.
  --foreground          Do not detach; stream server logs in foreground.
  --online              Set HF_HUB_OFFLINE=0 and HF_DATASETS_OFFLINE=0.
  --dry-run             Print the docker command without executing it.
  -h, --help            Show this help.

Examples:
  serve/serve_model.sh qwen3-omni-thinking --gpus 1,2 --tp 2 --port 8013 --name qwen3-omni-thinking
  serve/serve_model.sh qwen2-audio --gpus 6 --port 8020 --name qwen2-audio
  serve/serve_model.sh kimi-audio --gpus 5 --port 8017 --name kimi-audio
USAGE
}

if [[ $# -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

model_key="$1"
shift

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
cache_dir="${CACHE_DIR:-$HOME/.cache/msi-bench-eval}"
output_root="${OUTPUT_ROOT:-artifacts/serve}"
internal_port=8000
detach=1
dry_run=0
online=0
extra_args=()
omni=0
mimo_tokenizer=0

case "$model_key" in
  qwen3-omni-thinking)
    config_subdir="qwen3-omni"
    config_file="vllm-audio.yaml"
    image="registry.canada.boson.ai/eval-runner:vllm-audio-20260403"
    model_path="/models/Qwen3-Omni-30B-A3B-Thinking"
    gpus="0,1"
    tp="2"
    port="8013"
    name="qwen3-omni-thinking"
    ;;
  qwen3-omni-instruct)
    config_subdir="qwen3-omni"
    config_file="vllm-audio.yaml"
    image="registry.canada.boson.ai/eval-runner:vllm-audio-20260403"
    model_path="/models/Qwen3-Omni-30B-A3B-Instruct"
    gpus="0,1"
    tp="2"
    port="8013"
    name="qwen3-omni-instruct"
    ;;
  gemma4-12b)
    config_subdir="gemma4-12b"
    config_file="vllm-gemma4.yaml"
    image="registry.canada.boson.ai/eval-runner:vllm-gemma4-20260418"
    model_path="/models/gemma-4-12B-it"
    gpus="0"
    tp="1"
    port="8014"
    name="gemma4-12b"
    ;;
  mimo-audio)
    config_subdir="mimo-audio"
    config_file="vllm-omni.yaml"
    image="registry.canada.boson.ai/eval-runner:vllm-omni-20260421"
    model_path="/models/MiMo-Audio-7B-Instruct"
    gpus="0"
    tp="1"
    port="8015"
    name="mimo-audio"
    omni=1
    mimo_tokenizer=1
    ;;
  voxtral-small-24b)
    config_subdir="voxtral-small"
    config_file="vllm-audio.yaml"
    image="registry.canada.boson.ai/eval-runner:vllm-audio-20260403"
    model_path="/models/Voxtral-Small-24B-2507"
    gpus="0,1"
    tp="2"
    port="8016"
    name="voxtral-small-24b"
    ;;
  kimi-audio)
    config_subdir="kimi-audio"
    config_file="vllm-nightly-audio.yaml"
    image="registry.canada.boson.ai/eval-runner:vllm-nightly-audio-20260413"
    model_path="/models/Kimi-Audio-7B-Instruct"
    gpus="0"
    tp="1"
    port="8017"
    name="kimi-audio"
    ;;
  qwen2.5-omni)
    config_subdir="qwen2.5-omni"
    config_file="vllm-audio.yaml"
    image="registry.canada.boson.ai/eval-runner:vllm-audio-20260403"
    model_path="/models/Qwen2.5-Omni-7B"
    gpus="0"
    tp="1"
    port="8018"
    name="qwen2.5-omni"
    ;;
  phi4-multimodal)
    config_subdir="phi4-multimodal"
    config_file="vllm-audio.yaml"
    image="registry.canada.boson.ai/eval-runner:vllm-audio-20260403"
    model_path="/models/Phi-4-multimodal-instruct"
    gpus="0"
    tp="1"
    port="8019"
    name="phi4-multimodal"
    ;;
  qwen2-audio)
    config_subdir="qwen2-audio"
    config_file="vllm-audio.yaml"
    image="registry.canada.boson.ai/eval-runner:vllm-audio-20260403"
    model_path="/models/Qwen2-Audio-7B-Instruct"
    gpus="0"
    tp="1"
    port="8020"
    name="qwen2-audio"
    ;;
  *)
    echo "Unknown model: $model_key" >&2
    usage >&2
    exit 2
    ;;
esac

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) gpus="${2:?missing --gpus value}"; shift 2 ;;
    --tp) tp="${2:?missing --tp value}"; shift 2 ;;
    --port) port="${2:?missing --port value}"; shift 2 ;;
    --name) name="${2:?missing --name value}"; shift 2 ;;
    --model-path) model_path="${2:?missing --model-path value}"; shift 2 ;;
    --image) image="${2:?missing --image value}"; shift 2 ;;
    --extra-arg) extra_args+=("${2:?missing --extra-arg value}"); shift 2 ;;
    --foreground) detach=0; shift ;;
    --online) online=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

config_dir="$script_dir/configs/$config_subdir"
if [[ ! -f "$config_dir/$config_file" ]]; then
  echo "Missing repo-local config: $config_dir/$config_file" >&2
  exit 1
fi

hf_offline="1"
if [[ "$online" -eq 1 ]]; then
  hf_offline="0"
fi

mkdir -p "$cache_dir"

cmd=(docker run --rm)
if [[ "$detach" -eq 1 ]]; then
  cmd+=(-d)
fi
cmd+=(--name "$name" --runtime nvidia --gpus all --ipc host)

add_mount_if_exists() {
  local src="$1"
  local dst="${2:-$1}"
  if [[ -e "$src" ]]; then
    cmd+=(-v "$src:$dst")
  fi
}

add_mount_if_exists /ceph /ceph
add_mount_if_exists /ceph/models /models
add_mount_if_exists /fsx /fsx
add_mount_if_exists /dev/infiniband /dev/infiniband
cmd+=(-v "$cache_dir:/cache")
if [[ -d "$output_root" ]]; then
  cmd+=(-v "$output_root:/output")
fi
cmd+=(-v "$config_dir:/config:ro")
cmd+=(-p "$port:$internal_port")
cmd+=(-e "CUDA_VISIBLE_DEVICES=$gpus")
cmd+=(-e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1)
cmd+=(-e HF_HOME=/cache/hf_hub)
cmd+=(-e "HF_HUB_OFFLINE=$hf_offline")
cmd+=(-e "HF_DATASETS_OFFLINE=$hf_offline")
if [[ "$mimo_tokenizer" -eq 1 ]]; then
  cmd+=(-e MIMO_AUDIO_TOKENIZER_PATH=/models/MiMo-Audio-Tokenizer)
fi
cmd+=(--entrypoint vllm "$image")
cmd+=(serve --config "/config/$config_file")
if [[ "$omni" -eq 1 ]]; then
  cmd+=(--omni)
fi
cmd+=(--tensor-parallel-size "$tp" --model "$model_path")
cmd+=("${extra_args[@]}")

if [[ "$dry_run" -eq 1 ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

printf 'Starting %s on http://localhost:%s/v1 using GPUs %s (tp=%s)\n' "$model_key" "$port" "$gpus" "$tp" >&2
"${cmd[@]}"
