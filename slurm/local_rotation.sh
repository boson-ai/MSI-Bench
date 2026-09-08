#!/usr/bin/env bash
# Calling spec:
#   slurm/local_rotation.sh <model-key>... [--gpu N] [--suite DIR] [--output-root DIR] [--limit N]
# Inputs: slurm/models.d/<key>.env + .models.yaml (same contracts as the sbatch path);
#         rootless enroot on the local host; one free GPU.
# Outputs: serves each model sequentially on the chosen GPU via enroot, runs ib eval
#          against it, writes predictions + reports to <output-root>/<key>/<scene>/.
# Side effects: extracts sqsh images under ENROOT_DATA_PATH on first use; no leaderboard refresh.
set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"

# Parallel lanes: three vLLM servers of OpenBLAS threads exhaust the default
# per-user RLIMIT_NPROC (2048); raise to the hard limit and pin BLAS threads.
ulimit -u 65536 2>/dev/null || ulimit -u "$(ulimit -Hu)" 2>/dev/null || true

export ENROOT_DATA_PATH="${ENROOT_DATA_PATH:-$HOME/.local/share/enroot/data}"
export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH:-$HOME/.local/share/enroot/cache}"
export ENROOT_RUNTIME_PATH="${ENROOT_RUNTIME_PATH:-/tmp/enroot-runtime-$USER}"
mkdir -p "$ENROOT_DATA_PATH" "$ENROOT_CACHE_PATH" "$ENROOT_RUNTIME_PATH"

gpu=5
port=18000
suite="$repo_root/slurm/suites/logic_preview_20260716"
output_root="$repo_root/artifacts/eval/logic_preview_20260716"
limit=""
keys=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) gpu="${2:?missing --gpu value}"; shift 2 ;;
        --port) port="${2:?missing --port value}"; shift 2 ;;
        --suite) suite="${2:?missing --suite value}"; shift 2 ;;
        --output-root) output_root="${2:?missing --output-root value}"; shift 2 ;;
        --limit) limit="${2:?missing --limit value}"; shift 2 ;;
        *) keys+=("$1"); shift ;;
    esac
done
[[ ${#keys[@]} -ge 1 ]] || { echo "usage: local_rotation.sh <model-key>... [--gpu N]" >&2; exit 2; }

CACHE_DIR=${IB_EVAL_HOTDATA:?}/project/evaluation/cache
LOG_DIR="$output_root/logs"
mkdir -p "$LOG_DIR"

# The shared host runs other people's services (e.g. port 8000); a stale
# listener would fake the readiness probe, so demand a free dedicated port.
if curl -s --max-time 2 "http://localhost:$port/" > /dev/null 2>&1; then
    echo "[rotation] FATAL: port $port already in use on this host" >&2
    exit 1
fi

server_is_ready() {
    curl -sf --max-time 5 "http://localhost:$port/v1/models" 2>/dev/null | grep -q '"data"'
}

SERVER_PID=""
cleanup() {
    [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
}
trap cleanup EXIT

declare -A results

for key in "${keys[@]}"; do
    env_file="$script_dir/models.d/$key.env"
    models_yaml="$script_dir/models.d/$key.models.yaml"
    if [[ ! -f "$env_file" || ! -f "$models_yaml" ]]; then
        echo "[rotation] $key SKIPPED: missing models.d config"
        results[$key]="skipped"
        continue
    fi
    # Reset per-model vars so one .env never leaks into the next.
    IMAGE="" CONFIG_DIR="" CONFIG_FILE="" MODEL_PATH="" TP=1 OMNI=0 CONTAINER_ENV="" VLLM_EXTRA_ARGS=""
    source "$env_file"

    container="$(basename "$IMAGE" .sqsh)"
    if ! enroot list | grep -qx "$container"; then
        echo "[rotation] $key: extracting image $container..."
        if ! enroot create --name "$container" "$IMAGE" >> "$LOG_DIR/$key.serve.log" 2>&1; then
            echo "[rotation] $key FAILED: enroot create"
            results[$key]="create-failed"
            continue
        fi
    fi

    omni_flag=""
    [[ "${OMNI:-0}" == "1" ]] && omni_flag="--omni"
    env_args=(--env "NVIDIA_VISIBLE_DEVICES=$gpu"
              --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
              --env OPENBLAS_NUM_THREADS=1
              --env HF_HOME=/cache/hf_hub --env HF_HUB_OFFLINE=1 --env HF_DATASETS_OFFLINE=1)
    for kv in ${CONTAINER_ENV:-}; do
        env_args+=(--env "$kv")
    done

    echo "[rotation] $key: starting server (gpu $gpu, image $container)"
    enroot start --rw \
        --rc "$script_dir/enroot_exec.rc" \
        --mount /ceph:/ceph \
        --mount /ceph/models:/models \
        --mount ${IB_EVAL_HOTDATA:?}:${IB_EVAL_HOTDATA:?} \
        --mount "$CACHE_DIR:/cache" \
        --mount "$repo_root/$CONFIG_DIR:/config" \
        "${env_args[@]}" \
        "$container" \
        vllm serve --config "/config/$CONFIG_FILE" $omni_flag \
            --tensor-parallel-size "$TP" --model "$MODEL_PATH" --port "$port" ${VLLM_EXTRA_ARGS:-} \
        >> "$LOG_DIR/$key.serve.log" 2>&1 &
    SERVER_PID=$!

    ready=0
    elapsed=0
    while [[ $ready -eq 0 ]]; do
        if server_is_ready; then
            ready=1
        elif ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[rotation] $key FAILED: server died during startup (see logs/$key.serve.log)"
            results[$key]="server-died"
            break
        elif [[ $elapsed -ge 3600 ]]; then
            echo "[rotation] $key FAILED: server not ready after 3600s"
            results[$key]="server-timeout"
            kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null
            break
        else
            sleep 5; elapsed=$((elapsed + 5))
        fi
    done
    if [[ $ready -eq 0 ]]; then
        SERVER_PID=""
        continue
    fi
    echo "[rotation] $key: server ready after ${elapsed}s, running eval"

    limit_args=()
    [[ -n "$limit" ]] && limit_args=(--limit "$limit")
    IB_EVAL_AUTO_LEADERBOARD=0 \
    LOCAL_AUDIO_UNDERSTANDING_BASE_URL="http://localhost:$port/v1" \
    "$repo_root/.venv/bin/python" -m ib eval \
        --run "$suite" \
        --models "$models_yaml" \
        --output "$output_root/$key" \
        --target-workers 1 \
        "${limit_args[@]}" \
        >> "$LOG_DIR/$key.eval.log" 2>&1
    rc=$?
    if [[ $rc -eq 0 ]]; then
        echo "[rotation] $key DONE"
        results[$key]="done"
    else
        echo "[rotation] $key FAILED: ib eval rc=$rc (see logs/$key.eval.log)"
        results[$key]="eval-failed"
    fi

    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
    SERVER_PID=""
    sleep 5
done

echo "[rotation] all models processed:"
for key in "${keys[@]}"; do
    echo "[rotation]   $key: ${results[$key]:-unknown}"
done
