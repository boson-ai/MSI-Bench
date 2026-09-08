#!/bin/bash
# Calling spec:
#   sbatch --job-name=... --gpus-per-node=N --cpus-per-task=C --mem=M --time=T \
#     slurm/ib_eval_local.sbatch <model-key> [suite-dir] [output-root] [limit]
# Inputs: slurm/models.d/<model-key>.env (serving contract) and .models.yaml (eval targets);
#         suite-dir with <run>/manifest/manifest.jsonl symlinks; enroot sqsh image.
# Outputs: predictions + deterministic eval reports under <output-root>/<model-key>/<scene>/.
# Side effects: serves one vLLM container on the node (port 8000), no leaderboard refresh.
# Prefer slurm/submit.sh, which fills the sbatch resource flags from the .env file.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm-logs/slurm-%j.out
#SBATCH --error=slurm-logs/slurm-%j.err

set -euo pipefail

# Clear TMPDIR inherited from VS Code / other sandboxed shells
unset TMPDIR
ulimit -u 65536 2>/dev/null || true
export OPENBLAS_NUM_THREADS=1

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
MODEL_KEY="${1:?usage: ib_eval_local.sbatch <model-key> [suite-dir] [output-root] [limit]}"
SUITE="${2:-$REPO/slurm/suites/logic_preview_20260716}"
OUTPUT_ROOT="${3:-$REPO/artifacts/eval/logic_preview_20260716}"
LIMIT="${4:-}"

MODEL_ENV="$REPO/slurm/models.d/$MODEL_KEY.env"
MODELS_YAML="$REPO/slurm/models.d/$MODEL_KEY.models.yaml"
[[ -f "$MODEL_ENV" ]] || { echo "missing $MODEL_ENV" >&2; exit 2; }
[[ -f "$MODELS_YAML" ]] || { echo "missing $MODELS_YAML" >&2; exit 2; }
source "$MODEL_ENV"

# Unique per-job port: compute nodes are shared and another job may already
# serve port 8000; a foreign listener would fake the readiness probe.
PORT=$((20000 + SLURM_JOB_ID % 20000))
CACHE_DIR="${IB_EVAL_CACHE_DIR:?set IB_EVAL_CACHE_DIR to a shared HF cache dir}"
MOUNTS="/ceph:/ceph,/ceph/models:/models,${IB_EVAL_HOTDATA:?}:${IB_EVAL_HOTDATA:?},$CACHE_DIR:/cache,$REPO/$CONFIG_DIR:/config"

cleanup() {
    echo "[ib-eval] Cleaning up..."
    kill "${WATCHDOG_PID:-}" 2>/dev/null || true
    kill "${SERVER_PID:-}" 2>/dev/null || true
    wait "${SERVER_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT

# Background watchdog: if the server step dies mid-eval, kill the job
server_watchdog() {
    while kill -0 "$SERVER_PID" 2>/dev/null; do
        sleep 10
    done
    echo "[ib-eval] ERROR: server process died unexpectedly (OOM or crash)"
    kill -TERM $$ 2>/dev/null || true
}

echo "=== ib eval local START: $MODEL_KEY $(date) ==="
echo "[ib-eval] suite=$SUITE output=$OUTPUT_ROOT/$MODEL_KEY limit=${LIMIT:-none}"

SERVE_ENV="export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 HF_HOME=/cache/hf_hub HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1"
if [[ -n "${CONTAINER_ENV:-}" ]]; then
    SERVE_ENV="$SERVE_ENV $CONTAINER_ENV"
fi
OMNI_FLAG=""
[[ "${OMNI:-0}" == "1" ]] && OMNI_FLAG="--omni"

srun --ntasks=1 \
    --container-image "$IMAGE" \
    --container-mounts "$MOUNTS" \
    --no-container-mount-home \
    bash -c "$SERVE_ENV; vllm serve --config /config/$CONFIG_FILE $OMNI_FLAG --tensor-parallel-size $TP --model $MODEL_PATH --port $PORT ${VLLM_EXTRA_ARGS:-}" &
SERVER_PID=$!

TIMEOUT=3600
ELAPSED=0
echo "[ib-eval] Waiting for server on port $PORT (timeout: ${TIMEOUT}s)..."
while ! curl -sf --max-time 5 "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q '"data"'; do
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[ib-eval] ERROR: server process died during startup" >&2
        exit 1
    fi
    if [[ $ELAPSED -ge $TIMEOUT ]]; then
        echo "[ib-eval] ERROR: server timeout (${TIMEOUT}s)" >&2
        exit 1
    fi
done
echo "[ib-eval] Server ready after ${ELAPSED}s."

server_watchdog &
WATCHDOG_PID=$!

export LOCAL_AUDIO_UNDERSTANDING_BASE_URL="http://localhost:${PORT}/v1"
export IB_EVAL_AUTO_LEADERBOARD=0

LIMIT_ARGS=()
[[ -n "$LIMIT" ]] && LIMIT_ARGS=(--limit "$LIMIT")
# Optional ib eval flag pass-through (e.g. "--case-modes speak --speak-probe-approx"),
# set by slurm/submit.sh --extra-args; word-split on purpose.
read -r -a EXTRA_ARGS <<< "${IB_EVAL_EXTRA_ARGS:-}"

cd "$REPO"
# Use a venv whose interpreter is reachable from compute nodes; override
# with IB_EVAL_PYTHON when the default repo venv is not shared.
"${IB_EVAL_PYTHON:-$REPO/.venv/bin/python}" -m ib eval \
    --run "$SUITE" \
    --models "$MODELS_YAML" \
    --output "$OUTPUT_ROOT/$MODEL_KEY" \
    --target-workers 1 \
    "${LIMIT_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"

echo "=== ib eval local END: $MODEL_KEY $(date) ==="
