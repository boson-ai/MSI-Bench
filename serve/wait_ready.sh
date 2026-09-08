#!/usr/bin/env bash
# Calling spec:
#   wait_ready.sh [port] [timeout_seconds]
# Inputs: a host port for an OpenAI-compatible server.
# Outputs: prints /v1/models JSON when ready; exits nonzero on timeout.
# Side effects: none.
set -euo pipefail

port="${1:-8013}"
timeout_seconds="${2:-600}"
base_url="http://localhost:${port}/v1"
deadline=$((SECONDS + timeout_seconds))

while true; do
  if body="$(curl -fsS "$base_url/models" 2>/dev/null)"; then
    printf '%s\n' "$body"
    exit 0
  fi
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for $base_url/models after ${timeout_seconds}s" >&2
    exit 1
  fi
  echo "waiting for $base_url/models ..." >&2
  sleep 10
done
