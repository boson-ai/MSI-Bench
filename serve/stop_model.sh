#!/usr/bin/env bash
# Calling spec:
#   stop_model.sh <container-name>
# Inputs: Docker container name.
# Outputs: Docker removal result.
# Side effects: stops and removes the named container.
set -euo pipefail

name="${1:-}"
if [[ -z "$name" || "$name" == "-h" || "$name" == "--help" ]]; then
  echo "Usage: serve/stop_model.sh <container-name>" >&2
  exit 2
fi

docker rm -f "$name"
