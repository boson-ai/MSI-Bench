"""site_paths — where local model weights live.

Calling spec:
    model_path(name, env=None) -> Path      $IB_MODELS_ROOT/<name>, default ./models/<name>

Only informational in the evaluation code (it is recorded in reports); the evaluated
model is reached through its OpenAI-compatible base URL, not through this path.
No side effects; reads the environment only.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path


def model_path(name: str, env: Mapping[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    return Path(env.get("IB_MODELS_ROOT") or "models") / name
