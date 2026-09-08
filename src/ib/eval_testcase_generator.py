"""eval_testcase_generator — read actual testcase-generator metadata.

Calling spec:
    testcase_generator_metadata(manifest_path) -> {provider: str, model: str}

Inputs: a canonical scene manifest path.
Outputs: the accepted generation-attempt provider/model, or ``unknown``.
Side effects: reads sibling ``generation/attempt_log.jsonl`` when it exists.
"""

from __future__ import annotations

import json
from pathlib import Path


UNKNOWN_GENERATOR = "unknown"


def testcase_generator_metadata(manifest_path: str | Path) -> dict[str, str]:
    """Return the actual routed model recorded for accepted generation attempts."""
    scene_root = _scene_root(Path(manifest_path))
    attempt_metadata = _accepted_attempt_metadata(scene_root / "generation" / "attempt_log.jsonl")
    return attempt_metadata if attempt_metadata is not None else _unknown_metadata()


def _scene_root(manifest_path: Path) -> Path:
    """Return the scene artifact root containing a canonical manifest."""
    return (
        manifest_path.parent.parent
        if manifest_path.parent.name == "manifest"
        else manifest_path.parent
    )


def _accepted_attempt_metadata(path: Path) -> dict[str, str] | None:
    """Return the one provider/model used by accepted attempts, if recorded."""
    if not path.is_file():
        return None
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError):
        return _unknown_metadata()
    generators = {
        (_nonempty_string(row.get("provider")), _nonempty_string(row.get("model")))
        for row in rows
        if isinstance(row, dict) and row.get("accepted") is True
    }
    if len(generators) != 1:
        return _unknown_metadata()
    provider, model = generators.pop()
    return {"provider": provider, "model": model}


def _nonempty_string(value: object) -> str:
    return value if isinstance(value, str) and value else UNKNOWN_GENERATOR


def _unknown_metadata() -> dict[str, str]:
    return {"provider": UNKNOWN_GENERATOR, "model": UNKNOWN_GENERATOR}
