"""report_policy - verify persisted scorer reports before cache reuse.

Calling spec:
    summary_report_has_policy(root, reports, name, expected_policy) -> bool

Reads one JSON report referenced by a summary. It never modifies artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def summary_report_has_policy(
    root: str | Path,
    reports: Mapping[str, Any],
    name: str,
    expected_policy: str,
) -> bool:
    """Return whether a referenced JSON report declares the expected policy."""
    relative = reports.get(name)
    if not isinstance(relative, str):
        return False
    try:
        payload = json.loads((Path(root) / relative).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(payload, dict) and payload.get("policy_version") == expected_policy
