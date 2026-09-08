"""probe_sampling — deterministic per-case speak-probe parameters.

Calling spec:
    truncate_fraction_for_row(row) -> float in [0.1, 0.5]
    inject_offset_s_for_row(row) -> float in [1.0, 2.0]

Inputs are manifest rows (cell_id plus optional per-row override fields below).
The same cell_id always maps to the same value (md5-based, one salt per
parameter), so every model sees an identical stimulus and per-case resume stays
stable, while values vary across cases. Deterministic, no side effects.
"""

from __future__ import annotations

import hashlib
from typing import Any

TRUNCATE_FRACTION_FIELD = "logic_speak_truncate_fraction"
INJECT_OFFSET_FIELD = "logic_speak_inject_offset_s"

TRUNCATE_FRACTION_RANGE = (0.1, 0.5)
INJECT_OFFSET_RANGE_S = (1.0, 2.0)


def truncate_fraction_for_row(row: dict[str, Any]) -> float:
    """Barge-in truncation point: per-row override or per-case sample."""
    return _resolve(row, TRUNCATE_FRACTION_FIELD, TRUNCATE_FRACTION_RANGE, "speak_truncate")


def inject_offset_s_for_row(row: dict[str, Any]) -> float:
    """Fullduplex injection offset into assistant speech, in seconds."""
    return _resolve(row, INJECT_OFFSET_FIELD, INJECT_OFFSET_RANGE_S, "speak_inject")


def _resolve(
    row: dict[str, Any],
    field: str,
    bounds: tuple[float, float],
    salt: str,
) -> float:
    low, high = bounds
    raw = row.get(field)
    if isinstance(raw, (int, float)):
        return min(max(float(raw), low), high)
    return low + (high - low) * _unit(str(row.get("cell_id") or ""), salt)


def _unit(key: str, salt: str) -> float:
    digest = hashlib.md5(f"{salt}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0x100000000
