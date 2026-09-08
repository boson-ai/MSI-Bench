"""scoring.progress — terminal progress for eval prediction generation.

Calling spec:
    print_eval_case_progress(provider, model, cell_id, index, total) -> None
    line = format_eval_case_progress(provider, model, cell_id, index, total)

Side effects: print_eval_case_progress writes thread-safe stderr progress.
"""

from __future__ import annotations

import os
import sys
import threading
from typing import TextIO

_PROGRESS_LOCK = threading.Lock()
_BAR_WIDTH = 20
_LIVE_STATE: dict[str, str] = {}
_LIVE_RENDERED_LINES = 0


def format_eval_case_progress(
    provider: str,
    model: str,
    cell_id: str,
    index: int,
    total: int,
) -> str:
    """Return one stable per-case eval progress line with a compact progress bar."""
    safe_total = max(total, 1)
    safe_index = min(max(index, 0), safe_total)
    filled = round((safe_index / safe_total) * _BAR_WIDTH)
    bar = "#" * filled + "-" * (_BAR_WIDTH - filled)
    percent = round((safe_index / safe_total) * 100)
    return f"ib eval: {provider}:{model} case {index}/{total} [{bar}] {percent:3d}% {cell_id}"


def print_eval_case_progress(
    provider: str,
    model: str,
    cell_id: str,
    index: int,
    total: int,
    *,
    stream: TextIO | None = None,
) -> None:
    """Print or repaint per-case eval progress without interleaving concurrent targets."""
    target = stream or sys.stderr
    line = format_eval_case_progress(provider, model, cell_id, index, total)
    with _PROGRESS_LOCK:
        if not _use_live_progress(target):
            print(line, file=target, flush=True)
            return
        _LIVE_STATE[f"{provider}:{model}"] = line
        _render_live_block(target)


def _use_live_progress(stream: TextIO) -> bool:
    mode = os.getenv("IB_EVAL_PROGRESS", "live").strip().lower()
    if mode in {"0", "false", "off", "line", "lines", "plain"}:
        return False
    is_tty = getattr(stream, "isatty", None)
    return bool(is_tty and is_tty())


def _render_live_block(stream: TextIO) -> None:
    global _LIVE_RENDERED_LINES
    lines = list(_LIVE_STATE.values())
    if _LIVE_RENDERED_LINES:
        stream.write(f"\x1b[{_LIVE_RENDERED_LINES}A")
    for line in lines:
        stream.write(f"\r\x1b[2K{line}\n")
    _LIVE_RENDERED_LINES = len(lines)
    stream.flush()
