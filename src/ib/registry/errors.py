"""errors — the explicit deferred-path signal for declared registries.

Calling spec:
    raise UnsupportedHandlerError(registry="plane", key="some_future_plane",
                                  reason="declared but not implemented")

`UnsupportedHandlerError` is the single, loud failure for any declared-but-not-yet
-implemented path (proposal-v2 Option A). A registry NEVER returns None or falls
through silently for a deferred or unknown key — it raises this instead.

Side effects: none.
"""

from __future__ import annotations


class UnsupportedHandlerError(Exception):
    """Raised when a declared registry path has no implemented handler yet."""

    def __init__(self, registry: str, key: str, reason: str = "") -> None:
        self.registry = registry
        self.key = key
        self.reason = reason
        detail = f": {reason}" if reason else ""
        super().__init__(f"{registry} registry: path {key!r} is not supported yet{detail}")
