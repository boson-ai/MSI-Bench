"""usage — async call and token counter for LLM calls.

Calling spec:
    tracker = UsageTracker()
    await tracker.record(LlmUsage(model="gpt-4.1-mini", module="planner.director",
                                  input_tokens=120, output_tokens=80))
    snapshot = tracker.summary()        # dict (token totals + call counts per module and model)
    tracker.write(path)                 # write the snapshot as JSON

No pricing: spend is read from the API key's billing dashboard.

Side effects: ``write`` writes a JSON file.
Dependencies: ib.io.write_json
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ib.io import write_json


@dataclass(frozen=True)
class LlmUsage:
    """Token usage for one completed LLM call (success or failure)."""

    model: str
    module: str
    input_tokens: int
    output_tokens: int


class UsageTracker:
    """Aggregates call counts and tokens across a run. Coroutine-safe via an asyncio.Lock."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._call_count = 0
        self._calls_by_module: dict[str, int] = defaultdict(int)
        self._calls_by_model: dict[str, int] = defaultdict(int)

    async def record(self, usage: LlmUsage) -> None:
        """Record one call's tokens and count it under its module and model."""
        async with self._lock:
            self._total_input_tokens += usage.input_tokens
            self._total_output_tokens += usage.output_tokens
            self._call_count += 1
            self._calls_by_module[usage.module] += 1
            self._calls_by_model[usage.model] += 1

    def summary(self) -> dict:
        """Return a plain-dict snapshot of current aggregation."""
        return {
            "schema_version": "ib.llm_calls.v1",
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "call_count": self._call_count,
            "calls_by_module": dict(sorted(self._calls_by_module.items())),
            "calls_by_model": dict(sorted(self._calls_by_model.items())),
        }

    def write(self, path: str | Path) -> dict:
        """Write the summary as JSON and return it."""
        summary = self.summary()
        write_json(path, summary)
        return summary
