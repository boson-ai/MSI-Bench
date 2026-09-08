"""cost — token pricing table + async cost tracker for LLM calls.

Calling spec:
    tracker = CostTracker(budget_usd=None, warn_at_percent=80.0, default_cost_per_1m=10.0)
    await tracker.record(LlmUsage(model="gpt-4.1-mini", module="planner.director",
                                  input_tokens=120, output_tokens=80))
    snapshot = tracker.summary()        # dict (totals + per-module + per-model)
    tracker.write(path)                 # write the snapshot as JSON

Prices are USD per 1M tokens (industry standard). Unknown models silently fall
back to ``default_cost_per_1m`` for both input and output. Call counts and token
usage remain authoritative even when pricing is unavailable. Budget is advisory:
it warns once at ``warn_at_percent`` of ``budget_usd`` and never raises.

Side effects: logs warnings via the stdlib logger; ``write`` writes a JSON file.
Dependencies: ib.io.write_json
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ib.io import write_json

logger = logging.getLogger("ib.llm.cost")

# Keyed by model id (the only live provider is OpenAI). USD per 1M tokens.
PRICING_TABLE: dict[str, dict[str, float]] = {
    "gpt-4o": {"input_per_1m": 2.5, "output_per_1m": 10.0},
    "gpt-4o-mini": {"input_per_1m": 0.15, "output_per_1m": 0.6},
    "gpt-4.1": {"input_per_1m": 2.0, "output_per_1m": 8.0},
    "gpt-4.1-mini": {"input_per_1m": 0.4, "output_per_1m": 1.6},
    "gpt-4.1-nano": {"input_per_1m": 0.1, "output_per_1m": 0.4},
    "o3": {"input_per_1m": 2.0, "output_per_1m": 8.0},
    "o4-mini": {"input_per_1m": 1.1, "output_per_1m": 4.4},
}


def price_lookup(model: str, default_cost_per_1m: float) -> dict[str, float]:
    """Return pricing for a model, silently using the configured fallback if unknown."""
    pricing = PRICING_TABLE.get(model)
    if pricing is not None:
        return dict(pricing)
    return {"input_per_1m": default_cost_per_1m, "output_per_1m": default_cost_per_1m}


@dataclass(frozen=True)
class LlmUsage:
    """Token usage for one completed LLM call (success or failure)."""

    model: str
    module: str
    input_tokens: int
    output_tokens: int


def _compute_cost(usage: LlmUsage, pricing: dict[str, float]) -> float:
    input_cost = (usage.input_tokens / 1_000_000) * pricing["input_per_1m"]
    output_cost = (usage.output_tokens / 1_000_000) * pricing["output_per_1m"]
    return input_cost + output_cost


class CostTracker:
    """Aggregates token cost across a run. Coroutine-safe via an asyncio.Lock."""

    def __init__(
        self,
        budget_usd: float | None = None,
        warn_at_percent: float = 80.0,
        default_cost_per_1m: float = 10.0,
    ) -> None:
        self._budget = budget_usd
        self._warn_at_percent = warn_at_percent
        self._default_cost_per_1m = default_cost_per_1m
        self._lock = asyncio.Lock()
        self._warned = False

        self._total_cost = 0.0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._call_count = 0
        self._by_module: dict[str, float] = defaultdict(float)
        self._by_model: dict[str, float] = defaultdict(float)
        self._calls_by_module: dict[str, int] = defaultdict(int)
        self._calls_by_model: dict[str, int] = defaultdict(int)

    async def record(self, usage: LlmUsage) -> None:
        """Record one call's cost. Advisory budget warning, never raises."""
        pricing = price_lookup(usage.model, self._default_cost_per_1m)
        cost = _compute_cost(usage, pricing)
        async with self._lock:
            self._total_cost += cost
            self._total_input_tokens += usage.input_tokens
            self._total_output_tokens += usage.output_tokens
            self._call_count += 1
            self._by_module[usage.module] += cost
            self._by_model[usage.model] += cost
            self._calls_by_module[usage.module] += 1
            self._calls_by_model[usage.model] += 1
            self._maybe_warn()

    def _maybe_warn(self) -> None:
        if self._budget is None or self._warned:
            return
        if self._total_cost >= self._budget * (self._warn_at_percent / 100.0):
            logger.warning(
                "LLM budget warning: $%.4f of $%.2f used (%.1f%%)",
                self._total_cost,
                self._budget,
                (self._total_cost / self._budget) * 100.0,
            )
            self._warned = True

    def summary(self) -> dict:
        """Return a plain-dict snapshot of current aggregation."""
        return {
            "schema_version": "ib.llm_cost.v1",
            "total_cost_usd": round(self._total_cost, 6),
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "call_count": self._call_count,
            "calls_by_module": dict(sorted(self._calls_by_module.items())),
            "calls_by_model": dict(sorted(self._calls_by_model.items())),
            "by_module": {k: round(v, 6) for k, v in sorted(self._by_module.items())},
            "by_model": {k: round(v, 6) for k, v in sorted(self._by_model.items())},
        }

    def write(self, path: str | Path) -> dict:
        """Write the summary as JSON and return it."""
        summary = self.summary()
        write_json(path, summary)
        return summary
