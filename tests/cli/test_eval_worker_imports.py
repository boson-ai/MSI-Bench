"""Eval target worker import-safety tests."""

from __future__ import annotations

import builtins
from typing import Any

from ib.cli.eval import _preload_target_worker_imports
from ib.eval_targets import EvalTarget


def test_preload_target_worker_imports_imports_openai_modules_for_parallel_generation(
    monkeypatch,
) -> None:
    targets = [
        EvalTarget(mode="turn_based", provider="openai", model="gpt-audio-1.5"),
        EvalTarget(mode="fullduplex", provider="openai_realtime", model="gpt-realtime-1.5"),
    ]
    imports: list[str] = []

    def fake_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        del globals, locals, fromlist, level
        imports.append(name)
        return object()

    monkeypatch.setattr(builtins, "__import__", fake_import)

    _preload_target_worker_imports(targets, worker_count=2)

    assert imports == ["openai.resources.chat", "openai.resources.realtime"]


def test_preload_target_worker_imports_skips_serial_or_existing_predictions(
    monkeypatch,
) -> None:
    def fail_import(*args: Any, **kwargs: Any) -> object:
        raise AssertionError("unexpected import")

    monkeypatch.setattr(builtins, "__import__", fail_import)

    _preload_target_worker_imports(
        [EvalTarget(mode="turn_based", provider="openai", model="gpt-audio-1.5")],
        worker_count=1,
    )
    _preload_target_worker_imports(
        [EvalTarget(mode="turn_based", provider="openai", predictions="predictions.jsonl")],
        worker_count=2,
    )
