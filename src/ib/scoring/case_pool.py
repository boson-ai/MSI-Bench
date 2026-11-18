"""case_pool — run one target's per-case predictions N at a time, resumable.

Calling spec:
    case_workers() -> int
        IB_CASE_WORKERS (set by `ib eval --case-workers`), default 1; must be >= 1.
    PROVIDER_LOCK
        held by providers around their call_count / usage updates and their lazy client creation.
    predict_rows(rows, existing, predict_one, *, output, progress) -> list[PredictionRow]
    await apredict_rows(rows, existing, apredict_one, *, output, progress) -> list[PredictionRow]

Inputs:  rows, the manifest rows in order; existing, {cell_id: PredictionRow} already in `output`;
         predict_one(row) -> PredictionRow (a provider's own per-case call, including its per-case
         error rows; an exception aborts the run); progress(cell_id, done, total).
Outputs: one prediction per row, in row order; existing ones are reused, not re-run.
Side effects: appends each new prediction to `output` when it completes. Only the calling thread writes
         (sync) and only the event loop writes (async), so lines never interleave. On an exception the
         cases already running finish and are written, queued ones are not started, and the exception
         propagates, so a rerun resumes. With one worker, cases run one after another in row order, as before.
"""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Awaitable, Callable
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from ib.io import canonical_json
from ib.scoring.predictions import PredictionRow

CASE_WORKERS_ENV = "IB_CASE_WORKERS"
PROVIDER_LOCK = threading.Lock()  # guards provider state (counters, the lazy client) while cases run on threads
Row = dict[str, Any]
Progress = Callable[[str, int, int], None]


def case_workers() -> int:
    raw = os.environ.get(CASE_WORKERS_ENV, "1")
    workers = int(raw)
    if workers < 1:
        raise ValueError(f"{CASE_WORKERS_ENV} must be >= 1, got {raw!r}")
    return workers


def predict_rows(rows: list[Row], existing: dict[str, PredictionRow],
                 predict_one: Callable[[Row], PredictionRow], *, output: Path,
                 progress: Progress) -> list[PredictionRow]:
    done: dict[str, PredictionRow] = {}
    todo = _reuse(rows, existing, done, progress)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(case_workers()) as pool:
        pending = {pool.submit(predict_one, row): row["cell_id"] for row in todo}
        while pending:
            finished, _ = wait(pending, return_when=FIRST_EXCEPTION)
            failed = [future for future in finished if future.exception() is not None]
            if failed:  # stop queued cases, keep the ones already running: they are paid for
                running = [future for future in pending if not future.cancel()]
                finished = finished | wait(running).done
            for future in finished:
                cell_id = pending.pop(future)
                if future.exception() is None:
                    _record(handle, done, cell_id, future.result(), progress, len(rows))
            if failed:
                raise failed[0].exception()
    return [done[row["cell_id"]] for row in rows]


async def apredict_rows(rows: list[Row], existing: dict[str, PredictionRow],
                        apredict_one: Callable[[Row], Awaitable[PredictionRow]], *, output: Path,
                        progress: Progress) -> list[PredictionRow]:
    done: dict[str, PredictionRow] = {}
    todo = _reuse(rows, existing, done, progress)
    gate = asyncio.Semaphore(case_workers())

    async def one(row: Row) -> tuple[str, PredictionRow]:
        async with gate:
            return row["cell_id"], await apredict_one(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    tasks = [asyncio.create_task(one(row)) for row in todo]
    try:
        with output.open("a", encoding="utf-8") as handle:
            for next_done in asyncio.as_completed(tasks):
                cell_id, prediction = await next_done
                _record(handle, done, cell_id, prediction, progress, len(rows))
    finally:
        for task in tasks:
            task.cancel()
    return [done[row["cell_id"]] for row in rows]


def _reuse(rows: list[Row], existing: dict[str, PredictionRow], done: dict[str, PredictionRow],
           progress: Progress) -> list[Row]:
    todo = []
    for row in rows:
        cell_id = row["cell_id"]
        if cell_id in existing:
            done[cell_id] = existing[cell_id]
            progress(cell_id, len(done), len(rows))
        else:
            todo.append(row)
    return todo


def _record(handle: Any, done: dict[str, PredictionRow], cell_id: str, prediction: PredictionRow,
            progress: Progress, total: int) -> None:
    handle.write(f"{canonical_json(prediction)}\n")
    handle.flush()
    done[cell_id] = prediction
    progress(cell_id, len(done), total)
