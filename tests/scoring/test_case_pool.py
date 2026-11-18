"""case_pool runs N cases at once, keeps row order, writes every finished case, and resumes."""

import asyncio
import threading
import time

import pytest

from ib.scoring import case_pool
from ib.scoring.predictions import PredictionRow, read_predictions


def pred(cell_id):
    return PredictionRow(cell_id=cell_id, predicted_action="silent", answer_text=None, tool_calls=[],
                         confidence=0.0, metadata={})


ROWS = [{"cell_id": f"c{i}"} for i in range(8)]


class Probe:
    def __init__(self, fail=None):
        self.now = self.peak = 0
        self.calls, self.fail, self.lock = [], fail, threading.Lock()

    def __call__(self, row):
        with self.lock:
            self.calls.append(row["cell_id"])
            self.now += 1
            self.peak = max(self.peak, self.now)
        time.sleep(0.05 if row["cell_id"] != "c0" else 0.15)  # c0 finishes last
        with self.lock:
            self.now -= 1
        if row["cell_id"] == self.fail:
            raise RuntimeError("outage")
        return pred(row["cell_id"])


def run(tmp_path, probe, existing=None):
    out = tmp_path / "predictions.jsonl"
    return out, case_pool.predict_rows(ROWS, existing or {}, probe, output=out, progress=lambda *a: None)


def test_four_workers_run_four_at_once_and_return_rows_in_order(tmp_path, monkeypatch):
    monkeypatch.setenv(case_pool.CASE_WORKERS_ENV, "4")
    probe = Probe()
    out, got = run(tmp_path, probe)
    assert probe.peak == 4
    assert [p.cell_id for p in got] == [r["cell_id"] for r in ROWS], "row order, though c0 finished last"
    assert sorted(p.cell_id for p in read_predictions(out)) == sorted(r["cell_id"] for r in ROWS)


def test_one_worker_is_the_old_serial_loop(tmp_path, monkeypatch):
    monkeypatch.delenv(case_pool.CASE_WORKERS_ENV, raising=False)
    probe = Probe()
    run(tmp_path, probe)
    assert probe.peak == 1 and probe.calls == [r["cell_id"] for r in ROWS]


def test_existing_predictions_are_reused_not_rerun(tmp_path, monkeypatch):
    monkeypatch.setenv(case_pool.CASE_WORKERS_ENV, "3")
    probe = Probe()
    _, got = run(tmp_path, probe, existing={"c2": pred("c2"), "c5": pred("c5")})
    assert "c2" not in probe.calls and "c5" not in probe.calls and len(got) == 8


def test_an_outage_keeps_finished_cases_and_the_rerun_resumes(tmp_path, monkeypatch):
    monkeypatch.setenv(case_pool.CASE_WORKERS_ENV, "2")
    with pytest.raises(RuntimeError, match="outage"):
        run(tmp_path, Probe(fail="c3"))
    out = tmp_path / "predictions.jsonl"
    written = {p.cell_id for p in read_predictions(out)}
    assert written and "c3" not in written and "c7" not in written, "queued cases did not start"
    probe = Probe()
    _, got = run(tmp_path, probe, existing={p.cell_id: p for p in read_predictions(out)})
    assert set(probe.calls) == {r["cell_id"] for r in ROWS} - written and len(got) == 8


def test_async_rows_run_n_at_once_in_order(tmp_path, monkeypatch):
    monkeypatch.setenv(case_pool.CASE_WORKERS_ENV, "3")
    state = {"now": 0, "peak": 0}

    async def one(row):
        state["now"] += 1
        state["peak"] = max(state["peak"], state["now"])
        await asyncio.sleep(0.03 if row["cell_id"] != "c0" else 0.1)
        state["now"] -= 1
        return pred(row["cell_id"])

    out = tmp_path / "p.jsonl"
    got = asyncio.run(case_pool.apredict_rows(ROWS, {}, one, output=out, progress=lambda *a: None))
    assert state["peak"] == 3 and [p.cell_id for p in got] == [r["cell_id"] for r in ROWS]
    assert len(read_predictions(out)) == 8


def test_cases_that_finish_with_the_failing_one_are_still_written(tmp_path, monkeypatch):
    monkeypatch.setenv(case_pool.CASE_WORKERS_ENV, "2")
    gate = threading.Barrier(2)

    def together(row):  # c0 and c1 finish at the same instant; c1 is the outage
        gate.wait()
        if row["cell_id"] == "c1":
            raise RuntimeError("outage")
        return pred(row["cell_id"])

    out = tmp_path / "p.jsonl"
    with pytest.raises(RuntimeError):
        case_pool.predict_rows(ROWS[:2], {}, together, output=out, progress=lambda *a: None)
    assert [p.cell_id for p in read_predictions(out)] == ["c0"]


def test_zero_workers_is_refused(monkeypatch):
    monkeypatch.setenv(case_pool.CASE_WORKERS_ENV, "0")
    with pytest.raises(ValueError):
        case_pool.case_workers()
