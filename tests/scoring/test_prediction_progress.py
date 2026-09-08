"""Prediction generation progress formatting tests."""

from __future__ import annotations

import io

from ib.scoring import progress as progress_module
from ib.scoring.progress import format_eval_case_progress, print_eval_case_progress


def test_format_eval_case_progress_keeps_case_prefix_and_bar() -> None:
    line = format_eval_case_progress(
        "openai_realtime", "gpt-realtime-1.5", "commerce-c1", 2, 4
    )

    assert line == (
        "ib eval: openai_realtime:gpt-realtime-1.5 "
        "case 2/4 [##########----------]  50% commerce-c1"
    )


def test_print_eval_case_progress_repaints_tty_block(monkeypatch) -> None:
    class TtyStream(io.StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.setenv("IB_EVAL_PROGRESS", "live")
    progress_module._LIVE_STATE.clear()
    progress_module._LIVE_RENDERED_LINES = 0
    stream = TtyStream()

    print_eval_case_progress("gemini", "gemini-3.5-flash", "c1", 1, 20, stream=stream)
    print_eval_case_progress("gemini", "gemini-3.5-flash", "c2", 2, 20, stream=stream)

    output = stream.getvalue()
    assert "\x1b[1A" in output
    assert "case 1/20" in output
    assert "case 2/20" in output


def test_print_eval_case_progress_uses_lines_for_non_tty(capsys) -> None:
    progress_module._LIVE_STATE.clear()
    progress_module._LIVE_RENDERED_LINES = 0

    print_eval_case_progress("openai", "gpt-audio-1.5", "c1", 1, 1)

    assert "\x1b[" not in capsys.readouterr().err
