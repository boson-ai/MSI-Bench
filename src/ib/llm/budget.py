"""budget — recognise a provider saying the money is gone, and remember it for this process.

Calling spec:
    is_budget_refusal(status, text) -> bool      402, or a 400/403/429 whose text speaks of credit, a key limit or a budget
    budget_error(text) -> LlmBudgetError         remembers the refusal and returns the error to raise
    refuse_if_tripped() -> None                  raises LlmBudgetError once this process has seen a refusal
    tripped() -> str | None                      the first refusal's text
    reset() -> None                              for tests
Side effects: the process-wide latch only.

Once a key is spent every further call is refused too, so the latch stops them before they reach the network, and a batch
drains in seconds without marking dialogues failed by chance of timing. A new process tries again, which is how a run
started after a top-up works.
"""

from __future__ import annotations

import re
import threading

from ib.llm.types import LlmBudgetError

_WORDS = re.compile(r"credit|key limit|budget|quota|insufficient (?:funds|balance)|spend limit", re.I)
_LOCK = threading.Lock()
_first: str | None = None


def is_budget_refusal(status: int | None, text: str) -> bool:
    return status == 402 or (status in (400, 403, 429) and bool(_WORDS.search(text)))


def budget_error(text: str) -> LlmBudgetError:
    global _first
    with _LOCK:
        _first = _first or text[:300]
    return LlmBudgetError(text)


def refuse_if_tripped() -> None:
    if _first is not None:
        raise LlmBudgetError(_first)


def tripped() -> str | None:
    return _first


def reset() -> None:
    global _first
    with _LOCK:
        _first = None
