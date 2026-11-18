"""Sync complete() honors the configured retry attempts (default 3) and waits out 429s first."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ib.llm import service
from ib.llm.service import complete, set_sync_retry_max_attempts
from ib.llm.rate_limit_wait import JITTER_S, RATE_LIMIT_BUDGET_S, rate_limit_wait
from ib.llm.types import LlmProviderError, LlmRateLimitError, LlmRequest, LlmResponse, LlmTransientError


class _Provider:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        outcome = self.outcomes.pop(0) if self.outcomes else LlmTransientError("timeout")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(service, "_sleep", lambda seconds: None)
    monkeypatch.setattr(service, "_sync_retry_max_attempts", service._sync_retry_max_attempts)


def _call(monkeypatch, provider):
    monkeypatch.setitem(service.PROVIDER_REGISTRY, "fake", provider)
    return complete(LlmRequest(provider="fake", model="m", messages=[]))


def test_default_is_three_calls(monkeypatch):
    provider = _Provider([])
    with pytest.raises(LlmTransientError):
        _call(monkeypatch, provider)
    assert provider.calls == 3


def test_setting_of_one_calls_once(monkeypatch):
    set_sync_retry_max_attempts(1)
    provider = _Provider([])
    with pytest.raises(LlmTransientError):
        _call(monkeypatch, provider)
    assert provider.calls == 1


def test_transient_then_success(monkeypatch):
    provider = _Provider([LlmTransientError("x"), LlmResponse(text="ok")])
    assert _call(monkeypatch, provider).text == "ok"
    assert provider.calls == 2


def _recording_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr(service, "_sleep", slept.append)
    return slept


def test_429s_are_waited_out_without_using_attempts_and_the_call_succeeds_once_they_stop(monkeypatch):
    slept = _recording_sleep(monkeypatch)
    provider = _Provider([*(LlmRateLimitError("Rate limit exceeded") for _ in range(5)), LlmResponse(text="ok")])
    assert _call(monkeypatch, provider).text == "ok"
    assert provider.calls == 6 and len(slept) == 5  # more 429s than the 3 attempts, still no failure
    assert all(2.5 <= s <= 90 for s in slept) and sum(slept) <= RATE_LIMIT_BUDGET_S


def test_a_429_waits_at_least_its_retry_after(monkeypatch):
    slept = _recording_sleep(monkeypatch)
    provider = _Provider([LlmRateLimitError("429", retry_after=42.0), LlmResponse(text="ok")])
    assert _call(monkeypatch, provider).text == "ok"
    assert 42.0 <= slept[0] <= 42.0 + JITTER_S


def test_endless_429s_stop_after_the_wait_budget_and_then_the_transient_attempts(monkeypatch):
    slept = _recording_sleep(monkeypatch)
    provider = _Provider([LlmRateLimitError("Rate limit exceeded") for _ in range(100)])
    with pytest.raises(LlmRateLimitError):
        _call(monkeypatch, provider)
    waits = sum(slept[: len(slept) - 2])  # the last two sleeps are the ordinary transient backoff
    assert waits == pytest.approx(RATE_LIMIT_BUDGET_S) and provider.calls == len(slept) + 1


def test_a_terminal_error_is_not_retried(monkeypatch):
    slept = _recording_sleep(monkeypatch)
    provider = _Provider([LlmProviderError("401 bad key")])
    with pytest.raises(LlmProviderError):
        _call(monkeypatch, provider)
    assert provider.calls == 1 and slept == []


def test_the_proxy_reset_time_sets_the_wait():
    error = LlmRateLimitError("Rate limit exceeded for user: x. Limit resets at: 2026-10-29 20:14:43 UTC")
    now = datetime(2026, 10, 29, 20, 14, 13, tzinfo=timezone.utc)
    assert 31.0 <= rate_limit_wait(error, 0, 0.0, now=now) <= 30.0 + JITTER_S
    assert rate_limit_wait(error, 0, 290.0, now=now) == pytest.approx(10.0)  # cut to the budget left
    assert rate_limit_wait(error, 0, RATE_LIMIT_BUDGET_S, now=now) is None
    assert rate_limit_wait(LlmTransientError("timeout"), 0, 0.0) is None
