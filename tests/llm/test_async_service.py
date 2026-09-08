"""Async LLM service: rate limiting, retries, cost tracking, and concurrency."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from ib.llm import (
    CostTracker,
    LlmConfig,
    LlmProviderError,
    LlmRequest,
    LlmService,
    LlmUsage,
    RateLimiter,
    complete_json,
    parse_json_object,
    price_lookup,
)
from ib.llm.cost import PRICING_TABLE
from ib.llm.qwen import (
    QWEN35_BASE_URL,
    QWEN35_MAX_OUTPUT_TOKENS,
    QWEN35_MODEL,
    qwen_request_options,
)
from ib.llm.retry import RetryPolicy, retry_async
from ib.llm.service import LlmRateLimitError, LlmTransientError


def _fake_client(create) -> SimpleNamespace:
    """Wrap an async ``create`` coroutine into an openai-shaped client double."""
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _sync_fake_client(create) -> SimpleNamespace:
    """Wrap a sync ``create`` function into an openai-shaped client double."""
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _gemini_fake_client(create) -> SimpleNamespace:
    """Wrap an async ``create`` coroutine into a Gemini interactions-shaped client."""
    return SimpleNamespace(interactions=SimpleNamespace(create=create))


def _response(text: str, *, prompt_tokens: int = 10, completion_tokens: int = 5, model="gpt-4.1-mini"):
    return SimpleNamespace(
        id="resp-1",
        model=model,
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
    )


def _req(client, *, model="gpt-4.1-mini", content="hi"):
    return LlmRequest(
        provider="openai",
        model=model,
        messages=[{"role": "user", "content": content}],
        client=client,
    )


# --- RateLimiter -----------------------------------------------------------


def test_rate_limiter_caps_concurrency() -> None:
    limiter = RateLimiter(max_concurrency=3, rpm=100_000.0)
    inflight = 0
    peak = 0

    async def worker() -> None:
        nonlocal inflight, peak
        async with limiter.slot():
            inflight += 1
            peak = max(peak, inflight)
            await asyncio.sleep(0.01)
            inflight -= 1

    asyncio.run(_gather(worker, 30))
    assert peak <= 3


def test_token_bucket_throttles_burst_beyond_rpm() -> None:
    # rpm=60 -> bucket holds 60 tokens, refills 1/sec. 61 immediate acquires must
    # force at least one ~1s wait beyond the burst.
    limiter = RateLimiter(max_concurrency=100, rpm=60.0)

    async def drain() -> float:
        loop = asyncio.get_event_loop()
        start = loop.time()
        for _ in range(62):
            async with limiter.slot():
                pass
        return loop.time() - start

    elapsed = asyncio.run(drain())
    assert elapsed >= 1.0  # the 61st+ token had to wait for a refill


async def _gather(worker, n) -> None:
    await asyncio.gather(*(worker() for _ in range(n)))


# --- CostTracker -----------------------------------------------------------


def test_cost_tracker_aggregates_by_module_and_model() -> None:
    tracker = CostTracker(budget_usd=None, default_cost_per_1m=10.0)

    async def run() -> dict:
        await tracker.record(LlmUsage("gpt-4.1-mini", "planner.director", 1_000_000, 0))
        await tracker.record(LlmUsage("gpt-4.1-mini", "planner.actor", 0, 1_000_000))
        return tracker.summary()

    summary = asyncio.run(run())
    # gpt-4.1-mini: input 0.4/1M, output 1.6/1M
    assert summary["total_cost_usd"] == pytest.approx(0.4 + 1.6)
    assert summary["total_input_tokens"] == 1_000_000
    assert summary["total_output_tokens"] == 1_000_000
    assert summary["call_count"] == 2
    assert summary["calls_by_module"] == {"planner.actor": 1, "planner.director": 1}
    assert summary["calls_by_model"] == {"gpt-4.1-mini": 2}
    assert summary["by_module"]["planner.director"] == pytest.approx(0.4)
    assert summary["by_module"]["planner.actor"] == pytest.approx(1.6)
    assert summary["by_model"]["gpt-4.1-mini"] == pytest.approx(2.0)


def test_pricing_fallback_for_unknown_model_is_silent(caplog) -> None:
    assert price_lookup("gpt-4.1-mini", 99.0) == PRICING_TABLE["gpt-4.1-mini"]
    with caplog.at_level("WARNING", logger="ib.llm.cost"):
        fallback = price_lookup("totally-unknown", 7.5)
    assert fallback == {"input_per_1m": 7.5, "output_per_1m": 7.5}
    assert not caplog.records


def test_budget_warns_once(caplog) -> None:
    tracker = CostTracker(budget_usd=1.0, warn_at_percent=80.0, default_cost_per_1m=1_000_000.0)

    async def run() -> None:
        # default cost is $1/token here; one token at input rate = $1.0 -> over 80%.
        with caplog.at_level("WARNING", logger="ib.llm.cost"):
            await tracker.record(LlmUsage("unpriced", "m", 1, 0))
            await tracker.record(LlmUsage("unpriced", "m", 1, 0))

    asyncio.run(run())
    budget_warnings = [r for r in caplog.records if "budget warning" in r.getMessage().lower()]
    assert len(budget_warnings) == 1  # warns once, never again, never raises


def test_cost_write(tmp_path) -> None:
    tracker = CostTracker(budget_usd=None)

    async def run() -> None:
        await tracker.record(LlmUsage("gpt-4.1-mini", "x", 100, 50))

    asyncio.run(run())
    out = tmp_path / "cost.json"
    written = tracker.write(out)
    on_disk = json.loads(out.read_text())
    assert on_disk == written
    assert on_disk["call_count"] == 1


# --- retry -----------------------------------------------------------------


def test_llm_config_defaults_to_100_second_request_timeout() -> None:
    assert LlmConfig().request_timeout_s == 100.0


def test_llm_config_defaults_to_two_second_retry_base_delay() -> None:
    assert LlmConfig().retry_base_delay_s == 2.0


def test_retry_recovers_after_transient_failures() -> None:
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise LlmTransientError("503")
        return "ok"

    policy = RetryPolicy(max_attempts=3, base_delay_s=0.001, max_delay_s=0.002)
    assert asyncio.run(retry_async(policy, flaky)) == "ok"
    assert calls["n"] == 3


def test_retry_does_not_retry_terminal_error() -> None:
    calls = {"n": 0}

    async def boom():
        calls["n"] += 1
        raise LlmProviderError("bad request")  # terminal, not a transient subclass

    policy = RetryPolicy(max_attempts=5, base_delay_s=0.001)
    with pytest.raises(LlmProviderError):
        asyncio.run(retry_async(policy, boom))
    assert calls["n"] == 1


def test_retry_exhausts_and_raises_last_transient() -> None:
    async def always():
        raise LlmRateLimitError("429", retry_after=0.0)

    policy = RetryPolicy(max_attempts=2, base_delay_s=0.001, max_delay_s=0.002)
    with pytest.raises(LlmRateLimitError):
        asyncio.run(retry_async(policy, always))


# --- LlmService ------------------------------------------------------------


def test_acomplete_captures_tokens_and_records_cost() -> None:
    async def create(**kwargs):
        assert kwargs["model"] == "gpt-4.1-mini"
        return _response(json.dumps({"ok": True}), prompt_tokens=120, completion_tokens=30)

    async def run() -> dict:
        svc = LlmService(LlmConfig())
        payload = await svc.acomplete_json(_req(_fake_client(create)), module="planner.director")
        await svc.aclose()
        return {"payload": payload, "summary": svc.cost.summary()}

    out = asyncio.run(run())
    assert out["payload"] == {"ok": True}
    s = out["summary"]
    assert s["call_count"] == 1
    assert s["total_input_tokens"] == 120
    assert s["total_output_tokens"] == 30
    assert s["by_module"]["planner.director"] > 0


def test_acomplete_routes_gemini_to_interactions_client() -> None:
    captured = {}

    async def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            id="gemini-call-1",
            output_text=json.dumps({"ok": True}),
            usage_metadata={"promptTokenCount": 9, "candidatesTokenCount": 4},
            model_version="gemini-3.1-pro-preview",
        )

    async def run() -> dict:
        svc = LlmService(LlmConfig())
        request = LlmRequest(
            provider="gemini",
            model="gemini-3.1-pro-preview",
            messages=[{"role": "user", "content": "return json"}],
            response_format={"type": "json_object"},
            client=_gemini_fake_client(create),
        )
        payload = await svc.acomplete_json(request, module="planner.logic_director")
        await svc.aclose()
        return {"payload": payload, "summary": svc.cost.summary()}

    out = asyncio.run(run())

    assert out["payload"] == {"ok": True}
    assert captured["model"] == "gemini-3.1-pro-preview"
    assert captured["input"] == "return json"
    assert captured["response_format"]["mime_type"] == "application/json"
    assert out["summary"]["calls_by_model"] == {"gemini-3.1-pro-preview": 1}


def test_qwen_request_forwards_base_model_and_thinking_options() -> None:
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return _response(json.dumps({"ok": True}), model=QWEN35_MODEL)

    request = LlmRequest(
        provider="qwen3_5",
        model=QWEN35_MODEL,
        messages=[{"role": "user", "content": "judge this"}],
        response_format={"type": "json_object"},
        max_tokens=QWEN35_MAX_OUTPUT_TOKENS,
        base_url=QWEN35_BASE_URL,
        extra_body=qwen_request_options(enable_thinking=False),
        client=_sync_fake_client(create),
    )

    assert complete_json(request) == {"ok": True}
    assert captured["model"] == QWEN35_MODEL
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["max_tokens"] == QWEN35_MAX_OUTPUT_TOKENS
    assert captured["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_parse_json_object_extracts_first_balanced_object() -> None:
    text = 'thinking...\n```json\n{"ok": true, "text": "brace } inside"}\n```\ntrailing'

    assert parse_json_object(text) == {"ok": True, "text": "brace } inside"}


def test_async_qwen_request_forwards_thinking_options() -> None:
    captured = {}

    async def create(**kwargs):
        captured.update(kwargs)
        return _response(json.dumps({"ok": True}), model=QWEN35_MODEL)

    request = LlmRequest(
        provider="qwen3_5_thinking",
        model=QWEN35_MODEL,
        messages=[{"role": "user", "content": "reason"}],
        response_format={"type": "json_object"},
        max_tokens=QWEN35_MAX_OUTPUT_TOKENS,
        base_url=QWEN35_BASE_URL,
        extra_body=qwen_request_options(enable_thinking=True),
        client=_fake_client(create),
    )

    async def run() -> dict:
        svc = LlmService(LlmConfig())
        try:
            return await svc.acomplete_json(request, module="planner.director")
        finally:
            await svc.aclose()

    assert asyncio.run(run()) == {"ok": True}
    assert captured["max_tokens"] == QWEN35_MAX_OUTPUT_TOKENS
    assert captured["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}


def test_acomplete_retries_then_succeeds_and_counts_one_call() -> None:
    state = {"n": 0}

    async def create(**kwargs):
        state["n"] += 1
        if state["n"] < 3:
            raise LlmTransientError("temporary 503")
        return _response("{}")

    async def run() -> dict:
        svc = LlmService(LlmConfig(retry_base_delay_s=0.001, retry_max_delay_s=0.002))
        await svc.acomplete(_req(_fake_client(create)))
        await svc.aclose()
        return svc.cost.summary()

    summary = asyncio.run(run())
    assert state["n"] == 3  # retried twice
    assert summary["call_count"] == 1  # cost recorded once, on success


def test_acomplete_json_rejects_non_json() -> None:
    async def create(**kwargs):
        return _response("not json")

    async def run() -> None:
        svc = LlmService(LlmConfig())
        try:
            await svc.acomplete_json(_req(_fake_client(create)))
        finally:
            await svc.aclose()

    with pytest.raises(LlmProviderError, match="invalid JSON"):
        asyncio.run(run())


def test_service_throttles_global_concurrency_across_calls() -> None:
    inflight = 0
    peak = 0

    async def create(**kwargs):
        nonlocal inflight, peak
        inflight += 1
        peak = max(peak, inflight)
        await asyncio.sleep(0.01)
        inflight -= 1
        return _response("{}")

    async def run() -> int:
        svc = LlmService(LlmConfig(max_concurrency=4, rpm=100_000.0))
        client = _fake_client(create)
        await asyncio.gather(*(svc.acomplete(_req(client)) for _ in range(20)))
        await svc.aclose()
        return svc.cost.summary()["call_count"]

    count = asyncio.run(run())
    assert peak <= 4
    assert count == 20
