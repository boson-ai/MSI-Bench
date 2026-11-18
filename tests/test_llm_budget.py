"""A spent key: the real OpenAI SDK and the raw HTTP path against a local stub that answers like OpenRouter."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import openai
import pytest

from ib.llm import budget
from ib.llm.service import complete
from ib.llm.types import LlmBudgetError, LlmProviderError, LlmRateLimitError, LlmRequest

OK = {"id": "x", "object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
      "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


@pytest.fixture(autouse=True)
def fresh_latch():
    budget.reset()
    yield
    budget.reset()


@pytest.fixture
def stub():
    state = {"status": 402, "message": "Insufficient credits", "calls": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length", 0)))
            state["calls"] += 1
            ok = state["status"] == 200
            body = json.dumps(OK if ok else {"error": {"code": state["status"], "message": state["message"]}}).encode()
            self.send_response(state["status"])
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{server.server_port}/v1"
    yield state
    server.shutdown()


def via_sdk(url):
    client = openai.OpenAI(base_url=url, api_key="k", max_retries=0)
    return LlmRequest(provider="openai", model="m", messages=[{"role": "user", "content": "x"}], client=client)


def via_http(url):
    return LlmRequest(provider="openai", model="m", messages=[{"role": "user", "content": "x"}], base_url=url, api_key="k")


@pytest.mark.parametrize("route", [via_sdk, via_http])
@pytest.mark.parametrize("status, message", [
    (402, "Insufficient credits"),
    (403, "Key limit exceeded (weekly limit). Manage it using https://openrouter.ai/settings/keys"),
    (400, "Your credit balance is too low to access the API"),
    (429, "Budget has been exceeded"),
])
def test_a_spent_key_is_one_call_and_every_later_call_stays_off_the_network(stub, route, status, message):
    stub.update(status=status, message=message)
    with pytest.raises(LlmBudgetError):
        complete(route(stub["url"]))
    assert stub["calls"] == 1, "no retry"
    assert message in budget.tripped()
    stub["status"] = 200
    with pytest.raises(LlmBudgetError):
        complete(route(stub["url"]))
    assert stub["calls"] == 1


@pytest.mark.parametrize("route", [via_sdk, via_http])
def test_a_refusal_that_is_not_about_money_does_not_stop_the_process(stub, route):
    stub.update(status=403, message="This model is not available in your region")
    with pytest.raises(LlmProviderError) as caught:
        complete(route(stub["url"]))
    assert not isinstance(caught.value, LlmBudgetError) and budget.tripped() is None
    stub["status"] = 200
    assert complete(route(stub["url"])).text == "hi"


def test_an_ordinary_rate_limit_stays_a_rate_limit(stub):
    from ib.llm.service import OpenAiLlmProvider
    stub.update(status=429, message="Rate limit exceeded: too many requests")
    with pytest.raises(LlmRateLimitError):
        OpenAiLlmProvider().complete(via_sdk(stub["url"]))
    assert budget.tripped() is None
