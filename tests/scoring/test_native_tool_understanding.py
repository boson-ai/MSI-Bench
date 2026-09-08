"""Provider-native tool transport for turn-based understanding evals."""

from __future__ import annotations

import json
import wave
from io import BytesIO
from types import SimpleNamespace

from ib.io import read_jsonl
from ib.predictions import PredictionRow
from ib.scoring.gemini_understanding import run_gemini_understanding_predictions
from ib.scoring.openai_understanding import run_openai_understanding_predictions


def _wav_bytes() -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24_000)
        handle.writeframes((b"\x00\x00") * 16)
    return buffer.getvalue()


def _native_run(tmp_path) -> tuple[dict, object]:
    run_root = tmp_path / "run"
    (run_root / "manifest").mkdir(parents=True)
    (run_root / "audio").mkdir()
    (run_root / "audio" / "final.wav").write_bytes(_wav_bytes())
    row = {
        "cell_id": "en-distributed-1",
        "scene": "commerce_service",
        "logic_pattern": "distributed_parameters",
        "expected_action": "respond",
        "mixed_testcase_audio_paths": ["audio/final.wav"],
        "user_turn_1_audio": "audio/final.wav",
        "logic_available_functions": [
            {
                "name": "update_pickup_order_constraints",
                "arguments": {
                    "pickup_name": "string_or_null",
                    "temperature": "enum[ambient,refrigerated]",
                    "rush": "boolean",
                },
            }
        ],
    }
    manifest = run_root / "manifest" / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return row, manifest


class _OpenAiCompletions:
    def __init__(self, response) -> None:
        self.response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class _OpenAiClient:
    def __init__(self, response) -> None:
        self.chat = SimpleNamespace(completions=_OpenAiCompletions(response))


class _GeminiModels:
    def __init__(self, response) -> None:
        self.response = response
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class _GeminiClient:
    def __init__(self, response) -> None:
        self.models = _GeminiModels(response)


def test_openai_native_tools_override_textual_tool_calls(tmp_path) -> None:
    _, manifest = _native_run(tmp_path)
    model_message = SimpleNamespace(
        content=json.dumps(
            {
                "predicted_action": "silent",
                "answer_text": "I will update the settled pickup constraints.",
                "tool_calls": [{"name": "invented_text_tool", "arguments": {}}],
            }
        ),
        tool_calls=[
            SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(
                    name="update_pickup_order_constraints",
                    arguments=json.dumps(
                        {
                            "pickup_name": "Avery",
                            "temperature": "refrigerated",
                            "rush": True,
                        }
                    ),
                ),
            )
        ],
    )
    response = SimpleNamespace(
        id="resp_1",
        choices=[SimpleNamespace(message=model_message)],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    client = _OpenAiClient(response)
    output = tmp_path / "openai-native.jsonl"

    report = run_openai_understanding_predictions(
        manifest,
        output,
        model="gpt-audio-1.5",
        client=client,
        tool_protocol="native",
    )

    prediction = read_jsonl(output, PredictionRow)[0]
    request = client.chat.completions.calls[0]
    system_prompt = request["messages"][0]["content"]
    assert report["tool_protocol"] == "native"
    assert request["tool_choice"] == "auto"
    assert request["tools"][0]["function"]["name"] == "update_pickup_order_constraints"
    assert request["tools"][0]["function"]["parameters"]["additionalProperties"] is False
    assert "update_pickup_order_constraints" not in system_prompt
    assert "Available functions for this conversation" not in system_prompt
    assert prediction.predicted_action.value == "respond"
    assert prediction.answer_text == "I will update the settled pickup constraints."
    assert [call.model_dump(mode="json") for call in prediction.tool_calls] == [
        {
            "name": "update_pickup_order_constraints",
            "arguments": {
                "pickup_name": "Avery",
                "temperature": "refrigerated",
                "rush": True,
            },
        }
    ]
    assert prediction.metadata["tool_protocol"] == "native"


def test_gemini_native_function_only_response_is_a_respond_action(tmp_path) -> None:
    _, manifest = _native_run(tmp_path)
    response = SimpleNamespace(
        response_id="gemini_1",
        text=None,
        function_calls=[
            SimpleNamespace(
                name="update_pickup_order_constraints",
                args={"pickup_name": "Avery", "temperature": "ambient", "rush": False},
            )
        ],
        candidates=[],
        usage_metadata=SimpleNamespace(
            prompt_token_count=11,
            candidates_token_count=6,
            total_token_count=17,
        ),
    )
    client = _GeminiClient(response)
    output = tmp_path / "gemini-native.jsonl"

    report = run_gemini_understanding_predictions(
        manifest,
        output,
        model="gemini-3.5-flash",
        client=client,
        tool_protocol="native",
    )

    prediction = read_jsonl(output, PredictionRow)[0]
    config = client.models.calls[0]["config"]
    declaration = config.tools[0].function_declarations[0]
    assert report["tool_protocol"] == "native"
    assert declaration.name == "update_pickup_order_constraints"
    assert declaration.parameters_json_schema["properties"]["rush"] == {"type": "boolean"}
    assert config.tool_config.function_calling_config.mode.value == "AUTO"
    assert config.automatic_function_calling.disable is True
    assert config.response_mime_type is None
    assert "update_pickup_order_constraints" not in config.system_instruction
    assert prediction.predicted_action.value == "respond"
    assert prediction.answer_text is None
    assert prediction.tool_calls[0].arguments == {
        "pickup_name": "Avery",
        "temperature": "ambient",
        "rush": False,
    }
    assert prediction.metadata["native_tool_count"] == 1


def test_gemini_native_thought_summary_is_not_answer_text(tmp_path) -> None:
    _, manifest = _native_run(tmp_path)
    function_call = SimpleNamespace(
        name="update_pickup_order_constraints",
        args={"pickup_name": "Avery", "temperature": "ambient", "rush": False},
    )
    response = SimpleNamespace(
        response_id="gemini_thought_and_tool",
        text=None,
        function_calls=[function_call],
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            text="I should verify every tool argument before acting.",
                            thought=True,
                            function_call=None,
                        ),
                        SimpleNamespace(
                            text=None,
                            thought=None,
                            function_call=function_call,
                        ),
                    ]
                )
            )
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=11,
            candidates_token_count=6,
            total_token_count=17,
        ),
    )
    output = tmp_path / "gemini-thought-and-tool.jsonl"

    run_gemini_understanding_predictions(
        manifest,
        output,
        model="gemini-3.5-flash",
        client=_GeminiClient(response),
        tool_protocol="native",
        thinking_mode="enabled",
    )

    prediction = read_jsonl(output, PredictionRow)[0]
    assert prediction.predicted_action.value == "respond"
    assert prediction.answer_text is None
    assert prediction.tool_calls[0].arguments == {
        "pickup_name": "Avery",
        "temperature": "ambient",
        "rush": False,
    }


def test_gemini_native_keeps_final_text_after_thought_summary(tmp_path) -> None:
    _, manifest = _native_run(tmp_path)
    function_call = SimpleNamespace(
        name="update_pickup_order_constraints",
        args={"pickup_name": "Avery", "temperature": "ambient", "rush": False},
    )
    final_text = json.dumps(
        {
            "predicted_action": "respond",
            "answer_text": "I will update the settled pickup constraints.",
        }
    )
    response = SimpleNamespace(
        response_id="gemini_thought_answer_and_tool",
        text=None,
        function_calls=[function_call],
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            text="I should verify every tool argument before acting.",
                            thought=True,
                            function_call=None,
                        ),
                        SimpleNamespace(
                            text=final_text,
                            thought=False,
                            function_call=None,
                        ),
                        SimpleNamespace(
                            text=None,
                            thought=None,
                            function_call=function_call,
                        ),
                    ]
                )
            )
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=11,
            candidates_token_count=6,
            total_token_count=17,
        ),
    )
    output = tmp_path / "gemini-thought-answer-and-tool.jsonl"

    run_gemini_understanding_predictions(
        manifest,
        output,
        model="gemini-3.5-flash",
        client=_GeminiClient(response),
        tool_protocol="native",
        thinking_mode="enabled",
    )

    prediction = read_jsonl(output, PredictionRow)[0]
    raw_response = json.loads(
        (tmp_path / "raw_responses.jsonl").read_text(encoding="utf-8")
    )["response"]
    raw_thought = raw_response["candidates"][0]["content"]["parts"][0]
    assert raw_thought == {
        "text": "I should verify every tool argument before acting.",
        "thought": True,
        "function_call": None,
    }
    assert prediction.answer_text == "I will update the settled pickup constraints."
    assert len(prediction.tool_calls) == 1


def test_native_text_without_function_call_cannot_smuggle_tool_calls(tmp_path) -> None:
    _, manifest = _native_run(tmp_path)
    response = SimpleNamespace(
        id="resp_1",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(
                        {
                            "predicted_action": "respond",
                            "answer_text": "I need confirmation before acting.",
                            "tool_calls": [
                                {"name": "update_pickup_order_constraints", "arguments": {}}
                            ],
                        }
                    ),
                    tool_calls=[],
                )
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    client = _OpenAiClient(response)
    output = tmp_path / "no-native-call.jsonl"

    run_openai_understanding_predictions(
        manifest,
        output,
        model="gpt-audio-1.5",
        client=client,
        tool_protocol="native",
    )

    prediction = read_jsonl(output, PredictionRow)[0]
    assert prediction.answer_text == "I need confirmation before acting."
    assert prediction.tool_calls == []


def test_gemini_native_explicit_silent_decision_abstains(tmp_path) -> None:
    _, manifest = _native_run(tmp_path)
    silent_part = SimpleNamespace(
        text=json.dumps({"predicted_action": "silent", "answer_text": None}),
        function_call=None,
    )
    response = SimpleNamespace(
        response_id="gemini_silent_decision",
        text=None,
        function_calls=[],
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[silent_part]))],
        usage_metadata=SimpleNamespace(
            prompt_token_count=5, candidates_token_count=2, total_token_count=7
        ),
    )
    client = _GeminiClient(response)
    output = tmp_path / "gemini-decision-silent.jsonl"

    run_gemini_understanding_predictions(
        manifest, output, model="gemini-3.5-flash", client=client, tool_protocol="native"
    )

    prediction = read_jsonl(output, PredictionRow)[0]
    assert prediction.predicted_action.value == "silent"
    assert prediction.answer_text is None
    assert prediction.tool_calls == []


def test_native_json_can_explicitly_choose_silence(tmp_path) -> None:
    _, manifest = _native_run(tmp_path)
    response = SimpleNamespace(
        id="resp_1",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(
                        {"predicted_action": "silent", "answer_text": None}
                    ),
                    tool_calls=[],
                )
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    client = _OpenAiClient(response)
    output = tmp_path / "native-silent.jsonl"

    run_openai_understanding_predictions(
        manifest,
        output,
        model="gpt-audio-1.5",
        client=client,
        tool_protocol="native",
    )

    prediction = read_jsonl(output, PredictionRow)[0]
    assert prediction.predicted_action.value == "silent"
    assert prediction.answer_text is None
    assert prediction.tool_calls == []
