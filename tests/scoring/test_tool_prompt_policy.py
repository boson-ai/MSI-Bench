"""Prompt-only policy tests for tool-call utterances and participant refs."""

from __future__ import annotations

from ib.llm import render_prompt
from ib.scoring.openai_understanding import _system_prompt_with_setup


def test_turn_based_tool_prompts_request_pre_action_answer_text() -> None:
    row = {
        "scene": "commerce_service",
        "logic_pattern": "constraint_attribution_under_interleaving",
        "logic_available_functions": [
            {
                "name": "process_return",
                "arguments": {
                    "requester_ref": "enum[S1,S2]",
                    "amount": "number",
                },
            }
        ],
    }

    for prompt in (
        _system_prompt_with_setup(row),
        _system_prompt_with_setup(row, native_tools=True),
    ):
        assert "brief user-facing pre-action utterance" in prompt
        assert "do not claim the action has already succeeded" in prompt
        assert "decisive attribution, final modification, or controlling constraint" in prompt
        assert "Do not recite every parameter, internal reasoning, function name, or JSON" in prompt
        assert "distinct human voices in chronological order of first appearance" in prompt
        assert "Do not guess how a spoken name is spelled" in prompt


def test_live_tool_prompt_uses_the_same_pre_action_boundary() -> None:
    prompt = render_prompt(
        "understanding/system.j2",
        {"kind": "live", "scene": "commerce service"},
    )

    assert "what you are about to do" in prompt
    assert "do not claim success before a function result" in prompt
    assert "decisive attribution, final modification, or controlling constraint" in prompt
    assert "speak function names or JSON" in prompt
