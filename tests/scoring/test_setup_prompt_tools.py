"""The eval setup prompt lists a case's tools as the training system message does.

prompt_json targets read Anthropic tool definitions (name, description, input_schema)
in the setup prompt; native targets get tools only through the provider API.
"""

from __future__ import annotations

import json

import pytest

from ib.manifest.tool_definitions import available_functions_text, tool_definitions, tools_text
from ib.scoring.multiturn_audio import setup_prompt_for
from ib.scoring.native_tool_schema import build_native_tool_bundle

V2_FUNCTION = {
    "name": "book_table",
    "description": "Reserve a restaurant table.",
    "arguments": {"party_size": "integer 1..12", "time": "string HH:MM 24h"},
    "argument_descriptions": {"party_size": "How many diners.", "time": "Arrival time."},
}
V1_FUNCTION = {
    "name": "prepare_wheelchair_zone",
    "arguments": {"door_area": "enum[front_door,rear_door]", "fold_seats_up": "boolean"},
}
V1_NO_ARGUMENTS = {"name": "get_queue", "arguments": {}}


def _row(functions: list[dict]) -> dict:
    return {"logic_pattern": "distributed_parameters", "logic_available_functions": functions}


def _listed_tools(prompt: str) -> list[dict]:
    """Parse the JSON array that follows "Available functions:" in the prompt."""
    tail = prompt.split("Available functions:\n", 1)[1]
    tools, _ = json.JSONDecoder().raw_decode(tail)
    return tools


def test_prompt_json_setup_lists_descriptions_and_input_schema() -> None:
    prompt = setup_prompt_for(_row([V2_FUNCTION]))

    [tool] = _listed_tools(prompt)
    assert tool["name"] == "book_table"
    assert tool["description"] == "Reserve a restaurant table."
    schema = tool["input_schema"]
    assert schema["type"] == "object"
    assert schema["required"] == ["party_size", "time"]
    assert schema["properties"]["party_size"]["description"] == "How many diners."
    assert schema["properties"]["party_size"]["type"] == "integer"
    assert schema["properties"]["time"]["description"].startswith("Arrival time.")
    assert "input_schema properties as keys" in prompt


def test_prompt_json_setup_is_the_training_system_message_listing() -> None:
    functions = [V2_FUNCTION, V1_FUNCTION]

    assert available_functions_text(functions) in setup_prompt_for(_row(functions))


def test_v1_function_without_descriptions_renders_with_empty_description() -> None:
    tools = _listed_tools(setup_prompt_for(_row([V1_FUNCTION, V1_NO_ARGUMENTS])))

    assert tools[0]["description"] == ""
    assert tools[0]["input_schema"]["properties"]["door_area"] == {
        "type": "string",
        "enum": ["front_door", "rear_door"],
    }
    assert tools[0]["input_schema"]["required"] == ["door_area", "fold_seats_up"]
    assert tools[1] == {
        "name": "get_queue",
        "description": "",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    }


def test_row_without_functions_lists_an_empty_array() -> None:
    assert _listed_tools(setup_prompt_for(_row([]))) == []


def test_native_setup_has_no_listing_and_bundle_keeps_parameters() -> None:
    prompt = setup_prompt_for(_row([V2_FUNCTION]), tool_protocol="native")
    assert "Available functions:" not in prompt
    assert "native tools interface" in prompt

    [declaration] = build_native_tool_bundle([V2_FUNCTION]).declarations
    assert declaration["name"] == "book_table"
    assert declaration["description"] == "Reserve a restaurant table."
    assert declaration["parameters"]["additionalProperties"] is False
    assert declaration["parameters"]["required"] == ["party_size", "time"]
    assert declaration["parameters"]["properties"]["party_size"]["description"] == "How many diners."
    assert "input_schema" not in declaration


def test_gemini_live_instructions_list_each_tool_once() -> None:
    from ib.scoring.gemini_fullduplex import _live_instructions

    prompt = _live_instructions(_row([V2_FUNCTION]), [])

    assert prompt.count("book_table") == 1
    assert prompt.count("Available functions:") == 1


def test_openai_live_instructions_list_each_tool_once() -> None:
    from ib.scoring.openai_fullduplex import _live_instructions

    prompt = _live_instructions(_row([V2_FUNCTION]), [])

    assert prompt.count("book_table") == 1
    assert prompt.count("Available functions:") == 1


HOLDER_TOOL = {
    "name": "set_prep_reminder",
    "description": "Set a prep reminder.",
    "input_schema": {
        "type": "object",
        "properties": {
            "holder": {
                "type": "array",
                "items": {"type": "string", "pattern": "^(?:<\\|S[1-9]\\d*\\|>|all)$"},
                "description": 'Lists the voice labels of the people the value binds, or ["all"] for everyone present.',
            },
            "prep_time": {"type": "string", "description": "HH:MM."},
        },
        "required": ["holder", "prep_time"],
    },
}
LEGACY_RULE = "- No function argument refers to a participant; never put a speaker's name or label in an argument."
VOICE_RULE = "- Never put a speaker's name in a function argument; an argument that takes voice labels"


def _v2_row(tools: list[dict]) -> dict:
    return {"cell_id": "msi2-x", "logic_pattern": "distributed_parameters", "logic_tools": tools,
            "audio_history": [{"role": "user", "audio": "blocks/00.wav", "block_index": 0}]}


def test_logic_tools_are_listed_verbatim() -> None:
    prompt = setup_prompt_for(_v2_row([HOLDER_TOOL]))

    assert _listed_tools(prompt) == [HOLDER_TOOL]
    assert tools_text([HOLDER_TOOL]) in prompt


def test_logic_tools_render_the_same_prompt_as_their_legacy_descriptors() -> None:
    functions = [V2_FUNCTION, V1_FUNCTION]

    legacy = setup_prompt_for(_row(functions))
    v2 = {k: v for k, v in _v2_row(tool_definitions(functions)).items() if k != "audio_history"}

    assert setup_prompt_for(v2) == legacy
    assert LEGACY_RULE in legacy


def test_logic_tools_win_over_legacy_descriptors() -> None:
    row = {**_row([V2_FUNCTION]), "logic_tools": [HOLDER_TOOL]}

    assert [tool["name"] for tool in _listed_tools(setup_prompt_for(row))] == ["set_prep_reminder"]


def test_voice_label_argument_replaces_the_no_participant_rule() -> None:
    prompt = setup_prompt_for(_v2_row([HOLDER_TOOL]))

    assert VOICE_RULE in prompt
    assert LEGACY_RULE not in prompt


def test_tools_without_voice_labels_keep_the_no_participant_rule() -> None:
    plain = {**HOLDER_TOOL, "input_schema": {"type": "object", "properties": {"prep_time": {"type": "string"}}}}

    prompt = setup_prompt_for(_v2_row([plain]))

    assert LEGACY_RULE in prompt
    assert VOICE_RULE not in prompt


def test_native_protocol_declares_logic_tools() -> None:
    from ib.scoring.native_tool_schema import native_bundle_for_row

    prompt = setup_prompt_for(_v2_row([HOLDER_TOOL]), tool_protocol="native")
    bundle = native_bundle_for_row(_v2_row([HOLDER_TOOL]))

    assert "native tools interface" in prompt and "Available functions:" not in prompt
    assert [d["name"] for d in bundle.declarations] == [HOLDER_TOOL["name"]]
    assert bundle.declarations[0]["parameters"]["properties"] == HOLDER_TOOL["input_schema"]["properties"]
    assert bundle.declarations[0]["parameters"]["additionalProperties"] is False
    assert bundle.name_map == {HOLDER_TOOL["name"]: HOLDER_TOOL["name"]}


def test_timeline_rows_get_the_multi_line_rule_and_final_note() -> None:
    from ib.scoring.multiturn_audio import FINAL_TURN_NOTE, final_turn_note

    row = _v2_row([HOLDER_TOOL])
    assert "Decide about the most recent line" not in setup_prompt_for(row)  # the rule is opt-in (IB_MULTI_LINE_RULE=1)
    assert final_turn_note(row) == FINAL_TURN_NOTE
    assert final_turn_note(row, tool_protocol="native") is None
    legacy = {k: v for k, v in row.items() if k != "audio_history"}
    legacy["logic_available_functions"] = []
    legacy.pop("logic_tools", None)
    assert "Decide about the most recent line" not in setup_prompt_for(legacy)
    assert final_turn_note(legacy) is None
