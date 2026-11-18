"""tool_definitions — Anthropic-format tool definitions from harness function descriptors; pure.

Calling spec:
    definition = tool_definition(function)
        function: {"name", "description", "arguments": {arg: descriptor},
                   "argument_descriptions": {arg: sentence}}
        -> {"name": ..., "description": ...,
            "input_schema": {"type": "object",
                             "properties": {arg: {"type": ..., "description": ..., "enum"?, "minimum"?,
                                                  "maximum"?, "pattern"?}},
                             "required": [...]}}
    definitions = tool_definitions(functions) -> [definition, ...] in input order
    text = tools_text(definitions)
        -> "Available functions:\n" + json.dumps(definitions, ensure_ascii=False, indent=2)
    text = available_functions_text(functions) -> tools_text(tool_definitions(functions))

Each property is ib.scoring.native_tool_schema.argument_schema of the argument's
descriptor and description (descriptor_json_schema plus the description), and
`required` lists the arguments whose schema does not allow null, in argument order
(the same rule as build_native_tool_bundle). No return values are described.
A function without a "description" (v1 manifests) gets description "".

The training sample's `tools` field and the "Available functions:" JSON of its
system message (ib.manifest.system_message) are tool_definitions of the contract's
available_functions. The eval setup prompt of the prompt_json tool protocol
(ib.scoring.multiturn_audio.setup_prompt_for) carries the same available_functions_text,
so training and eval show the model identical tool listings.

Side effects: none.
"""

from __future__ import annotations

import json
from typing import Any

from ib.scoring.native_tool_schema import argument_schema, schema_allows_null


def tool_definition(function: dict[str, Any]) -> dict[str, Any]:
    """Return {"name", "description", "input_schema"} for one harness function descriptor.

    input_schema is {"type": "object", "properties": {...}, "required": [...]}: one
    property per argument, in argument order, built by argument_schema from the
    argument's descriptor and its entry in `argument_descriptions`; `required` is the
    arguments whose schema does not allow null. The input is not mutated.
    """
    descriptions = function.get("argument_descriptions") or {}
    properties = {
        name: argument_schema(descriptor, descriptions.get(name))
        for name, descriptor in (function.get("arguments") or {}).items()
    }
    return {
        "name": function["name"],
        "description": function.get("description", ""),
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": [name for name, schema in properties.items() if not schema_allows_null(schema)],
        },
    }


def tool_definitions(functions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return tool_definition of each function, in input order."""
    return [tool_definition(function) for function in functions]


def tools_text(definitions: list[dict[str, Any]]) -> str:
    """Return "Available functions:\n" + json.dumps(definitions, ensure_ascii=False, indent=2)."""
    return "Available functions:\n" + json.dumps(definitions, ensure_ascii=False, indent=2)


def available_functions_text(functions: list[dict[str, Any]]) -> str:
    """Return tools_text(tool_definitions(functions))."""
    return tools_text(tool_definitions(functions))
