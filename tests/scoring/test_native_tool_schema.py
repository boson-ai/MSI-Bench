"""Canonical JSON Schema conversion for provider-native eval tools."""

from __future__ import annotations

from ib.scoring.native_tool_schema import (
    build_native_tool_bundle,
    descriptor_json_schema,
    gemini_function_declarations,
    gemini_live_tools,
    openai_chat_tools,
    openai_realtime_tools,
)


def test_descriptor_json_schema_covers_manifest_descriptor_forms() -> None:
    assert descriptor_json_schema("string_or_null HH:MM") == {
        "type": ["string", "null"],
        "pattern": r"^(?:[01]\d|2[0-3]):[0-5]\d$",
        "description": "24-hour time in HH:MM format",
    }
    assert descriptor_json_schema("string YYYY-MM-DD") == {
        "type": "string",
        "pattern": r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$",
        "description": "calendar date in YYYY-MM-DD format",
    }
    assert descriptor_json_schema("boolean") == {"type": "boolean"}
    assert descriptor_json_schema("integer range[-5,10]") == {
        "type": "integer",
        "minimum": -5,
        "maximum": 10,
    }
    assert descriptor_json_schema("number_or_null range[16,30]") == {
        "type": ["number", "null"],
        "minimum": 16.0,
        "maximum": 30.0,
    }
    assert descriptor_json_schema("enum[Friday,Saturday]") == {
        "type": "string",
        "enum": ["Friday", "Saturday"],
    }
    assert descriptor_json_schema("enum[warm_amber,cool_white] or null") == {
        "type": ["string", "null"],
        "enum": ["warm_amber", "cool_white", None],
    }
    assert descriptor_json_schema("array[string]_or_null") == {
        "type": ["array", "null"],
        "items": {"type": "string"},
    }


def test_object_array_descriptor_preserves_nested_enum_fields() -> None:
    schema = descriptor_json_schema(
        "array of objects{holder:string_or_null,scope:enum[personal,group],active:boolean}"
    )

    assert schema == {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "holder": {"type": ["string", "null"]},
                "scope": {"type": "string", "enum": ["personal", "group"]},
                "active": {"type": "boolean"},
            },
            "additionalProperties": False,
            "required": ["scope", "active"],
        },
    }


def test_bundle_restores_original_names_and_wraps_provider_shapes() -> None:
    bundle = build_native_tool_bundle(
        [
            {
                "name": "修改订餐选项",
                "arguments": {
                    "meal_type": "enum[standard,vegan]",
                    "quantity": "integer range[1,8]",
                },
            }
        ]
    )

    assert bundle.name_map == {"tool_1": "修改订餐选项"}
    assert bundle.declarations[0]["parameters"]["additionalProperties"] is False
    assert bundle.declarations[0]["parameters"]["required"] == [
        "meal_type",
        "quantity",
    ]
    assert openai_chat_tools(bundle)[0] == {
        "type": "function",
        "function": bundle.declarations[0],
    }
    assert openai_realtime_tools(bundle)[0] == {
        "type": "function",
        **bundle.declarations[0],
    }
    assert (
        gemini_function_declarations(bundle)[0]["parameters_json_schema"]
        == (bundle.declarations[0]["parameters"])
    )
    assert gemini_live_tools(bundle) == [
        {"function_declarations": gemini_function_declarations(bundle)}
    ]


def test_bundle_requires_non_nullable_arguments_but_not_nullable_ones() -> None:
    bundle = build_native_tool_bundle(
        [
            {
                "name": "update_booking",
                "arguments": {
                    "count": "integer range[1,8]",
                    "note": "string_or_null",
                },
            }
        ]
    )

    assert bundle.declarations[0]["parameters"]["required"] == ["count"]
