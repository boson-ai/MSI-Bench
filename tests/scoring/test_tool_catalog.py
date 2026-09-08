"""Tool catalog prompt safety tests."""

from __future__ import annotations

from ib.scoring.tool_catalog import render_available_function_catalog, render_tool_catalog


def test_render_available_function_catalog_uses_harness_schema_not_gold_values() -> None:
    block = render_available_function_catalog(
        [
            {
                "name": "set_navigation",
                "arguments": {
                    "destination": "string_or_null",
                    "route_mode": "enum[fastest,quietest]",
                },
            },
            {
                "name": "set_audio",
                "arguments": {"volume": "integer range[0,40]"},
            },
        ]
    )

    assert "set_navigation arguments: destination: string_or_null" in block
    assert "route_mode: enum[fastest,quietest]" in block
    assert "set_audio arguments: volume: integer range[0,40]" in block
    assert "Use only these function names" in block
    assert "SECRET" not in block
    assert "logic_expected_tool_calls" not in block


def test_render_tool_catalog_keeps_explicit_name_compatibility() -> None:
    block = render_tool_catalog(("set_navigation",))

    assert "set_navigation arguments: destination" in block
    assert "Use only these function names" in block
