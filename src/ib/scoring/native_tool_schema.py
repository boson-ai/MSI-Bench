"""native_tool_schema - canonical schemas for provider-native function tools.

Calling spec:
    bundle = build_native_tool_bundle(manifest_row.get("logic_available_functions"))
    tools = openai_chat_tools(bundle)
    realtime_tools = openai_realtime_tools(bundle)
    declarations = gemini_function_declarations(bundle)
    live_tools = gemini_live_tools(bundle)

Inputs are model-visible harness function descriptors. Outputs are deterministic
JSON Schema declarations plus a provider-name-to-harness-name map. No gold calls
are consumed, and the module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_MAX_TOOL_NAME_LENGTH = 64
_RANGE_RE = re.compile(r"range\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", re.I)
_ENUM_RE = re.compile(r"^enum(?:_or_null)?\[(.*)\](?:\s+or\s+null)?$", re.I)
_OBJECT_ARRAY_RE = re.compile(
    r"^array\s+of\s+objects\{(.*)\}(?:_or_null|\s+or\s+null)?$",
    re.I,
)
_TIME_24H_PATTERN = r"^(?:[01]\d|2[0-3]):[0-5]\d$"
_DATE_ISO_PATTERN = r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$"
# Canonical string formats. A gold-callable string argument must match one of
# these hints; anything else compiles to a bare string and the tool-contract
# gate rejects it. Add a format by adding one entry.
STRING_FORMATS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"\bHH:MM\b", re.I), _TIME_24H_PATTERN, "24-hour time in HH:MM format"),
    (
        re.compile(r"\bYYYY-MM-DD\b", re.I),
        _DATE_ISO_PATTERN,
        "calendar date in YYYY-MM-DD format",
    ),
)
STRING_FORMAT_HINTS: tuple[str, ...] = ("string HH:MM 24h", "string YYYY-MM-DD")


@dataclass(frozen=True)
class NativeToolBundle:
    """Canonical function declarations and provider-to-original name mapping."""

    declarations: list[dict[str, Any]]
    name_map: dict[str, str]


def build_native_tool_bundle(available_functions: Any) -> NativeToolBundle:
    """Return canonical native function declarations for one manifest row."""

    if available_functions is None:
        return NativeToolBundle(declarations=[], name_map={})
    if not isinstance(available_functions, list):
        raise ValueError("logic_available_functions must be a list")
    declarations: list[dict[str, Any]] = []
    name_map: dict[str, str] = {}
    used_names: set[str] = set()
    for index, raw_function in enumerate(available_functions):
        if not isinstance(raw_function, dict):
            raise ValueError(f"available function #{index + 1} must be an object")
        original_name = raw_function.get("name")
        if not isinstance(original_name, str) or not original_name.strip():
            raise ValueError(f"available function #{index + 1} needs a non-blank name")
        original_name = original_name.strip()
        raw_arguments = raw_function.get("arguments", {})
        if not isinstance(raw_arguments, dict):
            raise ValueError(f"available function {original_name!r} arguments must be an object")
        invalid_keys = [key for key in raw_arguments if not isinstance(key, str) or not key.strip()]
        if invalid_keys:
            raise ValueError(f"available function {original_name!r} has invalid argument names")
        provider_name = _safe_tool_name(original_name, index=index, used_names=used_names)
        name_map[provider_name] = original_name
        properties = {
            key: descriptor_json_schema(value) for key, value in raw_arguments.items()
        }
        parameters: dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
        }
        required = [key for key, schema in properties.items() if not schema_allows_null(schema)]
        if required:
            parameters["required"] = required
        declarations.append(
            {
                "name": provider_name,
                "description": (
                    f"Execute {original_name} for the current assistant request. "
                    f"The original action name is {original_name}."
                ),
                "parameters": parameters,
            }
        )
    return NativeToolBundle(declarations=declarations, name_map=name_map)


def openai_chat_tools(bundle: NativeToolBundle) -> list[dict[str, Any]]:
    """Return Chat Completions tool wrappers for a canonical bundle."""

    return [{"type": "function", "function": declaration} for declaration in bundle.declarations]


def openai_realtime_tools(bundle: NativeToolBundle) -> list[dict[str, Any]]:
    """Return flattened OpenAI Realtime function declarations."""

    return [{"type": "function", **declaration} for declaration in bundle.declarations]


def gemini_function_declarations(bundle: NativeToolBundle) -> list[dict[str, Any]]:
    """Return google-genai FunctionDeclaration-compatible dictionaries."""

    return [
        {
            "name": declaration["name"],
            "description": declaration["description"],
            "parameters_json_schema": declaration["parameters"],
        }
        for declaration in bundle.declarations
    ]


def gemini_live_tools(bundle: NativeToolBundle) -> list[dict[str, Any]]:
    """Return Gemini Live tool wrappers for a canonical bundle."""

    declarations = gemini_function_declarations(bundle)
    return [{"function_declarations": declarations}] if declarations else []


def descriptor_json_schema(value: Any) -> dict[str, Any]:
    """Convert one compact harness argument descriptor into JSON Schema."""

    raw = str(value).strip()
    lowered = raw.lower()
    nullable = bool(
        lowered.endswith("_or_null")
        or re.search(r"\s+or\s+null$", lowered)
        or re.match(r"^(?:string|number|integer|boolean)_or_null(?:\s|$)", lowered)
    )
    object_match = _OBJECT_ARRAY_RE.fullmatch(raw)
    if object_match:
        fields = _object_fields_schema(object_match.group(1))
        item_schema: dict[str, Any] = {
            "type": "object",
            "properties": fields,
            "additionalProperties": False,
        }
        required = [key for key, schema in fields.items() if not schema_allows_null(schema)]
        if required:
            item_schema["required"] = required
        return _nullable_schema(
            {
                "type": "array",
                "items": item_schema,
            },
            nullable,
        )
    if lowered.startswith("array["):
        item_text = raw[raw.find("[") + 1 : raw.rfind("]")].strip()
        item_schema = descriptor_json_schema(item_text or "string")
        return _nullable_schema({"type": "array", "items": item_schema}, nullable)
    enum_match = _ENUM_RE.fullmatch(raw)
    if enum_match:
        options = [option.strip() for option in enum_match.group(1).split(",") if option.strip()]
        schema: dict[str, Any] = {"type": "string", "enum": options}
        if nullable:
            schema["enum"] = [*options, None]
        return _nullable_schema(schema, nullable)
    if "boolean" in lowered:
        return _nullable_schema({"type": "boolean"}, nullable)
    if "integer" in lowered:
        return _numeric_schema("integer", raw, nullable=nullable)
    if "number" in lowered:
        return _numeric_schema("number", raw, nullable=nullable)
    schema = {"type": "string"}
    for hint, pattern, description in STRING_FORMATS:
        if hint.search(raw):
            schema.update({"pattern": pattern, "description": description})
            break
    return _nullable_schema(schema, nullable)


def _object_fields_schema(raw_fields: str) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for raw_field in _split_top_level(raw_fields):
        name, separator, descriptor = raw_field.partition(":")
        if not separator or not name.strip() or not descriptor.strip():
            raise ValueError(f"invalid object-array field descriptor: {raw_field!r}")
        fields[name.strip()] = descriptor_json_schema(descriptor.strip())
    return fields


def _split_top_level(value: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(value):
        if char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(value[start:index].strip())
            start = index + 1
    parts.append(value[start:].strip())
    return [part for part in parts if part]


def _numeric_schema(kind: str, descriptor: str, *, nullable: bool) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": kind}
    range_match = _RANGE_RE.search(descriptor)
    if range_match:
        cast = int if kind == "integer" else float
        schema["minimum"] = cast(range_match.group(1))
        schema["maximum"] = cast(range_match.group(2))
    return _nullable_schema(schema, nullable)


def _nullable_schema(schema: dict[str, Any], nullable: bool) -> dict[str, Any]:
    if not nullable:
        return schema
    schema = dict(schema)
    value_type = schema.get("type")
    if isinstance(value_type, str):
        schema["type"] = [value_type, "null"]
    return schema


def schema_allows_null(schema: dict[str, Any]) -> bool:
    """Return whether one canonical JSON Schema accepts null."""

    value_type = schema.get("type")
    return value_type == "null" or (
        isinstance(value_type, list) and "null" in value_type
    )


def _safe_tool_name(original_name: str, *, index: int, used_names: set[str]) -> str:
    base = original_name if _TOOL_NAME_RE.fullmatch(original_name) else ""
    if not base:
        base = re.sub(r"[^A-Za-z0-9_-]+", "_", original_name).strip("_-")
    if not base:
        base = f"tool_{index + 1}"
    base = base[:_MAX_TOOL_NAME_LENGTH].strip("_-") or f"tool_{index + 1}"
    candidate = base
    suffix = 2
    while candidate in used_names:
        ending = f"_{suffix}"
        candidate = f"{base[: _MAX_TOOL_NAME_LENGTH - len(ending)]}{ending}"
        suffix += 1
    used_names.add(candidate)
    return candidate
