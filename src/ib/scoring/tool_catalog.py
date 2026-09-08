"""tool_catalog — available-action hints for model-facing prompts.

Calling spec:
    render_available_function_catalog([{"name": "set_navigation", "arguments": {...}}]) -> str

Inputs are harness-provided function schemas. Output is a compact prompt block
with argument type/range descriptors only; it never consumes or reveals
row-level expected tool call values, gold IDs, codes, or deadlines.

Side effects: none.
"""

from __future__ import annotations

from ib.llm import render_prompt

TOOL_ARGUMENT_HINTS: dict[str, tuple[str, ...]] = {
    "apply_coupon": ("coupon_code", "target", "order_id"),
    "assign_group": ("student", "group", "role", "due_time"),
    "book_room": ("room", "date", "time_window", "attendees", "purpose"),
    "contact_emergency_support": ("service", "reason", "location", "contact"),
    "create_schedule_or_reminder": ("title", "date", "time", "recipients", "notes"),
    "draft_email": ("recipient", "subject", "body", "send_time", "attachments"),
    "file_document": ("document", "folder", "label", "visibility"),
    "modify_order": ("order_id", "action", "item", "quantity", "notes"),
    "notify_caregiver": ("caregiver", "message", "patient", "urgency"),
    "place_order": ("merchant", "items", "order_type", "pickup_time", "instructions"),
    "provide_public_info": ("topic", "location", "language", "format"),
    "reserve_slot": ("service", "date", "time_window", "party_size", "notes"),
    "send_message": ("recipient", "message", "channel", "send_time"),
    "set_accessibility_mode": ("mode", "device", "duration", "notes"),
    "set_appliance": ("appliance", "mode", "start_time", "duration", "setting"),
    "set_audio": ("source", "volume", "zone", "mode", "until"),
    "set_cabin_camera": ("mode", "zone", "duration", "privacy_scope"),
    "set_cargo_mode": ("mode", "item", "route_constraint", "duration"),
    "set_climate": ("zone", "temperature", "mode", "until"),
    "set_contact_preference": ("contact", "preference", "channel", "time_window"),
    "set_device_state": ("device", "state", "location", "duration"),
    "set_medication_reminder": ("medication", "time", "person", "instructions"),
    "set_meeting_mode": ("room", "mode", "audio", "display", "privacy"),
    "set_navigation": ("destination", "route", "arrival_time", "avoid", "stops"),
    "set_party_mode": ("mode", "location", "guest_scope", "until"),
    "set_pickup": ("order_id", "location", "date", "time_window", "contact"),
    "set_playlist": ("playlist", "source", "queue_rules", "audience"),
    "set_queue_preference": ("service", "preference", "language", "accessibility"),
    "set_room_lighting": ("room", "brightness", "color", "until"),
    "set_security_mode": ("mode", "location", "duration", "authorized_by"),
    "set_study_mode": ("mode", "subject", "duration", "participants"),
    "set_thermostat": ("zone", "temperature", "mode", "until"),
    "set_window": ("window", "position", "zone", "until"),
    "share_assignment": ("assignment", "recipients", "version", "due_time"),
    "share_document": ("document", "recipients", "permission", "version"),
    "share_invite": ("event", "recipients", "visibility", "message"),
    "share_location": ("recipient", "location", "duration", "reason"),
    "start_stream": ("service", "title", "start_time", "device"),
    "update_calendar": ("event", "date", "time_window", "attendees", "change"),
}


def render_available_function_catalog(available_functions: list[dict]) -> str:
    """Return compact, available-action hints from harness function schemas."""
    functions: list[dict[str, str]] = []
    for function in available_functions:
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            continue
        tool_name = function["name"]
        arguments = function.get("arguments")
        if isinstance(arguments, dict) and arguments:
            rendered = ", ".join(
                f"{key}: {value}" for key, value in sorted(arguments.items())
            )
        else:
            hints = TOOL_ARGUMENT_HINTS.get(tool_name, ("arguments_from_spoken_evidence",))
            rendered = ", ".join(hints)
        functions.append({"name": tool_name, "arguments": rendered})
    return render_prompt("understanding/system.j2", {"kind": "tool_catalog", "functions": functions})


def render_tool_catalog(tool_names: tuple[str, ...]) -> str:
    """Return compact, available-action hints for explicit tool-name inputs."""
    functions: list[dict[str, str]] = []
    for tool_name in tool_names:
        hints = TOOL_ARGUMENT_HINTS.get(tool_name, ("arguments_from_spoken_evidence",))
        functions.append({"name": tool_name, "arguments": ", ".join(hints)})
    return render_prompt("understanding/system.j2", {"kind": "tool_catalog", "functions": functions})
