"""setup_prompt_v1 — the MSI-Bench v1 paper's evaluation setup prompt.

Calling spec:
    setup_prompt_v1(row, tool_protocol="prompt_json") -> str

The function bodies are copied verbatim from the public MSI-Bench eval tree that
produced the v1 paper numbers (ib.scoring.multiturn_audio at boson-ai/MSI-Bench dfdc118);
only the template name differs (understanding/system_v1.j2, the same file verbatim).
ib.scoring.multiturn_audio.setup_prompt_for dispatches here for prompt_version "v1".
Do not edit it to follow the current prompt: v1 numbers are only reproducible with it.

Deterministic; no side effects.
"""

from __future__ import annotations

from typing import Any

from ib.llm import render_prompt
from ib.scoring.scene_prompt_profiles import profile_for_scene, scene_date_weekday
from ib.scoring.tool_catalog import render_available_function_catalog


def setup_prompt_v1(row: dict[str, Any], *, tool_protocol: str = "prompt_json") -> str:
    """Return scenario setup that belongs in the system instruction."""

    return render_prompt(
        "understanding/system_v1.j2",
        {
            "kind": "setup",
            "authorization": _authorization_prompt_data(row),
            "logic": _logic_prompt_data(row, tool_protocol=tool_protocol),
            "private_context": _private_memory_for_row(row),
            "speak_probe": bool(row.get("logic_speak_probe_render")),
        },
    )


def _logic_prompt_data(row: dict[str, Any], *, tool_protocol: str) -> dict[str, Any]:
    """Return template data for scenario-facing tool guidance."""

    pattern = row.get("logic_pattern")
    if not isinstance(pattern, str) or not pattern:
        return {"enabled": False, "participant_context": None, "native_tools": False}
    profile = profile_for_scene(str(row.get("scene") or "mobility_transport"))
    available_functions = row.get("logic_available_functions")
    native_tools = tool_protocol == "native"
    function_catalog = None
    if not native_tools:
        function_catalog = (
            render_available_function_catalog(available_functions)
            if isinstance(available_functions, list) and available_functions
            else "- Available functions for this conversation: none provided."
        )
    scene_date = row.get("logic_scene_date")
    return {
        "enabled": True,
        "assistant_role": profile.assistant_role,
        "setting_phrase": profile.setting_phrase,
        "function_catalog": function_catalog,
        "native_tools": native_tools,
        "scene_date": scene_date if isinstance(scene_date, str) and scene_date else None,
        "scene_weekday": (
            scene_date_weekday(scene_date)
            if isinstance(scene_date, str) and scene_date
            else None
        ),
        "participant_context": _participant_context_prompt_data(row),
    }


def _participant_context_prompt_data(row: dict[str, Any]) -> dict[str, Any] | None:
    """Return model-visible participant identity context for in_text rows."""
    context = row.get("participant_context")
    if not isinstance(context, dict):
        return None
    if context.get("visibility") != "model_visible":
        return None
    participants = context.get("participants")
    if not isinstance(participants, list) or not participants:
        return None
    return {
        "identity_mode": context.get("identity_mode") or "in_text",
        "authority_context": bool(context.get("authority_context")),
        "authority_rule": context.get("authority_rule"),
        "participants": [item for item in participants if isinstance(item, dict)],
    }


def _private_memory_for_row(row: dict[str, Any]) -> str | None:
    """Return disclosure-only private memory for the evaluated assistant."""
    if (
        row.get("logic_pattern") != "disclosure_clause_in_instruction"
        and row.get("logic_base_pattern") != "disclosure_clause_in_instruction"
    ):
        return None
    context = row.get("logic_private_memory")
    if not isinstance(context, str):
        return None
    text = context.strip()
    return text or None


def _authorization_prompt_data(row: dict[str, Any]) -> dict[str, Any]:
    """Return template data for authorization-aware manifest rows."""

    scope = row.get("authorization_scope")
    if not isinstance(scope, str) or not scope:
        return {"scope": None}
    speaker_profiles: list[dict[str, Any]] = []
    profiles = row.get("speaker_profiles")
    if isinstance(profiles, list):
        for profile in profiles:
            if isinstance(profile, dict):
                speaker_profiles.append(profile)
    return {
        "scope": scope,
        "capability": row.get("authorization_capability"),
        "policy_text": row.get("authorization_policy_text"),
        "speaker_profiles": speaker_profiles,
    }
