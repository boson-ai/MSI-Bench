"""multiturn_audio — render manifest rows as interleaved audio chat history.

Calling spec:
    setup_prompt_for(row, tool_protocol="prompt_json") -> str
    history_messages(row, audio_paths) -> list[AudioHistoryMessage]
    history_prompt_for(row, audio_paths) -> str

Inputs are built manifest rows and resolved testcase audio paths. Outputs describe
the same conversation as AudioMultiChallenge-style history: each user turn is one
audio message, and prior assistant turns are assistant text messages.

Side effects: none.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ib.llm import render_prompt
from ib.scoring.scene_prompt_profiles import profile_for_scene, scene_date_weekday
from ib.scoring.tool_catalog import render_available_function_catalog


HistoryRole = Literal["user", "assistant"]


@dataclass(frozen=True)
class AudioHistoryMessage:
    """One chronological message sent to the evaluated model."""

    role: HistoryRole
    turn_index: int | None = None
    audio_path: Path | None = None
    assistant_text: str | None = None
    final_user_turn: bool = False


def setup_prompt_for(row: dict[str, Any], *, tool_protocol: str = "prompt_json") -> str:
    """Return scenario setup that belongs in the system instruction."""

    return render_prompt(
        "understanding/system.j2",
        {
            "kind": "setup",
            "authorization": _authorization_prompt_data(row),
            "logic": _logic_prompt_data(row, tool_protocol=tool_protocol),
            "private_context": _private_memory_for_row(row),
            "speak_probe": bool(row.get("logic_speak_probe_render")),
        },
    )


def history_messages(row: dict[str, Any], audio_paths: list[Path]) -> list[AudioHistoryMessage]:
    """Return chronological user-audio and assistant-text history messages."""

    messages: list[AudioHistoryMessage] = []
    used_audio_indexes: set[int] = set()
    for turn_index in range(1, 9):
        audio_index = _audio_index_for(row.get(f"user_turn_{turn_index}_audio"), audio_paths)
        if audio_index is not None:
            messages.append(
                AudioHistoryMessage(
                    role="user",
                    turn_index=turn_index,
                    audio_path=audio_paths[audio_index],
                )
            )
            used_audio_indexes.add(audio_index)
        assistant_text = _clean_text(row.get(f"assistant_turn_{turn_index}_transcript"))
        if assistant_text is not None:
            messages.append(
                AudioHistoryMessage(
                    role="assistant",
                    turn_index=turn_index,
                    assistant_text=assistant_text,
                )
            )
    if not messages:
        messages = [
            AudioHistoryMessage(role="user", turn_index=index, audio_path=path)
            for index, path in enumerate(audio_paths, start=1)
        ]
    else:
        for index, path in enumerate(audio_paths):
            if index not in used_audio_indexes:
                messages.append(
                    AudioHistoryMessage(role="user", turn_index=None, audio_path=path)
                )
    return _mark_final_user_turn(messages)


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


def _mark_final_user_turn(messages: list[AudioHistoryMessage]) -> list[AudioHistoryMessage]:
    final_user_index = next(
        (index for index in range(len(messages) - 1, -1, -1) if messages[index].role == "user"),
        None,
    )
    if final_user_index is None:
        return messages
    return [
        AudioHistoryMessage(
            role=message.role,
            turn_index=message.turn_index,
            audio_path=message.audio_path,
            assistant_text=message.assistant_text,
            final_user_turn=index == final_user_index,
        )
        for index, message in enumerate(messages)
    ]


def _audio_index_for(value: Any, audio_paths: list[Path]) -> int | None:
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value)
    candidate_text = candidate.as_posix().lstrip("/")
    for index, path in enumerate(audio_paths):
        path_text = path.as_posix().lstrip("/")
        if candidate.is_absolute() and path == candidate:
            return index
        if path_text == candidate_text or path_text.endswith(f"/{candidate_text}"):
            return index
        if path.name == candidate.name:
            return index
    return None


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None
