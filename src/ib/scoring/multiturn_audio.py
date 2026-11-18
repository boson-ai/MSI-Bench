"""multiturn_audio — render manifest rows as interleaved audio chat history.

Calling spec:
    setup_prompt_for(row, tool_protocol="prompt_json") -> str
        dispatches on ib.scoring.prompt_version.prompt_version_for(row) through
        SETUP_PROMPT_VARIANTS: "v1" -> the v1 paper prompt (ib.scoring.setup_prompt_v1),
        "v2" and absent -> the current prompt described below.
        prompt_json lists row["logic_tools"] (Anthropic tool definitions, a v2 release row) as
        ib.manifest.tool_definitions.tools_text when it is a non-empty list, else
        row["logic_available_functions"] (legacy descriptors) as available_functions_text;
        both are the training system message's listing. "native" leaves tools to the native
        API and raises ValueError for a row with logic_tools (native declarations are built
        from legacy descriptors only). When a logic_tools property description mentions a
        "voice label", the setup allows voice labels in arguments instead of forbidding
        participant references.
    history_messages(row, audio_paths) -> list[AudioHistoryMessage]
    history_prompt_for(row, audio_paths) -> str

Inputs are built manifest rows and resolved testcase audio paths. Outputs describe
the same conversation as AudioMultiChallenge-style history: each user turn is one
audio message, and prior assistant turns are assistant text messages.

A row with `audio_history` (a timeline benchmark case, ib.manifest.timeline_case) is replayed
from that ordered list, unbounded, in the row's `history_style` (default text_turns):
    HISTORY_STYLES = {
        "text_turns": one audio message per room block, assistant turns as text (training parity),
        "single_mix": one audio message, the whole-case room mix (row["case_mix"]["room"]), for
                      full-duplex or turn-limited targets,
    }
A history audio path must be one of the resolved `audio_paths` (the single mix is resolved
next to them); a missing one raises ValueError. Rows without `audio_history` keep the v1
wide-column walk over user_turn_N_audio / assistant_turn_N_transcript.

Side effects: none.
"""

from __future__ import annotations

import os

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ib.llm import render_prompt
from ib.manifest.tool_definitions import available_functions_text, tools_text
from ib.scoring.scene_prompt_profiles import scene_date_weekday
from ib.scoring.prompt_version import PromptVersion, prompt_version_for
from ib.scoring.setup_prompt_v1 import setup_prompt_v1


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
    """Return scenario setup that belongs in the system instruction, in the row's prompt version."""

    return SETUP_PROMPT_VARIANTS[prompt_version_for(row)](row, tool_protocol=tool_protocol)


def _current_setup_prompt(row: dict[str, Any], *, tool_protocol: str = "prompt_json") -> str:
    """The current setup prompt (MSI-Bench v2 and every row without prompt_version).

    When the row has a non-blank "assistant_name" (the contract's assistant name, from
    logic_manifest_fields), the setup tells the model its name and that the speakers call
    it by this name; a row without one gets no name line.
    """

    return render_prompt(
        "understanding/system.j2",
        {
            "kind": "setup",
            "assistant_name": _assistant_name_for(row),
            "logic": _logic_prompt_data(row, tool_protocol=tool_protocol),
            "voice_label_args": _takes_voice_labels(row.get("logic_tools")),
            "private_context": _private_memory_for_row(row),
            "multi_line_clips": MULTI_LINE_RULE and bool(row.get("audio_history")),  # timeline rows: a clip holds several lines
            "speak_probe": bool(row.get("logic_speak_probe_render")),
        },
    )


SETUP_PROMPT_VARIANTS: dict[PromptVersion | None, Any] = {
    "v1": setup_prompt_v1,
    "v2": _current_setup_prompt,
    None: _current_setup_prompt,
}


# IB_FINAL_TURN_NOTE overrides the note (empty string disables it); IB_MULTI_LINE_RULE=0 drops the
# multi-line-clip rule from the setup prompt. Both exist for prompt ablations on timeline rows.
FINAL_TURN_NOTE = os.environ.get(
    "IB_FINAL_TURN_NOTE",
    "(Instruction to the assistant, not spoken by anyone: the audio above is the latest speech in the room. "
    "Decide now and return your decision as the single JSON object described in the system prompt, with no other text.)",
)
MULTI_LINE_RULE = os.environ.get("IB_MULTI_LINE_RULE", "0") == "1"  # off: it made gemma-4 go silent twice as often


def final_turn_note(row: dict[str, Any], *, tool_protocol: str = "prompt_json") -> str | None:
    """Text appended to the final user turn of a timeline row (audio_history) under prompt_json: the clips
    hold whole stretches of talk, and without a closing cue models answer the last request in prose."""

    if tool_protocol == "native" or not row.get("audio_history") or not FINAL_TURN_NOTE:
        return None
    return FINAL_TURN_NOTE


def history_messages(row: dict[str, Any], audio_paths: list[Path]) -> list[AudioHistoryMessage]:
    """Return chronological user-audio and assistant-text history messages."""

    if row.get("audio_history"):
        style = row.get("history_style") or "text_turns"
        if style not in HISTORY_STYLES:
            raise ValueError(f"unknown history_style {style!r}; expected one of {sorted(HISTORY_STYLES)}")
        return _mark_final_user_turn(HISTORY_STYLES[style](row, audio_paths))
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


def _text_turns_history(row: dict[str, Any], audio_paths: list[Path]) -> list[AudioHistoryMessage]:
    """Return the row's audio_history as messages: a block per user entry, text per assistant entry."""
    messages = []
    for slot, entry in enumerate(row["audio_history"], start=1):
        if entry["role"] == "user":
            messages.append(
                AudioHistoryMessage(role="user", turn_index=slot, audio_path=_resolve_history_audio(entry["audio"], audio_paths))
            )
        elif entry["role"] == "assistant":
            messages.append(AudioHistoryMessage(role="assistant", turn_index=slot, assistant_text=entry["text"]))
        else:
            raise ValueError(f"audio_history entry {slot} has unknown role {entry.get('role')!r}")
    return messages


def _single_mix_history(row: dict[str, Any], audio_paths: list[Path]) -> list[AudioHistoryMessage]:
    """Return one user message carrying the whole-case room mix."""
    first = next(entry for entry in row["audio_history"] if entry["role"] == "user")
    block = _resolve_history_audio(first["audio"], audio_paths)
    relative = Path(first["audio"]).as_posix().lstrip("/")
    if not block.as_posix().endswith(relative):
        raise ValueError(f"cannot place the case mix next to {block}")
    root = Path(block.as_posix()[: -len(relative)])
    mix = root / row["case_mix"]["room"]
    if not mix.exists():
        raise ValueError(f"missing case mix {mix}")
    return [AudioHistoryMessage(role="user", turn_index=1, audio_path=mix)]


HISTORY_STYLES = {"text_turns": _text_turns_history, "single_mix": _single_mix_history}


def _resolve_history_audio(value: str, audio_paths: list[Path]) -> Path:
    index = _audio_index_for(value, audio_paths)
    if index is None:
        raise ValueError(f"audio_history names {value!r}, which is not among the row's resolved audio paths")
    return audio_paths[index]


def _logic_prompt_data(row: dict[str, Any], *, tool_protocol: str) -> dict[str, Any]:
    """Return template data for scenario-facing tool guidance."""

    pattern = row.get("logic_pattern")
    if not isinstance(pattern, str) or not pattern:
        return {"enabled": False, "native_tools": False}
    available_functions = row.get("logic_available_functions")
    logic_tools = row.get("logic_tools")
    native_tools = tool_protocol == "native"
    functions_text = None
    if not native_tools:
        functions_text = (
            tools_text(logic_tools)
            if isinstance(logic_tools, list) and logic_tools
            else available_functions_text(available_functions if isinstance(available_functions, list) else [])
        )
    scene_date = row.get("logic_scene_date")
    return {
        "enabled": True,
        "available_functions_text": functions_text,
        "native_tools": native_tools,
        "scene_date": scene_date if isinstance(scene_date, str) and scene_date else None,
        "scene_weekday": (
            scene_date_weekday(scene_date)
            if isinstance(scene_date, str) and scene_date
            else None
        ),
    }



def _takes_voice_labels(tools: Any) -> bool:
    """Return whether any input_schema property description of the Anthropic tools mentions a voice label."""
    if not isinstance(tools, list):
        return False
    return any(
        "voice label" in str(prop.get("description", "")).lower()
        for tool in tools
        for prop in ((tool.get("input_schema") or {}).get("properties") or {}).values()
    )


def _assistant_name_for(row: dict[str, Any]) -> str | None:
    """Return the row's non-blank "assistant_name" string, else None."""
    name = row.get("assistant_name")
    return name if isinstance(name, str) and name.strip() else None


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
