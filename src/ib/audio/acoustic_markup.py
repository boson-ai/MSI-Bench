"""acoustic_markup — normalize inline SFX markup into SoundPlan events.

Calling spec:
    result = normalize_acoustic_markup("<engine>I need [honk] Dundas", line_index=0)
    clean = clean_spoken_text(text)

Angle tags (`<...>`) become sustained `kind="ambient"` intents. Square tags
(`[... ]`) become burst/localizable `kind="event"` intents. Boson/Higgs speech
control tags (`<|...|>`) are stripped from clean text but never become acoustic
events. Tags are removed from clean text so transcripts never expose markup.

Side effects: none.
"""

from __future__ import annotations

import re

from pydantic import Field

from ib.models.base import IbBaseModel
from ib.models.sound_plan import SoundAnchorPolicy, SoundPlanEvent, SoundPlanLine

_TAG_RE = re.compile(r"(<\|[^<>]*?>|<[^<>]+>|\[[^\[\]]+\])")


class AcousticMarkupResult(IbBaseModel):
    """Clean spoken text plus normalized line-level acoustic events."""

    clean_text: str
    line_plan: SoundPlanLine


class _RawTag(IbBaseModel):
    marker: str = Field(min_length=2)
    description: str
    before_text: str
    after_text: str


def clean_spoken_text(text: str) -> str:
    """Return text with acoustic markup removed and whitespace normalized."""
    return normalize_acoustic_markup(text).clean_text


def normalize_acoustic_markup(text: str, *, line_index: int = 0) -> AcousticMarkupResult:
    """Parse inline acoustic markup into structured events and clean text."""
    raw_tags = _raw_tags(text)
    clean_text = _normalize_text(_TAG_RE.sub(" ", text))
    events = [_event_for_tag(index, tag) for index, tag in enumerate(raw_tags)]
    return AcousticMarkupResult(
        clean_text=clean_text,
        line_plan=SoundPlanLine(line_index=line_index, clean_text=clean_text, events=events),
    )


def _raw_tags(text: str) -> list[_RawTag]:
    tags: list[_RawTag] = []
    for match in _TAG_RE.finditer(text):
        marker = match.group(0)
        if _is_higgs_speech_tag(marker):
            continue
        description = marker[1:-1].strip()
        if not description:
            continue
        tags.append(
            _RawTag(
                marker=marker,
                description=description,
                before_text=_normalize_text(_TAG_RE.sub(" ", text[: match.start()])),
                after_text=_normalize_text(_TAG_RE.sub(" ", text[match.end() :])),
            )
        )
    return tags


def _is_higgs_speech_tag(marker: str) -> bool:
    return marker.startswith("<|")


def _event_for_tag(index: int, tag: _RawTag) -> SoundPlanEvent:
    kind = "ambient" if tag.marker.startswith("<") else "event"
    anchor_policy = _anchor_policy(tag, kind=kind)
    description_slug = re.sub(r"[^a-z0-9]+", "_", tag.description.lower()).strip("_") or kind
    return SoundPlanEvent(
        event_id=f"{kind}_{description_slug}_{index + 1}",
        kind=kind,
        description=tag.description,
        anchor_policy=anchor_policy,
        anchor_text_before=_last_word(tag.before_text),
        anchor_text_after=_first_word(tag.after_text),
    )


def _anchor_policy(tag: _RawTag, *, kind: str) -> SoundAnchorPolicy:
    if kind == "ambient" and not tag.before_text.strip():
        return "line"
    if not tag.before_text.strip():
        return "prefix"
    if not tag.after_text.strip():
        return "suffix"
    return "infix"


def _normalize_text(text: str) -> str:
    collapsed = " ".join(text.split())
    collapsed = re.sub(r"\s+([,.;:!?])", r"\1", collapsed)
    return re.sub(r"([({])\s+", r"\1", collapsed).strip()


def _last_word(text: str) -> str | None:
    words = _anchor_tokens(text)
    return words[-1] if words else None


def _first_word(text: str) -> str | None:
    words = _anchor_tokens(text)
    return words[0] if words else None


def _anchor_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    ascii_word: list[str] = []
    for char in text:
        if char.isascii() and (char.isalnum() or char == "'"):
            ascii_word.append(char)
            continue
        if ascii_word:
            tokens.append("".join(ascii_word))
            ascii_word = []
        if char.isalnum():
            tokens.append(char)
    if ascii_word:
        tokens.append("".join(ascii_word))
    return tokens
