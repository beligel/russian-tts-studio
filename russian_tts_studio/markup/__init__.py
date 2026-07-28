"""RTTS Markup — inline narration control from ``{{...}}`` commands.

Public API::

    from russian_tts_studio.markup import parse, ParsedDocument, NarrationSegment

    doc = parse("{{chapter \\"Урок 1\\"}} Привет. {{pause 700ms}} Пока. {{speed 0.9}} Медленно.")
    for seg in doc.segments:
        print(seg.text, seg.state.speed, seg.pause_after)

Phase 1 scope: pause, speed, volume, chapter, alias, reset.
Sound events (laugh, cough, bgm, humming, ...) and stress marks
(``{{stress "за́мок"}}``) are added in the higgs-audio-inspired
extension. Voice/language/cmd/preset/play/stop are intentionally
absent (see ``commands.py`` docstring for rationale).
"""

from __future__ import annotations

from .commands import (
    Alias,
    Chapter,
    Command,
    NarrationState,
    Pause,
    Reset,
    SoundEvent,
    Speed,
    Stress,
    Unknown,
    Volume,
)
from .document import ChapterMarker, NarrationSegment, ParsedDocument, parse
from .parser import find_markup, parse_command

__all__ = [
    "parse",
    "ParsedDocument",
    "NarrationSegment",
    "ChapterMarker",
    "parse_command",
    "find_markup",
    # Command dataclasses (for type-narrowing in callers)
    "Command",
    "Pause",
    "Speed",
    "Volume",
    "Chapter",
    "Alias",
    "Reset",
    "SoundEvent",
    "Stress",
    "Unknown",
    "NarrationState",
]