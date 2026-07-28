"""Document model: parsed markup → list of narration segments.

A :class:`ParsedDocument` splits the source text into a sequence of
:class:`NarrationSegment` objects, each carrying:

- ``text``: the raw narration text for this segment (markup stripped,
  aliases not yet applied — the pipeline applies aliases at synthesis
  time so the source stays editable).
- ``state``: a snapshot of the active narration state at the start of
  this segment (speed, volume, chapter, accumulated aliases).
- ``pause_after``: a :class:`Pause` to insert *after* this segment
  (from a trailing ``{{pause ...}}``). ``None`` means no explicit pause.

The document also collects ``warnings`` (unknown commands) and
``chapters`` (ordered list of chapter titles with their char offsets in
the *original* source, for UI navigation).

Segmentation rule: every command (except ``pause``) starts a new
segment. ``pause`` attaches to the preceding segment's ``pause_after``
without splitting. A leading command before any text produces an empty
initial segment with the state already set — the pipeline skips empty
segments at synthesis time.

Example::

    doc = parse("{{chapter \\"Урок 1\\"}} Привет. {{pause 700ms}} Пока.")
    assert len(doc.segments) == 2
    assert doc.segments[0].state.chapter == "Урок 1"
    assert doc.segments[0].pause_after.ms == 700
    assert doc.segments[1].text == "Пока."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .commands import (
    Alias,
    Chapter,
    Command,
    NarrationState,
    Pause,
    Reset,
    SoundEvent,
    Stress,
    Unknown,
)
from .parser import find_markup, parse_command


@dataclass
class NarrationSegment:
    """One chunk of narration with its active state and trailing pause."""

    text: str
    state: NarrationState = field(default_factory=NarrationState)
    pause_after: Optional[Pause] = None
    # Index of the first char of this segment's text in the *original*
    # source (including markup). Useful for UI highlighting / error
    # reporting.
    source_start: int = 0


@dataclass
class ChapterMarker:
    """Chapter title + its position in the original source text."""

    title: str
    source_offset: int = 0


@dataclass
class ParsedDocument:
    """Result of parsing markup-annotated text.

    Use :func:`parse` to build one. Then iterate ``segments`` to
    synthesise, or read ``warnings`` / ``chapters`` for UI feedback.
    """

    segments: list[NarrationSegment] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    chapters: list[ChapterMarker] = field(default_factory=list)
    # The original source text (with markup intact) — for round-trip
    # editing in the UI.
    source: str = ""
    # Set by ``parse()`` — True iff the source contained any ``{{...}}``
    # blocks. ``segments`` is always non-empty (at least one segment), so
    # we can't use ``bool(self.segments)`` as the signal.
    has_markup: bool = False


def parse(text: str) -> ParsedDocument:
    """Parse markup-annotated text into a :class:`ParsedDocument`.

    Empty input → one empty segment (callers should handle the
    no-text case themselves; we don't filter here so the segment list
    always has at least one entry and indexing is predictable).
    """
    doc = ParsedDocument(source=text)
    state = NarrationState()
    segments: list[NarrationSegment] = []

    marks = find_markup(text)

    # If no markup at all, return a single segment with the whole text.
    if not marks:
        segments.append(NarrationSegment(text=text, state=state.snapshot()))
        doc.segments = segments
        doc.has_markup = False
        return doc

    doc.has_markup = True

    # Walk the text, accumulating narration between commands.
    cursor = 0
    current_text_start = 0
    current_text_parts: list[str] = []

    def _flush_segment() -> None:
        """Emit a segment from accumulated text + current state."""
        raw = "".join(current_text_parts)
        # Strip surrounding whitespace but preserve internal structure.
        stripped = raw.strip()
        if stripped or segments:  # always emit if we already have segments
            seg = NarrationSegment(
                text=stripped,
                state=state.snapshot(),
                source_start=current_text_start,
            )
            # Attach a pending pause if one was queued by the previous
            # command. We attach to the *preceding* segment, not the
            # following one — a pause after "Привет." sits between
            # "Привет." and "Пока.", so it delays the boundary after
            # "Привет.".
            segments.append(seg)
        current_text_parts.clear()

    pending_pause: Optional[Pause] = None

    for start, end, body in marks:
        # Text between cursor and this command belongs to the current segment.
        if start > cursor:
            chunk = text[cursor:start]
            if current_text_parts == []:
                current_text_start = cursor
            current_text_parts.append(chunk)

        cmd = parse_command(body)

        if isinstance(cmd, Unknown):
            doc.warnings.append(cmd.warning or f"unknown command: {body!r}")
            # Unknown command doesn't split — treat as a no-op marker.
            # But we still flush the accumulated text into a segment so
            # the warning position is attributable. Actually, to keep
            # semantics simple: unknown commands are invisible to
            # segmentation (like they don't exist). Continue accumulating.
            cursor = end
            continue

        if isinstance(cmd, Pause):
            # Pause attaches to the *preceding* text. Flush the current
            # text as a segment, then set pause_after on it.
            _flush_segment()
            if not segments:
                # Leading pause before any text → create an empty
                # segment so the pipeline can emit initial silence.
                segments.append(
                    NarrationSegment(text="", state=state.snapshot(), source_start=start)
                )
            segments[-1].pause_after = cmd
            pending_pause = None
            cursor = end
            continue

        if isinstance(cmd, Chapter):
            doc.chapters.append(
                ChapterMarker(title=cmd.title, source_offset=start)
            )
            # Chapter starts a new segment boundary.
            _flush_segment()
            state.apply(cmd)
            cursor = end
            continue

        if isinstance(cmd, Reset):
            _flush_segment()
            state.apply(cmd)
            cursor = end
            continue

        if isinstance(cmd, (Alias, Stress)):
            # Aliases and stress marks don't split segments — they
            # accumulate into state and apply to all following text.
            # Stress auto-detect warnings surface into doc.warnings.
            if isinstance(cmd, Stress) and cmd.warning:
                doc.warnings.append(cmd.warning)
            state.apply(cmd)
            cursor = end
            continue

        if isinstance(cmd, SoundEvent):
            if cmd.is_inline():
                # Inline event (laugh, cough, ...): emit the token into
                # the text stream at this position. We don't split the
                # segment — the token becomes part of the current
                # segment's text. Engines that support sound events
                # (Higgs) render it; engines that don't (VoxCPM2) have
                # the pipeline strip it before synthesis.
                if current_text_parts == []:
                    current_text_start = start
                current_text_parts.append(cmd.token)
                cursor = end
                continue
            # Span event: ``start`` opens a region, ``end`` closes it.
            # We don't split the segment on a span marker — the span
            # state attaches to every subsequent segment's snapshot
            # until the matching ``end`` arrives. This way the pipeline
            # can emit wrapping tokens (or strip them) per segment.
            if cmd.is_span_start():
                _flush_segment()
                state.apply(cmd)
                cursor = end
                continue
            if cmd.is_span_end():
                _flush_segment()
                state.apply(cmd)
                cursor = end
                continue
            # Defensive: unknown span marker shape — treat as unknown.
            doc.warnings.append(f"sound event: malformed {cmd.name!r}")
            cursor = end
            continue

        # Speed / Volume: start a new segment so the new param applies
        # only to the following text.
        _flush_segment()
        state.apply(cmd)
        cursor = end

    # Trailing text after the last command.
    if cursor < len(text):
        chunk = text[cursor:]
        if current_text_parts == []:
            current_text_start = cursor
        current_text_parts.append(chunk)
    _flush_segment()

    # Edge case: input was all commands, no text. Ensure at least one
    # segment exists so callers don't Indexerror.
    if not segments:
        segments.append(NarrationSegment(text="", state=state.snapshot()))

    doc.segments = segments
    return doc