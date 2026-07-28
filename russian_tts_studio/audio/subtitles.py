"""Subtitle generation from Whisper word-level timestamps.

Generates:
- ``.srt`` — standard SubRip subtitles, grouped into readable cues.
- ``.ass`` — karaoke-style ASS subtitles with word-level timing.

Both formats use the same word-timestamp input. The grouping algorithm
merges consecutive words into cues of ``max_duration`` seconds, breaking
on sentence-ending punctuation (``.``, ``!``, ``?``) when possible.

Offset support: when narration starts after a music intro (podcast
mix), pass ``offset_sec`` to shift all timestamps forward.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class WordTimestamp:
    """A single word with its time alignment."""

    word: str
    start: float  # seconds
    end: float  # seconds


@dataclass
class SubtitleCue:
    """A grouped subtitle cue (one or more words)."""

    text: str
    start: float
    end: float
    words: list[WordTimestamp]


# ---------------------------------------------------------------------------
# SRT format
# ---------------------------------------------------------------------------


def _format_srt_time(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS,mmm`` for SRT."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _group_cues(
    words: list[WordTimestamp],
    max_duration: float = 5.0,
    max_chars: int = 80,
) -> list[SubtitleCue]:
    """Group word timestamps into readable subtitle cues.

    Breaks on:
    1. Sentence-ending punctuation (``.``, ``!``, ``?``).
    2. ``max_duration`` seconds.
    3. ``max_chars`` characters.

    Returns a list of :class:`SubtitleCue`.
    """
    if not words:
        return []

    cues: list[SubtitleCue] = []
    current_words: list[WordTimestamp] = []

    def _flush() -> None:
        if not current_words:
            return
        text = " ".join(w.word for w in current_words)
        cues.append(SubtitleCue(
            text=text,
            start=current_words[0].start,
            end=current_words[-1].end,
            words=list(current_words),
        ))
        current_words.clear()

    for w in words:
        current_words.append(w)
        # Check break conditions.
        duration = w.end - (current_words[0].start if current_words else w.start)
        text_len = sum(len(ww.word) + 1 for ww in current_words)
        ends_sentence = bool(re.search(r"[.!?]$", w.word.rstrip()))

        if ends_sentence or duration >= max_duration or text_len >= max_chars:
            _flush()

    _flush()
    return cues


def write_srt(
    words: list[WordTimestamp],
    output_path: str | Path,
    offset_sec: float = 0.0,
    max_duration: float = 5.0,
    max_chars: int = 80,
) -> Path:
    """Write an SRT subtitle file from word timestamps.

    ``offset_sec`` shifts all timestamps (useful when narration starts
    after a music intro in a podcast mix).
    """
    output_path = Path(output_path)
    cues = _group_cues(words, max_duration=max_duration, max_chars=max_chars)

    lines: list[str] = []
    for i, cue in enumerate(cues, 1):
        start = cue.start + offset_sec
        end = cue.end + offset_sec
        lines.append(str(i))
        lines.append(f"{_format_srt_time(start)} --> {_format_srt_time(end)}")
        lines.append(cue.text)
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote SRT: %s (%d cues)", output_path, len(cues))
    return output_path


# ---------------------------------------------------------------------------
# ASS format (karaoke-style)
# ---------------------------------------------------------------------------


def _format_ass_time(seconds: float) -> str:
    """Format seconds as ``H:MM:SS.cc`` for ASS (centiseconds)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds - int(seconds)) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_header() -> str:
    return """[Script Info]
Title: Russian TTS Studio Subtitles
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,1,2,10,10,40,1
Style: Karaoke,Arial,48,&H0000FFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,1,2,10,10,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def write_ass(
    words: list[WordTimestamp],
    output_path: str | Path,
    offset_sec: float = 0.0,
    max_duration: float = 5.0,
    max_chars: int = 80,
) -> Path:
    r"""Write an ASS (Advanced SubStation Alpha) subtitle file.

    Each word is timed individually for karaoke-style highlighting.
    Cues are grouped the same way as SRT, but each word within a cue
    gets its own ``\kf`` (karaoke fill) tag.
    """
    output_path = Path(output_path)
    cues = _group_cues(words, max_duration=max_duration, max_chars=max_chars)

    events: list[str] = []
    for cue in cues:
        start = cue.start + offset_sec
        end = cue.end + offset_sec
        # Build karaoke text with \kf tags for word-level timing.
        karaoke_parts: list[str] = []
        prev_end = cue.start
        for w in cue.words:
            # \kf duration is in centiseconds from the previous word end.
            dur_cs = int((w.start - prev_end) * 100)
            karaoke_parts.append(f"\\kf{dur_cs} {w.word}")
            prev_end = w.end
        karaoke_parts.append("\\kf0")  # final flush

        text = "".join(karaoke_parts)
        events.append(
            f"Dialogue: 0,{_ass_format_time(start)},{_ass_format_time(end)}"
            f",Default,,0,0,0,,{text}"
        )

    content = _ass_header() + "\n".join(events) + "\n"
    output_path.write_text(content, encoding="utf-8")
    logger.info("Wrote ASS: %s (%d cues)", output_path, len(cues))
    return output_path


def _ass_format_time(seconds: float) -> str:
    """Format for ASS dialogue lines: ``H:MM:SS.cc``."""
    return _format_ass_time(seconds)


# ---------------------------------------------------------------------------
# Convenience: extract words from Whisper result dict
# ---------------------------------------------------------------------------


def words_from_whisper(result: dict) -> list[WordTimestamp]:
    """Extract :class:`WordTimestamp` list from a Whisper result dict.

    Whisper's ``transcribe`` output has ``result["segments"][i]["words"]``
    with ``word``, ``start``, ``end`` keys. This helper flattens all
    segments into a single word list.
    """
    words: list[WordTimestamp] = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            words.append(WordTimestamp(
                word=w["word"].strip(),
                start=w["start"],
                end=w["end"],
            ))
    return words


def words_from_timestamps_list(
    timestamps: list[dict],
) -> list[WordTimestamp]:
    """Extract words from a flat list of ``{word, start, end}`` dicts.

    Useful when timestamps are stored as a JSON blob in the project
    store (``segment.metrics_json`` or a dedicated field).
    """
    return [
        WordTimestamp(word=w["word"].strip(), start=w["start"], end=w["end"])
        for w in timestamps
        if w.get("word")
    ]


# ---------------------------------------------------------------------------
# Chapter-aware subtitle export
# ---------------------------------------------------------------------------


def write_chapter_subtitles(
    words: list[WordTimestamp],
    chapters: list[dict],
    output_dir: str | Path,
    format: str = "srt",
    max_duration: float = 5.0,
    max_chars: int = 80,
) -> list[Path]:
    """Write separate subtitle files for each chapter.

    ``chapters`` is a list of ``{title, char_start, char_end}`` dicts
    (as returned by ``text.longform.detect_chapters``). Words are
    assigned to chapters based on their temporal position relative to
    the overall narration duration.

    Returns a list of paths to the generated subtitle files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not words or not chapters:
        return []

    # Determine total duration for mapping chapters to time ranges.
    total_duration = words[-1].end if words else 0
    total_chars = max(c.get("char_end", 0) for c in chapters) if chapters else 1

    paths: list[Path] = []
    for i, ch in enumerate(chapters):
        title = ch.get("title", f"chapter_{i}")
        char_start = ch.get("char_start", 0)
        char_end = ch.get("char_end", total_chars)

        # Map character range to time range.
        t_start = (char_start / total_chars) * total_duration
        t_end = (char_end / total_chars) * total_duration

        # Filter words in this time range.
        ch_words = [w for w in words if w.start >= t_start - 0.1 and w.end <= t_end + 0.1]

        # Offset words so chapter starts at 0.
        offset = t_start
        ch_words = [
            WordTimestamp(word=w.word, start=w.start - offset, end=w.end - offset)
            for w in ch_words
        ]

        # Build filename from title.
        safe_title = "".join(c if c.isalnum() or c in "-_ " else "_" for c in title).strip()
        if not safe_title:
            safe_title = f"chapter_{i}"
        ext = "ass" if format == "ass" else "srt"
        out_path = output_dir / f"{safe_title}.{ext}"

        if format == "ass":
            write_ass(ch_words, out_path, max_duration=max_duration, max_chars=max_chars)
        else:
            write_srt(ch_words, out_path, max_duration=max_duration, max_chars=max_chars)
        paths.append(out_path)

    # Also write a combined file with all chapters.
    combined_path = output_dir / f"all_chapters.{ext}"
    if format == "ass":
        write_ass(words, combined_path, max_duration=max_duration, max_chars=max_chars)
    else:
        write_srt(words, combined_path, max_duration=max_duration, max_chars=max_chars)
    paths.append(combined_path)

    logger.info("Wrote %d chapter subtitle files + combined", len(chapters))
    return paths