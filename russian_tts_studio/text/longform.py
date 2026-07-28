"""Long-form text processing: chapter detection + paragraph-aware chunking.

This module sits *above* ``utils.text_utils.chunk_text_for_tts`` and
*below* the markup parser. It handles plain-text / Markdown / DOCX
inputs and produces a chapter tree with paragraph-aware chunks that
respect markup ``{{chapter "..."}}`` markers when present.

Pipeline role::

    raw text ──→ detect_chapters() ──→ Chapter[]
                                        │
                                        ▼
               chunk_with_paragraphs() ──→ Chunk[] (per chapter)

The chunks are then fed to the TTS pipeline (one ``synthesize`` call
per chunk, or via ``synthesize_markup`` when ``{{...}}`` is present).

Chapter detection signals (in priority order):
1. Explicit markup: ``{{chapter "Title"}}`` (handled by the markup
   parser — we detect the *positions* here for UI navigation).
2. Markdown headings: ``#``, ``##``, ``###`` (ATX style).
3. Setext headings: underlined with ``=`` or ``-``.
4. Uppercase short heading: a line in CAPS, ≤ 60 chars, followed by
   a blank line or body text. Common in Russian textbooks
   ("ГЛАВА ПЕРВАЯ", "УРОК 3. ЗАГАДКИ").

Paragraph-aware chunking:
- Split on blank lines (``\\n\\s*\\n``) → paragraphs.
- Never break inside a sentence.
- Accumulate sentences into a chunk until ``max_chars`` or
  ``max_sentences`` is reached.
- Paragraph boundary = preferred chunk boundary (flush current chunk
  even if not full, so paragraph pauses land naturally).
- Chapter boundary = hard chunk boundary (never mix two chapters in
  one chunk).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator, Optional


@dataclass
class Chapter:
    """A detected chapter in the source text.

    ``char_start`` / ``char_end`` are indices into the *original* source
    text (including markup). ``title`` is the heading text (stripped of
    ``#`` / underlines / markup braces). For the implicit "preamble"
    before the first heading, ``title`` is ``""`` and ``is_preamble``
    is ``True``.
    """

    title: str
    char_start: int
    char_end: int = 0
    is_preamble: bool = False
    # Chunks produced from this chapter's text (filled by
    # ``chunk_with_paragraphs`` when called with ``fill_chapters=True``).
    chunks: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Chapter detection
# ---------------------------------------------------------------------------

# Markdown ATX headings: 1-6 ``#`` followed by space + text.
_ATX_HEADING_RE = re.compile(
    r"^(#{1,6})\s+(.+?)\s*#*\s*$",
    re.MULTILINE,
)

# Setext headings: a line of text followed by a line of ``=`` or ``-``.
_SETEXT_HEADING_RE = re.compile(
    r"^(.+?)\n(=+|-+)\s*$",
    re.MULTILINE,
)

# ``{{chapter "..."}}`` markup — we detect the *position* here; the
# markup parser handles the actual state transition at synthesis time.
_MARKUP_CHAPTER_RE = re.compile(
    r"\{\{\s*chapter\s*[\"']([^\"']*)[\"']\s*\}\}",
    re.IGNORECASE,
)

# Uppercase short heading: a line that is mostly uppercase, ≤ 60 chars,
# and contains at least one letter. Lines like "ГЛАВА 1" or "УРОК 3."
# match; "Hello World" (mixed case) or a 200-char all-caps paragraph
# do not.
_UPPERCASE_HEADING_MAX_LEN = 60

# ``[SPEAKER0]``, ``[SPEAKER1]``, ... tags in transcript text. Mirrors
# Higgs Audio's multi-speaker format. A line starting with a SPEAKER
# tag opens a new utterance; continuation lines belong to the current
# speaker until the next tag. Defined here (before _is_uppercase_heading)
# so the heading detector can reject SPEAKER-tagged lines.
_SPEAKER_TAG_RE = re.compile(r"^\s*(\[(SPEAKER\d+)\])\s*(.*)$", re.MULTILINE)


def _is_uppercase_heading(line: str) -> bool:
    """Heuristic: is this line an uppercase short heading?

    Rejects lines that look like multi-speaker dialog tags
    (``[SPEAKER0]``) so they aren't mistaken for chapter headings —
    the speaker-chunking path handles those separately.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > _UPPERCASE_HEADING_MAX_LEN:
        return False
    # Skip multi-speaker dialog tags — they're narration, not headings.
    if _SPEAKER_TAG_RE.match(stripped):
        return False
    # Must contain at least one letter (Cyrillic or Latin).
    if not re.search(r"[а-яА-Яa-zA-ZёЁ]", stripped):
        return False
    # At least 60% of letters are uppercase.
    letters = [c for c in stripped if c.isalpha()]
    if not letters:
        return False
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters) >= 0.6


def detect_chapters(text: str) -> list[Chapter]:
    """Detect chapters in source text.

    Returns a list of :class:`Chapter` covering the entire text
    (``char_end`` of chapter N == ``char_start`` of chapter N+1). The
    first chapter may be a "preamble" (text before the first heading)
    with ``title=""`` and ``is_preamble=True``.

    Signals, in priority order (first match wins at a given position):
    1. ``{{chapter "..."}}`` markup
    2. Markdown ATX headings (``#``, ``##``, …)
    3. Setext headings (underlined with ``=`` or ``-``)
    4. Uppercase short heading lines
    """
    # Collect all heading positions as (char_start, title).
    headings: list[tuple[int, str, int]] = []  # (start, title, end)

    # 1. Markup chapters
    for m in _MARKUP_CHAPTER_RE.finditer(text):
        headings.append((m.start(), m.group(1).strip(), m.end()))

    # 2. Markdown ATX
    for m in _ATX_HEADING_RE.finditer(text):
        # Skip if this position is inside a markup chapter match
        if any(abs(m.start() - h[0]) < 3 for h in headings):
            continue
        headings.append((m.start(), m.group(2).strip(), m.end()))

    # 3. Setext
    for m in _SETEXT_HEADING_RE.finditer(text):
        if any(abs(m.start() - h[0]) < 3 for h in headings):
            continue
        title = m.group(1).strip()
        # Setext underline ``=`` → H1, ``-`` → H2. We don't distinguish
        # levels here — both are chapter boundaries.
        headings.append((m.start(), title, m.end()))

    # 4. Uppercase short headings — scan line by line.
    for m in re.finditer(r"^(.+)$", text, re.MULTILINE):
        line = m.group(0)
        if _is_uppercase_heading(line):
            if any(abs(m.start() - h[0]) < 3 for h in headings):
                continue
            # Only accept uppercase headings that are followed by a
            # blank line or different text (avoids matching all-caps
            # paragraphs that happen to be ≤ 60 chars on the first line).
            after = text[m.end():m.end() + 2]
            if after.startswith("\n\n") or after.startswith("\n") or m.end() >= len(text):
                headings.append((m.start(), line, m.end()))

    if not headings:
        # No headings → single chapter covering everything.
        return [Chapter(title="", char_start=0, char_end=len(text), is_preamble=True)]

    # Sort by position and build chapter spans.
    headings.sort(key=lambda h: h[0])

    chapters: list[Chapter] = []
    # Preamble before the first heading?
    if headings[0][0] > 0:
        preamble_text = text[: headings[0][0]]
        if preamble_text.strip():
            chapters.append(
                Chapter(
                    title="",
                    char_start=0,
                    char_end=headings[0][0],
                    is_preamble=True,
                )
            )

    for i, (start, title, end) in enumerate(headings):
        chapter_end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
        chapters.append(Chapter(title=title, char_start=start, char_end=chapter_end))

    return chapters


# ---------------------------------------------------------------------------
# Paragraph-aware chunking
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """One TTS-sized chunk of narration text.

    ``text`` is the raw narration (no markup, no chapter heading).
    ``chapter_title`` is the chapter this chunk belongs to (empty for
    preamble). ``is_chapter_start`` is True for the first chunk of a
    chapter.

    For multi-speaker dialog chunking (``chunk_method="speaker"``),
    ``speaker`` records which speaker this chunk belongs to (e.g.
    ``"SPEAKER0"``) and ``turn_index`` is the utterance ordinal within
    that speaker's turn sequence. For the default paragraph-aware
    chunking, ``speaker`` is ``""`` and ``turn_index`` is 0.
    """

    text: str
    chapter_title: str = ""
    is_chapter_start: bool = False
    char_start: int = 0  # offset in the original source
    # Multi-speaker metadata — only populated by ``chunk_method="speaker"``.
    speaker: str = ""
    turn_index: int = 0


def _split_paragraphs(text: str) -> list[tuple[int, str]]:
    """Split text into paragraphs on blank lines.

    Returns ``[(char_start, paragraph_text), ...]``. ``char_start`` is
    the index of the paragraph's first non-whitespace char in the
    original text. Whitespace-only lines between paragraphs are
    discarded.
    """
    paras: list[tuple[int, str]] = []
    # Match blocks of non-empty lines separated by blank lines.
    for m in re.finditer(r"(?:(?!^)\n\s*\n|^\s*\n)?([^\n]+(?:\n[^\n]+)*?)(?=\n\s*\n|\Z)", text):
        body = m.group(1).strip()
        if body:
            # Find the actual start of the body within the match.
            start = m.start(1) + (len(m.group(1)) - len(m.group(1).lstrip()))
            paras.append((start, body))
    return paras


# ---------------------------------------------------------------------------
# Multi-speaker dialog chunking
# ---------------------------------------------------------------------------

# ``_SPEAKER_TAG_RE`` is defined above (near _is_uppercase_heading) so
# the heading detector can reject SPEAKER-tagged lines. The multi-speaker
# chunking function below uses it.


def split_by_speakers(
    text: str,
    chunk_max_num_turns: int = 1,
    chapter_title: str = "",
    is_chapter_start: bool = False,
) -> list[Chunk]:
    """Split multi-speaker dialog text into per-utterance chunks.

    Inspired by Higgs Audio's ``prepare_chunk_text(chunk_method="speaker")``
    (``examples/generation.py``). A line starting with ``[SPEAKER0]`` or
    ``[SPEAKER1]`` opens a new utterance; following lines (until the next
    tag) belong to that speaker. Each utterance becomes a :class:`Chunk`
    tagged with the speaker name and turn index.

    ``chunk_max_num_turns > 1`` groups consecutive utterances into a
    single chunk — useful for engines that can render a short dialog
    turn-pair in one call (e.g. Higgs with multi-voice cloning). With
    ``chunk_max_num_turns=2``, two consecutive utterances are merged
    into one chunk; ``chunk_max_num_turns=1`` (default) keeps them
    separate.

    Lines without any SPEAKER tag before the first tag are treated as
    a preamble chunk (speaker="" / turn_index=-1) — they're narration
    outside the dialog.
    """
    chunks: list[Chunk] = []
    # Walk the text line by line, accumulating utterances.
    speaker_utterances: list[tuple[str, str, int]] = []  # (speaker, text, char_start)
    # Preamble (text before the first SPEAKER tag).
    preamble_lines: list[str] = []
    preamble_start = 0
    first_tag_pos = _SPEAKER_TAG_RE.search(text)
    if first_tag_pos and first_tag_pos.start() > 0:
        preamble_text = text[: first_tag_pos.start()].strip()
        if preamble_text:
            chunks.append(
                Chunk(
                    text=preamble_text,
                    chapter_title=chapter_title,
                    is_chapter_start=is_chapter_start,
                    char_start=0,
                    speaker="",
                    turn_index=-1,
                )
            )
        first_chunk_pending = False
    else:
        first_chunk_pending = is_chapter_start

    current_speaker = ""
    current_utterance_lines: list[str] = []
    current_utterance_start = 0
    turn_counter: dict[str, int] = {}

    def _flush_utterance(start_offset: int) -> None:
        nonlocal current_utterance_lines, current_speaker, first_chunk_pending
        if current_utterance_lines and current_speaker:
            body = " ".join(line.strip() for line in current_utterance_lines if line.strip())
            if body:
                speaker_utterances.append(
                    (current_speaker, body, start_offset, turn_counter[current_speaker])
                )
        current_utterance_lines = []

    # Re-scan with char offsets.
    for m in _SPEAKER_TAG_RE.finditer(text):
        # Flush the previous utterance before starting a new one.
        if current_speaker:
            _flush_utterance(current_utterance_start)
        current_speaker = m.group(2)  # "SPEAKER0" without brackets
        turn_counter.setdefault(current_speaker, 0)
        current_utterance_start = m.start()
        # The rest of the tag line (after the tag) is the start of the utterance.
        rest_of_line = m.group(3).strip()
        current_utterance_lines = [rest_of_line] if rest_of_line else []
        # Consume continuation lines until the next tag or end.
        # We do this by scanning forward from m.end() to the next tag.
        next_tag = _SPEAKER_TAG_RE.search(text, m.end())
        continuation_end = next_tag.start() if next_tag else len(text)
        continuation = text[m.end():continuation_end]
        for line in continuation.split("\n"):
            line = line.strip()
            if line:
                current_utterance_lines.append(line)
        turn_counter[current_speaker] += 1

    # Flush the last utterance.
    if current_speaker:
        _flush_utterance(current_utterance_start)

    # Group utterances by ``chunk_max_num_turns``.
    if chunk_max_num_turns <= 1:
        for spk, body, start, turn_idx in speaker_utterances:
            chunks.append(
                Chunk(
                    text=f"[{spk}] {body}",
                    chapter_title=chapter_title,
                    is_chapter_start=first_chunk_pending,
                    char_start=start,
                    speaker=spk,
                    turn_index=turn_idx,
                )
            )
            first_chunk_pending = False
    else:
        # Merge ``chunk_max_num_turns`` consecutive utterances into one chunk.
        for i in range(0, len(speaker_utterances), chunk_max_num_turns):
            group = speaker_utterances[i : i + chunk_max_num_turns]
            if not group:
                continue
            merged_lines = [f"[{spk}] {body}" for spk, body, _, _ in group]
            merged_text = "\n".join(merged_lines)
            chunks.append(
                Chunk(
                    text=merged_text,
                    chapter_title=chapter_title,
                    is_chapter_start=first_chunk_pending,
                    char_start=group[0][2],
                    speaker=group[0][0],  # primary speaker of the group
                    turn_index=group[0][3],
                )
            )
            first_chunk_pending = False

    # Edge case: no SPEAKER tags found at all → fall back to a single chunk.
    if not chunks and text.strip():
        chunks.append(
            Chunk(
                text=text.strip(),
                chapter_title=chapter_title,
                is_chapter_start=is_chapter_start,
                char_start=0,
            )
        )

    return chunks


# ---------------------------------------------------------------------------
# Paragraph-aware chunking
# ---------------------------------------------------------------------------


def chunk_with_paragraphs(
    text: str,
    max_chars: int = 200,
    max_sentences: int = 4,
    chapter_title: str = "",
    is_chapter_start: bool = False,
    chunk_method: str = "paragraph",
    chunk_max_num_turns: int = 1,
) -> list[Chunk]:
    """Split ``text`` into TTS-safe chunks respecting paragraph boundaries.

    ``chunk_method`` selects the algorithm:
    - ``"paragraph"`` (default): paragraph-aware sentence accumulation.
      Splits on blank lines, accumulates sentences until ``max_chars``
      or ``max_sentences``, flushes at paragraph boundaries.
    - ``"speaker"``: multi-speaker dialog chunking via
      :func:`split_by_speakers`. Detects ``[SPEAKER0]``/``[SPEAKER1]``
      tags, splits by utterance, optionally groups via
      ``chunk_max_num_turns``. Each chunk is tagged with ``.speaker``
      and ``.turn_index``.

    Paragraph algorithm:
    1. Split into paragraphs (blank-line separated).
    2. For each paragraph, split into sentences (reusing
       ``utils.text_utils.split_into_sentences``).
    3. Accumulate sentences into a chunk until ``max_chars`` or
       ``max_sentences`` is hit.
    4. At a paragraph boundary, *flush* the current chunk even if it
       isn't full — so paragraph pauses land naturally between chunks
       rather than mid-chunk.
    5. A single sentence longer than ``max_chars`` is hard-split via
       the existing ``chunk_text_for_tts`` fallback (comma/word split).
    """
    if chunk_method == "speaker":
        return split_by_speakers(
            text,
            chunk_max_num_turns=chunk_max_num_turns,
            chapter_title=chapter_title,
            is_chapter_start=is_chapter_start,
        )

    from ..utils.text_utils import chunk_text_for_tts, split_into_sentences

    chunks: list[Chunk] = []
    paras = _split_paragraphs(text)

    current_sentences: list[str] = []
    current_len = 0
    current_start = 0
    # Mutable flag: ``is_chapter_start`` is True only for the first
    # chunk emitted from this call. We can't reassign the function
    # parameter, so we use a separate local that _flush can clear.
    first_chunk_pending = is_chapter_start

    def _flush() -> None:
        nonlocal current_sentences, current_len, current_start, first_chunk_pending
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    chapter_title=chapter_title,
                    is_chapter_start=first_chunk_pending,
                    char_start=current_start,
                )
            )
            first_chunk_pending = False
            current_sentences = []
            current_len = 0

    for para_start, para_text in paras:
        sentences = split_into_sentences(para_text)

        for sentence in sentences:
            sent_len = len(sentence)

            # Overlong single sentence → hard-split and flush as its own chunk(s).
            if sent_len > max_chars:
                _flush()
                sub_chunks = chunk_text_for_tts(sentence, max_chars=max_chars, max_sentences=1)
                for sc in sub_chunks:
                    chunks.append(
                        Chunk(
                            text=sc,
                            chapter_title=chapter_title,
                            is_chapter_start=first_chunk_pending,
                            char_start=para_start,
                        )
                    )
                    first_chunk_pending = False
                continue

            if (current_len + sent_len + 1 > max_chars or len(current_sentences) >= max_sentences) and current_sentences:
                _flush()

            if not current_sentences:
                current_start = para_start
            current_sentences.append(sentence)
            current_len += sent_len + 1

        # Paragraph boundary → flush (natural pause point).
        _flush()

    # Edge case: no paragraphs detected (single-line input).
    if not chunks and text.strip():
        chunks.append(
            Chunk(
                text=text.strip(),
                chapter_title=chapter_title,
                is_chapter_start=first_chunk_pending,
                char_start=0,
            )
        )

    return chunks


def chunk_document(
    text: str,
    max_chars: int = 200,
    max_sentences: int = 4,
    chunk_method: str = "auto",
    chunk_max_num_turns: int = 1,
) -> list[Chunk]:
    """Detect chapters, then chunk each chapter's body.

    This is the top-level entry point for long-form text: feed it a
    whole book / lesson / article and get back a flat list of chunks
    with chapter metadata. Each chunk is safe to pass to a single
    ``synthesize`` call.

    Chapter heading lines (``#``, ``{{chapter ...}}``, uppercase) are
    *stripped* from the chunk text — they're metadata, not narration.

    ``chunk_method``:
    - ``"auto"`` (default): auto-detect. If the text contains
      ``[SPEAKER0]``/``[SPEAKER1]`` tags → use ``"speaker"``; otherwise
      ``"paragraph"``. This lets callers pass dialog-or-narration text
      without choosing the method themselves.
    - ``"paragraph"``: paragraph-aware sentence accumulation.
    - ``"speaker"``: multi-speaker dialog chunking (see
      :func:`split_by_speakers`).

    ``chunk_max_num_turns`` is forwarded to the speaker method only —
    it groups that many consecutive utterances into one chunk.
    """
    # Auto-detect: speaker tags anywhere in the text → speaker method.
    if chunk_method == "auto":
        if _SPEAKER_TAG_RE.search(text):
            chunk_method = "speaker"
        else:
            chunk_method = "paragraph"

    chapters = detect_chapters(text)
    all_chunks: list[Chunk] = []

    for ch in chapters:
        # Extract the body text, stripping the heading line(s).
        body = text[ch.char_start:ch.char_end]
        if not ch.is_preamble and ch.title:
            # Remove the first line if it's the heading (ATX/setext/
            # uppercase). Markup ``{{chapter}}`` is handled separately
            # by the markup parser — we just remove the markup block.
            body = _strip_heading(body, ch.title)

        chunks = chunk_with_paragraphs(
            body,
            max_chars=max_chars,
            max_sentences=max_sentences,
            chapter_title=ch.title,
            is_chapter_start=not ch.is_preamble,
            chunk_method=chunk_method,
            chunk_max_num_turns=chunk_max_num_turns,
        )
        all_chunks.extend(chunks)

    return all_chunks


def _strip_heading(body: str, title: str) -> str:
    """Remove the heading line(s) from the start of ``body``.

    Handles ATX (``# Title``), setext (``Title\\n===``), uppercase
    (``TITLE``), and markup (``{{chapter "TITLE"}}``). Falls back to
    removing the first line if it contains the title.
    """
    # Markup chapter block
    body = re.sub(r"^\s*\{\{\s*chapter\s*[\"'][^\"']*[\"']\s*\}\}\s*\n?", "", body, count=1)
    # ATX heading
    body = re.sub(r"^\s*#{1,6}\s+" + re.escape(title) + r"\s*#*\s*\n?", "", body, count=1)
    # Setext heading (title + underline)
    body = re.sub(r"^\s*" + re.escape(title) + r"\s*\n[=]+\s*\n?", "", body, count=1, flags=re.MULTILINE)
    body = re.sub(r"^\s*" + re.escape(title) + r"\s*\n[-]+\s*\n?", "", body, count=1, flags=re.MULTILINE)
    # Uppercase heading (exact line match)
    body = re.sub(r"^\s*" + re.escape(title) + r"\s*\n?", "", body, count=1)
    return body