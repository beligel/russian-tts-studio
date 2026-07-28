"""Long-form text processing for RTTS.

Phase 2 (LTV-inspired):
- Chapter detection (Markdown headings, uppercase short headings,
  ``{{chapter "..."}}`` markup).
- Paragraph-aware chunking that respects chapter and paragraph
  boundaries.
- Multi-speaker dialog chunking (``[SPEAKER0]``/``[SPEAKER1]`` tags)
  inspired by Higgs Audio's ``prepare_chunk_text``.
- File importers: .txt, .md, .docx.

Public API::

    from russian_tts_studio.text import (
        detect_chapters, chunk_document, chunk_with_paragraphs,
        split_by_speakers, import_file, Chapter, Chunk,
    )

    chapters = detect_chapters(text)
    chunks = chunk_document(text)
"""

from __future__ import annotations

from .importers import import_docx, import_file, import_md, import_txt
from .longform import (
    Chapter,
    Chunk,
    chunk_document,
    chunk_with_paragraphs,
    detect_chapters,
    split_by_speakers,
)

__all__ = [
    "detect_chapters",
    "chunk_document",
    "chunk_with_paragraphs",
    "split_by_speakers",
    "import_file",
    "import_txt",
    "import_md",
    "import_docx",
    "Chapter",
    "Chunk",
]