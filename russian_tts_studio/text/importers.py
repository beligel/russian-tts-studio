"""File importers for long-form text: .txt, .md, .docx.

Each importer returns ``(text, source_format)`` where ``text`` is the
raw string content ready for ``detect_chapters`` / ``chunk_document``,
and ``source_format`` is one of ``"txt"`` / ``"md"`` / ``"docx"``.

DOCX import is optional — it requires ``python-docx`` which is not in
the minimal requirements. If the package is missing, ``import_docx``
raises :class:`ImportError` with a helpful pip hint, mirroring the
``num2words`` pattern in ``utils.text_utils.normalize_numbers``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

SourceFormat = Literal["txt", "md", "docx"]


def import_txt(path: str | Path) -> tuple[str, SourceFormat]:
    """Read a plain-text file (UTF-8, falls back to cp1251 for legacy RU)."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Legacy Russian .txt files are often in cp1251.
        text = p.read_text(encoding="cp1251", errors="replace")
    return text, "txt"


def import_md(path: str | Path) -> tuple[str, SourceFormat]:
    """Read a Markdown file as raw text (no rendering — we detect
    ``#`` headings downstream in ``detect_chapters``)."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    return text, "md"


def import_docx(path: str | Path) -> tuple[str, SourceFormat]:
    """Extract text from a Word .docx file.

    Requires ``python-docx``: ``pip install python-docx``.
    Paragraphs are joined with double newlines so ``chunk_document``
    sees them as paragraph boundaries.
    """
    try:
        from docx import Document  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "DOCX import requires python-docx. Install with:\n"
            "  pip install python-docx\n"
            f"Underlying error: {e}"
        ) from e

    doc = Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    text = "\n\n".join(paragraphs)
    return text, "docx"


def import_file(path: str | Path) -> tuple[str, SourceFormat]:
    """Auto-dispatch by file extension.

    Supported: ``.txt``, ``.md``, ``.markdown``, ``.docx``.
    Raises :class:`ValueError` for unknown extensions.
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".txt":
        return import_txt(p)
    if ext in (".md", ".markdown"):
        return import_md(p)
    if ext == ".docx":
        return import_docx(p)
    raise ValueError(
        f"Unsupported file extension: {ext!r}. "
        f"Supported: .txt, .md, .markdown, .docx"
    )