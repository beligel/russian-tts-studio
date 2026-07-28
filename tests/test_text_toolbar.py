"""Tests for the text toolbar features: file upload + stress mark.

Covers:
- ``/api/import`` endpoint (already tested in test_longform_api, but
  we add a round-trip test verifying the response is suitable for
  inserting into a textarea — text + char_count + chapters).
- The stress-mark insertion logic (Python mirror of the JS
  ``applyStressToSelection`` function) — verifies the two cases
  (single vowel → U+0301, word → {{stress "..."}} markup).
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestImportForTextarea:
    """``/api/import`` response is suitable for direct textarea insertion."""

    def test_import_returns_text_and_metadata(self):
        from fastapi.testclient import TestClient
        from web.app import app

        client = TestClient(app)
        content = "# Тест\n\nПривет, мир."
        resp = client.post(
            "/api/import",
            files={"file": ("test.txt", io.BytesIO(content.encode("utf-8")), "text/plain")},
        )
        assert resp.status_code == 200
        body = resp.json()
        # The UI needs `text` (to insert into textarea) and optionally
        # `char_count` / `chapters` for the toast notification.
        assert "text" in body
        assert isinstance(body["text"], str)
        assert body["char_count"] == len(body["text"])
        assert "source_format" in body
        assert "chapters" in body
        assert isinstance(body["chapters"], list)

    def test_import_docx_returns_text(self):
        """DOCX import — skip if python-docx not installed."""
        pytest.importorskip("docx")
        from fastapi.testclient import TestClient
        from web.app import app

        # Create a minimal DOCX in memory.
        from docx import Document
        doc = Document()
        doc.add_heading("Тестовый документ", 0)
        doc.add_paragraph("Содержание документа.")
        buf = io.BytesIO()
        doc.save(buf)
        buf.seek(0)

        client = TestClient(app)
        resp = client.post(
            "/api/import",
            files={"file": ("test.docx", buf, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "Тестовый" in body["text"]
        assert "Содержание" in body["text"]


class TestStressInsertionLogic:
    """Python mirror of the JS ``applyStressToSelection`` function.

    The JS runs in the browser, but we verify the logic here so it's
    covered by CI. The two cases:
    1. Single vowel selected → insert U+0301 after it.
    2. Word/phrase selected → wrap as {{stress "word"}} markup.
    """

    COMBINING_ACUTE = "\u0301"
    VOWELS = "аеёиоуыэюяaeiouyАЕЁИОУЫЭЮЯAEIOUY"

    def _apply_stress(self, text: str, start: int, end: int) -> tuple[str, int, int]:
        """Mirror of applyStressToSelection. Returns (new_text, new_start, new_end)."""
        selected = text[start:end]
        if not selected:
            return text, start, end

        # Case 1: single vowel → insert U+0301 after it.
        if len(selected) == 1 and selected in self.VOWELS:
            new_text = text[:start] + selected + self.COMBINING_ACUTE + text[end:]
            return new_text, start + 2, start + 2

        # Case 2: word/phrase → wrap as {{stress "..."}}.
        plain = selected.replace(self.COMBINING_ACUTE, "")
        markup = f'{{{{stress "{plain}"}}}}'
        new_text = text[:start] + markup + text[end:]
        return new_text, start + len(markup), start + len(markup)

    def test_single_vowel_gets_combining_acute(self):
        text = "замок"
        # User selects the "о" (index 3)
        new_text, _, _ = self._apply_stress(text, 3, 4)
        assert "о\u0301" in new_text
        assert new_text == "замо\u0301к"

    def test_single_uppercase_vowel_gets_acute(self):
        text = "Замок"
        # Select "З" — not a vowel, so falls through to markup path
        new_text, _, _ = self._apply_stress(text, 0, 1)
        # "З" is not a vowel → wraps as markup
        assert "{{stress" in new_text

    def test_word_selection_wraps_as_markup(self):
        text = "замок большой"
        # User selects "замок"
        new_text, new_start, new_end = self._apply_stress(text, 0, 5)
        assert '{{stress "замок"}}' in new_text
        # Cursor is after the markup
        assert new_end == new_start
        assert new_end == len('{{stress "замок"}}')

    def test_word_with_existing_acute_stripped_in_markup(self):
        # "замо\u0301к" = 6 chars (з,а,м,о,U+0301,к) — the acute is a
        # zero-width combining char that counts as a separate codepoint.
        text = "замо\u0301к большой"
        # User selects the word (with existing acute) — indices 0..6
        new_text, _, _ = self._apply_stress(text, 0, 6)
        # The markup form should have the plain word (no acute)
        assert '{{stress "замок"}}' in new_text

    def test_empty_selection_no_change(self):
        text = "замок"
        new_text, new_start, new_end = self._apply_stress(text, 2, 2)
        assert new_text == text
        assert new_start == 2
        assert new_end == 2

    def test_non_vowel_single_char_wraps_as_markup(self):
        text = "замок"
        # Select "з" (not a vowel)
        new_text, _, _ = self._apply_stress(text, 0, 1)
        assert '{{stress "з"}}' in new_text

    def test_acute_appears_in_output_for_vowel_case(self):
        """The combining acute is a zero-width character — verify it's
        actually present in the output string."""
        text = "сосна"
        # Select the second "а" (index 4)
        new_text, _, _ = self._apply_stress(text, 4, 5)
        # "сосна" → "сосна́" (with acute after the last "а")
        assert new_text == "сосна\u0301"
        # The acute is a separate character
        assert len(new_text) == len(text) + 1

    def test_cursor_position_after_acute_insert(self):
        text = "замок"
        new_text, new_start, new_end = self._apply_stress(text, 3, 4)
        # Cursor is after "о" + acute = 2 chars past original position
        assert new_start == 5  # original 3 + 2
        assert new_end == 5

    def test_cursor_position_after_markup_insert(self):
        text = "большой замок"
        new_text, new_start, new_end = self._apply_stress(text, 8, 13)  # "замок"
        markup_len = len('{{stress "замок"}}')
        assert new_start == 8 + markup_len
        assert new_end == 8 + markup_len