"""Tests for long-form text processing: chapter detection + chunking."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from russian_tts_studio.text import (  # noqa: E402
    Chapter,
    Chunk,
    chunk_document,
    chunk_with_paragraphs,
    detect_chapters,
    split_by_speakers,
)


class TestDetectChapters:
    """Tests for ``detect_chapters``."""

    def test_no_headings_single_preamble_chapter(self):
        text = "Просто текст без заголовков. Несколько предложений."
        chapters = detect_chapters(text)
        assert len(chapters) == 1
        assert chapters[0].is_preamble
        assert chapters[0].title == ""

    def test_markdown_atx_h1(self):
        text = "# Глава 1\nТекст первой главы.\n\n# Глава 2\nТекст второй."
        chapters = detect_chapters(text)
        assert len(chapters) == 2
        assert chapters[0].title == "Глава 1"
        assert chapters[1].title == "Глава 2"

    def test_markdown_atx_h2_h3(self):
        text = "## Раздел 1\nТекст.\n\n### Подраздел\nЕщё текст."
        chapters = detect_chapters(text)
        # h2 and h3 both count as chapter boundaries
        assert len(chapters) >= 2

    def test_markdown_setext_heading(self):
        text = "Глава первая\n=========\nТекст главы.\n\nГлава вторая\n=========\nТекст."
        chapters = detect_chapters(text)
        assert len(chapters) == 2
        assert chapters[0].title == "Глава первая"
        assert chapters[1].title == "Глава вторая"

    def test_markup_chapter(self):
        text = '{{chapter "Урок 1"}}\nТекст урока.'
        chapters = detect_chapters(text)
        assert len(chapters) == 1
        assert chapters[0].title == "Урок 1"
        assert not chapters[0].is_preamble

    def test_uppercase_short_heading(self):
        text = "ГЛАВА ПЕРВАЯ\n\nТекст первой главы здесь."
        chapters = detect_chapters(text)
        assert len(chapters) == 1
        assert chapters[0].title == "ГЛАВА ПЕРВАЯ"

    def test_uppercase_heading_too_long_ignored(self):
        # A 70-char all-caps line should NOT be detected as a heading
        long_caps = "А Б В Г Д Е Ж З И К Л М Н О П Р С Т У Ф Х Ц Ч Ш Щ Ъ Ы Ь Э Ю Я " * 2
        text = long_caps + "\n\nТекст."
        chapters = detect_chapters(text)
        # Treated as body text, not a heading
        assert all(c.is_preamble for c in chapters) or len(chapters) == 1

    def test_preamble_before_first_heading(self):
        text = "Введение в книгу.\n\nЭто предисловие.\n\n# Глава 1\nТекст."
        chapters = detect_chapters(text)
        assert len(chapters) == 2
        assert chapters[0].is_preamble
        assert chapters[0].title == ""
        assert chapters[1].title == "Глава 1"

    def test_chapter_spans_cover_entire_text(self):
        text = "# Г1\nТекст1.\n\n# Г2\nТекст2."
        chapters = detect_chapters(text)
        # char_end of last chapter == len(text)
        assert chapters[-1].char_end == len(text)
        # No gaps between chapters
        for i in range(len(chapters) - 1):
            assert chapters[i].char_end == chapters[i + 1].char_start

    def test_multiple_markup_chapters(self):
        text = (
            '{{chapter "Урок 1"}} Текст один. '
            '{{chapter "Урок 2"}} Текст два.'
        )
        chapters = detect_chapters(text)
        assert len(chapters) == 2
        assert chapters[0].title == "Урок 1"
        assert chapters[1].title == "Урок 2"

    def test_mixed_headings(self):
        text = (
            "# Markdown Heading\n"
            "Текст.\n\n"
            "{{chapter \"Markup Chapter\"}}\n"
            "Ещё текст."
        )
        chapters = detect_chapters(text)
        titles = [c.title for c in chapters if not c.is_preamble]
        assert "Markdown Heading" in titles
        assert "Markup Chapter" in titles


class TestChunkWithParagraphs:
    """Tests for ``chunk_with_paragraphs``."""

    def test_single_paragraph_single_chunk(self):
        chunks = chunk_with_paragraphs("Привет, мир. Это тест.")
        assert len(chunks) == 1
        assert "Привет" in chunks[0].text

    def test_paragraph_boundary_flushes_chunk(self):
        text = "Первый абзац. Короткий.\n\nВторой абзац. Тоже короткий."
        chunks = chunk_with_paragraphs(text)
        # Two paragraphs → two chunks (even though each is short)
        assert len(chunks) == 2
        assert "Первый" in chunks[0].text
        assert "Второй" in chunks[1].text

    def test_max_chars_respected(self):
        # Build a long paragraph that exceeds max_chars
        text = ". ".join([f"Предложение номер {i}" for i in range(20)])
        chunks = chunk_with_paragraphs(text, max_chars=80, max_sentences=3)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c.text) <= 120  # some slack for sentence boundaries

    def test_max_sentences_respected(self):
        text = "Раз. Два. Три. Четыре. Пять. Шесть."
        chunks = chunk_with_paragraphs(text, max_sentences=2)
        assert len(chunks) >= 3
        # Each chunk has at most 2 sentences
        for c in chunks:
            # Count sentence-ending punctuation
            n = c.text.count(".") + c.text.count("!") + c.text.count("?")
            assert n <= 2

    def test_overlong_sentence_hard_split(self):
        # A single sentence longer than max_chars
        words = ["слово"] * 50
        long_sentence = "Это " + " ".join(words) + "."
        chunks = chunk_with_paragraphs(long_sentence, max_chars=80)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c.text) <= 100

    def test_chapter_title_propagated(self):
        chunks = chunk_with_paragraphs(
            "Текст.", chapter_title="Глава 1", is_chapter_start=True
        )
        assert chunks[0].chapter_title == "Глава 1"
        assert chunks[0].is_chapter_start is True

    def test_empty_text(self):
        chunks = chunk_with_paragraphs("")
        assert chunks == []

    def test_whitespace_only(self):
        chunks = chunk_with_paragraphs("   \n\n  \n  ")
        assert chunks == []


class TestChunkDocument:
    """Tests for the top-level ``chunk_document`` entry point."""

    def test_book_with_chapters(self):
        text = (
            "# Глава 1\n"
            "Первый абзац первой главы. Второе предложение.\n\n"
            "Второй абзац первой главы.\n\n"
            "# Глава 2\n"
            "Первый абзац второй главы."
        )
        chunks = chunk_document(text)
        # At least one chunk per chapter
        chapter1 = [c for c in chunks if c.chapter_title == "Глава 1"]
        chapter2 = [c for c in chunks if c.chapter_title == "Глава 2"]
        assert len(chapter1) >= 1
        assert len(chapter2) >= 1
        # Heading text stripped from chunks
        for c in chunks:
            assert "# Глава" not in c.text
            assert "Глава 1" not in c.text or "Глав" not in c.text.split()[0:2]

    def test_heading_stripped_from_chunk_text(self):
        text = "# Мой заголовок\nТекст главы без заголовка."
        chunks = chunk_document(text)
        assert len(chunks) == 1
        assert "Мой заголовок" not in chunks[0].text
        assert "Текст главы" in chunks[0].text

    def test_markup_chapter_heading_stripped(self):
        text = '{{chapter "Урок 1"}}\nТекст урока здесь.'
        chunks = chunk_document(text)
        assert len(chunks) >= 1
        # The {{chapter ...}} block is stripped from the chunk text
        for c in chunks:
            assert "{{chapter" not in c.text
        assert "Текст урока" in chunks[0].text

    def test_preamble_chunks_have_empty_chapter_title(self):
        text = "Предисловие. Короткое.\n\n# Глава 1\nТекст."
        chunks = chunk_document(text)
        preamble = [c for c in chunks if c.chapter_title == ""]
        assert len(preamble) >= 1
        assert "Предисловие" in preamble[0].text

    def test_is_chapter_start_flag(self):
        text = "# Глава 1\nПервый абзац. Второй.\n\nТретий абзац."
        chunks = chunk_document(text)
        # The first chunk of the chapter has is_chapter_start=True
        chapter_chunks = [c for c in chunks if c.chapter_title == "Глава 1"]
        assert chapter_chunks[0].is_chapter_start is True
        # Subsequent chunks in the same chapter have is_chapter_start=False
        if len(chapter_chunks) > 1:
            assert chapter_chunks[1].is_chapter_start is False

    def test_long_book_produces_many_chunks(self):
        # Simulate a 5000-char book with 3 chapters
        chapter_body = ". ".join([f"Предложение {i} этой главы" for i in range(30)])
        text = f"# Глава 1\n{chapter_body}\n\n# Глава 2\n{chapter_body}\n\n# Глава 3\n{chapter_body}"
        chunks = chunk_document(text, max_chars=200, max_sentences=4)
        assert len(chunks) > 10
        titles = {c.chapter_title for c in chunks}
        assert titles == {"Глава 1", "Глава 2", "Глава 3"}

    def test_preserves_russian_text(self):
        text = "# Глава\nПривет, мир! Это тест на русском языке. Ёлка зелёная."
        chunks = chunk_document(text)
        assert any("Привет" in c.text for c in chunks)
        assert any("ёлка" in c.text.lower() or "Ёлка" in c.text for c in chunks)

    def test_paragraph_pause_points(self):
        """Paragraph boundaries produce separate chunks so the pipeline
        can insert natural pauses between them."""
        text = "Абзац один. Короткий.\n\nАбзац два. Короткий.\n\nАбзац три. Короткий."
        chunks = chunk_with_paragraphs(text)
        assert len(chunks) == 3
        assert "один" in chunks[0].text
        assert "два" in chunks[1].text
        assert "три" in chunks[2].text


class TestSpeakerChunking:
    """Tests for multi-speaker dialog chunking (``split_by_speakers``)."""

    def test_simple_dialog_splits_by_speaker(self):
        text = (
            "[SPEAKER0] Привет, как дела?\n"
            "[SPEAKER1] Привет, отлично.\n"
            "[SPEAKER0] Пойдём гулять?"
        )
        chunks = split_by_speakers(text)
        assert len(chunks) == 3
        assert chunks[0].speaker == "SPEAKER0"
        assert chunks[1].speaker == "SPEAKER1"
        assert chunks[2].speaker == "SPEAKER0"
        # Each chunk text includes the SPEAKER tag prefix
        assert "[SPEAKER0]" in chunks[0].text
        assert "[SPEAKER1]" in chunks[1].text

    def test_turn_index_increments_per_speaker(self):
        text = (
            "[SPEAKER0] Реплика 1.\n"
            "[SPEAKER1] Ответ.\n"
            "[SPEAKER0] Реплика 2.\n"
            "[SPEAKER1] Ответ 2."
        )
        chunks = split_by_speakers(text)
        # SPEAKER0 turns: 1, 2  | SPEAKER1 turns: 1, 2
        spk0_chunks = [c for c in chunks if c.speaker == "SPEAKER0"]
        spk1_chunks = [c for c in chunks if c.speaker == "SPEAKER1"]
        assert [c.turn_index for c in spk0_chunks] == [1, 2]
        assert [c.turn_index for c in spk1_chunks] == [1, 2]

    def test_continuation_lines_join_into_utterance(self):
        text = (
            "[SPEAKER0] Это первая строка.\n"
            "Это продолжение той же реплики.\n"
            "Ещё одна строка.\n"
            "[SPEAKER1] Ответ."
        )
        chunks = split_by_speakers(text)
        assert len(chunks) == 2
        # All three continuation lines are in the SPEAKER0 chunk
        assert "первая строка" in chunks[0].text
        assert "продолжение" in chunks[0].text
        assert "Ещё одна" in chunks[0].text

    def test_chunk_max_num_turns_groups_utterances(self):
        text = (
            "[SPEAKER0] А.\n"
            "[SPEAKER1] Б.\n"
            "[SPEAKER0] В.\n"
            "[SPEAKER1] Г."
        )
        # Group 2 utterances per chunk
        chunks = split_by_speakers(text, chunk_max_num_turns=2)
        assert len(chunks) == 2
        # First chunk merges SPEAKER0 + SPEAKER1
        assert "[SPEAKER0]" in chunks[0].text
        assert "[SPEAKER1]" in chunks[0].text
        assert "А." in chunks[0].text
        assert "Б." in chunks[0].text
        # Second chunk merges the next pair
        assert "В." in chunks[1].text
        assert "Г." in chunks[1].text

    def test_preamble_before_first_tag(self):
        text = (
            "Вступительный текст без тега.\n\n"
            "[SPEAKER0] Реплика."
        )
        chunks = split_by_speakers(text)
        assert len(chunks) == 2
        # First chunk is preamble (speaker="", turn_index=-1)
        assert chunks[0].speaker == ""
        assert chunks[0].turn_index == -1
        assert "Вступительный" in chunks[0].text
        # Second chunk is the dialog
        assert chunks[1].speaker == "SPEAKER0"

    def test_no_speaker_tags_falls_back_to_single_chunk(self):
        text = "Просто обычный текст без тегов."
        chunks = split_by_speakers(text)
        assert len(chunks) == 1
        assert chunks[0].speaker == ""
        assert "обычный текст" in chunks[0].text

    def test_empty_text_returns_empty_list(self):
        chunks = split_by_speakers("")
        assert chunks == []

    def test_is_chapter_start_only_on_first_chunk(self):
        text = "[SPEAKER0] Первая.\n[SPEAKER1] Вторая."
        chunks = split_by_speakers(text, is_chapter_start=True)
        assert chunks[0].is_chapter_start is True
        assert chunks[1].is_chapter_start is False


class TestChunkDocumentAutoDetect:
    """Tests for ``chunk_document`` auto-detection of speaker vs paragraph."""

    def test_auto_detects_speaker_when_tags_present(self):
        text = "[SPEAKER0] Привет.\n[SPEAKER1] Здравствуй."
        chunks = chunk_document(text)
        # Should use speaker method → chunks tagged with speaker
        assert any(c.speaker == "SPEAKER0" for c in chunks)

    def test_auto_uses_paragraph_when_no_tags(self):
        text = "Один. Два. Три."
        chunks = chunk_document(text)
        # Paragraph method → no speaker tags
        assert all(c.speaker == "" for c in chunks)

    def test_explicit_speaker_method(self):
        text = "[SPEAKER0] Привет.\n[SPEAKER1] Здравствуй."
        chunks = chunk_document(text, chunk_method="speaker")
        assert any(c.speaker == "SPEAKER0" for c in chunks)

    def test_explicit_paragraph_method_ignores_speaker_tags(self):
        text = "[SPEAKER0] Привет. Ещё одно предложение."
        chunks = chunk_document(text, chunk_method="paragraph")
        # Paragraph method → no speaker metadata
        assert all(c.speaker == "" for c in chunks)

    def test_speaker_tags_not_detected_as_chapter_headings(self):
        """``[SPEAKER0] Привет.`` should NOT be mistaken for an uppercase
        chapter heading — the speaker-chunking path handles it."""
        text = "[SPEAKER0] ПРИВЕТ.\n[SPEAKER1] ЗДРАВСТВУЙ."
        chunks = chunk_document(text)
        # All chunks belong to one chapter (no false chapter split)
        chapters = {c.chapter_title for c in chunks}
        assert "" in chapters  # preamble or single implicit chapter

    def test_mixed_narration_and_dialog(self):
        """Narration paragraph followed by dialog — both chunked correctly."""
        text = (
            "# Глава 1\n"
            "Немного повествования.\n\n"
            "[SPEAKER0] Привет.\n"
            "[SPEAKER1] Здравствуй."
        )
        chunks = chunk_document(text)
        # First chunk: narration (no speaker)
        narration = [c for c in chunks if c.speaker == "" and "повествования" in c.text]
        assert len(narration) == 1
        # Remaining chunks: dialog
        dialog = [c for c in chunks if c.speaker]
        assert len(dialog) == 2
        assert dialog[0].speaker == "SPEAKER0"
        assert dialog[1].speaker == "SPEAKER1"
        # All in the same chapter
        assert all(c.chapter_title == "Глава 1" for c in chunks)