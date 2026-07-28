"""Tests for text normalization dictionaries and Whisper review."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestNormalization:
    """Tests for the normalization engine."""

    def test_abbreviations_expansion(self):
        from russian_tts_studio.text.normalization import normalize_for_tts, NormalizationConfig

        cfg = NormalizationConfig()
        result = normalize_for_tts("Он сказал и т.д. и т.п.", cfg)
        assert "и так далее" in result
        assert "и тому подобное" in result

    def test_address_expansion(self, tmp_path):
        from russian_tts_studio.text.normalization import normalize_for_tts, NormalizationConfig
        import russian_tts_studio.text.normalization as norm_mod
        import sqlite3

        # Fresh DB so the fixed default dictionaries are loaded
        db_path = tmp_path / "test_norm.db"
        conn = sqlite3.connect(str(db_path), timeout=10, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS dictionaries (
                id TEXT PRIMARY KEY, name TEXT, description TEXT DEFAULT '',
                enabled INTEGER DEFAULT 1, created_at TEXT DEFAULT '', updated_at TEXT DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT, dict_id TEXT,
                source TEXT, replacement TEXT, enabled INTEGER DEFAULT 1,
                sort_order INTEGER DEFAULT 0, UNIQUE(dict_id, source)
            );
        """)
        orig = norm_mod.get_db
        norm_mod.get_db = lambda *a, **kw: conn
        try:
            cfg = NormalizationConfig()
            result = normalize_for_tts("ул. Пушкина, д. 10", cfg, output_dir=tmp_path)
            assert "улица" in result
            assert "дом" in result
        finally:
            norm_mod.get_db = orig
            conn.close()

    def test_disabled_config_noop(self):
        from russian_tts_studio.text.normalization import normalize_for_tts, NormalizationConfig

        cfg = NormalizationConfig(enabled=False)
        text = "Он сказал и т.д."
        result = normalize_for_tts(text, cfg)
        assert result == text

    def test_longest_match_first(self):
        """'и т.д.' should match before 'т.д.'"""
        from russian_tts_studio.text.normalization import normalize_for_tts, NormalizationConfig

        cfg = NormalizationConfig()
        result = normalize_for_tts("и т.д.", cfg)
        assert "и так далее" in result

    def test_custom_dictionary(self, tmp_path):
        from russian_tts_studio.text.normalization import (
            create_dictionary, add_entry, normalize_for_tts,
            NormalizationConfig, get_db,
        )

        # Create a temp DB
        db_path = tmp_path / "test_norm.db"
        import sqlite3
        conn = sqlite3.connect(str(db_path), timeout=10, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS dictionaries (
                id TEXT PRIMARY KEY, name TEXT, description TEXT DEFAULT '',
                enabled INTEGER DEFAULT 1, created_at TEXT DEFAULT '', updated_at TEXT DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT, dict_id TEXT,
                source TEXT, replacement TEXT, enabled INTEGER DEFAULT 1,
                sort_order INTEGER DEFAULT 0, UNIQUE(dict_id, source)
            );
        """)

        import russian_tts_studio.text.normalization as norm_mod
        orig = norm_mod.get_db
        norm_mod.get_db = lambda *a, **kw: conn

        try:
            create_dictionary("custom", "Мой словарь", output_dir=tmp_path)
            add_entry("custom", "КТ", "компьютер", output_dir=tmp_path)

            cfg = NormalizationConfig(use_custom_dicts=True)
            result = normalize_for_tts("Это КТ.", cfg, output_dir=tmp_path)
            assert "компьютер" in result
        finally:
            norm_mod.get_db = orig
            conn.close()

    def test_entry_enable_disable(self, tmp_path):
        from russian_tts_studio.text.normalization import (
            create_dictionary, add_entry, update_entry,
            normalize_for_tts, NormalizationConfig,
        )

        db_path = tmp_path / "test_norm.db"
        import sqlite3
        conn = sqlite3.connect(str(db_path), timeout=10, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS dictionaries (
                id TEXT PRIMARY KEY, name TEXT, description TEXT DEFAULT '',
                enabled INTEGER DEFAULT 1, created_at TEXT DEFAULT '', updated_at TEXT DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT, dict_id TEXT,
                source TEXT, replacement TEXT, enabled INTEGER DEFAULT 1,
                sort_order INTEGER DEFAULT 0, UNIQUE(dict_id, source)
            );
        """)

        import russian_tts_studio.text.normalization as norm_mod
        orig = norm_mod.get_db
        norm_mod.get_db = lambda *a, **kw: conn

        try:
            create_dictionary("test", "Test", output_dir=tmp_path)
            entry_id = add_entry("test", "абр", "аббревиатура", output_dir=tmp_path)

            # Enabled → replacement happens
            cfg = NormalizationConfig(use_custom_dicts=True)
            result = normalize_for_tts("Это абр.", cfg, output_dir=tmp_path)
            assert "аббревиатура" in result

            # Disable → replacement doesn't happen
            update_entry(entry_id, enabled=False, output_dir=tmp_path)
            result = normalize_for_tts("Это абр.", cfg, output_dir=tmp_path)
            assert "аббревиатура" not in result
        finally:
            norm_mod.get_db = orig
            conn.close()

    def test_export_import_json(self, tmp_path):
        from russian_tts_studio.text.normalization import (
            create_dictionary, add_entry, export_dictionary_json,
            import_dictionary_json,
        )

        db_path = tmp_path / "test_norm.db"
        import sqlite3
        conn = sqlite3.connect(str(db_path), timeout=10, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS dictionaries (
                id TEXT PRIMARY KEY, name TEXT, description TEXT DEFAULT '',
                enabled INTEGER DEFAULT 1, created_at TEXT DEFAULT '', updated_at TEXT DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT, dict_id TEXT,
                source TEXT, replacement TEXT, enabled INTEGER DEFAULT 1,
                sort_order INTEGER DEFAULT 0, UNIQUE(dict_id, source)
            );
        """)

        import russian_tts_studio.text.normalization as norm_mod
        orig = norm_mod.get_db
        norm_mod.get_db = lambda *a, **kw: conn

        try:
            create_dictionary("export_test", "Export Test", output_dir=tmp_path)
            add_entry("export_test", "АБР", "аббревиатура", output_dir=tmp_path)

            json_str = export_dictionary_json("export_test", output_dir=tmp_path)
            assert "АБР" in json_str

            # Import into a new dictionary
            create_dictionary("import_test", "Import Test", output_dir=tmp_path)
            count = import_dictionary_json("import_test", json_str, output_dir=tmp_path)
            assert count >= 1
        finally:
            norm_mod.get_db = orig
            conn.close()


class TestReview:
    """Tests for the Whisper review system."""

    def test_review_result_to_dict(self):
        from russian_tts_studio.pipeline.review import ReviewResult, ReviewStatus

        r = ReviewResult(status=ReviewStatus.APPROVED, wer=0.05, cer=0.02)
        d = r.to_dict()
        assert d["status"] == "approved"
        assert d["wer"] == 0.05

    def test_review_config_to_dict(self):
        from russian_tts_studio.pipeline.review import ReviewConfig

        cfg = ReviewConfig(retry_max=3, tail_warning_ms=400)
        d = cfg.to_dict()
        assert d["retry_max"] == 3
        assert d["tail_warning_ms"] == 400

    def test_calculate_wer_perfect(self):
        from russian_tts_studio.pipeline.review import _calculate_wer

        assert _calculate_wer("привет мир", "привет мир") == 0.0

    def test_calculate_wer_total_mismatch(self):
        from russian_tts_studio.pipeline.review import _calculate_wer

        assert _calculate_wer("привет мир", "hello world") == 1.0

    def test_calculate_cer_perfect(self):
        from russian_tts_studio.pipeline.review import _calculate_cer

        assert _calculate_cer("тест", "тест") == 0.0

    def test_normalize_for_wer(self):
        from russian_tts_studio.pipeline.review import _normalize_for_wer

        assert _normalize_for_wer("Привет, мир!") == "привет мир"
        assert _normalize_for_wer("  ТЕСТ  ") == "тест"

    def test_compute_tail_ms(self):
        from russian_tts_studio.pipeline.review import _compute_tail_ms

        # Audio is 5s, last word ends at 3s → tail = 2s = 2000ms
        whisper_result = {
            "segments": [{"words": [{"end": 3.0}]}],
        }
        tail = _compute_tail_ms(5.0, whisper_result)
        assert tail == pytest.approx(2000.0)

    def test_compute_tail_ms_no_words(self):
        from russian_tts_studio.pipeline.review import _compute_tail_ms

        tail = _compute_tail_ms(5.0, {"segments": []})
        assert tail == 0.0

    def test_review_segment_missing_audio(self):
        from russian_tts_studio.pipeline.review import review_segment, ReviewStatus

        result = review_segment("/nonexistent/audio.wav", "text")
        assert result.status == ReviewStatus.ERROR
        assert "not found" in result.error.lower()

    def test_review_segment_without_whisper(self, tmp_path):
        """Review without Whisper installed should still produce a result."""
        from russian_tts_studio.pipeline.review import review_segment, ReviewStatus
        import numpy as np
        import soundfile as sf

        # Create a minimal WAV
        wav_path = tmp_path / "test.wav"
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        wav = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        sf.write(str(wav_path), wav, 16000, subtype="FLOAT")

        result = review_segment(wav_path, "Привет мир")
        # Without Whisper, transcript is empty, but status should be
        # at least not ERROR (WER/CER can't be computed from empty transcript).
        assert result.status in (ReviewStatus.APPROVED, ReviewStatus.NEEDS_REVIEW, ReviewStatus.NEEDS_RETRY)

    def test_review_status_enum(self):
        from russian_tts_studio.pipeline.review import ReviewStatus

        assert ReviewStatus.APPROVED.value == "approved"
        assert ReviewStatus.NEEDS_RETRY.value == "needs_retry"
        assert ReviewStatus.ERROR.value == "error"