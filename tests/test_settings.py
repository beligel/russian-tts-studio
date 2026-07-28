"""Tests for settings manager, Qwen3-TTS wrapper, and chapter subtitles."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestSettings:
    """Tests for the unified settings manager."""

    def test_default_settings(self, tmp_path):
        from russian_tts_studio.settings import Settings

        s = Settings(path=tmp_path / "config.json")
        assert s.get("tts.speed") == 0.9
        assert s.get("tts.engine") == "voxcpm"
        assert s.get("nonexistent", "fallback") == "fallback"

    def test_set_and_get(self, tmp_path):
        from russian_tts_studio.settings import Settings

        s = Settings(path=tmp_path / "config.json")
        s.set("tts.speed", 0.85)
        assert s.get("tts.speed") == 0.85

    def test_update_flat(self, tmp_path):
        from russian_tts_studio.settings import Settings

        s = Settings(path=tmp_path / "config.json")
        s.update({"tts.speed": 0.8, "tts.engine": "qwen3"})
        assert s.get("tts.speed") == 0.8
        assert s.get("tts.engine") == "qwen3"

    def test_update_nested(self, tmp_path):
        from russian_tts_studio.settings import Settings

        s = Settings(path=tmp_path / "config.json")
        s.update({"tts": {"speed": 0.7, "engine": "silero"}})
        assert s.get("tts.speed") == 0.7
        assert s.get("tts.engine") == "silero"

    def test_save_and_reload(self, tmp_path):
        from russian_tts_studio.settings import Settings

        p = tmp_path / "config.json"
        s = Settings(path=p)
        s.set("tts.speed", 0.75)
        s.save()

        s2 = Settings(path=p)
        assert s2.get("tts.speed") == 0.75

    def test_get_section(self, tmp_path):
        from russian_tts_studio.settings import Settings

        s = Settings(path=tmp_path / "config.json")
        section = s.get_section("tts")
        assert "speed" in section
        assert "engine" in section

    def test_to_dict(self, tmp_path):
        from russian_tts_studio.settings import Settings

        s = Settings(path=tmp_path / "config.json")
        d = s.to_dict()
        assert isinstance(d, dict)
        assert "tts.speed" in d

    def test_reset_single(self, tmp_path):
        from russian_tts_studio.settings import Settings

        s = Settings(path=tmp_path / "config.json")
        s.set("tts.speed", 0.5)
        s.reset("tts.speed")
        assert s.get("tts.speed") == 0.9

    def test_reset_all(self, tmp_path):
        from russian_tts_studio.settings import Settings

        s = Settings(path=tmp_path / "config.json")
        s.set("tts.speed", 0.5)
        s.reset()
        assert s.get("tts.speed") == 0.9

    def test_schema_version_migration(self, tmp_path):
        from russian_tts_studio.settings import Settings

        p = tmp_path / "config.json"
        # Write settings with old schema version
        old = {"schema_version": 0, "tts.speed": 0.6}
        p.write_text(json.dumps(old), encoding="utf-8")

        s = Settings(path=p)
        # Should have defaults merged + version bumped
        assert s.get("tts.speed") == 0.6  # preserved from old
        assert s.get("tts.engine") == "voxcpm"  # from defaults

    def test_corrupt_file_fallback(self, tmp_path):
        from russian_tts_studio.settings import Settings

        p = tmp_path / "config.json"
        p.write_text("not json {{{", encoding="utf-8")
        s = Settings(path=p)
        # Should fall back to defaults
        assert s.get("tts.speed") == 0.9

    def test_atomic_save(self, tmp_path):
        """Save should not leave .tmp files behind."""
        from russian_tts_studio.settings import Settings

        p = tmp_path / "config.json"
        s = Settings(path=p)
        s.save()
        assert p.exists()
        assert not (tmp_path / "config.json.tmp").exists()


class TestQwen3Synthesizer:
    """Tests for Qwen3-TTS wrapper (unit tests, no real model)."""

    def test_init_default(self):
        from russian_tts_studio.models.qwen3_synth import Qwen3Synthesizer

        s = Qwen3Synthesizer()
        assert s.model_name == "qwen3-1.7b-base"
        assert s.language == "Russian"

    def test_init_custom_model(self):
        from russian_tts_studio.models.qwen3_synth import Qwen3Synthesizer

        s = Qwen3Synthesizer(model_name="qwen3-0.6b-custom")
        assert "0.6B" in s.model_id

    def test_init_voice_design(self):
        from russian_tts_studio.models.qwen3_synth import Qwen3Synthesizer

        s = Qwen3Synthesizer(model_name="qwen3-1.7b-design")
        assert "VoiceDesign" in s.model_id

    def test_init_unknown_model_raises(self):
        from russian_tts_studio.models.qwen3_synth import Qwen3Synthesizer

        with pytest.raises(ValueError, match="Unknown Qwen3 model"):
            Qwen3Synthesizer(model_name="nonexistent")

    def test_not_loaded_state(self):
        from russian_tts_studio.models.qwen3_synth import Qwen3Synthesizer

        s = Qwen3Synthesizer()
        assert not s.is_loaded()

    def test_supported_models_dict(self):
        from russian_tts_studio.models.qwen3_synth import Qwen3Synthesizer

        assert len(Qwen3Synthesizer.SUPPORTED_MODELS) == 5
        assert "qwen3-1.7b-base" in Qwen3Synthesizer.SUPPORTED_MODELS

    def test_resolve_device_auto(self):
        from russian_tts_studio.models.qwen3_synth import Qwen3Synthesizer

        device = Qwen3Synthesizer._resolve_device("auto")
        assert device in ("cpu", "cuda", "mps")


class TestChapterSubtitles:
    """Tests for chapter-aware subtitle export."""

    def _sample_words(self):
        from russian_tts_studio.audio.subtitles import WordTimestamp
        return [
            WordTimestamp("Глава", 0.0, 0.3),
            WordTimestamp("первый", 0.3, 0.7),
            WordTimestamp("текст.", 0.7, 1.2),
            WordTimestamp("Глава", 1.5, 1.8),
            WordTimestamp("второй", 1.8, 2.3),
            WordTimestamp("текст.", 2.3, 2.8),
        ]

    def test_write_chapter_subtitles_srt(self, tmp_path):
        from russian_tts_studio.audio.subtitles import write_chapter_subtitles

        words = self._sample_words()
        chapters = [
            {"title": "Глава 1", "char_start": 0, "char_end": 20},
            {"title": "Глава 2", "char_start": 20, "char_end": 40},
        ]
        paths = write_chapter_subtitles(words, chapters, tmp_path / "subs", format="srt")
        assert len(paths) >= 3  # 2 chapters + combined
        for p in paths:
            assert p.exists()
            assert p.suffix == ".srt"

    def test_write_chapter_subtitles_ass(self, tmp_path):
        from russian_tts_studio.audio.subtitles import write_chapter_subtitles

        words = self._sample_words()
        chapters = [{"title": "Ch1", "char_start": 0, "char_end": 20}]
        paths = write_chapter_subtitles(words, chapters, tmp_path / "subs", format="ass")
        assert len(paths) >= 2
        for p in paths:
            assert p.suffix == ".ass"

    def test_chapter_subtitle_content(self, tmp_path):
        from russian_tts_studio.audio.subtitles import write_chapter_subtitles

        words = self._sample_words()
        chapters = [
            {"title": "Урок 1", "char_start": 0, "char_end": 20},
            {"title": "Урок 2", "char_start": 20, "char_end": 40},
        ]
        paths = write_chapter_subtitles(words, chapters, tmp_path / "subs")
        # Check that the combined file has all words
        combined = tmp_path / "subs" / "all_chapters.srt"
        assert combined.exists()
        content = combined.read_text(encoding="utf-8")
        assert "Глава" in content

    def test_empty_words(self, tmp_path):
        from russian_tts_studio.audio.subtitles import write_chapter_subtitles

        paths = write_chapter_subtitles([], [{"title": "Ch", "char_start": 0, "char_end": 10}], tmp_path / "subs")
        assert paths == []

    def test_empty_chapters(self, tmp_path):
        from russian_tts_studio.audio.subtitles import write_chapter_subtitles

        from russian_tts_studio.audio.subtitles import WordTimestamp
        words = [WordTimestamp("test", 0, 1)]
        paths = write_chapter_subtitles(words, [], tmp_path / "subs")
        assert paths == []


class TestSettingsAPI:
    """Tests for /api/settings endpoints."""

    def test_get_settings(self):
        from fastapi.testclient import TestClient
        from web.app import app

        client = TestClient(app)
        resp = client.get("/api/settings")
        assert resp.status_code == 200
        body = resp.json()
        assert "tts.speed" in body

    def test_update_settings(self):
        from fastapi.testclient import TestClient
        from web.app import app

        client = TestClient(app)
        resp = client.put(
            "/api/settings",
            json={"tts.speed": 0.75},
        )
        assert resp.status_code == 200

    def test_reset_settings(self):
        from fastapi.testclient import TestClient
        from web.app import app

        client = TestClient(app)
        resp = client.post("/api/settings/reset")
        assert resp.status_code == 200