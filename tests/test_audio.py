"""Tests for audio mixing and subtitle generation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from russian_tts_studio.audio.mix import MixConfig, _get_duration, mix_podcast  # noqa: E402
from russian_tts_studio.audio.subtitles import (  # noqa: E402
    WordTimestamp,
    _group_cues,
    write_ass,
    write_srt,
    words_from_timestamps_list,
)


def _write_test_wav(path: Path, duration: float = 3.0, sr: int = 22050, freq: float = 440) -> Path:
    """Write a simple sine WAV for testing."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    wav = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav, sr, subtype="FLOAT")
    return path


def _write_test_mp3(path: Path, duration: float = 10.0, sr: int = 44100) -> Path:
    """Write a test MP3 via FFmpeg (requires FFmpeg)."""
    import subprocess

    wav_tmp = path.with_suffix(".wav")
    _write_test_wav(wav_tmp, duration=duration, sr=sr, freq=330)
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", str(wav_tmp), "-codec:a", "libmp3lame", "-b:a", "128k", str(path)],
        check=True, capture_output=True,
    )
    wav_tmp.unlink()
    return path


class TestMixPodcast:
    """Tests for ``mix_podcast`` (requires FFmpeg)."""

    @pytest.fixture(autouse=True)
    def _tmp_audio(self, tmp_path):
        self.out_dir = tmp_path / "audio"
        self.out_dir.mkdir()

    def test_basic_mix_wav(self):
        """Mix narration + music → single WAV output."""
        voice = _write_test_wav(self.out_dir / "voice.wav", duration=3.0, freq=440)
        music = _write_test_wav(self.out_dir / "music.wav", duration=10.0, freq=330)
        cfg = MixConfig(output_format="wav", ducking=False, loudnorm=False)
        out = mix_podcast(voice, music, config=cfg, output_path=self.out_dir / "mix.wav")
        assert out.exists()
        assert out.suffix == ".wav"
        dur = _get_duration(out)
        assert dur > 2.5  # at least close to voice duration

    def test_basic_mix_mp3(self):
        """Mix narration + music → MP3 output."""
        voice = _write_test_wav(self.out_dir / "voice.wav", duration=3.0)
        music = _write_test_wav(self.out_dir / "music.wav", duration=10.0)
        cfg = MixConfig(output_format="mp3", ducking=False, loudnorm=False)
        out = mix_podcast(voice, music, config=cfg, output_path=self.out_dir / "mix.mp3")
        assert out.exists()
        assert out.suffix == ".mp3"

    def test_ducking_enabled(self):
        """Ducking should not crash and should produce output."""
        voice = _write_test_wav(self.out_dir / "voice.wav", duration=3.0)
        music = _write_test_wav(self.out_dir / "music.wav", duration=10.0)
        cfg = MixConfig(ducking=True, loudnorm=False, output_format="wav")
        out = mix_podcast(voice, music, config=cfg, output_path=self.out_dir / "mix.wav")
        assert out.exists()

    def test_loudnorm_enabled(self):
        """Loudnorm should not crash."""
        voice = _write_test_wav(self.out_dir / "voice.wav", duration=3.0)
        music = _write_test_wav(self.out_dir / "music.wav", duration=10.0)
        cfg = MixConfig(loudnorm=True, ducking=False, output_format="wav")
        out = mix_podcast(voice, music, config=cfg, output_path=self.out_dir / "mix.wav")
        assert out.exists()

    def test_intro_delay(self):
        """Intro_sec should extend total duration beyond voice length."""
        voice = _write_test_wav(self.out_dir / "voice.wav", duration=3.0)
        music = _write_test_wav(self.out_dir / "music.wav", duration=15.0)
        cfg = MixConfig(intro_sec=3.0, ducking=False, loudnorm=False, output_format="wav")
        out = mix_podcast(voice, music, config=cfg, output_path=self.out_dir / "mix.wav")
        dur = _get_duration(out)
        # Voice (3s) + intro (3s) + tail (0) = ~6s, music may extend further
        assert dur > 5.5

    def test_voice_gain(self):
        """Voice gain should not crash."""
        voice = _write_test_wav(self.out_dir / "voice.wav", duration=3.0)
        music = _write_test_wav(self.out_dir / "music.wav", duration=10.0)
        cfg = MixConfig(voice_db=-6.0, ducking=False, loudnorm=False, output_format="wav")
        out = mix_podcast(voice, music, config=cfg, output_path=self.out_dir / "mix.wav")
        assert out.exists()

    def test_missing_narration_raises(self):
        music = _write_test_wav(self.out_dir / "music.wav", duration=3.0)
        with pytest.raises(FileNotFoundError, match="Narration not found"):
            mix_podcast(self.out_dir / "nonexistent.wav", music)

    def test_missing_music_raises(self):
        voice = _write_test_wav(self.out_dir / "voice.wav", duration=3.0)
        with pytest.raises(FileNotFoundError, match="Music not found"):
            mix_podcast(voice, self.out_dir / "nonexistent.mp3")

    def test_music_loop_short_music(self):
        """Music shorter than narration should loop without crashing."""
        voice = _write_test_wav(self.out_dir / "voice.wav", duration=10.0, freq=440)
        music = _write_test_wav(self.out_dir / "music.wav", duration=2.0, freq=330)
        cfg = MixConfig(loop_music=True, ducking=False, loudnorm=False, output_format="wav")
        out = mix_podcast(voice, music, config=cfg, output_path=self.out_dir / "loop_mix.wav")
        assert out.exists()
        dur = _get_duration(out)
        assert dur > 9.0  # at least close to voice duration

    def test_voice_muted(self):
        """Voice mute should produce output (with music only)."""
        voice = _write_test_wav(self.out_dir / "voice.wav", duration=3.0)
        music = _write_test_wav(self.out_dir / "music.wav", duration=10.0)
        cfg = MixConfig(voice_muted=True, ducking=False, loudnorm=False, output_format="wav")
        out = mix_podcast(voice, music, config=cfg, output_path=self.out_dir / "muted.wav")
        assert out.exists()

    def test_music_muted(self):
        """Music mute should produce output (voice only)."""
        voice = _write_test_wav(self.out_dir / "voice.wav", duration=3.0)
        music = _write_test_wav(self.out_dir / "music.wav", duration=10.0)
        cfg = MixConfig(music_muted=True, ducking=False, loudnorm=False, output_format="wav")
        out = mix_podcast(voice, music, config=cfg, output_path=self.out_dir / "no_music.wav")
        assert out.exists()

    def test_ducking_strength_low(self):
        """Ducking preset 'low' should work."""
        voice = _write_test_wav(self.out_dir / "voice.wav", duration=3.0)
        music = _write_test_wav(self.out_dir / "music.wav", duration=10.0)
        cfg = MixConfig(ducking_strength="low", loudnorm=False, output_format="wav")
        out = mix_podcast(voice, music, config=cfg, output_path=self.out_dir / "duck_low.wav")
        assert out.exists()

    def test_ducking_strength_medium(self):
        voice = _write_test_wav(self.out_dir / "voice.wav", duration=3.0)
        music = _write_test_wav(self.out_dir / "music.wav", duration=10.0)
        cfg = MixConfig(ducking_strength="medium", loudnorm=False, output_format="wav")
        out = mix_podcast(voice, music, config=cfg, output_path=self.out_dir / "duck_med.wav")
        assert out.exists()

    def test_ducking_strength_high(self):
        voice = _write_test_wav(self.out_dir / "voice.wav", duration=3.0)
        music = _write_test_wav(self.out_dir / "music.wav", duration=10.0)
        cfg = MixConfig(ducking_strength="high", loudnorm=False, output_format="wav")
        out = mix_podcast(voice, music, config=cfg, output_path=self.out_dir / "duck_high.wav")
        assert out.exists()

    def test_mp3_metadata(self):
        """MP3 metadata tags should be embedded."""
        voice = _write_test_wav(self.out_dir / "voice.wav", duration=3.0)
        music = _write_test_wav(self.out_dir / "music.wav", duration=10.0)
        cfg = MixConfig(
            ducking=False, loudnorm=False, output_format="mp3",
            meta_title="Test Podcast", meta_artist="RTTS", meta_album="Test Album",
        )
        out = mix_podcast(voice, music, config=cfg, output_path=self.out_dir / "meta.mp3")
        assert out.exists()
        assert out.suffix == ".mp3"

    def test_preview_segment(self):
        """Preview window should produce a short output."""
        voice = _write_test_wav(self.out_dir / "voice.wav", duration=10.0)
        music = _write_test_wav(self.out_dir / "music.wav", duration=15.0)
        cfg = MixConfig(
            ducking=False, loudnorm=False, output_format="wav",
            preview_start=2.0, preview_duration=3.0,
        )
        out = mix_podcast(voice, music, config=cfg, output_path=self.out_dir / "preview.wav")
        assert out.exists()
        dur = _get_duration(out)
        assert dur < 5.0  # preview should be shorter than full mix

    def test_ducking_params_presets(self):
        cfg = MixConfig(ducking_strength="low")
        params = cfg.get_ducking_params()
        assert params[0] == 0.035  # threshold
        assert params[1] == 3.0    # ratio

        cfg = MixConfig(ducking_strength="medium")
        params = cfg.get_ducking_params()
        assert params[1] == 6.0

        cfg = MixConfig(ducking_strength="high")
        params = cfg.get_ducking_params()
        assert params[1] == 10.0

    def test_ducking_params_custom(self):
        cfg = MixConfig(ducking_strength="", duck_depth_db=6.0, duck_attack=0.2, duck_release=0.8)
        params = cfg.get_ducking_params()
        assert params[2] == 0.2   # attack
        assert params[3] == 0.8   # release

    def test_mix_config_to_dict(self):
        cfg = MixConfig()
        d = cfg.to_dict()
        assert "voice_db" in d
        assert "ducking" in d
        assert "target_lufs" in d


class TestSubtitles:
    """Tests for SRT and ASS subtitle generation."""

    def _sample_words(self) -> list[WordTimestamp]:
        return [
            WordTimestamp("Привет", 0.0, 0.5),
            WordTimestamp("мир", 0.5, 1.0),
            WordTimestamp("это", 1.0, 1.3),
            WordTimestamp("тест.", 1.3, 1.8),
            WordTimestamp("Второе", 2.0, 2.5),
            WordTimestamp("предложение.", 2.5, 3.2),
        ]

    def test_write_srt(self, tmp_path):
        words = self._sample_words()
        out = tmp_path / "test.srt"
        write_srt(words, out)
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "Привет" in content
        assert "-->" in content
        assert "00:00:00,000" in content

    def test_write_srt_with_offset(self, tmp_path):
        words = self._sample_words()
        out = tmp_path / "test_offset.srt"
        write_srt(words, out, offset_sec=5.0)
        content = out.read_text(encoding="utf-8")
        # First cue should start at 5.0s → 00:00:05,000
        assert "00:00:05,000" in content

    def test_write_ass(self, tmp_path):
        words = self._sample_words()
        out = tmp_path / "test.ass"
        write_ass(words, out)
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "[Script Info]" in content
        assert "[V4+ Styles]" in content
        assert "[Events]" in content
        assert "Dialogue:" in content
        assert "Привет" in content

    def test_write_ass_karaoke_tags(self, tmp_path):
        words = self._sample_words()
        out = tmp_path / "test_karaoke.ass"
        write_ass(words, out)
        content = out.read_text(encoding="utf-8")
        # Should contain \kf tags for word-level timing
        assert "\\kf" in content

    def test_group_cues_sentence_break(self):
        words = self._sample_words()
        cues = _group_cues(words, max_duration=5.0, max_chars=80)
        # "тест." ends with a period → should break here
        assert len(cues) >= 2
        assert "тест." in cues[0].text

    def test_group_cues_max_duration(self):
        words = [
            WordTimestamp(f"слово{i}", i * 0.5, (i + 1) * 0.5)
            for i in range(10)
        ]
        cues = _group_cues(words, max_duration=2.0, max_chars=200)
        # With max_duration=2.0, should break every ~4 words
        assert len(cues) > 1

    def test_group_cues_empty(self):
        cues = _group_cues([])
        assert cues == []

    def test_words_from_timestamps_list(self):
        data = [
            {"word": "Привет", "start": 0.0, "end": 0.5},
            {"word": "мир", "start": 0.5, "end": 1.0},
        ]
        words = words_from_timestamps_list(data)
        assert len(words) == 2
        assert words[0].word == "Привет"
        assert words[1].start == 0.5

    def test_srt_format_time(self):
        from russian_tts_studio.audio.subtitles import _format_srt_time
        assert _format_srt_time(0.0) == "00:00:00,000"
        assert _format_srt_time(61.5) == "00:01:01,500"
        assert _format_srt_time(3661.123) == "01:01:01,123"

    def test_ass_format_time(self):
        from russian_tts_studio.audio.subtitles import _format_ass_time
        assert _format_ass_time(0.0) == "0:00:00.00"
        assert _format_ass_time(61.5) == "0:01:01.50"


class TestAudioAPI:
    """Tests for /api/music and /api/mix endpoints."""

    def test_list_music_empty(self):
        from fastapi.testclient import TestClient
        from web.app import app

        client = TestClient(app)
        resp = client.get("/api/music")
        assert resp.status_code == 200
        assert "music" in resp.json()

    def test_upload_and_list_music(self, tmp_path):
        from fastapi.testclient import TestClient
        from web.app import app, MUSIC_DIR

        # Write a test WAV to the music library
        test_wav = MUSIC_DIR / "test_bg.wav"
        _write_test_wav(test_wav, duration=2.0)

        client = TestClient(app)
        resp = client.get("/api/music")
        assert resp.status_code == 200
        names = [m["name"] for m in resp.json()["music"]]
        assert "test_bg.wav" in names

        # Clean up
        test_wav.unlink(missing_ok=True)

    def test_delete_music(self):
        from fastapi.testclient import TestClient
        from web.app import app, MUSIC_DIR

        test_wav = MUSIC_DIR / "to_delete.wav"
        _write_test_wav(test_wav, duration=1.0)

        client = TestClient(app)
        resp = client.delete("/api/music/to_delete.wav")
        assert resp.status_code == 200
        assert not test_wav.exists()

    def test_delete_nonexistent_music(self):
        from fastapi.testclient import TestClient
        from web.app import app

        client = TestClient(app)
        resp = client.delete("/api/music/nonexistent.mp3")
        assert resp.status_code == 404

    def test_mix_endpoint(self):
        from fastapi.testclient import TestClient
        from web.app import app, SAMPLES_DIR

        # Create test narration and music files
        narr = SAMPLES_DIR / "test_narr.wav"
        music_file = SAMPLES_DIR / "test_music.wav"
        _write_test_wav(narr, duration=3.0, freq=440)
        _write_test_wav(music_file, duration=8.0, freq=330)

        client = TestClient(app)
        resp = client.post(
            "/api/mix",
            data={
                "narration_path": str(narr),
                "music_path": str(music_file),
                "ducking": "false",
                "loudnorm": "false",
                "output_format": "wav",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "audio_url" in body

    def test_mix_missing_narration(self):
        from fastapi.testclient import TestClient
        from web.app import app

        client = TestClient(app)
        resp = client.post(
            "/api/mix",
            data={
                "narration_path": "/nonexistent/narr.wav",
                "music_path": "/nonexistent/music.mp3",
            },
        )
        assert resp.status_code == 404

    def test_subtitles_preview(self):
        from fastapi.testclient import TestClient
        from web.app import app

        client = TestClient(app)
        resp = client.get(
            "/api/subtitles/preview",
            params={"text": "Привет мир это тест", "duration_sec": "2.0"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_words"] == 4
        assert len(body["cues"]) >= 1