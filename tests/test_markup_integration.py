"""Integration tests for markup → pipeline → API.

These tests verify the wiring between the markup parser and the TTS
pipeline's ``synthesize_markup`` method, plus the ``/api/markup/parse``
endpoint. They do NOT call real VoxCPM2/Silero inference — the
synthesizers are monkey-patched with stubs that write 1-second sine
WAVs so concatenation logic is exercised.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from russian_tts_studio.markup import parse  # noqa: E402


def _write_sine_wav(path: Path, duration_sec: float = 0.5, sr: int = 22050) -> Path:
    """Write a short sine WAV — stands in for a real TTS output."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    wav = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav, sr, subtype="FLOAT")
    return path


class TestSynthesizeMarkupWiring:
    """Test ``TTSPipeline.synthesize_markup`` with a stubbed synthesizer."""

    def test_markup_parse_and_segment_count(self):
        """Parse produces the expected segment structure."""
        doc = parse("Привет. {{pause 500ms}} Пока. {{speed 0.9}} Медленно.")
        non_empty = [s for s in doc.segments if s.text.strip()]
        assert len(non_empty) == 3

    def test_concat_wavs_interleaves_silence(self):
        """``_concat_wavs`` interleaves seg/silence in order."""
        from russian_tts_studio.pipeline.tts_pipeline import TTSPipeline

        tmp = Path("output/test_concat")
        seg0 = _write_sine_wav(tmp / "seg0.wav", 0.5)
        seg1 = _write_sine_wav(tmp / "seg1.wav", 0.5)
        sil0 = _write_sine_wav(tmp / "sil0.wav", 0.2)
        out = tmp / "out.wav"
        TTSPipeline._concat_wavs([seg0, seg1], [sil0], out)
        assert out.exists()
        wav, sr = sf.read(str(out), dtype="float32", always_2d=False)
        # 0.5 + 0.2 + 0.5 = 1.2 sec
        assert abs(len(wav) / sr - 1.2) < 0.05

    def test_concat_wavs_empty_inputs(self):
        from russian_tts_studio.pipeline.tts_pipeline import TTSPipeline

        out = Path("output/test_concat/empty.wav")
        TTSPipeline._concat_wavs([], [], out)
        assert out.exists()

    def test_concat_wavs_resamples_mismatched_sr(self):
        """Segments at 48 kHz get resampled to 22050 Hz."""
        from russian_tts_studio.pipeline.tts_pipeline import TTSPipeline

        tmp = Path("output/test_concat_sr")
        tmp.mkdir(parents=True, exist_ok=True)
        # Write at 48000 Hz (VoxCPM2 native rate)
        t = np.linspace(0, 0.5, int(48000 * 0.5), endpoint=False)
        wav = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        seg = tmp / "seg48k.wav"
        sf.write(str(seg), wav, 48000, subtype="FLOAT")
        out = tmp / "out.wav"
        TTSPipeline._concat_wavs([seg], [], out)
        info = sf.info(str(out))
        assert info.samplerate == 22050

    def test_apply_aliases(self):
        from russian_tts_studio.pipeline.tts_pipeline import TTSPipeline
        from russian_tts_studio.markup import Alias

        aliases = [Alias(target="GPT", replacement="gee pee tee")]
        out = TTSPipeline._apply_aliases("GPT — это хорошо", aliases)
        assert "gee pee tee" in out
        assert "GPT" not in out

    def test_context_window_passes_previous_segments(self):
        """``synthesize_markup(context_window=2)`` populates
        ``context_segments`` in per-segment metadata with the previous
        2 segment texts. We verify the metadata is built correctly by
        stubbing ``synthesize`` to capture what it receives."""
        from russian_tts_studio.pipeline.tts_pipeline import TTSPipeline, PipelineConfig
        from russian_tts_studio.markup import parse

        doc = parse("Первый. {{pause 500ms}} Второй. {{pause 500ms}} Третий.")
        # Build a pipeline with a stub synth that records metadata.
        pipe = TTSPipeline.__new__(TTSPipeline)
        pipe.engine = "voxcpm"
        pipe.config = PipelineConfig(enable_quality_check=False, enable_fallback=False, enable_postprocess=False)
        pipe.silero = None
        pipe.transcriber = None
        pipe.similarity_calc = None
        pipe._initialized = True

        captured_meta: list[dict] = []

        class _StubSynth:
            def synthesize(self, request):
                captured_meta.append(dict(request.metadata or {}))
                # Write a tiny sine WAV so _concat_wavs has something to read.
                import numpy as np
                import soundfile as sf
                p = Path(request.output_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(p), np.zeros(100, dtype="float32"), 22050)
                from russian_tts_studio.models.base_synth import SynthesisResult
                return SynthesisResult(
                    audio_path=p, duration_sec=1.0, generation_time_sec=0.1,
                    rtf=0.1, model="stub", text=request.text,
                )

        pipe.synth = _StubSynth()

        # Stub _strip_unsupported_markup (instance method) so it doesn't
        # try to query the stub synth's capabilities.
        pipe._strip_unsupported_markup = lambda text: text
        # Stub _postprocess so it doesn't run FFmpeg on the stub WAV.
        pipe._postprocess = lambda path, prosody_applied=False: path

        # Create a dummy reference audio file so ``synthesize`` doesn't
        # bail out with "no reference audio" (we stub the synth anyway).
        ref_wav = _write_sine_wav(Path("output/test_ctx/reference.wav"), 0.3)

        result = pipe.synthesize_markup(
            doc, reference_audio=str(ref_wav),
            output_path="output/test_ctx/out.wav",
            context_window=2,
        )
        # 3 segments → 3 captured metadata dicts.
        assert len(captured_meta) == 3
        # First segment: no context (no previous).
        assert "context_segments" not in captured_meta[0]
        # Second segment: 1 context segment (the first).
        assert "context_segments" in captured_meta[1]
        assert len(captured_meta[1]["context_segments"]) == 1
        assert "Первый" in captured_meta[1]["context_segments"][0]
        # Third segment: 2 context segments (window capped at 2).
        assert "context_segments" in captured_meta[2]
        assert len(captured_meta[2]["context_segments"]) == 2
        assert "Первый" in captured_meta[2]["context_segments"][0]
        assert "Второй" in captured_meta[2]["context_segments"][1]


class TestMarkupParseAPI:
    """Test the ``/api/markup/parse`` endpoint (no synthesis)."""

    def test_parse_endpoint_returns_structure(self):
        from fastapi.testclient import TestClient

        # Import the app lazily so module-level side effects (logging
        # setup, MMS_FA env) don't run at collection time for every
        # test in the file.
        from web.app import app

        client = TestClient(app)
        resp = client.post(
            "/api/markup/parse",
            data={"text": '{{chapter "Урок 1"}} Привет. {{pause 700ms}} Пока.'},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["has_markup"] is True
        assert len(body["chapters"]) == 1
        assert body["chapters"][0]["title"] == "Урок 1"
        # At least one segment has pause_after_ms set
        pauses = [s["pause_after_ms"] for s in body["segments"] if s["pause_after_ms"] is not None]
        assert 700 in pauses

    def test_parse_endpoint_plain_text(self):
        from fastapi.testclient import TestClient
        from web.app import app

        client = TestClient(app)
        resp = client.post("/api/markup/parse", data={"text": "Просто текст."})
        assert resp.status_code == 200
        body = resp.json()
        assert body["has_markup"] is False
        assert len(body["segments"]) == 1
        assert body["warnings"] == []

    def test_parse_endpoint_unknown_command_warning(self):
        from fastapi.testclient import TestClient
        from web.app import app

        client = TestClient(app)
        resp = client.post(
            "/api/markup/parse",
            data={"text": "Текст {{frobnicate 42}} здесь."},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["warnings"]) == 1
        assert "frobnicate" in body["warnings"][0]

    def test_parse_endpoint_speed_and_volume(self):
        from fastapi.testclient import TestClient
        from web.app import app

        client = TestClient(app)
        resp = client.post(
            "/api/markup/parse",
            data={"text": "{{speed 0.9}} Текст. {{volume -3db}} Тихо."},
        )
        assert resp.status_code == 200
        body = resp.json()
        speeds = [s["speed"] for s in body["segments"]]
        assert any(abs(s - 0.9) < 0.01 for s in speeds)
        vols = [s["volume"] for s in body["segments"] if s["volume"] is not None]
        assert any(v["gain_db"] == -3.0 for v in vols)

    def test_parse_endpoint_returns_sound_event_tokens(self):
        """Inline sound-event tokens appear in segment text; span events
        set ``active_span`` on the segments inside the span."""
        from fastapi.testclient import TestClient
        from web.app import app

        client = TestClient(app)
        resp = client.post(
            "/api/markup/parse",
            data={"text": "Привет. {{laugh}} Это смешно. {{bgm start}} Музыка. {{bgm end}} Конец."},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["has_markup"] is True
        # The [laugh] token is in the segment text
        inline_seg = next(s for s in body["segments"] if "[laugh]" in s["text"])
        assert "[laugh]" in inline_seg["text"]
        # The segment inside {{bgm start}}...{{bgm end}} has active_span
        span_segs = [s for s in body["segments"] if s.get("active_span")]
        assert len(span_segs) >= 1
        assert span_segs[0]["active_span"]["name"] == "bgm"
        assert span_segs[0]["active_span"]["token"] == "[music]"
        # The segment after {{bgm end}} has no active_span
        after_segs = [s for s in body["segments"] if s["text"] == "Конец."]
        assert len(after_segs) == 1
        assert after_segs[0]["active_span"] is None

    def test_parse_endpoint_returns_stresses(self):
        """``{{stress "за́мок"}}`` populates ``stresses`` on following segments."""
        from fastapi.testclient import TestClient
        from web.app import app

        client = TestClient(app)
        resp = client.post(
            "/api/markup/parse",
            data={"text": '{{stress "за́мок"}} Замок большой.'},
        )
        assert resp.status_code == 200
        body = resp.json()
        seg = body["segments"][0]
        assert len(seg["stresses"]) == 1
        assert seg["stresses"][0]["target"] == "замок"
        assert seg["stresses"][0]["stressed"] == "за́мок"

    def test_parse_endpoint_stress_warning_surfaces(self):
        """When auto-detect falls back, the warning appears in ``warnings``."""
        from fastapi.testclient import TestClient
        from web.app import app

        client = TestClient(app)
        resp = client.post(
            "/api/markup/parse",
            data={"text": '{{stress "замок" "я"}} Замок.'},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert any("not found" in w for w in body["warnings"])