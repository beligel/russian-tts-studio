"""Tests for the Higgs Audio v2 synthesizer wrapper.

These tests verify the wrapper's interface, constructor validation,
mode detection, and capability flags — they do NOT load the real
Higgs model (which requires the upstream ``boson-ai/higgs-audio`` repo
and a GPU with ≥24 GB). Model-loading paths are exercised by mocking
the ``boson_multimodal`` import.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestHiggsConstructor:
    """Constructor / device / dtype resolution — no model load."""

    def test_default_model_is_v2_3b(self):
        from russian_tts_studio.models.higgs_synth import HiggsAudioSynthesizer

        synth = HiggsAudioSynthesizer()
        assert synth.model_name == "higgs-v2-3b"
        assert synth.model_id == "bosonai/higgs-audio-v2-generation-3B-base"

    def test_unknown_model_raises(self):
        from russian_tts_studio.models.higgs_synth import HiggsAudioSynthesizer

        with pytest.raises(ValueError, match="Unknown Higgs model"):
            HiggsAudioSynthesizer(model_name="higgs-v9-nonexistent")

    def test_supported_models_list(self):
        from russian_tts_studio.models.higgs_synth import HiggsAudioSynthesizer

        keys = list(HiggsAudioSynthesizer.SUPPORTED_MODELS.keys())
        assert "higgs-v2-3b" in keys
        assert "higgs-v2.5" in keys

    def test_resolve_device_auto(self):
        from russian_tts_studio.models.higgs_synth import HiggsAudioSynthesizer

        dev = HiggsAudioSynthesizer._resolve_device("auto")
        assert dev in ("cuda", "mps", "cpu")

    def test_resolve_device_explicit(self):
        from russian_tts_studio.models.higgs_synth import HiggsAudioSynthesizer

        assert HiggsAudioSynthesizer._resolve_device("cpu") == "cpu"
        assert HiggsAudioSynthesizer._resolve_device("cuda") == "cuda"

    def test_resolve_dtype_auto(self):
        from russian_tts_studio.models.higgs_synth import HiggsAudioSynthesizer

        dt = HiggsAudioSynthesizer._resolve_dtype("auto")
        assert dt in ("bfloat16", "float32")

    def test_resolve_dtype_explicit(self):
        from russian_tts_studio.models.higgs_synth import HiggsAudioSynthesizer

        assert HiggsAudioSynthesizer._resolve_dtype("float16") == "float16"
        assert HiggsAudioSynthesizer._resolve_dtype("bfloat16") == "bfloat16"

    def test_not_loaded_by_default(self):
        from russian_tts_studio.models.higgs_synth import HiggsAudioSynthesizer

        synth = HiggsAudioSynthesizer()
        assert not synth.is_loaded()


class TestHiggsCapabilities:
    """The capability flags the pipeline queries to decide stripping."""

    def test_supports_sound_events_true(self):
        from russian_tts_studio.models.higgs_synth import HiggsAudioSynthesizer

        assert HiggsAudioSynthesizer.supports_sound_events() is True

    def test_supports_stress_marks_true(self):
        from russian_tts_studio.models.higgs_synth import HiggsAudioSynthesizer

        assert HiggsAudioSynthesizer.supports_stress_marks() is True

    def test_voxcpm_sound_events_false_stress_true(self):
        """VoxCPM2 does NOT support sound events (strips [laugh] etc.)
        but DOES keep stress marks (U+0301) — the tokeniser tolerates
        combining diacritics, and stripping them was worse."""
        from russian_tts_studio.models.voxcpm_synth import VoxCPMSynthesizer

        # Sound events: not supported → pipeline strips [laugh]/[music]
        assert not getattr(VoxCPMSynthesizer, "supports_sound_events", lambda: False)()
        # Stress marks: kept (True) — pipeline does NOT strip U+0301
        assert getattr(VoxCPMSynthesizer, "supports_stress_marks", lambda: False)() is True


class TestHiggsReferenceParsing:
    """``_parse_reference_list`` — multi-speaker comma-separated handling."""

    def _synth(self):
        from russian_tts_studio.models.higgs_synth import HiggsAudioSynthesizer

        return HiggsAudioSynthesizer()

    def test_none_returns_empty(self):
        assert self._synth()._parse_reference_list(None) == []

    def test_empty_string_returns_empty(self):
        assert self._synth()._parse_reference_list("") == []

    def test_single_path(self, tmp_path):
        ref = tmp_path / "ref.wav"
        ref.write_bytes(b"fake")
        result = self._synth()._parse_reference_list(str(ref))
        assert len(result) == 1
        assert result[0] == ref

    def test_comma_separated_multi(self, tmp_path):
        r1 = tmp_path / "a.wav"
        r2 = tmp_path / "b.wav"
        r1.write_bytes(b"fake")
        r2.write_bytes(b"fake")
        result = self._synth()._parse_reference_list(f"{r1},{r2}")
        assert len(result) == 2
        assert result[0] == r1
        assert result[1] == r2

    def test_list_input(self, tmp_path):
        r1 = tmp_path / "a.wav"
        r2 = tmp_path / "b.wav"
        r1.write_bytes(b"fake")
        r2.write_bytes(b"fake")
        result = self._synth()._parse_reference_list([str(r1), str(r2)])
        assert len(result) == 2

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Higgs reference audio not found"):
            self._synth()._parse_reference_list(str(tmp_path / "nonexistent.wav"))


class TestHiggsLoadImportError:
    """``load()`` raises a clear error when ``boson_multimodal`` is missing."""

    def test_load_raises_without_boson_multimodal(self, monkeypatch):
        # Force the import inside ``load()`` to fail by hiding the module.
        import builtins

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name.startswith("boson_multimodal"):
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        from russian_tts_studio.models.higgs_synth import HiggsAudioSynthesizer

        synth = HiggsAudioSynthesizer()
        with pytest.raises(RuntimeError, match="Cannot import the 'boson_multimodal'"):
            synth.load()


class TestPipelineEngineSelection:
    """TTSPipeline accepts 'higgs' as an engine and constructs the right synth."""

    def test_pipeline_accepts_higgs_engine(self):
        from russian_tts_studio.pipeline.tts_pipeline import TTSPipeline

        pipe = TTSPipeline(engine="higgs")
        assert pipe.engine == "higgs"

    def test_pipeline_rejects_unknown_engine(self):
        from russian_tts_studio.pipeline.tts_pipeline import TTSPipeline

        with pytest.raises(ValueError, match="Unknown engine"):
            TTSPipeline(engine="elevenlabs")

    def test_pipeline_rejects_old_qwen_name(self):
        # "qwen3" was never an accepted pipeline engine (Qwen3 goes
        # through its own get_qwen3_synthesizer path), but we check
        # that the engine validator only accepts voxcpm/higgs.
        from russian_tts_studio.pipeline.tts_pipeline import TTSPipeline

        with pytest.raises(ValueError, match="Unknown engine"):
            TTSPipeline(engine="qwen3")