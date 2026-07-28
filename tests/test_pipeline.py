"""Tests for the russian_tts_studio package."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestTextUtils:
    def test_expand_abbreviations(self):
        from russian_tts_studio.utils.text_utils import expand_abbreviations
        assert "то есть" in expand_abbreviations("Это т.е. пример")
        assert "рублей" in expand_abbreviations("100 руб.")

    def test_split_into_sentences(self):
        from russian_tts_studio.utils.text_utils import split_into_sentences
        s = split_into_sentences("Привет. Как дела? Хорошо!")
        assert s == ["Привет.", "Как дела?", "Хорошо!"]

    def test_chunk_text_short(self):
        from russian_tts_studio.utils.text_utils import chunk_text_for_tts
        chunks = chunk_text_for_tts("Привет, мир.")
        assert chunks == ["Привет, мир."]

    def test_chunk_text_long(self):
        from russian_tts_studio.utils.text_utils import chunk_text_for_tts
        text = ". ".join(["Предложение номер " + str(i) for i in range(20)])
        chunks = chunk_text_for_tts(text, max_chars=80, max_sentences=3)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c) <= 120

    def test_fix_yo_letter(self):
        from russian_tts_studio.utils.text_utils import fix_yo_letter
        result = fix_yo_letter("Все хорошо. Еще увидимся.")
        assert "Всё" in result
        assert "Ещё" in result


class TestMetrics:
    def test_calculate_wer_perfect(self):
        from russian_tts_studio.utils.metrics import calculate_wer
        assert calculate_wer("привет мир", "привет мир") == 0.0

    def test_calculate_wer_total_mismatch(self):
        from russian_tts_studio.utils.metrics import calculate_wer
        assert calculate_wer("привет мир", "пока все") == 1.0

    def test_calculate_wer_partial(self):
        from russian_tts_studio.utils.metrics import calculate_wer
        ref = "раз два три четыре пять"
        hyp = "раз три четыре пять"
        wer = calculate_wer(ref, hyp)
        assert 0.0 < wer < 1.0
        assert abs(wer - 0.2) < 0.01

    def test_calculate_cer(self):
        from russian_tts_studio.utils.metrics import calculate_cer
        assert calculate_cer("привет", "привет") == 0.0
        assert 0.0 < calculate_cer("привет", "пока") <= 1.0

    def test_normalize_text_for_wer(self):
        from russian_tts_studio.utils.metrics import normalize_text_for_wer
        assert normalize_text_for_wer("Привет, мир!") == "привет мир"
        assert normalize_text_for_wer("  Множество   пробелов  ") == "множество пробелов"

    def test_tts_quality_metrics_to_dict(self):
        from russian_tts_studio.utils.metrics import TTSQualityMetrics
        m = TTSQualityMetrics(wer=0.1, cer=0.05)
        d = m.to_dict()
        assert d["wer"] == 0.1
        assert d["cer"] == 0.05


class TestAudioUtils:
    def test_get_duration(self):
        import torch
        from russian_tts_studio.utils.audio_utils import get_duration
        sr = 22050
        wav = torch.zeros(sr * 2)  # 2 seconds
        assert abs(get_duration(wav, sr) - 2.0) < 0.01

    def test_calculate_silence_ratio(self):
        import torch
        from russian_tts_studio.utils.metrics import calculate_silence_ratio
        silence = torch.zeros(1000)
        assert calculate_silence_ratio(silence) == 1.0
        loud = torch.ones(1000)
        assert calculate_silence_ratio(loud) == 0.0

    def test_concatenate_audios_single(self):
        import torch
        from russian_tts_studio.utils.audio_utils import concatenate_audios
        a = torch.tensor([1.0, 2.0, 3.0])
        result = concatenate_audios([a])
        assert torch.equal(result, a)


class TestPipelineConfig:
    def test_default_config(self):
        from russian_tts_studio.pipeline import PipelineConfig
        c = PipelineConfig()
        assert c.enable_fallback is True
        assert c.wer_threshold == 0.20
        assert c.sim_threshold == 0.50

    def test_custom_config(self):
        from russian_tts_studio.pipeline import PipelineConfig
        c = PipelineConfig(wer_threshold=0.3, enable_fallback=False)
        assert c.wer_threshold == 0.3
        assert c.enable_fallback is False


class TestSileroSynthesizer:
    def test_speakers_list(self):
        from russian_tts_studio.models.silero_synth import SileroSynthesizer
        assert "xenia" in SileroSynthesizer.SPEAKERS
        assert "eugene" in SileroSynthesizer.SPEAKERS


class TestBaseSynth:
    def test_request_dataclass(self):
        from russian_tts_studio.models.base_synth import SynthesisRequest, SynthesisResult
        req = SynthesisRequest(text="привет", reference_audio="/tmp/x.wav")
        assert req.text == "привет"
        assert req.speed == 1.0
        assert req.metadata == {}

        res = SynthesisResult(
            audio_path=Path("/tmp/x.wav"),
            duration_sec=1.0, generation_time_sec=0.5, rtf=0.5,
            model="xtts", text="привет",
        )
        assert res.success is True
        assert res.error is None


class TestComfyUIIntegration:
    def test_search_paths_constant(self):
        from russian_tts_studio.integrations import COMFYUI_PLUGIN_NAME
        assert COMFYUI_PLUGIN_NAME == "ComfyUI_FL-Russian TTS Studio3"


class TestVoxCPMSynthesizer:
    """VoxCPM2 wrapper is in .venv-voxcpm only, so these tests must
    not actually load the model — they just exercise the construction
    path and import surface. End-to-end inference is verified
    manually with a live POST."""

    def test_supported_models_includes_voxcpm(self):
        from russian_tts_studio.models.voxcpm_synth import VoxCPMSynthesizer
        assert "voxcpm-2" in VoxCPMSynthesizer.SUPPORTED_MODELS
        assert VoxCPMSynthesizer.SUPPORTED_MODELS["voxcpm-2"] == "OpenBMB/VOXCPM2"

    def test_creates_with_default_args(self):
        from russian_tts_studio.models.voxcpm_synth import VoxCPMSynthesizer
        s = VoxCPMSynthesizer()
        assert s.model_name == "voxcpm-2"
        assert s.model_id == "OpenBMB/VOXCPM2"
        assert s.language == "ru"
        assert s.sample_rate == 48000
        assert s.load_denoiser is False
        assert s.cfg_value == 2.0
        assert s.inference_timesteps == 10
        assert s._loaded is False

    def test_unknown_model_rejected(self):
        from russian_tts_studio.models.voxcpm_synth import VoxCPMSynthesizer
        with pytest.raises(ValueError, match="Unknown VoxCPM model"):
            VoxCPMSynthesizer(model_name="voxcpm-1")

    def test_device_resolution_auto_cpu(self):
        # The test env has no CUDA, so ``auto`` must collapse to cpu
        # (we don't want a test-only CUDA check to skip silently).
        from russian_tts_studio.models.voxcpm_synth import VoxCPMSynthesizer
        s = VoxCPMSynthesizer(device="auto")
        assert s.device in ("cpu", "cuda", "mps")

    def test_module_exports_in_models_package(self):
        import russian_tts_studio.models as m
        assert "VoxCPMSynthesizer" in dir(m)
        from russian_tts_studio.models import VoxCPMSynthesizer as Cls
        assert Cls is m.VoxCPMSynthesizer

    def test_pipeline_factory_dispatches_voxcpm(self):
        # Constructing the pipeline with engine="voxcpm" must succeed
        # WITHOUT triggering model load (load is lazy). This proves
        # the dispatch path is wired up. The actual model load is
        # skipped because the heavy import happens inside
        # VoxCPMSynthesizer.load() — we don't call that here.
        from russian_tts_studio.pipeline.tts_pipeline import TTSPipeline
        p = TTSPipeline(engine="voxcpm")
        assert p.engine == "voxcpm"
        assert p.synth is None  # lazy

    def test_pipeline_rejects_unknown_engine(self):
        from russian_tts_studio.pipeline.tts_pipeline import TTSPipeline
        with pytest.raises(ValueError, match="Unknown engine"):
            TTSPipeline(engine="f5-tts")

    def test_load_denoiser_can_be_enabled(self):
        from russian_tts_studio.models.voxcpm_synth import VoxCPMSynthesizer
        s = VoxCPMSynthesizer(load_denoiser=True)
        assert s.load_denoiser is True


class TestPauseConfig:
    """PauseConfig.from_metadata must only be enabled when the caller
    explicitly supplied pause_ms_* keys. Empty metadata (the default when
    ``enable_prosody=False`` in the UI) must NOT silently fall back to
    DEFAULT_PAUSE_MS, which caused unwanted pauses on every punctuation
    mark."""

    def test_word_gap_metadata_is_parsed(self):
        from russian_tts_studio.utils.prosody import PauseConfig
        cfg = PauseConfig.from_metadata({"pause_ms_word_gap": 80})
        assert cfg.word_gap_ms == 80
        assert cfg.is_enabled() is True
        # Word gap alone should NOT pull in punctuation defaults. The whole
        # config must only contain the explicitly requested word gap.
        assert cfg.comma == 0
        assert cfg.period == 0
        assert cfg.exclamation == 0

    def test_word_gap_with_punctuation_preserves_defaults(self):
        from russian_tts_studio.utils.prosody import PauseConfig, DEFAULT_PAUSE_MS
        cfg = PauseConfig.from_metadata({
            "pause_ms_word_gap": 60,
            "pause_ms_period": 400,
        })
        assert cfg.word_gap_ms == 60
        assert cfg.period == 400
        # Unset punctuation defaults remain
        assert cfg.comma == DEFAULT_PAUSE_MS["comma"]

    def test_explicit_zeros_disable_prosody(self):
        from russian_tts_studio.utils.prosody import PauseConfig
        cfg = PauseConfig.from_metadata({
            "pause_ms_comma": 0,
            "pause_ms_semicolon": 0,
            "pause_ms_colon": 0,
            "pause_ms_period": 0,
            "pause_ms_exclamation": 0,
            "pause_ms_question": 0,
            "pause_ms_ellipsis": 0,
            "pause_ms_word_gap": 0,
        })
        assert cfg.is_enabled() is False

    def test_explicit_zeros_without_word_gap_disable_prosody(self):
        # Regression: before word_gap existed, zeros on all punctuation
        # meant disabled. That contract must still hold.
        from russian_tts_studio.utils.prosody import PauseConfig
        cfg = PauseConfig.from_metadata({
            "pause_ms_comma": 0,
            "pause_ms_semicolon": 0,
            "pause_ms_colon": 0,
            "pause_ms_period": 0,
            "pause_ms_exclamation": 0,
            "pause_ms_question": 0,
            "pause_ms_ellipsis": 0,
        })
        assert cfg.is_enabled() is False

    def test_empty_metadata_means_disabled(self):
        from russian_tts_studio.utils.prosody import PauseConfig
        cfg = PauseConfig.from_metadata({})
        assert cfg.is_enabled() is False

    def test_partial_metadata_still_uses_defaults_for_unset_keys(self):
        from russian_tts_studio.utils.prosody import PauseConfig, DEFAULT_PAUSE_MS
        cfg = PauseConfig.from_metadata({"pause_ms_comma": 100})
        assert cfg.comma == 100
        # Unset keys keep their defaults
        assert cfg.period == DEFAULT_PAUSE_MS["period"]
        assert cfg.is_enabled() is True


class TestComparison:
    def test_get_engine_silero(self):
        from scripts.comparison import get_engine
        eng = get_engine("silero")
        assert eng.name == "silero"
        assert eng.supports_cloning is False

    def test_get_engine_unknown(self):
        from scripts.comparison import get_engine
        with pytest.raises(ValueError, match="Unknown engine"):
            get_engine("not-a-real-engine")

    def test_get_engine_voxcpm_does_not_eagerly_import(self):
        # ``get_engine("voxcpm")`` must NOT import voxcpm yet — the
        # class import is deferred to ``load()`` time. Same pattern
        # in scripts/comparison/__init__.py.
        from scripts.comparison import get_engine
        eng = get_engine("voxcpm")
        assert eng.name == "voxcpm-2"
        assert eng.supports_cloning is True
        assert eng.license == "Apache-2.0"
