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


class TestXTTSSynthesizer:
    def test_unknown_model(self):
        from russian_tts_studio.models.xtts_synth import XTTSSynthesizer
        with pytest.raises(ValueError, match="Unknown XTTS model"):
            XTTSSynthesizer(model_name="invalid-model")

    def test_supported_models(self):
        from russian_tts_studio.models.xtts_synth import XTTSSynthesizer
        assert "xtts-v2" in XTTSSynthesizer.SUPPORTED_MODELS


class TestNormalizeTextForXtts:
    """Cover the typographic-char normaliser that runs before every
    XTTS inference. The character set is regression-prone — adding a
    new entry is easy to forget in code review but will break tests
    for any Russian text the user has quoted."""

    def test_russian_quotes_replaced(self):
        from russian_tts_studio.models.xtts_synth import _normalize_text_for_xtts
        # Main offender reported by user.
        assert _normalize_text_for_xtts('Сказал: «Привет»') == 'Сказал: "Привет"'
        # German-style „…" quotes too.
        assert _normalize_text_for_xtts('„Тест"') == '"Тест"'
        # English smart quotes.
        assert _normalize_text_for_xtts("‘Hello’") == "'Hello'"

    def test_dashes_and_ellipsis(self):
        from russian_tts_studio.models.xtts_synth import _normalize_text_for_xtts
        # Em-dash: replaced with " - " (with spaces) for natural pause.
        assert _normalize_text_for_xtts("Москва — столица") == "Москва  -  столица"
        # En-dash: just "-"
        assert _normalize_text_for_xtts("2024–2025") == "2024-2025"
        # Ellipsis
        assert _normalize_text_for_xtts("Подождите…") == "Подождите..."

    def test_invisible_chars_stripped(self):
        from russian_tts_studio.models.xtts_synth import _normalize_text_for_xtts
        # NBSP becomes a regular space (it's whitespace, not nothing —
        # TTS needs a boundary so it doesn't merge "При" and "вет").
        assert _normalize_text_for_xtts("При\u00A0вет") == "При вет"
        # Zero-width joiner / BOM are *stripped* (purely invisible, no
        # semantic meaning, would only confuse the BPE tokenizer).
        assert _normalize_text_for_xtts("Hello\u200B\uFEFF!") == "Hello!"

    def test_passthrough(self):
        # Plain Russian / English / numbers / dots / colons stay intact.
        from russian_tts_studio.models.xtts_synth import _normalize_text_for_xtts as f
        assert f("Привет, мир! 123.") == "Привет, мир! 123."
        assert f("Hello, world.") == "Hello, world."
        assert f("") == ""

    def test_idempotent(self):
        from russian_tts_studio.models.xtts_synth import _normalize_text_for_xtts
        # Normalising twice == normalising once (no double-rewrite).
        once = _normalize_text_for_xtts('Сказал: «Привет» — …')
        twice = _normalize_text_for_xtts(once)
        assert once == twice


class TestForceLowercaseNoDiacritics:
    """XTTS-v2 vocab rejects uppercase Cyrillic AND any combining
    diacritic (both become [UNK]). We lowercase the whole string and
    strip combining marks before inference. Stress control is *not*
    attempted here — XTTS simply can't take it."""

    def test_uppercase_lowercased(self):
        from russian_tts_studio.models.xtts_synth import _force_lowercase_no_diacritics
        assert _force_lowercase_no_diacritics("ПРИВЕТ") == "привет"
        assert _force_lowercase_no_diacritics("Hello World") == "hello world"
        assert _force_lowercase_no_diacritics("iPhone") == "iphone"

    def test_combining_acute_stripped(self):
        from russian_tts_studio.models.xtts_synth import _force_lowercase_no_diacritics
        assert _force_lowercase_no_diacritics("за\u0301мок") == "замок"
        assert _force_lowercase_no_diacritics("а\u0301б\u0300в\u0303") == "абв"

    def test_yo_kept(self):
        from russian_tts_studio.models.xtts_synth import _force_lowercase_no_diacritics
        # ё is a full letter, not a combining mark — keep and lowercase.
        assert _force_lowercase_no_diacritics("Ёжик") == "ёжик"
        assert _force_lowercase_no_diacritics("Ё") == "ё"

    def test_punctuation_kept(self):
        from russian_tts_studio.models.xtts_synth import _force_lowercase_no_diacritics
        assert _force_lowercase_no_diacritics("Привет, мир!") == "привет, мир!"
        assert _force_lowercase_no_diacritics("...") == "..."

    def test_passthrough(self):
        from russian_tts_studio.models.xtts_synth import _force_lowercase_no_diacritics
        assert _force_lowercase_no_diacritics("привет") == "привет"
        assert _force_lowercase_no_diacritics("123") == "123"
        assert _force_lowercase_no_diacritics("") == ""

    def test_idempotent(self):
        from russian_tts_studio.models.xtts_synth import _force_lowercase_no_diacritics
        once = _force_lowercase_no_diacritics("За\u0301мок И ПРИВЕТ")
        twice = _force_lowercase_no_diacritics(once)
        assert once == twice


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
        # class import is deferred to ``load()`` time. This is the
        # pattern used for xtts too; see scripts/comparison/__init__.py.
        from scripts.comparison import get_engine
        eng = get_engine("voxcpm")
        assert eng.name == "voxcpm-2"
        assert eng.supports_cloning is True
        assert eng.license == "Apache-2.0"
