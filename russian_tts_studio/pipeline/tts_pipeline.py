"""Main TTS pipeline: XTTS-v2 (primary, voice cloning) + Silero (fallback).

VoxCPM2 is supported as an alternative primary engine (engine="voxcpm").
It runs in a separate venv (.venv-voxcpm) because of torch version
incompatibility with the XTTS venv — the import of VoxCPMSynthesizer
will fail-fast if you started the web server from the wrong venv.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import torch
import torchaudio

from ..models.base_synth import SynthesisRequest, SynthesisResult
from ..models.silero_synth import SileroSynthesizer
from ..models.xtts_synth import XTTSSynthesizer
from ..utils.metrics import (
    SpeakerSimilarityCalculator,
    TTSQualityMetrics,
    Transcriber,
    calculate_cer,
    calculate_silence_ratio,
    calculate_wer,
    normalize_text_for_wer,
)
from ..utils.text_utils import chunk_text_for_tts, normalize_numbers

logger = logging.getLogger(__name__)


class QualityCheckOutcome(str, Enum):
    PASS = "pass"
    FALLBACK = "fallback"
    FAIL = "fail"


@dataclass
class PipelineConfig:
    """Configuration for the TTS pipeline."""

    enable_fallback: bool = True
    wer_threshold: float = 0.20
    cer_threshold: float = 0.15
    sim_threshold: float = 0.50
    enable_postprocess: bool = True
    target_dbfs: float = -20.0
    enable_quality_check: bool = True
    whisper_model: str = "base"
    similarity_model: str = "wavlm"
    device: str = "auto"
    xtts_model: str = "xtts-v2"
    # Whether WER/CER (from Whisper ASR) can block the result and trigger
    # Silero fallback. Whisper is unreliable on CPU for short synthesized
    # clips in Russian, so the default is to *report* WER/CER in metrics
    # but never let them cause a fallback. Speaker similarity and silence
    # ratio remain gating signals.
    wer_blocks: bool = False


class TTSPipeline:
    """Production TTS pipeline.

    Flow:
        1. Try XTTS-v2 with voice cloning (when a reference is provided)
        2. Quality check (WER + speaker similarity)
        3. Fallback to Silero if quality below threshold
        4. Post-processing (normalization, trimming)
        5. Final output
    """

    def __init__(self, config: Optional[PipelineConfig] = None, engine: str = "xtts"):
        self.config = config or PipelineConfig()
        # ``engine`` is a *constructor* parameter (NOT a PipelineConfig
        # field — that caused a 500 once). The synthesiser slot is
        # lazily filled in initialize() based on the engine name.
        # Supported values: "xtts" (default), "voxcpm".
        if engine not in ("xtts", "voxcpm"):
            raise ValueError(
                f"Unknown engine: {engine!r}. Supported: 'xtts', 'voxcpm'"
            )
        self.engine = engine
        self.synth: Optional[object] = None  # XTTSSynthesizer or VoxCPMSynthesizer
        self.silero: Optional[SileroSynthesizer] = None
        self.transcriber: Optional[Transcriber] = None
        self.similarity_calc: Optional[SpeakerSimilarityCalculator] = None
        self._initialized = False

    def initialize(self) -> None:
        """Lazy initialization of all components."""
        if self._initialized:
            return
        logger.info(
            "Initializing TTS pipeline (engine=%s, model=%s, device=%s)…",
            self.engine, self.config.xtts_model, self.config.device,
        )
        # Coqui TTS asks the user to confirm the CPML license on first
        # download. COQUI_TOS_AGREED=1 is checked in
        # TTS.utils.manage.ModelManager.tos_agreed — without it the
        # call blocks on stdin. start.sh already exports it.
        if self.engine == "voxcpm":
            from ..models.voxcpm_synth import VoxCPMSynthesizer
            self.synth = VoxCPMSynthesizer(
                device=self.config.device,
            )
        else:
            self.synth = XTTSSynthesizer(
                model_name=self.config.xtts_model,
                device=self.config.device,
            )
        if self.config.enable_fallback:
            self.silero = SileroSynthesizer()
        if self.config.enable_quality_check:
            self.transcriber = Transcriber(model_size=self.config.whisper_model)
            self.similarity_calc = SpeakerSimilarityCalculator(
                model_name=self.config.similarity_model,
            )
        self._initialized = True

    def synthesize(
        self,
        text: str,
        reference_audio: Optional[str | Path] = None,
        reference_text: Optional[str] = None,
        instruct: Optional[str] = None,
        output_path: Optional[str | Path] = None,
        speaker_fallback: str = "xenia",
        quality_check: Optional[bool] = None,
        speed: float = 0.9,
    ) -> dict:
        """Synthesize text with quality checks and fallback.

        Returns a dict with:
            - result: SynthesisResult
            - metrics: TTSQualityMetrics
            - outcome: QualityCheckOutcome
            - final_path: Path to final audio
        """
        self.initialize()

        do_check = quality_check if quality_check is not None else self.config.enable_quality_check
        text = normalize_numbers(text, language="ru")

        result: Optional[SynthesisResult] = None
        metrics = TTSQualityMetrics()
        outcome = QualityCheckOutcome.FAIL
        already_handled = False  # True if Silero was already used as the primary

        if reference_audio is not None and Path(reference_audio).exists():
            logger.info(
                "Attempting %s with reference %s", self.engine.upper(), reference_audio,
            )
            request = SynthesisRequest(
                text=text,
                reference_audio=reference_audio,
                reference_text=reference_text,
                instruct=instruct,
                output_path=output_path,
                speed=speed,
            )
            assert self.synth is not None
            result = self.synth.synthesize(request)

            if result.success and do_check:
                outcome, metrics = self._quality_check(result, reference_audio, text)
            elif result.success:
                outcome = QualityCheckOutcome.PASS
            else:
                outcome = QualityCheckOutcome.FAIL
        else:
            logger.info("No reference audio, using Silero")
            if self.silero is None or not self.config.enable_fallback:
                raise RuntimeError(
                    "No reference audio provided and Silero fallback is disabled — "
                    f"{self.engine.upper()} requires a reference clip to clone a voice."
                )
            fallback_path = self._make_fallback_path(output_path, speaker_fallback)
            assert self.silero is not None
            result = self.silero.synthesize(
                text=text,
                speaker=speaker_fallback,
                output_path=fallback_path,
            )
            outcome = QualityCheckOutcome.FALLBACK
            already_handled = True

        if not already_handled and \
                outcome in (QualityCheckOutcome.FALLBACK, QualityCheckOutcome.FAIL) and \
                self.config.enable_fallback and self.silero is not None:
            logger.info("Falling back to Silero (reason: %s)", outcome.value)
            fallback_path = self._make_fallback_path(output_path, speaker_fallback)
            assert self.silero is not None
            result = self.silero.synthesize(
                text=text,
                speaker=speaker_fallback,
                output_path=fallback_path,
            )
            outcome = QualityCheckOutcome.FALLBACK

        if result is None or not result.success:
            raise RuntimeError(f"Pipeline failed: {result.error if result else 'no result'}")

        final_path = result.audio_path
        if self.config.enable_postprocess:
            final_path = self._postprocess(final_path)

        return {
            "result": result,
            "metrics": metrics,
            "outcome": outcome,
            "final_path": final_path,
        }

    def _quality_check(
        self,
        result: SynthesisResult,
        reference_audio: str | Path,
        original_text: str,
    ) -> tuple[QualityCheckOutcome, TTSQualityMetrics]:
        """Run quality checks and decide on outcome."""
        assert self.transcriber is not None
        from ..utils.audio_utils import load_audio, get_duration

        metrics = TTSQualityMetrics()
        try:
            synth_wav = load_audio(result.audio_path, target_sr=16000, mono=True)
            ref_wav = load_audio(reference_audio, target_sr=16000, mono=True)

            metrics.transcript = self.transcriber.transcribe(synth_wav, language="ru")
            metrics.ref_transcript = original_text

            if metrics.transcript:
                ref_norm = normalize_text_for_wer(original_text)
                hyp_norm = normalize_text_for_wer(metrics.transcript)
                metrics.wer = calculate_wer(ref_norm, hyp_norm)
                metrics.cer = calculate_cer(
                    ref_norm.replace(" ", ""),
                    hyp_norm.replace(" ", ""),
                )

            if self.similarity_calc is not None:
                metrics.speaker_similarity = self.similarity_calc.similarity(
                    ref_waveform=ref_wav,
                    synth_waveform=synth_wav,
                )

            metrics.duration_sec = get_duration(synth_wav, 16000)
            metrics.ref_duration_sec = get_duration(ref_wav, 16000)
            metrics.silence_ratio = calculate_silence_ratio(synth_wav)

            notes: list[str] = []
            wer_bad = metrics.wer > self.config.wer_threshold
            cer_bad = metrics.cer > self.config.cer_threshold
            sim_bad = (
                metrics.speaker_similarity > 0
                and metrics.speaker_similarity < self.config.sim_threshold
            )
            if wer_bad:
                notes.append(f"High WER: {metrics.wer:.1%}")
            if cer_bad:
                notes.append(f"High CER: {metrics.cer:.1%}")
            if sim_bad:
                notes.append(f"Low speaker similarity: {metrics.speaker_similarity:.2f}")
            metrics.notes = notes

            # WER/CER are reported in metrics (for observability) but only
            # gate the outcome when ``wer_blocks`` is enabled. On CPU
            # Whisper often mis-transcribes short Russian clips, causing
            # spurious fallbacks; speaker similarity and silence ratio are
            # the reliable signals.
            gating_failures = [sim_bad]
            if self.config.wer_blocks:
                gating_failures.extend([wer_bad, cer_bad])
            else:
                if wer_bad or cer_bad:
                    notes.append(
                        "WER/CER reported but not gating (wer_blocks=False)"
                    )

            if any(gating_failures):
                outcome = QualityCheckOutcome.FALLBACK
            else:
                outcome = QualityCheckOutcome.PASS
                logger.info(
                    "Quality OK: WER=%.1f%%, CER=%.1f%%, SIM=%.2f",
                    metrics.wer * 100, metrics.cer * 100, metrics.speaker_similarity,
                )

            return outcome, metrics
        except Exception as e:
            logger.exception("Quality check failed: %s", e)
            return QualityCheckOutcome.PASS, metrics

    def _postprocess(self, audio_path: Path) -> Path:
        """Apply post-processing: trim silence, normalize loudness."""
        from ..utils.audio_utils import (
            load_audio, normalize_loudness, trim_silence, get_duration,
        )

        waveform = load_audio(audio_path, target_sr=self.sample_rate_for_postprocess(), mono=False)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        processed = trim_silence(waveform.squeeze(0), threshold=0.01)
        processed = normalize_loudness(processed, target_dbfs=self.config.target_dbfs)

        out_path = audio_path.with_name(audio_path.stem + "_processed.wav")
        torchaudio.save(str(out_path), processed.unsqueeze(0), self.sample_rate_for_postprocess())
        logger.info(
            "Post-processed: %s (%.2fs)",
            out_path.name, get_duration(processed, self.sample_rate_for_postprocess()),
        )
        return out_path

    @staticmethod
    def sample_rate_for_postprocess() -> int:
        return 22050

    def _make_fallback_path(
        self,
        primary: Optional[str | Path],
        speaker: str,
    ) -> Path:
        if primary:
            p = Path(primary)
        else:
            p = Path("output/samples") / "fallback.wav"
        return p.with_name(f"{p.stem}_silero_{speaker}{p.suffix or '.wav'}")

    def cleanup(self) -> None:
        if self.synth is not None:
            self.synth.cleanup()
        self.synth = None
        if self.silero is not None:
            try:
                self.silero.cleanup()
            except Exception:
                pass
        self._initialized = False
        logger.info("Pipeline cleaned up")
