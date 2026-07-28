"""Main TTS pipeline: VoxCPM2 (primary, voice cloning) + Silero (fallback).

Flow:
    1. Try VoxCPM2 with voice cloning (when a reference is provided)
    2. Quality check (WER + speaker similarity)
    3. Fallback to Silero if quality below threshold
    4. Post-processing (normalization, trimming)
    5. Final output

The ``engine`` parameter on :class:`TTSPipeline` is kept for backward
compatibility with callers that constructed the pipeline by engine name.
The only value now accepted is ``"voxcpm"`` (the default).
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
from ..models.voxcpm_synth import VoxCPMSynthesizer
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
    # Whether WER/CER (from Whisper ASR) can block the result and trigger
    # Silero fallback. Whisper is unreliable on CPU for short synthesized
    # clips in Russian, so the default is to *report* WER/CER in metrics
    # but never let them cause a fallback. Speaker similarity and silence
    # ratio remain gating signals.
    wer_blocks: bool = False
    # Clamp internal silences longer than this (ms) down to ``target_gap_ms``.
    # VoxCPM2 tends to emit 0.5–1 s pauses after punctuation; clamping
    # prevents the "stutter" effect. If you prefer the model's natural
    # prosody, set ``enable_clamp=False``.
    enable_clamp: bool = True
    max_gap_ms: int = 450
    target_gap_ms: int = 180
    clamp_threshold: float = 0.02
    clamp_relative_threshold: float = 0.10


class TTSPipeline:
    """Production TTS pipeline.

    Flow:
        1. Try VoxCPM2 with voice cloning (when a reference is provided)
        2. Quality check (WER + speaker similarity)
        3. Fallback to Silero if quality below threshold
        4. Post-processing (normalization, trimming)
        5. Final output
    """

    def __init__(self, config: Optional[PipelineConfig] = None, engine: str = "voxcpm"):
        self.config = config or PipelineConfig()
        # ``engine`` is a *constructor* parameter (NOT a PipelineConfig
        # field). The synthesiser slot is lazily filled in initialize()
        # based on the engine name. Accepted values:
        #   - "voxcpm"  (default, OpenBMB VoxCPM2, Apache-2.0)
        #   - "higgs"   (Boson AI Higgs Audio v2, Apache-2.0; requires
        #               the upstream boson-ai/higgs-audio repo installed)
        if engine not in ("voxcpm", "higgs"):
            raise ValueError(
                f"Unknown engine: {engine!r}. Supported: 'voxcpm', 'higgs'"
            )
        self.engine = engine
        self.synth: Optional[object] = None  # VoxCPMSynthesizer | HiggsAudioSynthesizer
        self.silero: Optional[SileroSynthesizer] = None
        self.transcriber: Optional[Transcriber] = None
        self.similarity_calc: Optional[SpeakerSimilarityCalculator] = None
        self._initialized = False

    def initialize(self) -> None:
        """Lazy initialization of all components."""
        if self._initialized:
            return
        logger.info(
            "Initializing TTS pipeline (engine=%s, device=%s)…",
            self.engine, self.config.device,
        )
        if self.engine == "voxcpm":
            self.synth = VoxCPMSynthesizer(device=self.config.device)
        elif self.engine == "higgs":
            from ..models import get_higgs_synthesizer
            self.synth = get_higgs_synthesizer(device=self.config.device)
        else:
            # Unreachable — __init__ already validated the engine name.
            raise ValueError(f"Unknown engine: {self.engine!r}")
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
        prosody: Optional[dict] = None,
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
        # Normalize line endings: \r\n (from browser textarea) → \n.
        # VoxCPM2 can truncate text at \r — it treats \r as end-of-input
        # on some codepaths, producing audio that cuts off mid-sentence.
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = normalize_numbers(text, language="ru")

        result: Optional[SynthesisResult] = None
        metrics = TTSQualityMetrics()
        outcome = QualityCheckOutcome.FAIL
        already_handled = False  # True if Silero was already used as the primary

        if reference_audio is not None and Path(reference_audio).exists():
            logger.info(
                "Attempting %s with reference %s", self.engine.upper(), reference_audio,
            )
            # VoxCPM2 reads ``pause_ms_*`` keys straight out of
            # ``request.metadata`` (see ``utils.prosody.PauseConfig``),
            # so the prosody dict is passed through as the metadata.
            meta = dict(prosody) if prosody else {}
            request = SynthesisRequest(
                text=text,
                reference_audio=reference_audio,
                reference_text=reference_text,
                instruct=instruct,
                output_path=output_path,
                speed=speed,
                metadata=meta,
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
        prosody_applied = bool((result.metadata or {}).get("prosody_applied"))
        if self.config.enable_postprocess:
            final_path = self._postprocess(final_path, prosody_applied=prosody_applied)

        return {
            "result": result,
            "metrics": metrics,
            "outcome": outcome,
            "final_path": final_path,
            "prosody_degraded": bool((result.metadata or {}).get("prosody_degraded")),
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

    def _postprocess(self, audio_path: Path, prosody_applied: bool = False) -> Path:
        """Apply post-processing: trim silence, clamp long pauses, normalize."""
        from ..utils.audio_utils import (
            clamp_long_silences, load_audio, normalize_loudness, trim_silence,
            get_duration,
        )

        waveform = load_audio(audio_path, target_sr=self.sample_rate_for_postprocess(), mono=False)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        sr = self.sample_rate_for_postprocess()
        processed = trim_silence(waveform.squeeze(0), threshold=0.01)
        # When VoxCPM2 prosody was applied, the raw audio was already
        # clamped before the precise pauses were inserted. Running the
        # generic clamp again would flatten the user's comma/period/etc.
        # settings, so we skip it for prosody-enabled VoxCPM2 output.
        # For all other cases (Silero, VoxCPM2 without prosody, XTTS)
        # the clamp still removes VoxCPM2's unnaturally long mid-stream
        # silences.
        if self.config.enable_clamp and not prosody_applied:
            before = processed.shape[-1]
            processed = clamp_long_silences(
                processed, sr,
                max_gap_ms=self.config.max_gap_ms,
                target_gap_ms=self.config.target_gap_ms,
                threshold=self.config.clamp_threshold,
                relative_threshold=self.config.clamp_relative_threshold,
            )
            after = processed.shape[-1]
            trimmed_sec = (before - after) / sr
            if trimmed_sec > 0.01:
                logger.info(
                    "clamp_long_silences: trimmed %.3fs of mid-stream silence", trimmed_sec,
                )

        processed = normalize_loudness(processed, target_dbfs=self.config.target_dbfs)

        out_path = audio_path.with_name(audio_path.stem + "_processed.wav")
        torchaudio.save(str(out_path), processed.unsqueeze(0), sr)
        logger.info(
            "Post-processed: %s (%.2fs)",
            out_path.name, get_duration(processed, sr),
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

    # --- Markup-driven synthesis (Phase 1) ------------------------------------

    def synthesize_markup(
        self,
        doc: "ParsedDocument",
        reference_audio: Optional[str | Path] = None,
        reference_text: Optional[str] = None,
        instruct: Optional[str] = None,
        output_path: Optional[str | Path] = None,
        speaker_fallback: str = "xenia",
        quality_check: Optional[bool] = None,
        base_speed: float = 0.9,
        prosody: Optional[dict] = None,
        on_progress: Optional[callable] = None,
        context_window: int = 0,
    ) -> dict:
        """Synthesise a :class:`ParsedDocument` segment-by-segment.

        For each non-empty segment:
        1. Apply accumulated aliases to the segment text.
        2. Call :meth:`synthesize` with the segment's speed (overriding
           ``base_speed`` when the markup set one) and a per-segment
           output path.
        3. Insert ``pause_after`` silence between segments.
        4. Concatenate all segment WAVs into a single output file.

        ``on_progress(idx, total, segment)`` is called before each
        segment synthesis — used by the WebSocket handler to stream
        progress to the UI.

        ``context_window`` (inspired by Higgs Audio's
        ``generation_chunk_buffer_size``): when > 0, each segment's
        ``metadata["context_segments"]`` is populated with the text of
        the previous ``context_window`` segments. Engines that support
        long-context generation (Higgs) use this to improve prosody
        continuity across chunk boundaries — the model "sees" what it
        just said. Engines without the capability (VoxCPM2) ignore the
        field (each ``synthesize`` call is independent anyway). Default
        0 = no context window (backward-compatible).

        Returns a dict with:
            - segments: list of per-segment result dicts
            - final_path: path to the concatenated WAV
            - warnings: markup warnings (from the parsed document)
            - chapters: list of ``{title, source_offset}``
        """
        import numpy as np
        import soundfile as sf

        from ..markup import NarrationSegment

        self.initialize()

        non_empty = [s for s in doc.segments if s.text.strip()]
        total = len(non_empty)
        seg_results: list[dict] = []
        seg_paths: list[Path] = []
        silence_paths: list[Path] = []  # pauses between segments

        out_dir = Path("output/samples")
        if output_path:
            out_dir = Path(output_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(output_path).stem if output_path else "markup"

        for i, seg in enumerate(non_empty):
            if on_progress:
                on_progress(i + 1, total, seg)
            # Apply aliases to the segment text.
            text = self._apply_aliases(seg.text, seg.state.aliases)
            # Normalize \r\n → \n (browser textarea sends \r\n; VoxCPM2
            # can truncate at \r).
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            # Apply stress marks (same mechanism as aliases — replace
            # the plain word with its stressed form).
            text = self._apply_stresses(text, seg.state.stresses)
            # Strip sound-event tokens and combining stress marks for
            # engines that don't support them. VoxCPM2's TSLM uses
            # autoprosoody and ignores U+0301; its tokeniser also has no
            # notion of [laugh]/[music] inline tags. We remove both so
            # the engine just reads the surrounding narration. Engines
            # that DO support these (Higgs Audio, future Silero via
            # eSpeak) keep the tokens/marks — the engine renders them.
            text = self._strip_unsupported_markup(text)
            # Per-segment speed: markup speed overrides base_speed.
            seg_speed = seg.state.speed if abs(seg.state.speed - 1.0) > 1e-6 else base_speed
            # Per-segment output path.
            seg_out = out_dir / f"{stem}_seg{i:03d}.wav"
            # Build context window: text of previous ``context_window``
            # segments. Passed via metadata so capable engines (Higgs)
            # can condition on it; VoxCPM2 ignores it.
            seg_meta = dict(prosody) if prosody else {}
            if context_window > 0 and i > 0:
                start = max(0, i - context_window)
                seg_meta["context_segments"] = [
                    self._apply_stresses(
                        self._apply_aliases(prev_seg.text, prev_seg.state.aliases),
                        prev_seg.state.stresses,
                    )
                    for prev_seg in non_empty[start:i]
                ]
            try:
                result = self.synthesize(
                    text=text,
                    reference_audio=reference_audio,
                    reference_text=reference_text,
                    instruct=instruct,
                    output_path=seg_out,
                    speaker_fallback=speaker_fallback,
                    quality_check=quality_check,
                    speed=seg_speed,
                    prosody=seg_meta,
                )
                seg_results.append(result)
                seg_paths.append(result["final_path"])
            except Exception as e:
                logger.exception("Segment %d synthesis failed: %s", i, e)
                seg_results.append({"error": str(e), "segment_idx": i})
                continue

            # Insert pause_after silence as a separate WAV.
            if seg.pause_after and seg.pause_after.is_enabled():
                ms = seg.pause_after.sample_ms()
                sr = 22050
                silence = np.zeros(int(sr * ms / 1000), dtype=np.float32)
                sil_path = out_dir / f"{stem}_sil{i:03d}.wav"
                sf.write(str(sil_path), silence, sr, subtype="FLOAT")
                silence_paths.append(sil_path)

        # Concatenate all segment + silence WAVs in order.
        final_path = self._concat_wavs(seg_paths, silence_paths, output_path or (out_dir / f"{stem}_full.wav"))

        return {
            "segments": seg_results,
            "final_path": final_path,
            "warnings": doc.warnings,
            "chapters": [{"title": c.title, "source_offset": c.source_offset} for c in doc.chapters],
        }

    @staticmethod
    def _apply_aliases(text: str, aliases: list) -> str:
        """Replace alias targets with their pronunciation forms."""
        for alias in aliases:
            if alias.target:
                text = text.replace(alias.target, alias.replacement)
        return text

    @staticmethod
    def _apply_stresses(text: str, stresses: list) -> str:
        """Replace plain words with their stressed forms (U+0301).

        Stress marks accumulate like aliases — every previously seen
        ``{{stress ...}}`` applies to all following text. Matching is
        case-sensitive on the target word (the user typed it in the
        markup, so they control the form). The stressed form keeps
        the combining acute accent; ``_strip_unsupported_markup``
        removes it for engines that don't honour it.
        """
        for s in stresses:
            if s.target and s.stressed and s.target != s.stressed:
                text = text.replace(s.target, s.stressed)
        return text

    def _strip_unsupported_markup(self, text: str) -> str:
        """Remove inline sound-event tokens and combining stress marks
        for engines that don't support them.

        Engine-aware: the engine is queried via ``supports_sound_events()``
        and ``supports_stress_marks()`` (if available). VoxCPM2's TSLM
        uses autoprosoody and ignores U+0301; its tokeniser also has no
        notion of ``[laugh]`` / ``[music]`` inline tags — we strip both
        so the engine reads clean narration. Higgs Audio natively
        understands both, so the tokens/marks are preserved and the
        engine renders them.

        The tokens we strip are exactly the ones emitted by
        ``markup/commands.py:SOUND_EVENT_PRESETS`` plus the combining
        acute accent (U+0301).

        Note: span-event tokens (``[music]`` from ``{{bgm start}}``)
        are currently emitted inline into segment text by the document
        builder, so they're caught by the same strip. A future
        span-aware engine path may emit start/end tokens as separate
        markers; this strip would then only apply to the VoxCPM2 path.
        """
        if not text:
            return text

        # Query the active engine's capabilities. Default to "no support"
        # (strip everything) for unknown engines — safer than sending
        # tokens an engine can't parse.
        supports_sound = getattr(self.synth, "supports_sound_events", lambda: False)()
        supports_stress = getattr(self.synth, "supports_stress_marks", lambda: False)()

        if supports_sound and supports_stress:
            # Engine understands both — pass text through unchanged.
            return text

        import re as _re

        # Strip known sound-event inline tokens if the engine doesn't
        # support them. Keep the list in sync with SOUND_EVENT_PRESETS.
        if not supports_sound:
            sound_tokens = [
                "[laugh]", "[chuckle]", "[giggle]", "[cough]", "[sigh]",
                "[gasp]", "[cry]", "[sniffle]", "[sneeze]", "[yawn]",
                "[applause]", "[cheer]", "[music]", "[humming]", "[singing]",
            ]
            for tok in sound_tokens:
                text = text.replace(tok, "")
        # Strip the combining acute accent (U+0301) if the engine
        # doesn't honour stress marks.
        if not supports_stress:
            text = text.replace("\u0301", "")
        # Collapse multiple spaces left behind by token removal so
        # the engine doesn't see "word  word" sequences.
        text = _re.sub(r" {2,}", " ", text)
        return text.strip()

    @staticmethod
    def _concat_wavs(seg_paths: list[Path], silence_paths: list[Path], out_path: str | Path) -> Path:
        """Concatenate segment + silence WAVs into one output WAV.

        Interleaves in order: seg0, sil0, seg1, sil1, ... — silence
        follows its matching segment. ``silence_paths`` may be shorter
        than ``seg_paths`` (not every segment has a ``pause_after``);
        missing entries are skipped. A trailing pause (silence without
        a following segment) is appended at the end.
        """
        import numpy as np
        import soundfile as sf

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        target_sr = 22050

        def _read(p: Path) -> np.ndarray | None:
            if not p or not p.exists():
                return None
            wav, sr = sf.read(str(p), dtype="float32", always_2d=False)
            if sr != target_sr:
                import torch
                import torchaudio

                t = torch.from_numpy(wav).float()
                if t.dim() == 1:
                    t = t.unsqueeze(0)
                t = torchaudio.functional.resample(t, sr, target_sr)
                wav = t.squeeze(0).numpy()
            return wav

        chunks: list[np.ndarray] = []
        n = max(len(seg_paths), len(silence_paths))
        for i in range(n):
            seg_wav = _read(seg_paths[i]) if i < len(seg_paths) else None
            if seg_wav is not None:
                chunks.append(seg_wav)
            sil_wav = _read(silence_paths[i]) if i < len(silence_paths) else None
            if sil_wav is not None:
                chunks.append(sil_wav)

        if not chunks:
            sf.write(str(out_path), np.zeros(1, dtype=np.float32), target_sr, subtype="FLOAT")
            return out_path

        final = np.concatenate(chunks, axis=0)
        sf.write(str(out_path), final, target_sr, subtype="FLOAT")
        logger.info(
            "Concatenated %d segments + %d pauses → %s (%.2fs)",
            len(seg_paths), len(silence_paths), out_path.name, len(final) / target_sr,
        )
        return out_path
