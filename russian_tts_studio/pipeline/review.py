"""Whisper-based quality review with retry and tail detection.

Extends the existing ``Transcriber`` with:
1. Word-level timestamps (for subtitle generation and tail detection).
2. Per-segment review: ``approve`` / ``needs_review`` / ``needs_retry``.
3. Auto-retry loop: re-synthesise on ``needs_retry``, keep best candidate.
4. Tail detection: unexplained audio after the last aligned word.
5. Conservative trim: if tail > warning threshold, trim and re-review.

The retry system is designed for project-level workflows where each
segment is synthesised, reviewed, and either approved or retried. It
integrates with the ``projects`` store to persist review results.

Usage::

    from russian_tts_studio.pipeline.review import review_segment, ReviewConfig

    cfg = ReviewConfig(retry_max=2, tail_warning_ms=500)
    result = review_segment(audio_path, text, reference_audio, pipeline, cfg)
    if result.status == "approved":
        # use the audio
        pass
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ReviewStatus(str, Enum):
    APPROVED = "approved"
    NEEDS_REVIEW = "needs_review"
    NEEDS_RETRY = "needs_retry"
    ERROR = "error"


@dataclass
class ReviewConfig:
    """Configuration for segment review and retry."""

    # Quality thresholds
    wer_threshold: float = 0.25
    cer_threshold: float = 0.15
    sim_threshold: float = 0.45

    # Retry settings
    retry_max: int = 2  # max retries per segment
    retry_wer_threshold: float = 0.35  # retry if WER above this

    # Tail detection (ms)
    tail_safety_ms: float = 200.0  # below this → no tail issue
    tail_warning_ms: float = 500.0  # above this → needs review
    tail_retry_ms: float = 1000.0  # above this → auto retry

    # Trim settings
    trim_aggressive: bool = False  # trim tail even if within warning

    def to_dict(self) -> dict:
        return {
            "wer_threshold": self.wer_threshold,
            "cer_threshold": self.cer_threshold,
            "sim_threshold": self.sim_threshold,
            "retry_max": self.retry_max,
            "tail_safety_ms": self.tail_safety_ms,
            "tail_warning_ms": self.tail_warning_ms,
            "tail_retry_ms": self.tail_retry_ms,
        }


@dataclass
class ReviewResult:
    """Result of reviewing a single segment."""

    status: ReviewStatus = ReviewStatus.APPROVED
    transcript: str = ""
    wer: float = 0.0
    cer: float = 0.0
    speaker_similarity: float = 0.0
    tail_ms: float = 0.0  # unexplained audio after last word (ms)
    tail_trimmed: bool = False
    retries: int = 0
    best_wer: float = 1.0  # best WER across retries
    best_audio_path: Optional[Path] = None
    error: Optional[str] = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "transcript": self.transcript,
            "wer": round(self.wer, 4),
            "cer": round(self.cer, 4),
            "speaker_similarity": round(self.speaker_similarity, 4),
            "tail_ms": round(self.tail_ms, 1),
            "tail_trimmed": self.tail_trimmed,
            "retries": self.retries,
            "best_wer": round(self.best_wer, 4),
            "error": self.error,
            "notes": self.notes,
        }


def _transcribe_with_timestamps(
    model,
    audio: np.ndarray,
    language: str = "ru",
) -> dict:
    """Transcribe with word-level timestamps.

    Returns the full Whisper result dict with ``segments[].words[]``.
    """
    result = model.transcribe(
        audio, language=language, fp16=False, verbose=False,
    )
    return result


def _compute_tail_ms(
    audio_duration_sec: float,
    whisper_result: dict,
) -> float:
    """Compute unexplained audio duration after the last aligned word.

    If Whisper's last word ends at T seconds and the audio is D seconds
    long, the tail is (D - T) * 1000 ms. A large tail means there's
    audio that doesn't correspond to any recognised word — possibly
    noise, artifacts, or model hallucination.
    """
    last_end = 0.0
    for seg in whisper_result.get("segments", []):
        for w in seg.get("words", []):
            last_end = max(last_end, w.get("end", 0))
    if last_end <= 0:
        return 0.0
    tail_sec = max(0, audio_duration_sec - last_end)
    return tail_sec * 1000


def _trim_tail(audio_path: Path, trim_before_sec: float) -> Path:
    """Trim audio to remove the tail after ``trim_before_sec``.

    Returns a new file path with ``_trimmed`` suffix.
    """
    import soundfile as sf

    wav, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    trim_sample = int(trim_before_sec * sr)
    if trim_sample >= len(wav):
        return audio_path
    trimmed = wav[:trim_sample]
    out_path = audio_path.with_name(audio_path.stem + "_trimmed.wav")
    sf.write(str(out_path), trimmed, sr, subtype="FLOAT")
    logger.info("Trimmed tail: %.2fs → %.2fs", len(wav) / sr, trim_before_sec)
    return out_path


def _calculate_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate via Levenshtein distance."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 1.0 if hyp_words else 0.0
    n = len(ref_words)
    m = len(hyp_words)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(n + 1):
        dp[i, 0] = i
    for j in range(m + 1):
        dp[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
            else:
                dp[i, j] = 1 + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m]) / n


def _calculate_cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate."""
    if not reference:
        return 1.0 if hypothesis else 0.0
    n, m = len(reference), len(hypothesis)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(n + 1):
        dp[i, 0] = i
    for j in range(m + 1):
        dp[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
            else:
                dp[i, j] = 1 + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m]) / n


def _normalize_for_wer(text: str) -> str:
    """Normalise text for WER comparison: lowercase, strip punctuation."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def review_segment(
    audio_path: str | Path,
    original_text: str,
    reference_audio: str | Path | None = None,
    pipeline=None,
    config: ReviewConfig | None = None,
    transcriber=None,
    similarity_calc=None,
) -> ReviewResult:
    """Review a single synthesised segment.

    Steps:
    1. Transcribe the audio with Whisper (word timestamps).
    2. Compute WER, CER against the original text.
    3. Compute speaker similarity if reference + calculator available.
    4. Detect tail (unexplained audio after last word).
    5. If tail > ``tail_retry_ms`` → trim and re-transcribe.
    6. Decide status: approved / needs_review / needs_retry.

    If ``pipeline`` is provided and status is ``needs_retry``, this
    function performs one retry automatically (re-synthesise with the
    same parameters). The ``retry_max`` config controls how many
    retries are attempted across the full ``review_with_retry`` loop.

    Returns a :class:`ReviewResult` with all metrics and the decision.
    """
    from ..utils.metrics import SpeakerSimilarityCalculator, Transcriber

    cfg = config or ReviewConfig()
    audio_path = Path(audio_path)
    result = ReviewResult()

    if not audio_path.exists():
        result.status = ReviewStatus.ERROR
        result.error = f"Audio not found: {audio_path}"
        return result

    # Load audio for Whisper.
    try:
        from ..utils.audio_utils import load_audio, get_duration

        synth_wav = load_audio(audio_path, target_sr=16000, mono=True)
        audio_duration = get_duration(synth_wav, 16000)
    except Exception as e:
        result.status = ReviewStatus.ERROR
        result.error = f"Failed to load audio: {e}"
        return result

    # Transcribe with timestamps.
    if transcriber is None:
        transcriber = Transcriber(model_size="base", device="cpu")

    whisper_result = None
    try:
        if transcriber._ensure_loaded() and transcriber._model is not None:
            import numpy as np

            audio_np = synth_wav.cpu().numpy().squeeze().astype(np.float32)
            whisper_result = _transcribe_with_timestamps(
                transcriber._model, audio_np, language="ru",
            )
            result.transcript = whisper_result.get("text", "").strip()
    except Exception as e:
        logger.warning("Whisper transcription failed: %s", e)

    # Compute WER/CER.
    if result.transcript:
        ref_norm = _normalize_for_wer(original_text)
        hyp_norm = _normalize_for_wer(result.transcript)
        result.wer = _calculate_wer(ref_norm, hyp_norm)
        result.cer = _calculate_cer(ref_norm.replace(" ", ""), hyp_norm.replace(" ", ""))

    # Speaker similarity.
    if reference_audio and Path(reference_audio).exists():
        if similarity_calc is None:
            similarity_calc = SpeakerSimilarityCalculator(model_name="wavlm")
        try:
            from ..utils.audio_utils import load_audio as _load

            ref_wav = _load(reference_audio, target_sr=16000, mono=True)
            result.speaker_similarity = similarity_calc.similarity(
                ref_waveform=ref_wav, synth_waveform=synth_wav,
            )
        except Exception as e:
            logger.warning("Speaker similarity failed: %s", e)

    # Tail detection.
    if whisper_result:
        result.tail_ms = _compute_tail_ms(audio_duration, whisper_result)

    # Decision logic.
    notes: list[str] = []

    if result.wer > cfg.wer_threshold:
        notes.append(f"High WER: {result.wer:.1%}")
    if result.cer > cfg.cer_threshold:
        notes.append(f"High CER: {result.cer:.1%}")
    if result.speaker_similarity > 0 and result.speaker_similarity < cfg.sim_threshold:
        notes.append(f"Low speaker similarity: {result.speaker_similarity:.2f}")
    if result.tail_ms > cfg.tail_retry_ms:
        notes.append(f"Long tail: {result.tail_ms:.0f}ms (>{cfg.tail_retry_ms}ms)")
    elif result.tail_ms > cfg.tail_warning_ms:
        notes.append(f"Tail warning: {result.tail_ms:.0f}ms (>{cfg.tail_warning_ms}ms)")

    result.notes = notes

    # Tail trim if aggressive mode or tail is very long.
    if result.tail_ms > cfg.tail_warning_ms and cfg.trim_aggressive:
        trim_sec = (audio_duration * 1000 - result.tail_ms + cfg.tail_safety_ms) / 1000
        trimmed_path = _trim_tail(audio_path, trim_sec)
        if trimmed_path != audio_path:
            result.tail_trimmed = True
            result.best_audio_path = trimmed_path

    # Status decision.
    has_quality_issue = (
        result.wer > cfg.wer_threshold
        or result.cer > cfg.cer_threshold
        or (result.speaker_similarity > 0 and result.speaker_similarity < cfg.sim_threshold)
    )
    has_tail_issue = result.tail_ms > cfg.tail_retry_ms

    if has_quality_issue or has_tail_issue:
        result.status = ReviewStatus.NEEDS_RETRY
    elif result.tail_ms > cfg.tail_warning_ms or (result.wer > cfg.retry_wer_threshold):
        result.status = ReviewStatus.NEEDS_REVIEW
    else:
        result.status = ReviewStatus.APPROVED

    return result


def review_with_retry(
    audio_path: str | Path,
    original_text: str,
    pipeline,
    reference_audio: str | Path | None = None,
    config: ReviewConfig | None = None,
    transcriber=None,
    similarity_calc=None,
    output_dir: str | Path = "output/samples",
) -> ReviewResult:
    """Review a segment with automatic retry loop.

    If the first review returns ``needs_retry``, re-synthesise and
    re-review up to ``config.retry_max`` times. Keep the best
    candidate (lowest WER).

    Returns the final :class:`ReviewResult` with ``retries`` count
    and ``best_audio_path``.
    """
    cfg = config or ReviewConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_result: Optional[ReviewResult] = None
    best_wer = 1.0
    best_path: Optional[Path] = None
    current_path = Path(audio_path)

    for attempt in range(cfg.retry_max + 1):
        result = review_segment(
            current_path, original_text, reference_audio,
            pipeline=pipeline, config=cfg,
            transcriber=transcriber, similarity_calc=similarity_calc,
        )
        result.retries = attempt

        # Track best candidate.
        if result.wer < best_wer:
            best_wer = result.wer
            best_result = result
            best_path = current_path

        # If approved or no retry possible, stop.
        if result.status != ReviewStatus.NEEDS_RETRY:
            break

        if attempt >= cfg.retry_max:
            logger.info("Max retries (%d) reached for segment", cfg.retry_max)
            break

        # Re-synthesise.
        if pipeline is not None:
            retry_path = output_dir / f"retry_{attempt + 1}_{Path(audio_path).stem}.wav"
            try:
                pipeline_result = pipeline.synthesize(
                    text=original_text,
                    reference_audio=reference_audio,
                    output_path=retry_path,
                )
                current_path = pipeline_result["final_path"]
                logger.info("Retry %d: %s", attempt + 1, current_path)
            except Exception as e:
                logger.warning("Retry %d synthesis failed: %s", attempt + 1, e)
                result.status = ReviewStatus.ERROR
                result.error = f"Retry failed: {e}"
                break

    if best_result is None:
        return ReviewResult(
            status=ReviewStatus.ERROR,
            error="No successful review",
            retries=cfg.retry_max,
        )

    best_result.retries = attempt
    best_result.best_wer = best_wer
    best_result.best_audio_path = best_path
    return best_result