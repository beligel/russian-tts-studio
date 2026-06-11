"""Quality metrics for TTS evaluation: WER, speaker similarity, MOS proxy."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)


@dataclass
class TTSQualityMetrics:
    """Container for TTS quality metrics."""

    wer: float = 0.0
    cer: float = 0.0
    speaker_similarity: float = 0.0
    duration_sec: float = 0.0
    ref_duration_sec: float = 0.0
    speed_ratio: float = 0.0
    silence_ratio: float = 0.0
    transcript: str = ""
    ref_transcript: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "wer": round(self.wer, 4),
            "cer": round(self.cer, 4),
            "speaker_similarity": round(self.speaker_similarity, 4),
            "duration_sec": round(self.duration_sec, 3),
            "ref_duration_sec": round(self.ref_duration_sec, 3),
            "speed_ratio": round(self.speed_ratio, 3),
            "silence_ratio": round(self.silence_ratio, 4),
            "transcript": self.transcript,
            "ref_transcript": self.ref_transcript,
            "notes": self.notes,
        }


def normalize_text_for_wer(text: str, language: str = "ru") -> str:
    """Normalize text for WER calculation.

    - Lowercase
    - Remove punctuation
    - Normalize whitespace
    - Optionally remove digits
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate using Levenshtein distance.

    WER = (S + D + I) / N, where:
        S = substitutions, D = deletions, I = insertions, N = words in reference.
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) > 0 else 0.0

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
                dp[i, j] = 1 + min(
                    dp[i - 1, j],      # deletion
                    dp[i, j - 1],      # insertion
                    dp[i - 1, j - 1],  # substitution
                )

    return float(dp[n, m]) / n


def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate."""
    if len(reference) == 0:
        return 1.0 if len(hypothesis) > 0 else 0.0

    n = len(reference)
    m = len(hypothesis)
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
                dp[i, j] = 1 + min(
                    dp[i - 1, j],
                    dp[i, j - 1],
                    dp[i - 1, j - 1],
                )

    return float(dp[n, m]) / n


def calculate_silence_ratio(waveform: torch.Tensor, threshold: float = 0.01) -> float:
    """Calculate fraction of audio that is silence."""
    if waveform.dim() > 1:
        waveform = waveform.squeeze(0)
    if len(waveform) == 0:
        return 0.0
    silent = (waveform.abs() < threshold).sum().item()
    return silent / len(waveform)


class SpeakerSimilarityCalculator:
    """Compute cosine similarity between speaker embeddings.

    Uses WavLM or Resemblyzer for speaker embeddings.
    Falls back to simple spectral features if models unavailable.
    """

    def __init__(self, device: str = "cpu", model_name: str = "wavlm"):
        self.device = device
        self.model_name = model_name
        self._model = None
        self._model_loaded = False

    def _try_load_model(self) -> bool:
        if self._model_loaded:
            return self._model is not None
        self._model_loaded = True

        try:
            if self.model_name == "wavlm":
                from transformers import WavLMForXVector
                self._model = WavLMForXVector.from_pretrained(
                    "microsoft/wavlm-base-plus-sv"
                ).to(self.device).eval()
                logger.info("Loaded WavLM for speaker similarity")
                return True
        except Exception as e:
            logger.warning("Could not load %s: %s. Using spectral fallback.", self.model_name, e)

        try:
            if self.model_name == "resemblyzer":
                from resemblyzer import VoiceEncoder
                self._model = VoiceEncoder(device=self.device)
                logger.info("Loaded Resemblyzer for speaker similarity")
                return True
        except Exception as e:
            logger.warning("Could not load Resemblyzer: %s", e)

        return False

    def compute_embedding(self, waveform: torch.Tensor, sample_rate: int = 16000) -> Optional[np.ndarray]:
        """Compute speaker embedding for audio."""
        if not self._try_load_model():
            return None

        try:
            if self.model_name == "wavlm" and self._model is not None:
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                    waveform = resampler(waveform)
                with torch.no_grad():
                    emb = self._model(waveform.to(self.device)).embeddings
                return emb.cpu().numpy().squeeze()
            elif self.model_name == "resemblyzer" and self._model is not None:
                import numpy as np
                wav_np = waveform.cpu().numpy().astype(np.float32)
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                    wav_np = resampler(waveform).cpu().numpy().squeeze().astype(np.float32)
                emb = self._model.embed_utterance(wav_np)
                return emb
        except Exception as e:
            logger.error("Embedding computation failed: %s", e)
            return None

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity between two embeddings."""
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    def similarity(
        self,
        ref_waveform: torch.Tensor,
        synth_waveform: torch.Tensor,
        sample_rate: int = 16000,
    ) -> float:
        """Compute speaker similarity between reference and synthesized audio."""
        ref_emb = self.compute_embedding(ref_waveform, sample_rate)
        synth_emb = self.compute_embedding(synth_waveform, sample_rate)
        if ref_emb is None or synth_emb is None:
            return 0.0
        return self.cosine_similarity(ref_emb, synth_emb)


class Transcriber:
    """Speech-to-text using Whisper for WER calculation."""

    def __init__(self, model_size: str = "base", device: str = "cpu"):
        self.model_size = model_size
        self.device = device
        self._model = None
        self._loaded = False

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return self._model is not None
        self._loaded = True
        try:
            import whisper
            self._model = whisper.load_model(self.model_size, device=self.device)
            logger.info("Loaded Whisper %s for transcription", self.model_size)
            return True
        except ImportError:
            logger.warning("openai-whisper not installed. pip install openai-whisper")
        except Exception as e:
            logger.warning("Could not load Whisper: %s", e)
        return False

    def transcribe(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000,
        language: str = "ru",
    ) -> str:
        """Transcribe audio to text. Returns empty string on failure."""
        if not self._ensure_loaded() or self._model is None:
            return ""
        try:
            import numpy as np
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            audio = waveform.cpu().numpy().squeeze().astype(np.float32)
            result = self._model.transcribe(
                audio, language=language, fp16=False, verbose=False
            )
            return result.get("text", "").strip()
        except Exception as e:
            logger.error("Transcription failed: %s", e)
            return ""
