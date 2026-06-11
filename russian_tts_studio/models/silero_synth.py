"""Silero TTS wrapper — fast Russian fallback without voice cloning."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torchaudio

from .base_synth import SynthesisRequest, SynthesisResult

logger = logging.getLogger(__name__)


class SileroSynthesizer:
    """Silero TTS v4 wrapper.

    Used as a reliable Russian TTS fallback when:
    - Russian TTS Studio3 quality is below threshold (WER > 20%)
    - No reference audio is available
    - Speed is critical (CPU real-time)

    Built-in speakers (v3_1_ru): aidar, baya, kseniya, xenia, eugene, random
    """

    SPEAKERS = ["aidar", "baya", "kseniya", "xenia", "eugene", "random"]

    def __init__(self, device: str = "auto", sample_rate: int = 48000):
        self.device = self._resolve_device(device)
        self.sample_rate = sample_rate
        self._model = None
        self._loaded = False

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cpu"
        return device

    def load(self) -> None:
        if self._loaded:
            return
        try:
            self._model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-models",
                model="silero_tts",
                language="ru",
                speaker="v3_1_ru",
                trust_repo=True,
            )
            self._loaded = True
            logger.info("Silero TTS loaded")
        except Exception as e:
            logger.error("Failed to load Silero: %s", e)
            raise

    def synthesize(
        self,
        text: str,
        speaker: str = "xenia",
        output_path: Optional[str | Path] = None,
    ) -> SynthesisResult:
        if not self._loaded:
            self.load()
        if speaker not in self.SPEAKERS:
            logger.warning("Unknown speaker %s, using 'xenia'", speaker)
            speaker = "xenia"

        start = time.time()
        try:
            self._model.save_wav(
                text=text,
                speaker=speaker,
                sample_rate=self.sample_rate,
                audio_path=str(output_path) if output_path else None,
            )

            if output_path is None:
                output_path = Path("output/samples") / f"silero_{speaker}_{hash(text)}.wav"
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            from ..utils.audio_utils import load_audio, get_duration
            waveform = load_audio(output_path, target_sr=self.sample_rate, mono=True)
            duration = get_duration(waveform, self.sample_rate)
            gen_time = time.time() - start

            return SynthesisResult(
                audio_path=output_path,
                duration_sec=duration,
                generation_time_sec=gen_time,
                rtf=gen_time / duration if duration > 0 else 0.0,
                model=f"silero-{speaker}",
                text=text,
                metadata={"speaker": speaker, "engine": "silero"},
            )
        except Exception as e:
            logger.exception("Silero synthesis failed: %s", e)
            return SynthesisResult(
                audio_path=Path(""),
                duration_sec=0.0,
                generation_time_sec=time.time() - start,
                rtf=0.0,
                model="silero",
                text=text,
                success=False,
                error=str(e),
            )
