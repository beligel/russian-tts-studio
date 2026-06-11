"""Silero TTS engine wrapper for comparison tests."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import torch

from .base import TTSEngine, TTSOutput

logger = logging.getLogger(__name__)


class SileroEngine(TTSEngine):
    name = "silero"
    license = "MIT"
    supports_cloning = False

    def __init__(self, speaker: str = "xenia", sample_rate: int = 48000):
        self.speaker = speaker
        self.sample_rate = sample_rate
        self._model = None

    def load(self) -> None:
        self._model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language="ru",
            speaker="v3_1_ru",
            trust_repo=True,
        )

    def synthesize(
        self,
        text: str,
        reference_audio: Optional[Path] = None,
        output_path: Optional[Path] = None,
        **kwargs,
    ) -> TTSOutput:
        if self._model is None:
            self.load()

        start = time.time()
        try:
            output_path = Path(output_path) if output_path else \
                Path("output/comparison") / f"silero_{hash(text)}.wav"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            speaker = kwargs.get("speaker", self.speaker)

            self._model.save_wav(
                text=text,
                speaker=speaker,
                sample_rate=self.sample_rate,
                audio_path=str(output_path),
            )

            from russian_tts_studio.utils.audio_utils import load_audio, get_duration
            wav = load_audio(output_path, target_sr=self.sample_rate, mono=True)
            duration = get_duration(wav, self.sample_rate)
            gen_time = time.time() - start

            return TTSOutput(
                audio_path=output_path,
                duration_sec=duration,
                generation_time_sec=gen_time,
                rtf=gen_time / duration if duration > 0 else 0.0,
                engine=f"{self.name}-{speaker}",
                text=text,
                extra={"speaker": speaker},
            )
        except Exception as e:
            logger.exception("Silero synthesis failed: %s", e)
            return TTSOutput(
                audio_path=Path(""),
                duration_sec=0.0,
                generation_time_sec=time.time() - start,
                rtf=0.0,
                engine=self.name,
                text=text,
                success=False,
                error=str(e),
            )
