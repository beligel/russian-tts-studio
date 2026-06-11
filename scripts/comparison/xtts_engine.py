"""XTTS-v2 (Coqui) engine wrapper for comparison tests.

⚠️ Coqui TTS uses CPML license — non-commercial by default.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from .base import TTSEngine, TTSOutput

logger = logging.getLogger(__name__)


class XTTSEngine(TTSEngine):
    name = "xtts-v2"
    license = "CPML (non-commercial)"
    supports_cloning = True

    def __init__(self, device: str = "auto"):
        self.device = "cuda" if device == "auto" and self._has_cuda() else "cpu"
        self._tts = None

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def load(self) -> None:
        from TTS.api import TTS
        logger.info("Loading XTTS-v2 on %s", self.device)
        self._tts = TTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            gpu=(self.device == "cuda"),
        )

    def synthesize(
        self,
        text: str,
        reference_audio: Optional[Path] = None,
        output_path: Optional[Path] = None,
        **kwargs,
    ) -> TTSOutput:
        if self._tts is None:
            self.load()
        if reference_audio is None:
            raise ValueError("XTTS requires a reference audio")

        start = time.time()
        try:
            output_path = Path(output_path) if output_path else \
                Path("output/comparison") / f"xtts_{hash(text)}.wav"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            self._tts.tts_to_file(
                text=text,
                file_path=str(output_path),
                speaker_wav=[str(reference_audio)],
                language=kwargs.get("language", "ru"),
            )

            from russian_tts_studio.utils.audio_utils import load_audio, get_duration
            wav = load_audio(output_path, target_sr=24000, mono=True)
            duration = get_duration(wav, 24000)
            gen_time = time.time() - start

            return TTSOutput(
                audio_path=output_path,
                duration_sec=duration,
                generation_time_sec=gen_time,
                rtf=gen_time / duration if duration > 0 else 0.0,
                engine=self.name,
                text=text,
            )
        except Exception as e:
            logger.exception("XTTS synthesis failed: %s", e)
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
