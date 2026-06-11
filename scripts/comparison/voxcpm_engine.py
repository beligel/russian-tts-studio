"""VoxCPM2 (OpenBMB) engine wrapper for comparison tests.

Apache-2.0 license. Must run in .venv-voxcpm — see voxcpm_synth.py
for the torch-version rationale.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .base import TTSEngine, TTSOutput

logger = logging.getLogger(__name__)


class VoxCPMEngine(TTSEngine):
    name = "voxcpm-2"
    license = "Apache-2.0"
    supports_cloning = True

    def __init__(self, device: str = "auto"):
        self.device = "cuda" if device == "auto" and self._has_cuda() else "cpu"
        self._model = None
        self._sample_rate: int = 48000

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def load(self) -> None:
        try:
            from voxcpm import VoxCPM
        except ImportError as exc:
            raise RuntimeError(
                "Cannot import the 'voxcpm' package. Install with:\n"
                "  pip install voxcpm==2.0.3\n"
                "VoxCPM2 must run in .venv-voxcpm (torch>=2.5).\n"
                f"Underlying error: {exc}"
            )
        logger.info("Loading VoxCPM2 on %s", self.device)
        self._model = VoxCPM.from_pretrained(
            hf_model_id="OpenBMB/VOXCPM2",
            load_denoiser=False,
            device=self.device,
        )
        sr = getattr(self._model.tts_model, "sample_rate", None)
        if sr:
            self._sample_rate = int(sr)

    def synthesize(
        self,
        text: str,
        reference_audio: Optional[Path] = None,
        output_path: Optional[Path] = None,
        **kwargs,
    ) -> TTSOutput:
        if self._model is None:
            self.load()
        if reference_audio is None:
            raise ValueError("VoxCPM2 requires a reference audio for cloning")

        start = time.time()
        try:
            output_path = Path(output_path) if output_path else \
                Path("output/comparison") / f"voxcpm_{hash(text)}.wav"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            import soundfile as sf
            gen_kwargs = {
                "text": text,
                "reference_wav_path": str(reference_audio),
            }
            # Ultimate mode if reference_text is passed.
            ref_text = kwargs.get("reference_text")
            if ref_text and ref_text.strip():
                gen_kwargs["prompt_wav_path"] = str(reference_audio)
                gen_kwargs["prompt_text"] = ref_text.strip()
            wav_np = self._model.generate(**gen_kwargs)
            sf.write(
                str(output_path),
                np.asarray(wav_np, dtype=np.float32),
                self._sample_rate,
                subtype="FLOAT",
            )

            from russian_tts_studio.utils.audio_utils import load_audio, get_duration
            wav = load_audio(output_path, target_sr=self._sample_rate, mono=True)
            duration = get_duration(wav, self._sample_rate)
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
            logger.exception("VoxCPM2 synthesis failed: %s", e)
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
