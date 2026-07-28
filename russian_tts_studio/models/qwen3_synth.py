"""Qwen3-TTS inference wrapper.

Qwen3-TTS (Qwen team, Alibaba Cloud) is a series of TTS models supporting:
- CustomVoice: 9 premium timbres with instruction control
- VoiceDesign: free-form voice design from natural language descriptions
- Base: 3-second voice clone from reference audio

Models: 0.6B and 1.7B parameters, 10 languages (including Russian).
Package: ``pip install qwen-tts``

License: Apache-2.0 (same as VoxCPM2).

This wrapper provides a unified interface compatible with
:class:`SynthesisRequest` / :class:`SynthesisResult` from
``base_synth.py``, so the TTS pipeline can switch between VoxCPM2
and Qwen3-TTS transparently.

Usage::

    synth = Qwen3Synthesizer(device="auto")
    result = synth.synthesize(SynthesisRequest(
        text="Привет, мир!",
        reference_audio="speaker.wav",
        reference_text="Привет, мир!",
    ))
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from .base_synth import SynthesisRequest, SynthesisResult

logger = logging.getLogger(__name__)


class Qwen3Synthesizer:
    """High-level wrapper around Qwen3-TTS.

    Supports three modes:
    - **clone**: voice cloning from reference audio (Base model)
    - **custom_voice**: preset speakers with instruction control
    - **voice_design**: natural-language voice description

    The mode is determined by the request:
    - If ``reference_audio`` is provided → clone mode
    - If ``request.metadata["qwen3_mode"] == "custom_voice"`` → custom voice
    - If ``request.metadata["qwen3_mode"] == "voice_design"`` → voice design
    - Otherwise → clone mode (default)
    """

    SUPPORTED_MODELS = {
        "qwen3-0.6b-base": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "qwen3-1.7b-base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "qwen3-0.6b-custom": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "qwen3-1.7b-custom": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "qwen3-1.7b-design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    }

    def __init__(
        self,
        model_name: str = "qwen3-1.7b-base",
        device: str = "auto",
        dtype: str = "auto",
        language: str = "Russian",
        speaker: str = "Serena",
        instruct: str = "",
        sample_rate: int = 24000,
    ):
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown Qwen3 model: {model_name}. "
                f"Supported: {list(self.SUPPORTED_MODELS.keys())}"
            )
        self.model_name = model_name
        self.model_id = self.SUPPORTED_MODELS[model_name]
        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(dtype)
        self.language = language
        self.speaker = speaker
        self.instruct = instruct
        self.sample_rate = sample_rate
        self._model = None
        self._loaded = False

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    @staticmethod
    def _resolve_dtype(dtype: str) -> str:
        if dtype == "auto":
            import torch

            if torch.cuda.is_available():
                return "bfloat16"
            return "float32"
        return dtype

    def load(self) -> None:
        """Load the Qwen3-TTS model. Idempotent."""
        if self._loaded:
            return

        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            raise RuntimeError(
                "Cannot import 'qwen_tts'. Install with:\n"
                "  pip install qwen-tts\n"
                f"Underlying error: {exc}"
            ) from exc

        import torch

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.dtype, torch.bfloat16)

        try:
            logger.info(
                "Loading Qwen3-TTS %s on %s (dtype=%s)…",
                self.model_id, self.device, self.dtype,
            )
            self._model = Qwen3TTSModel.from_pretrained(
                self.model_id,
                device_map=f"{self.device}:0" if self.device == "cuda" else self.device,
                dtype=torch_dtype,
            )
            self._loaded = True
            logger.info("Qwen3-TTS loaded successfully")
        except Exception as e:
            logger.error("Failed to load Qwen3-TTS: %s", e)
            raise

    def is_loaded(self) -> bool:
        return self._loaded

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        """Synthesize text via Qwen3-TTS.

        Mode is determined by request metadata:
        - ``qwen3_mode = "clone"`` (default) — voice clone from reference
        - ``qwen3_mode = "custom_voice"`` — preset speaker + instruct
        - ``qwen3_mode = "voice_design"`` — natural language voice description
        """
        if not self._loaded:
            self.load()

        meta = request.metadata or {}
        mode = meta.get("qwen3_mode", "clone")
        language = meta.get("language", self.language)
        speaker = meta.get("speaker", self.speaker)
        instruct = request.instruct or meta.get("instruct", self.instruct)

        start_time = time.time()

        try:
            import soundfile as _sf

            if mode == "custom_voice":
                wavs, sr = self._model.generate_custom_voice(
                    text=request.text,
                    language=language,
                    speaker=speaker,
                    instruct=instruct or None,
                )
            elif mode == "voice_design":
                wavs, sr = self._model.generate_voice_design(
                    text=request.text,
                    language=language,
                    instruct=instruct or request.text,
                )
            else:
                # Clone mode — needs reference audio
                ref_path = Path(request.reference_audio) if request.reference_audio else None
                if ref_path and not ref_path.exists():
                    raise FileNotFoundError(f"Reference audio not found: {ref_path}")

                voice_clone_prompt = None
                if ref_path:
                    ref_text = request.reference_text or meta.get("ref_text", "")
                    voice_clone_prompt = self._model.create_voice_clone_prompt(
                        ref_audio=str(ref_path),
                        ref_text=ref_text,
                        x_vector_only_mode=not ref_text,
                    )

                wavs, sr = self._model.generate_voice_clone(
                    text=request.text,
                    language=language,
                    ref_audio=str(ref_path) if ref_path else None,
                    ref_text=request.reference_text or meta.get("ref_text", ""),
                    voice_clone_prompt=voice_clone_prompt,
                )

            wav_np = np.array(wavs[0], dtype=np.float32)
            output_path = self._resolve_output_path(request)

            sf.write(str(output_path), wav_np, sr, subtype="FLOAT")
            self.sample_rate = sr

            duration = len(wav_np) / sr
            gen_time = time.time() - start_time
            rtf = gen_time / duration if duration > 0 else 0.0

            logger.info(
                "Qwen3-TTS %s done: %.2fs audio in %.2fs (RTF=%.3f, mode=%s)",
                self.model_name, duration, gen_time, rtf, mode,
            )

            return SynthesisResult(
                audio_path=output_path,
                duration_sec=duration,
                generation_time_sec=gen_time,
                rtf=rtf,
                model=f"qwen3-{mode}",
                text=request.text,
                reference=Path(request.reference_audio) if request.reference_audio else None,
                metadata=meta,
            )

        except Exception as e:
            logger.exception("Qwen3-TTS synthesis failed: %s", e)
            return SynthesisResult(
                audio_path=Path(""),
                duration_sec=0.0,
                generation_time_sec=time.time() - start_time,
                rtf=0.0,
                model=f"qwen3-{mode}",
                text=request.text,
                reference=Path(request.reference_audio) if request.reference_audio else None,
                metadata=meta,
                success=False,
                error=str(e),
            )

    def _resolve_output_path(self, request: SynthesisRequest) -> Path:
        if request.output_path:
            p = Path(request.output_path)
        else:
            safe_text = "".join(
                c if c.isalnum() else "_" for c in request.text[:30]
            )
            p = Path("output/samples") / f"qwen3_{safe_text}.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def cleanup(self) -> None:
        """Free model from memory."""
        if self._loaded and self._model is not None:
            del self._model
            self._model = None
            self._loaded = False
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Qwen3-TTS unloaded")
