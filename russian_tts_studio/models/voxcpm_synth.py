"""VoxCPM2 (OpenBMB) inference wrapper with caching.

VoxCPM2 is OpenBMB's 2-billion-parameter TTS model with a tokenizer-free
diffusion-autoregressive architecture (LocEnc → TSLM → RALM → LocDiT).
It supports ~30 languages (Russian included), outputs 48 kHz mono WAV,
and has three cloning modes of escalating quality:

    1. basic:       reference_wav_path alone
    2. controllable: + voice-design via the ``instruct`` string
    3. ultimate:    + prompt_wav_path + prompt_text (lets the LLM see
                    both the reference voice and the *style* of a
                    longer context clip)

The wrapper exposes the same interface as :class:`XTTSSynthesizer` so
the pipeline can switch engines without changing the call site.

API used here matches ``voxcpm==2.0.3``:
    from voxcpm import VoxCPM
    model = VoxCPM.from_pretrained("OpenBMB/VOXCPM2", load_denoiser=False)
    wav = model.generate(
        text="...",
        reference_wav_path="speaker.wav",        # basic cloning
        # OR for ultimate:
        # prompt_wav_path="speaker.wav",
        # prompt_text="transcript of speaker.wav",
    )
    # ``wav`` is a 1-D numpy float32 array at model.tts_model.sample_rate
    # (48 kHz for VoxCPM2). Use soundfile to persist it.

⚠️ VoxCPM2 weights are distributed under the **Apache-2.0** license,
which is permissive for commercial use. Note that VoxCPM2 lives in
its own venv (``.venv-voxcpm``) because it requires ``torch>=2.5``
while XTTS needs ``torch<=2.4`` — the two cannot coexist in one
Python environment. The :mod:`russian_tts_studio.models.voxcpm_synth`
module is therefore imported lazily by :class:`TTSPipeline` and
**fails fast** if ``voxcpm`` isn't installed (i.e. you started the
web server from the wrong venv).

Stress control: VoxCPM2 has no input mechanism for explicit stress
marks. The model uses autoprosoody — its TSLM picks stress from
context. If you need to mark stress, do it in your own pre-processing
(e.g. transliterate to IPA) and pass the result as ``text``. The
wrapper does not modify Cyrillic case or diacritics for this reason.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

from .base_synth import SynthesisRequest, SynthesisResult
from ..utils.prosody import PauseConfig, insert_pauses

logger = logging.getLogger(__name__)


class VoxCPMSynthesizer:
    """High-level wrapper around OpenBMB VoxCPM2.

    Mirrors the shape of :class:`XTTSSynthesizer` so it can be swapped
    in transparently by the pipeline.

    Features:
    - Lazy model load (VoxCPM2 + AudioVAE, ~2 GB on first run)
    - Automatic device resolution (cuda / cpu)
    - Output written at 48 kHz mono (VoxCPM2 native sample rate) — the
      ``_postprocess`` step in ``tts_pipeline`` resamples to 22050 Hz
      anyway, so this round-trips cleanly.
    - Graceful degradation: failures are returned as ``SynthesisResult``
      with ``success=False`` (matches the other engines).
    - ``load_denoiser=False`` by default: the denoiser (``zipenhancer``)
      adds ~30 s to first-load and a model download; most studio
      references are already clean. Turn it on via ``__init__`` only
      if you have noisy field recordings.
    """

    SUPPORTED_MODELS = {
        "voxcpm-2": "OpenBMB/VOXCPM2",
    }

    def __init__(
        self,
        model_name: str = "voxcpm-2",
        device: str = "auto",
        cache_dir: str | Path = "models/cache",
        sample_rate: int = 48000,
        language: str = "ru",
        load_denoiser: bool = False,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
    ):
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown VoxCPM model: {model_name}. "
                f"Supported: {list(self.SUPPORTED_MODELS.keys())}"
            )
        self.model_name = model_name
        self.model_id = self.SUPPORTED_MODELS[model_name]
        self.device = self._resolve_device(device)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # ``sample_rate`` is the model's native output rate (48 kHz for
        # VoxCPM2). We don't expose it as a parameter the user picks —
        # the post-processor in tts_pipeline handles resampling to
        # 22050 Hz before writing the final file.
        self.sample_rate = sample_rate
        self.language = language
        self.load_denoiser = load_denoiser
        self.cfg_value = cfg_value
        self.inference_timesteps = inference_timesteps
        self._model: Optional["VoxCPM"] = None  # type: ignore[name-defined]
        self._loaded = False

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    def load(self) -> None:
        """Load the VoxCPM2 model into memory. Idempotent."""
        if self._loaded:
            return
        # VoxCPM2 needs torch>=2.5 — the SDPA forward pass uses
        # ``enable_gqa`` (torch 2.5+). The XTTS venv (.venv) ships
        # torch 2.3.1, so accidentally trying to load VoxCPM2 from
        # there crashes deep inside the model with a confusing
        # "got an unexpected keyword argument 'enable_gqa'" instead
        # of a useful "wrong venv" message. Catch it here.
        torch_major, torch_minor = (int(x) for x in torch.__version__.split(".")[:2])
        if (torch_major, torch_minor) < (2, 5):
            raise RuntimeError(
                f"VoxCPM2 requires torch>=2.5, but found torch=={torch.__version__}. "
                "This usually means the web server was started from the XTTS "
                "venv (.venv) instead of the VoxCPM2 venv (.venv-voxcpm). "
                "Start the server with:\n"
                "  source .venv-voxcpm/bin/activate && make start\n"
                "or run the binary directly: .venv-voxcpm/bin/python -m web.app"
            )
        try:
            from voxcpm import VoxCPM  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "Cannot import the 'voxcpm' package. Install with:\n"
                "  pip install voxcpm==2.0.3\n"
                "VoxCPM2 must run in a dedicated venv (.venv-voxcpm) "
                "because it requires torch>=2.5 and is incompatible with "
                "the XTTS venv's torch<=2.4.\n"
                f"Underlying error: {exc}"
            )

        try:
            logger.info(
                "Loading VoxCPM2 on %s (model=%s, denoiser=%s)…",
                self.device, self.model_id, self.load_denoiser,
            )
            self._model = VoxCPM.from_pretrained(
                hf_model_id=self.model_id,
                load_denoiser=self.load_denoiser,
                device=self.device,
            )
            # VoxCPM2 reports its native audio rate on the inner model
            # (48 kHz for v2). If the model returned a different rate
            # we'd be in trouble, so surface that here instead of
            # silently mis-saving.
            actual_sr = getattr(self._model.tts_model, "sample_rate", None)
            if actual_sr and actual_sr != self.sample_rate:
                logger.info(
                    "VoxCPM2 reports sample_rate=%d (overriding constructor default %d)",
                    actual_sr, self.sample_rate,
                )
                self.sample_rate = int(actual_sr)
            self._loaded = True
            logger.info(
                "VoxCPM2 loaded successfully (sample_rate=%d)", self.sample_rate,
            )
        except Exception as e:
            logger.error("Failed to load VoxCPM2: %s", e)
            raise

    def is_loaded(self) -> bool:
        return self._loaded

    def synthesize(
        self,
        request: SynthesisRequest,
    ) -> SynthesisResult:
        """Synthesise a single text request via VoxCPM2 inference.

        VoxCPM2 takes care of long-input chunking internally (its
        TSLM has a 4096-token context), so a single ``generate`` call
        is enough for any input length. ``cfg_value`` and
        ``inference_timesteps`` are set at init time — tweaking them
        per-request is rarely useful and would change the cached
        model's compiled shape, so we don't expose them.

        Stress marks: VoxCPM2 has no explicit stress input. The TSLM
        handles prosody automatically. We therefore do NOT modify the
        caller's text — Cyrillic case is preserved, combining diacritics
        are preserved, Russian typographic quotes are preserved. If
        you want to force a particular stress pattern, pre-process the
        text yourself (e.g. IPA transliteration) before calling.
        """
        if not self._loaded:
            self.load()

        ref_path = Path(request.reference_audio)
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {ref_path}")

        start_time = time.time()
        meta = request.metadata or {}
        language = meta.get("language") or self.language

        # Build the generate() kwargs. Three modes are supported:
        #   * ultimate  — prompt_wav_path + prompt_text are set
        #     (the LLM gets both the reference voice and a longer
        #     context clip with its transcript)
        #   * basic     — only reference_wav_path is set
        #   * zero-shot — no reference at all (VoxCPM2 falls back to
        #     its built-in default voice)
        #
        # Our pipeline always supplies a reference_audio (XTTS-style
        # cloning flow), so we use reference_wav_path for cloning.
        # If reference_text is also provided, we forward it as
        # prompt_text for the "ultimate" mode — this gives noticeably
        # cleaner prosody on long Russian sentences.
        generate_kwargs: dict = {
            "text": request.text,
            "reference_wav_path": str(ref_path),
            "cfg_value": self.cfg_value,
            "inference_timesteps": self.inference_timesteps,
        }
        if request.reference_text and request.reference_text.strip():
            generate_kwargs["prompt_wav_path"] = str(ref_path)
            generate_kwargs["prompt_text"] = request.reference_text.strip()
            logger.debug(
                "VoxCPM2 ultimate mode: ref_text len=%d chars",
                len(request.reference_text),
            )
        # NOTE: ``instruct`` (voice-design channel — "(slow, deep male) …")
        # was a planned feature in earlier voxcpm SDKs, but voxcpm==2.0.3
        # dropped it from ``VoxCPM._generate``. If we forward it, the call
        # raises ``unexpected keyword argument 'instruct'`` and the request
        # falls back to Silero. We only do zero-shot cloning here, so we
        # safely drop it. Re-introduce when the SDK adds the param back.
        # if request.instruct:
        #     generate_kwargs["instruct"] = request.instruct

        try:
            logger.info(
                "VoxCPM2 synthesising %.80s… (ref=%s, language=%s, "
                "cfg=%.2f, steps=%d, mode=%s)",
                request.text, ref_path.name, language,
                self.cfg_value, self.inference_timesteps,
                "ultimate" if "prompt_text" in generate_kwargs else "basic",
            )

            output_path = self._resolve_output_path(request)
            assert self._model is not None
            wav_np = self._model.generate(**generate_kwargs)
            # ``wav_np`` is 1-D float32 (mono) at self.sample_rate.
            # soundfile handles float32 directly. We don't touch
            # channel layout (VoxCPM2 is mono by design).
            sf.write(
                str(output_path),
                np.asarray(wav_np, dtype=np.float32),
                self.sample_rate,
                subtype="FLOAT",
            )
            # Optional prosody: splice silence at comma/period/etc. by
            # forced-aligning the wav against the original Cyrillic text.
            # ``insert_pauses`` is a no-op if all pause_ms_* are 0; any
            # alignment failure is swallowed inside (warns + returns the
            # path unchanged) so a broken aligner never breaks synthesis.
            try:
                pause_cfg = PauseConfig.from_metadata(
                    request.metadata or {}
                )
                if pause_cfg.is_enabled():
                    insert_pauses(output_path, request.text, pause_cfg)
            except Exception as e:
                logger.warning(
                    "Prosody post-processing skipped: %s: %s",
                    type(e).__name__, e,
                )
            # Re-measure from disk so reported duration matches the
            # post-pause file (insert_pauses may have appended silences).
            try:
                f_info = sf.info(str(output_path))
                duration = float(f_info.frames) / float(f_info.samplerate)
            except Exception:
                duration = len(wav_np) / float(self.sample_rate)

            gen_time = time.time() - start_time
            rtf = gen_time / duration if duration > 0 else 0.0

            result = SynthesisResult(
                audio_path=output_path,
                duration_sec=duration,
                generation_time_sec=gen_time,
                rtf=rtf,
                model=self.model_name,
                text=request.text,
                reference=ref_path,
                metadata=request.metadata,
            )
            logger.info(
                "VoxCPM2 synthesis done: %.2fs audio in %.2fs (RTF=%.3f, sr=%d)",
                duration, gen_time, rtf, int(self.sample_rate),
            )
            return result

        except Exception as e:
            logger.exception("VoxCPM2 synthesis failed: %s", e)
            return SynthesisResult(
                audio_path=Path(""),
                duration_sec=0.0,
                generation_time_sec=time.time() - start_time,
                rtf=0.0,
                model=self.model_name,
                text=request.text,
                reference=ref_path,
                metadata=request.metadata,
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
            p = Path("output/samples") / f"{self.model_name}_{safe_text}.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def cleanup(self) -> None:
        """Free model from memory."""
        if self._loaded and self._model is not None:
            del self._model
            self._model = None
            self._loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("VoxCPM2 unloaded")
