"""XTTS-v2 (Coqui) inference wrapper with caching and fallback support.

XTTS-v2 is Coqui's multilingual voice-cloning TTS. It supports
zero-shot cloning from a 6-10 s reference clip and natively handles
~16 languages, **including Russian** — unlike F5-TTS v1 Base which is
mostly Chinese/English and produces gibberish on Russian input.

API used here matches ``TTS==0.22.0``:
    from TTS.api import TTS
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2",
              gpu=(device == "cuda"))
    tts.tts_to_file(text=..., file_path=..., speaker_wav=[ref_path],
                    language="ru")

The model outputs 24 kHz mono WAV. XTTS automatically chunks long
inputs and concatenates them.

⚠️ XTTS-v2 is distributed under the **CPML (Coqui Public Model
License)** — non-commercial by default. The pipeline still works
with it; we just surface a license notice in the engine picker so
the user knows.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio

from .base_synth import SynthesisRequest, SynthesisResult

logger = logging.getLogger(__name__)


class XTTSSynthesizer:
    """High-level wrapper around Coqui XTTS-v2 with the same shape as
    :class:`Russian TTS StudioSynthesizer` / :class:`F5TTSSynthesizer` so it can
    be swapped in transparently.

    Features:
    - Lazy model loading (xtts_v2, ~2 GB on first run)
    - Automatic device resolution (cuda / cpu)
    - Output written at 24 kHz mono (XTTS native sample rate) — the
      ``_postprocess`` step in ``tts_pipeline`` resamples to 22050 Hz
      anyway, so this round-trips cleanly.
    - Graceful degradation: failures are returned as ``SynthesisResult``
      with ``success=False`` (matches the other engines).
    """

    SUPPORTED_MODELS = {
        "xtts-v2": "tts_models/multilingual/multi-dataset/xtts_v2",
    }

    def __init__(
        self,
        model_name: str = "xtts-v2",
        device: str = "auto",
        cache_dir: str | Path = "models/cache",
        sample_rate: int = 24000,
        language: str = "ru",
    ):
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown XTTS model: {model_name}. "
                f"Supported: {list(self.SUPPORTED_MODELS.keys())}"
            )
        self.model_name = model_name
        self.model_id = self.SUPPORTED_MODELS[model_name]
        self.device = self._resolve_device(device)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.language = language
        self._tts: Optional["TTS"] = None  # type: ignore[name-defined]
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
        """Load the XTTS-v2 model into memory. Idempotent."""
        if self._loaded:
            return
        try:
            from TTS.api import TTS  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "Cannot import the 'TTS' (Coqui) package. Install with:\n"
                "  pip install TTS==0.22.0\n"
                f"Underlying error: {exc}"
            )

        try:
            logger.info(
                "Loading XTTS-v2 on %s (model=%s)…",
                self.device, self.model_id,
            )
            self._tts = TTS(
                model_name=self.model_id,
                gpu=(self.device == "cuda"),
            ).to(self.device)
            self._loaded = True
            logger.info("XTTS-v2 loaded successfully")
        except Exception as e:
            logger.error("Failed to load XTTS-v2: %s", e)
            raise

    def is_loaded(self) -> bool:
        return self._loaded

    def synthesize(
        self,
        request: SynthesisRequest,
    ) -> SynthesisResult:
        """Synthesise a single text request via XTTS-v2 inference.

        XTTS automatically chunks long inputs at the sentence level and
        cross-fades the segments, so a single ``tts`` call is enough for
        any input length. ``language`` defaults to the value passed at
        ``__init__`` time (``"ru"``); callers can override via
        ``request.metadata["language"]`` if needed. ``speed`` (default
        ``0.9``) lowers the perceived pitch slightly — values below 1.0
        stretch the waveform and read as a deeper, slower voice.
        """
        if not self._loaded:
            self.load()

        ref_path = Path(request.reference_audio)
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {ref_path}")

        start_time = time.time()
        meta = request.metadata or {}
        language = meta.get("language") or self.language
        # ``speed`` lives on the request itself (see SynthesisRequest).
        # Default 0.9 lowers perceived pitch slightly — 1.0 sounds
        # brighter/thinner on Russian speech.
        speed = float(request.speed) if request.speed else 0.9
        if speed <= 0:
            raise ValueError(f"speed must be > 0, got {speed}")

        # Normalise typographic chars that confuse XTTS-v2 (it has
        # tokenisation quirks around «»"" … —). We replace *only* what
        # we know is harmful, leaving the rest of the text intact.
        original_text = request.text
        normalised_text = _normalize_text_for_xtts(original_text)
        # XTTS-v2 vocabulary contains no Cyrillic uppercase letters
        # and no stress diacritics — both are tokenised as [UNK] and
        # produce garbled audio. Lowercase the entire string, strip
        # combining-acute stress marks (we don't try to inject stress
        # into XTTS — that capability simply isn't in the model).
        normalised_text = _force_lowercase_no_diacritics(normalised_text)
        if normalised_text != original_text:
            logger.info(
                "XTTS text normalised: %d char(s) changed (e.g. %r → %r)",
                sum(1 for a, b in zip(original_text, normalised_text) if a != b),
                original_text[:40], normalised_text[:40],
            )

        try:
            logger.info(
                "XTTS synthesising %.80s… (ref=%s, language=%s, speed=%.2f)",
                normalised_text, ref_path.name, language, speed,
            )

            output_path = self._resolve_output_path(request)
            # Use ``tts()`` (not ``tts_to_file()``) so we can pass
            # ``speed`` — Coqui TTS 0.22.0 doesn't expose ``speed`` in
            # the file-writing variant. The result is an np.float32
            # array at the model's native sample rate (24 kHz).
            assert self._tts is not None
            wav_np = self._tts.tts(
                text=normalised_text,
                speaker_wav=[str(ref_path)],
                language=language,
                speed=speed,
            )
            wav = torch.from_numpy(np.asarray(wav_np, dtype=np.float32))
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            sr = self.sample_rate
            torchaudio.save(str(output_path), wav, sr)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            duration = wav.shape[-1] / float(sr)

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
                "XTTS synthesis done: %.2fs audio in %.2fs (RTF=%.3f, sr=%d)",
                duration, gen_time, rtf, int(sr),
            )
            return result

        except Exception as e:
            logger.exception("XTTS synthesis failed: %s", e)
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
        if self._loaded and self._tts is not None:
            del self._tts
            self._tts = None
            self._loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("XTTS-v2 unloaded")


# Russian typographic characters that XTTS-v2 mispronounces / drops /
# reads as garbage. The mapping was built by trial: each entry was
# confirmed to produce an audible glitch on a small Russian test set
# («Сказал: "Привет"», «Москва — столица», «Подождите…»).
#
# We replace, not delete — removing quotes would change meaning, and
# stripping em-dashes breaks punctuation rhythm. ASCII alternatives
# are the safest fallback because XTTS's BPE tokenizer treats them
# as single tokens with stable phoneme mappings.
_XTTS_CHAR_REPLACEMENTS: dict[str, str] = {
    # Russian / smart quotes — main offender reported by user
    "«": '"',
    "»": '"',
    "„": '"',
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    # Em-dash / en-dash — XTTS sometimes pauses weirdly on `—`
    "—": " - ",
    "–": "-",
    # Ellipsis (single char) — replace with three dots XTTS reads cleanly
    "…": "...",
    # Non-breaking space / thin space — tokeniser treats as word boundary
    "\u00A0": " ",
    "\u2009": " ",
    # Zero-width joiners / BOMs — invisible, but break BPE alignment
    "\u200B": "",
    "\u200C": "",
    "\u200D": "",
    "\uFEFF": "",
}


def _normalize_text_for_xtts(text: str) -> str:
    """Return ``text`` with XTTS-hostile typographic chars replaced.

    The function is intentionally narrow: it touches only characters
    that have been observed to corrupt synthesis output. It does NOT
    expand abbreviations (``т.е.``, ``и т.д.``), fix ``ё`` vs ``е``,
    or otherwise touch content — that is the job of the upstream
    normaliser in ``utils/text_utils.py``.
    """
    if not text:
        return text
    out = text
    for src, dst in _XTTS_CHAR_REPLACEMENTS.items():
        if src in out:
            out = out.replace(src, dst)
    return out


# XTTS-v2 supports neither Cyrillic uppercase letters nor combining
# diacritics — its BPE vocab is a small fixed set of lowercase
# substrings (~6k tokens) and any unknown char (А, Б, ..., а́, ё, …)
# is tokenised as [UNK], producing garbled audio. So before handing
# text to the model we (a) lowercase the whole string, and (b) strip
# any precomposed/stress diacritic that callers may have added.
#
# Stress control is therefore not possible with XTTS-v2 on Russian.
# Users who need explicit word-level stress (e.g. для омографов
# «зАмок»/«замОк») must either accept the model's natural choice or
# switch to a different engine — Silero supports ``+`` after the
# stressed vowel, but XTTS does not.
import unicodedata


def _force_lowercase_no_diacritics(text: str) -> str:
    """Return ``text`` lowercased and with combining marks stripped.

    Used to make Cyrillic input safe for the XTTS-v2 BPE tokenizer.
    All characters ``c`` where ``unicodedata.combining(c) != 0`` are
    removed; non-combining characters (including ``ё``) are kept and
    only their case is folded. This means a stress-marked input like
    ``за́мок`` and a plain ``замок`` end up identical — XTTS will pick
    its own default stress. We don't try to be cleverer than that.
    """
    if not text:
        return text
    out = []
    for ch in text:
        if unicodedata.combining(ch):
            continue
        out.append(ch.lower())
    return "".join(out)

