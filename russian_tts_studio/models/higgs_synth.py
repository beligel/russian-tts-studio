"""Higgs Audio v2 (Boson AI) inference wrapper.

Higgs Audio v2 is a text-audio foundation model pretrained on 10M+ hours.
The "generation variant" produces expressive speech with zero-shot voice
cloning, multi-speaker dialogs, smart-voice (no reference), and emergent
capabilities (humming, BGM, cross-lingual clone). 24 kHz output.

Models:
    - ``bosonai/higgs-audio-v2-generation-3B-base``  (3B, Apache-2.0)
    - ``bosonai/higgs-audio-v2.5`` (1B, see README_V2.md — GRPO-aligned)
Audio tokenizer:
    - ``bosonai/higgs-audio-v2-tokenizer``

⚠️ License: Higgs Audio **v2** (3B base) ships under Apache-2.0 —
permissive for commercial use, same as VoxCPM2. Higgs Audio **v3** is
non-commercial-only and is NOT supported by this wrapper (v3 doesn't
need this repo at all — it's served via SGLang-Omni or the Boson API).

The model requires the ``boson_multimodal`` package, which is vendored
in the upstream ``boson-ai/higgs-audio`` repo. We install it lazily
(``pip install -e <repo>`` or the published wheel when available). The
wrapper raises a clear ``RuntimeError`` with install instructions if
the import fails — matching the pattern in ``voxcpm_synth.py`` /``qwen3_synth.py``.

Stress control: Higgs Audio honours the combining acute accent (U+0301)
in input text — unlike VoxCPM2, which ignores it. The pipeline therefore
preserves ``{{stress "за́мок"}}`` marks when routing to this engine (see
``pipeline/tts_pipeline._strip_unsupported_markup`` for the VoxCPM2 path
that strips them).

Sound events: Higgs Audio natively understands inline tokens like
``[laugh]``, ``[music]``, ``[humming]`` in its transcript — the same
scheme RTTS markup emits. The pipeline therefore preserves sound-event
tokens when routing to this engine.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf

from .base_synth import SynthesisRequest, SynthesisResult

logger = logging.getLogger(__name__)


# Default HF model + tokenizer IDs. Override via constructor args or
# ``request.metadata["higgs_model"]`` / ``["higgs_audio_tokenizer"]``.
DEFAULT_MODEL = "bosonai/higgs-audio-v2-generation-3B-base"
DEFAULT_AUDIO_TOKENIZER = "bosonai/higgs-audio-v2-tokenizer"

# System prompt the upstream ``examples/generation.py`` uses by default.
# The ``<|scene_desc_start|>...<|scene_desc_end|>`` block is Higgs's
# scene-conditioning channel — "quiet room" biases the model toward
# clean studio output. Callers can override via ``request.metadata``
# (``higgs_scene_prompt``).
DEFAULT_SCENE_PROMPT = "Audio is recorded from a quiet room."

# Placeholder token the Higgs chat template uses to mark where the
# reference audio goes inside the system message. Mirrors upstream.
AUDIO_PLACEHOLDER_TOKEN = "<|__AUDIO_PLACEHOLDER__|>"

# Inline sound-event tags Higgs understands natively. RTTS markup emits
# the same tokens, so when the pipeline routes to Higgs we DON'T strip
# them — the model renders them as non-speech events. Keep this list in
# sync with ``markup/commands.py:SOUND_EVENT_PRESETS``.
_HIGGS_SOUND_TOKENS = {
    "[laugh]", "[chuckle]", "[giggle]", "[cough]", "[sigh]", "[gasp]",
    "[cry]", "[sniffle]", "[sneeze]", "[yawn]", "[applause]", "[cheer]",
    "[music]", "[humming]", "[singing]",
}


class HiggsAudioSynthesizer:
    """High-level wrapper around Boson AI Higgs Audio v2 / v2.5.

    Provides a unified interface compatible with
    :class:`SynthesisRequest` / :class:`SynthesisResult` so the TTS
    pipeline can switch between VoxCPM2, Qwen3, and Higgs transparently.

    Three modes (auto-selected from the request):
    - **clone**: zero-shot voice clone from reference audio (default
      when ``request.reference_audio`` is set)
    - **smart_voice**: no reference — model picks a voice from the text
      (default when no reference; ``request.metadata["higgs_smart_voice"]``
      can force this even if a reference is set)
    - **multi_speaker**: multi-voice dialog. Detect ``[SPEAKER0]`` /
      ``[SPEAKER1]`` tags in the text, OR pass a comma-separated list
      of reference audio paths in ``request.reference_audio``.
    """

    SUPPORTED_MODELS = {
        "higgs-v2-3b": "bosonai/higgs-audio-v2-generation-3B-base",
        "higgs-v2.5": "bosonai/higgs-audio-v2.5",
    }

    def __init__(
        self,
        model_name: str = "higgs-v2-3b",
        audio_tokenizer: str = DEFAULT_AUDIO_TOKENIZER,
        device: str = "auto",
        dtype: str = "auto",
        language: str = "ru",
        sample_rate: int = 24000,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        scene_prompt: str = DEFAULT_SCENE_PROMPT,
    ):
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown Higgs model: {model_name}. "
                f"Supported: {list(self.SUPPORTED_MODELS.keys())}"
            )
        self.model_name = model_name
        self.model_id = self.SUPPORTED_MODELS[model_name]
        self.audio_tokenizer_id = audio_tokenizer
        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(dtype)
        self.language = language
        self.sample_rate = sample_rate  # 24 kHz for Higgs v2
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.scene_prompt = scene_prompt
        self._serve_engine = None
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
        """Load the Higgs Audio serve engine. Idempotent.

        Raises ``RuntimeError`` with install instructions if the
        ``boson_multimodal`` package is not available — matching the
        pattern in ``voxcpm_synth.py`` and ``qwen3_synth.py``.
        """
        if self._loaded:
            return

        try:
            import torch  # noqa: F401  (needed for dtype resolution below)
            from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
        except ImportError as exc:
            raise RuntimeError(
                "Cannot import the 'boson_multimodal' package.\n"
                "Higgs Audio v2 requires the upstream repo:\n"
                "  git clone https://github.com/boson-ai/higgs-audio.git\n"
                "  cd higgs-audio && pip install -r requirements.txt && pip install -e .\n"
                f"Underlying error: {exc}"
            ) from exc

        import torch

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.dtype, torch.bfloat16)

        # MPS note: upstream disables static KV cache + CUDA graphs on
        # MPS (they rely on CUDA graphs). The serve engine handles this
        # internally, but we surface a warning so the user knows the
        # MPS path is slower.
        use_device = self.device
        if use_device == "mps":
            logger.warning(
                "Higgs Audio on MPS: static KV cache / CUDA graphs are "
                "disabled (CUDA-only). Expect lower throughput than on GPU."
            )

        try:
            logger.info(
                "Loading Higgs Audio %s on %s (dtype=%s, audio_tokenizer=%s)…",
                self.model_id, use_device, self.dtype, self.audio_tokenizer_id,
            )
            self._serve_engine = HiggsAudioServeEngine(
                model_name_or_path=self.model_id,
                audio_tokenizer_name_or_path=self.audio_tokenizer_id,
                device=use_device,
                torch_dtype=torch_dtype,
            )
            self._loaded = True
            logger.info("Higgs Audio loaded successfully")
        except Exception as e:
            logger.error("Failed to load Higgs Audio: %s", e)
            raise

    def is_loaded(self) -> bool:
        return self._loaded

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        """Synthesize text via Higgs Audio v2.

        Mode is determined by the request:
        - ``request.reference_audio`` set (single path) → clone
        - ``request.reference_audio`` comma-separated (multi) → multi-speaker
        - No reference, or ``metadata["higgs_smart_voice"]=True`` → smart voice
        - ``metadata["higgs_multi_speaker"]=True`` forces multi-speaker parsing
          of the text (looks for ``[SPEAKER0]``/``[SPEAKER1]`` tags)
        """
        if not self._loaded:
            self.load()
        assert self._serve_engine is not None

        meta = request.metadata or {}
        temperature = meta.get("higgs_temperature", self.temperature)
        top_p = meta.get("higgs_top_p", self.top_p)
        top_k = meta.get("higgs_top_k", self.top_k)
        max_new_tokens = meta.get("higgs_max_new_tokens", self.max_new_tokens)
        scene_prompt = meta.get("higgs_scene_prompt", self.scene_prompt)
        seed = meta.get("higgs_seed")

        # Detect multi-speaker from ``[SPEAKER0]`` tags in text OR
        # from a comma-separated reference list.
        speaker_tags = sorted(set(re.findall(r"\[(SPEAKER\d+)\]", request.text)))
        # Resolve voice profiles (``profile:name`` references) before
        # parsing file paths. A profile replaces the audio reference
        # with a text description that Higgs renders natively.
        profile_descriptions = self._resolve_profiles(request.reference_audio)
        ref_paths = self._parse_reference_list(request.reference_audio)
        multi_speaker = (
            bool(speaker_tags)
            or meta.get("higgs_multi_speaker", False)
            or len(ref_paths) > 1
            or len(profile_descriptions) > 1
        )

        start_time = time.time()
        try:
            from boson_multimodal.data_types import (
                AudioContent,
                ChatMLSample,
                Message,
            )

            messages, audio_ids = self._build_context(
                scene_prompt=scene_prompt,
                ref_paths=ref_paths,
                speaker_tags=speaker_tags,
                multi_speaker=multi_speaker,
                smart_voice=meta.get("higgs_smart_voice", False) and not ref_paths and not profile_descriptions,
                profile_descriptions=profile_descriptions,
            )

            # Higgs's generate() takes a ChatMLSample. We pass the full
            # text as a single user message — the serve engine handles
            # internal chunking for long inputs via its collator.
            messages = messages + [Message(role="user", content=request.text)]
            sample = ChatMLSample(messages=messages)

            response = self._serve_engine.generate(
                chat_ml_sample=sample,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
            )

            if response.audio is None:
                raise RuntimeError("Higgs Audio returned no audio (model produced text only)")

            wav_np = np.asarray(response.audio, dtype=np.float32)
            sr = response.sampling_rate or self.sample_rate
            output_path = self._resolve_output_path(request)

            sf.write(str(output_path), wav_np, sr, subtype="FLOAT")
            self.sample_rate = sr

            duration = len(wav_np) / sr
            gen_time = time.time() - start_time
            rtf = gen_time / duration if duration > 0 else 0.0
            mode = "multi_speaker" if multi_speaker else ("smart_voice" if not ref_paths else "clone")

            logger.info(
                "Higgs Audio %s done: %.2fs audio in %.2fs (RTF=%.3f, mode=%s)",
                self.model_name, duration, gen_time, rtf, mode,
            )

            return SynthesisResult(
                audio_path=output_path,
                duration_sec=duration,
                generation_time_sec=gen_time,
                rtf=rtf,
                model=f"higgs-{mode}",
                text=request.text,
                reference=Path(request.reference_audio) if request.reference_audio else None,
                metadata={
                    **meta,
                    "engine": "higgs",
                    "higgs_mode": mode,
                    "sampling_rate": sr,
                },
            )

        except Exception as e:
            logger.exception("Higgs Audio synthesis failed: %s", e)
            return SynthesisResult(
                audio_path=Path(""),
                duration_sec=0.0,
                generation_time_sec=time.time() - start_time,
                rtf=0.0,
                model="higgs",
                text=request.text,
                reference=Path(request.reference_audio) if request.reference_audio else None,
                metadata=meta,
                success=False,
                error=str(e),
            )

    # --- context building -------------------------------------------------

    def _parse_reference_list(self, reference_audio) -> list[Path]:
        """Parse ``reference_audio`` into a list of paths.

        Accepts a single path, a comma-separated string of paths
        (multi-speaker), or ``None``/missing. ``profile:<name>``
        references are NOT paths — they're handled by
        :meth:`_resolve_profiles` and excluded here.
        """
        if not reference_audio:
            return []
        if isinstance(reference_audio, (list, tuple)):
            raw_items = [str(p) for p in reference_audio]
        else:
            raw = str(reference_audio).strip()
            if not raw:
                return []
            raw_items = [p.strip() for p in raw.split(",") if p.strip()]
        # Filter out profile: references — those are text descriptions,
        # not file paths.
        from .voice_profiles import is_profile_reference
        path_items = [p for p in raw_items if not is_profile_reference(p)]
        paths = [Path(p) for p in path_items]
        # Validate existence; raise clear error for missing files.
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"Higgs reference audio not found: {p}")
        return paths

    def _resolve_profiles(self, reference_audio) -> list[str]:
        """Resolve ``profile:<name>`` references to their description text.

        Returns a list of descriptions (one per profile reference, in
        order). Non-profile references (file paths) are skipped — they
        go through :meth:`_parse_reference_list` instead. Unknown
        profile names produce a warning and are dropped.
        """
        if not reference_audio:
            return []
        if isinstance(reference_audio, (list, tuple)):
            raw_items = [str(p) for p in reference_audio]
        else:
            raw = str(reference_audio).strip()
            if not raw:
                return []
            raw_items = [p.strip() for p in raw.split(",") if p.strip()]
        from .voice_profiles import is_profile_reference, resolve_reference
        descriptions: list[str] = []
        for item in raw_items:
            if is_profile_reference(item):
                desc = resolve_reference(item)
                if desc:
                    descriptions.append(desc)
                else:
                    logger.warning("Unknown voice profile in %r — skipped", item)
        return descriptions

    def _build_context(
        self,
        scene_prompt: str,
        ref_paths: list[Path],
        speaker_tags: list[str],
        multi_speaker: bool,
        smart_voice: bool,
        profile_descriptions: list[str] = None,
    ):
        """Build the ChatML messages + audio_ids for Higgs generation.

        Mirrors ``examples/generation.py:prepare_generation_context`` but
        simplified to our use case (RTTS supplies reference audio files,
        not the upstream ``voice_prompts/<name>.wav`` lookup). Returns
        ``(messages, audio_ids)`` where ``messages`` is the system +
        reference-turn prefix and ``audio_ids`` is the list of tokenised
        reference audio tensors.

        Three voice-source paths (mutually exclusive on the speaker slot):
        - ``ref_paths`` non-empty → audio cloning (audio_ids populated)
        - ``profile_descriptions`` non-empty → text-described voices
          (no audio_ids; descriptions go in the speaker slot)
        - both empty → smart voice (model picks from transcript)
        """
        from boson_multimodal.data_types import AudioContent, Message

        messages: list[Message] = []
        audio_ids: list = []
        profile_descriptions = profile_descriptions or []

        # Text-described voices (``profile:<name>``) — no audio needed.
        if profile_descriptions and not ref_paths:
            system_parts = ["Generate audio following instruction."]
            scene_block_parts = [scene_prompt] if scene_prompt else []
            speaker_descs = []
            num_speakers = len(profile_descriptions)
            for spk_id, desc in enumerate(profile_descriptions):
                if multi_speaker or num_speakers > 1:
                    speaker_descs.append(f"SPEAKER{spk_id}: {desc}")
                else:
                    speaker_descs.append(f"SPEAKER0: {desc}")
            scene_block_parts.extend(speaker_descs)
            if scene_block_parts:
                system_parts.append(
                    f"<|scene_desc_start|>\n" + "\n\n".join(scene_block_parts) + "\n<|scene_desc_end|>"
                )
            system_content = "\n\n".join(system_parts)
            messages.insert(0, Message(role="system", content=system_content))
            return messages, audio_ids

        if not ref_paths:
            # Smart-voice path: no reference and no profiles. Model
            # picks a voice from the transcript. For multi-speaker
            # without refs, we let the model alternate voices (matches
            # upstream default).
            system_parts = ["Generate audio following instruction."]
            if scene_prompt:
                system_parts.append(
                    f"<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>"
                )
            if multi_speaker and len(speaker_tags) > 1:
                # Describe speakers alternately feminine/masculine
                # (upstream default when no reference audio is supplied).
                speaker_descs = []
                for idx, tag in enumerate(speaker_tags):
                    desc = "feminine" if idx % 2 == 0 else "masculine"
                    speaker_descs.append(f"{tag}: {desc}")
                joined = "\n".join(speaker_descs)
                system_parts.append(
                    f"<|scene_desc_start|>\n{joined}\n<|scene_desc_end|>"
                )
            system_content = "\n\n".join(system_parts)
            messages.insert(0, Message(role="system", content=system_content))
            return messages, audio_ids

        # Clone path: encode each reference audio and build either:
        #  - multi-speaker: put refs in system message with SPEAKER tags
        #  - single-speaker: put ref in system message with placeholder
        num_speakers = len(ref_paths)
        speaker_descs = []
        for spk_id, ref_path in enumerate(ref_paths):
            if multi_speaker:
                speaker_descs.append(f"SPEAKER{spk_id}: {AUDIO_PLACEHOLDER_TOKEN}")
            else:
                speaker_descs.append(f"SPEAKER0: {AUDIO_PLACEHOLDER_TOKEN}")
        system_parts = ["Generate audio following instruction."]
        scene_block = scene_prompt + "\n\n" + "\n".join(speaker_descs)
        system_parts.append(
            f"<|scene_desc_start|>\n{scene_block}\n<|scene_desc_end|>"
        )
        system_content = "\n\n".join(system_parts)
        # Build the system message with audio placeholders — Higgs's
        # ChatML template replaces ``<|__AUDIO_PLACEHOLDER__|>`` with
        # an ``AudioContent`` slot that the serve engine fills from
        # ``audio_ids``.
        sys_msg = self._build_system_message_with_audio(system_content)
        messages.insert(0, sys_msg)

        # Tokenise each reference audio.
        for ref_path in ref_paths:
            audio_ids.append(self._encode_reference(ref_path))

        return messages, audio_ids

    def _build_system_message_with_audio(self, system_content: str):
        """Split a system message at ``AUDIO_PLACEHOLDER_TOKEN`` positions
        into alternating ``TextContent`` / ``AudioContent`` blocks.

        Mirrors ``examples/generation.py:_build_system_message_with_audio``.
        """
        from boson_multimodal.data_types import AudioContent, Message, TextContent

        contents: list = []
        remaining = system_content
        while AUDIO_PLACEHOLDER_TOKEN in remaining:
            loc = remaining.find(AUDIO_PLACEHOLDER_TOKEN)
            if loc > 0:
                contents.append(TextContent(remaining[:loc]))
            contents.append(AudioContent(audio_url=""))
            remaining = remaining[loc + len(AUDIO_PLACEHOLDER_TOKEN):]
        if remaining:
            contents.append(TextContent(remaining))
        return Message(role="system", content=contents)

    def _encode_reference(self, ref_path: Path):
        """Tokenise a reference audio file via the Higgs audio tokenizer."""
        assert self._serve_engine is not None
        import librosa

        raw_audio, _ = librosa.load(
            str(ref_path), sr=self._serve_engine.audio_tokenizer.sampling_rate,
        )
        audio_ids = self._serve_engine.audio_tokenizer.encode(
            raw_audio, self._serve_engine.audio_tokenizer.sampling_rate,
        )
        return audio_ids.squeeze(0).cpu()

    # --- helpers -----------------------------------------------------------

    def _resolve_output_path(self, request: SynthesisRequest) -> Path:
        if request.output_path:
            p = Path(request.output_path)
        else:
            safe_text = "".join(
                c if c.isalnum() else "_" for c in request.text[:30]
            )
            p = Path("output/samples") / f"higgs_{safe_text}.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def cleanup(self) -> None:
        """Free model from memory."""
        if self._loaded and self._serve_engine is not None:
            del self._serve_engine
            self._serve_engine = None
            self._loaded = False
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Higgs Audio unloaded")

    @staticmethod
    def supports_sound_events() -> bool:
        """Higgs Audio natively understands ``[laugh]``/``[music]`` tokens.

        The pipeline uses this to decide whether to strip sound-event
        tokens before calling ``synthesize``. See
        ``pipeline/tts_pipeline._strip_unsupported_markup``.
        """
        return True

    @staticmethod
    def supports_stress_marks() -> bool:
        """Higgs Audio honours combining acute accent (U+0301) in text.

        The pipeline uses this to decide whether to strip stress marks
        before calling ``synthesize``.
        """
        return True