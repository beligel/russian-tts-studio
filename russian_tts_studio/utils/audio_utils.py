"""Audio loading, normalization, and resampling utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)


def _load_wav_via_wave(path: Path) -> tuple[torch.Tensor, int]:
    """Fallback WAV loader using Python stdlib (avoids torchcodec dep)."""
    import wave

    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        sample_width = wf.getsampwidth()
        raw = wf.readframes(n_frames)

    if sample_width == 2:
        dtype = np.int16
    elif sample_width == 4:
        dtype = np.int32
    elif sample_width == 1:
        dtype = np.uint8
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    audio = np.frombuffer(raw, dtype=dtype)
    if dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128.0) / 128.0

    audio = audio.reshape(-1, n_channels).T
    return torch.from_numpy(audio), sample_rate


def load_audio(
    path: str | Path,
    target_sr: int = 16000,
    mono: bool = True,
) -> torch.Tensor:
    """Load audio file and resample to target sample rate.

    Args:
        path: Path to audio file (wav, mp3, flac, ogg).
        target_sr: Target sample rate in Hz.
        mono: Convert to mono if True.

    Returns:
        Tensor of shape (channels, samples) or (samples,) if mono.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        waveform, sample_rate = torchaudio.load(str(path))
    except (ImportError, RuntimeError, OSError) as e:
        if path.suffix.lower() == ".wav":
            logger.debug("torchaudio.load failed (%s), falling back to stdlib wave", e)
            waveform, sample_rate = _load_wav_via_wave(path)
        else:
            raise

    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_sr
        )
        waveform = resampler(waveform)
        logger.debug("Resampled %s: %d Hz → %d Hz", path.name, sample_rate, target_sr)

    if mono:
        waveform = waveform.squeeze(0)

    return waveform


def normalize_loudness(
    waveform: torch.Tensor,
    target_dbfs: float = -20.0,
    sample_rate: int = 22050,
) -> torch.Tensor:
    """Normalize audio to target dBFS using simple peak normalization.

    For production-grade loudness normalization use pyloudnorm.
    """
    waveform_np = waveform.numpy() if isinstance(waveform, torch.Tensor) else waveform
    rms = np.sqrt(np.mean(waveform_np ** 2))
    if rms < 1e-6:
        logger.warning("Audio is silent, skipping normalization")
        return waveform

    current_dbfs = 20 * np.log10(rms)
    gain_db = target_dbfs - current_dbfs
    gain_linear = 10 ** (gain_db / 20)

    return waveform * gain_linear


def trim_silence(
    waveform: torch.Tensor,
    threshold: float = 0.01,
    frame_length: int = 1024,
    hop_length: int = 256,
) -> torch.Tensor:
    """Trim leading and trailing silence from audio."""
    if waveform.dim() > 1:
        waveform = waveform.squeeze(0)

    abs_wave = waveform.abs()
    n_frames = (len(waveform) - frame_length) // hop_length + 1

    if n_frames <= 0:
        return waveform

    frames = abs_wave.unfold(0, frame_length, hop_length)
    frame_energy = frames.pow(2).mean(dim=1).sqrt()
    voiced = frame_energy > threshold

    if not voiced.any():
        return waveform

    indices = torch.nonzero(voiced).squeeze()
    start_frame = indices[0].item()
    end_frame = indices[-1].item() + 1

    start_sample = start_frame * hop_length
    end_sample = min(end_frame * hop_length + frame_length, len(waveform))

    return waveform[start_sample:end_sample]


def get_duration(
    waveform: torch.Tensor,
    sample_rate: int = 22050,
) -> float:
    """Get audio duration in seconds."""
    if waveform.dim() > 1:
        n_samples = waveform.shape[-1]
    else:
        n_samples = waveform.shape[0]
    return n_samples / sample_rate


def split_into_chunks(
    waveform: torch.Tensor,
    max_duration_sec: float = 15.0,
    sample_rate: int = 22050,
    silence_threshold: float = 0.005,
) -> list[torch.Tensor]:
    """Split long audio into chunks at silence points.

    Useful for processing long synthesized audio without artifacts.
    """
    if waveform.dim() > 1:
        waveform = waveform.squeeze(0)

    max_samples = int(max_duration_sec * sample_rate)

    if len(waveform) <= max_samples:
        return [waveform]

    abs_wave = waveform.abs()
    chunk_samples = max_samples

    chunks: list[torch.Tensor] = []
    start = 0

    while start < len(waveform):
        end = min(start + chunk_samples, len(waveform))

        if end < len(waveform):
            search_start = max(start + chunk_samples // 2, end - sample_rate)
            search_region = abs_wave[search_start:end]
            if search_region.numel() > 0:
                quiet = torch.nonzero(search_region < silence_threshold).squeeze()
                if quiet.numel() > 0:
                    if quiet.dim() == 0:
                        offset = quiet.item()
                    else:
                        offset = quiet[len(quiet) // 2].item()
                    end = search_start + offset

        chunks.append(waveform[start:end])
        start = end

    return chunks


def concatenate_audios(
    audios: list[torch.Tensor],
    crossfade_ms: int = 50,
    sample_rate: int = 22050,
) -> torch.Tensor:
    """Concatenate audio chunks with crossfade for smooth transitions."""
    if not audios:
        raise ValueError("Empty audio list")
    if len(audios) == 1:
        return audios[0]

    crossfade_samples = int(crossfade_ms * sample_rate / 1000)
    result = audios[0]

    for audio in audios[1:]:
        if len(result) < crossfade_samples or len(audio) < crossfade_samples:
            result = torch.cat([result, audio], dim=-1)
            continue

        fade_out = torch.linspace(1.0, 0.0, crossfade_samples)
        fade_in = torch.linspace(0.0, 1.0, crossfade_samples)

        result_end = result[-crossfade_samples:] * fade_out
        audio_start = audio[:crossfade_samples] * fade_in
        crossfade = result_end + audio_start

        result = torch.cat([
            result[:-crossfade_samples],
            crossfade,
            audio[crossfade_samples:],
        ], dim=-1)

    return result
