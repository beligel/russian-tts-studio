"""Prosody enhancement for VoxCPM2 output.

VoxCPM2 has no SSML/prosody input — prosody is autoprosoody only.
This module compensates by post-processing the synthesised audio:
* run forced alignment of the synthesised wav against the original
  text (via torchaudio's ``MMS_FA`` model + ``uroman`` for Cyrillic
  → ASCII romanisation)
* locate the time positions of punctuation marks in the wav
* splice in silence of user-configurable duration at each location

Only VoxCPM2 uses this — XTTS-v2 already has its own (poor) prosody
sensitivity and the chunking would interfere with its inference.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio

logger = logging.getLogger(__name__)

# --- punctuation rules --------------------------------------------------------
# What we consider a "pause point" and the default duration for each kind.
# These defaults were chosen as a conservative compromise — long enough
# to be heard, short enough not to sound robotic.

DEFAULT_PAUSE_MS: dict[str, int] = {
    "comma": 500,
    "semicolon": 700,
    "colon": 700,
    "period": 900,
    "exclamation": 1000,
    "question": 1000,
    "ellipsis": 1300,
}

# Mapping char → rule key. Order matters for the ellipsis check (we test
# "..." first because three dots also match the period rule individually).
_PUNCT_RULES: list[tuple[str, str]] = [
    ("…", "ellipsis"),  # single-char ellipsis
    (".", "period"),
    ("!", "exclamation"),
    ("?", "question"),
    (";", "semicolon"),
    (":", "colon"),
    (",", "comma"),
]


@dataclass
class PauseConfig:
    """Per-punctuation pause durations in milliseconds. 0 disables that mark."""
    comma: int = DEFAULT_PAUSE_MS["comma"]
    semicolon: int = DEFAULT_PAUSE_MS["semicolon"]
    colon: int = DEFAULT_PAUSE_MS["colon"]
    period: int = DEFAULT_PAUSE_MS["period"]
    exclamation: int = DEFAULT_PAUSE_MS["exclamation"]
    question: int = DEFAULT_PAUSE_MS["question"]
    ellipsis: int = DEFAULT_PAUSE_MS["ellipsis"]

    def is_enabled(self) -> bool:
        return any(
            getattr(self, name) > 0
            for name in ("comma", "semicolon", "colon", "period",
                         "exclamation", "question", "ellipsis")
        )

    def duration_for_char(self, ch: str) -> Optional[int]:
        """Return pause duration in ms for the given punctuation char, or None."""
        for punct, name in _PUNCT_RULES:
            if ch == punct:
                ms = getattr(self, name)
                return ms if ms > 0 else None
        return None

    @classmethod
    def from_metadata(cls, meta: dict) -> "PauseConfig":
        """Build a PauseConfig from a request.metadata dict (all keys optional)."""
        kwargs: dict = {}
        for name in ("comma", "semicolon", "colon", "period",
                     "exclamation", "question", "ellipsis"):
            val = meta.get(f"pause_ms_{name}")
            if val is not None:
                try:
                    kwargs[name] = max(0, int(val))
                except (TypeError, ValueError):
                    pass
        return cls(**kwargs)


# --- MMS_FA + uroman lazy loader ---------------------------------------------

_aligner = None
_tokenizer = None
_model = None
_device = None
_load_lock = threading.Lock()


def _load_aligner():
    """Load torchaudio MMS_FA aligner + uroman. Cached for the process lifetime.

    Downloads are automatic on first call (~1 GB for MMS_FA, ~10 MB for
    uroman). Subsequent calls reuse the in-process model.

    Honours the ``MMS_FA_SKIP_DOWNLOAD`` env var: when set to ``1``,
    the loader refuses to download and raises ``ModelUnavailable`` so
    the caller can fall back to proportional placement. This is useful
    on networks where ``dl.fbaipublicfiles.com`` is throttled (~1 KB/s
    here — a full download would take 14+ days).
    """
    global _aligner, _tokenizer, _model, _device
    if _aligner is not None:
        return
    with _load_lock:
        if _aligner is not None:
            return
        if os.environ.get("MMS_FA_SKIP_DOWNLOAD") == "1":
            raise ModelUnavailable(
                "MMS_FA_SKIP_DOWNLOAD=1 set; using proportional fallback"
            )
        from torchaudio.pipelines import MMS_FA as bundle
        import uroman

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = bundle.get_model().to(_device)
        _model.eval()
        _aligner = bundle.get_aligner()
        _tokenizer = bundle.get_tokenizer()
        # uroman instance is reentrant / thread-safe for romanize_string
        _uroman = uroman.Uroman()
        # Stash on globals so _align_text can reach it
        globals()["_uroman_instance"] = _uroman  # type: ignore[var-annotated]
        logger.info(
            "Loaded MMS_FA aligner (device=%s, sample_rate=%d)",
            _device, bundle.sample_rate,
        )


class ModelUnavailable(RuntimeError):
    """Raised when the forced-aligner model can't be loaded (network
    blocked, file missing, or MMS_FA_SKIP_DOWNLOAD=1). Callers should
    catch this and use a proportional fallback."""


def _normalise_ru(text: str) -> str:
    """Lowercase + replace apostrophes + strip everything that isn't a
    latin letter, apostrophe, or space.

    Mirrors the torchaudio multilingual forced-alignment tutorial.
    """
    text = text.lower().replace("’", "'")
    text = re.sub(r"([^a-z' ])", " ", text)
    return re.sub(r" +", " ", text).strip()


def _char_to_word_mapping(text: str) -> list[tuple[int, int, str]]:
    """Return list of (char_start, char_end, word) spans in the *original* text.

    A "word" is a maximal run of letters/digits/apostrophes — same
    definition uroman preserves through romanisation. We use this to map
    from the aligned romanised transcript back to character positions in
    the original Cyrillic text.
    """
    spans: list[tuple[int, int, str]] = []
    for m in re.finditer(r"[^\W_]+(?:['’][^\W_]+)*", text, flags=re.UNICODE):
        spans.append((m.start(), m.end(), m.group()))
    return spans


def _romanize_words(words: list[str]) -> list[str]:
    """Romanise a list of words preserving order. Stripped of any
    uroman-added angle brackets / diacritics since MMS_FA's vocab is
    pure ASCII."""
    urom = globals().get("_uroman_instance")
    if urom is None:
        return [_normalise_ru(w) for w in words]
    out: list[str] = []
    for w in words:
        r = urom.romanize_string(w, lcode="rus")
        out.append(_normalise_ru(r))
    return out


def _align_text(
    wav_path: Path,
    text: str,
) -> dict[int, tuple[float, float]]:
    """Return ``{char_index: (start_sec, end_sec)}`` for *every* character
    in ``text`` that participates in a word (i.e. alphanumerics). Punctuation
    and whitespace are not in the dict — call sites must look up by the
    preceding character's end-time to place a pause after it.

    Raises RuntimeError if the aligner could not be loaded.
    """
    _load_aligner()
    assert _aligner is not None and _tokenizer is not None and _model is not None
    assert _device is not None

    spans = _char_to_word_mapping(text)
    if not spans:
        return {}

    # 1) Resample to 16 kHz (MMS_FA native rate)
    waveform, sr = torchaudio.load(str(wav_path))
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # mono
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    waveform = waveform.to(_device)

    # 2) Romanise and split into words for the tokenizer
    original_words = [s[2] for s in spans]
    romanised = _romanize_words(original_words)
    # Build the tokenised transcript the aligner expects: list of word strings
    transcript_words: list[str] = []
    for r in romanised:
        transcript_words.extend(r.split())
    if not transcript_words:
        return {}

    # 3) Forward + align
    with torch.inference_mode():
        emission, _ = _model(waveform)
        # token_spans: list[list[TokenSpan]] — outer = words, inner = tokens in word
        token_spans = _aligner(emission[0], _tokenizer(transcript_words))

    # 4) Convert frame indices to seconds
    ratio = waveform.size(1) / emission.size(1) / 16000.0

    # 5) Map romanised words back to original text char positions.
    #    Strategy: zip original_words (original text) with token_spans
    #    (after uroman, the WORD count matches — uroman keeps words
    #    intact, just transliterates). For each original word, take the
    #    first token's start and the last token's end as the word's
    #    time span. Then expand word span to per-char spans.
    if len(token_spans) != len(original_words):
        # Uroman or aligner dropped/added a word — fall back to proportional
        # mapping by word count.
        logger.warning(
            "Alignment word count mismatch: %d original vs %d aligned — "
            "falling back to proportional mapping",
            len(original_words), len(token_spans),
        )
        n = max(1, len(original_words))
        total_sec = waveform.size(1) / 16000.0
        result: dict[int, tuple[float, float]] = {}
        for idx, (cs, ce, _w) in enumerate(spans):
            t0 = (idx / n) * total_sec
            t1 = ((idx + 1) / n) * total_sec
            for i in range(cs, ce):
                result[i] = (t0, t1)
        return result

    result = {}
    for (cs, ce, _orig_w), word_token_spans in zip(spans, token_spans):
        if not word_token_spans:
            continue
        word_start = word_token_spans[0].start * ratio
        word_end = word_token_spans[-1].end * ratio
        # Linear split: char `cs + k` occupies [word_start + k*Δ, ... + (k+1)*Δ]
        n_chars = max(1, ce - cs)
        delta = (word_end - word_start) / n_chars
        for k in range(n_chars):
            t0 = word_start + k * delta
            t1 = t0 + delta
            result[cs + k] = (t0, t1)
    return result


# --- Main entry point ---------------------------------------------------------

def insert_pauses(
    wav_path: Path,
    text: str,
    config: PauseConfig,
) -> tuple[Path, bool]:
    """Insert silence after punctuation marks in ``wav_path`` per ``config``.

    Returns ``(output_path, degraded)`` where ``degraded`` is True if the
    forced-aligner failed and we fell back to proportional placement.

    The original file at ``wav_path`` is left untouched (the caller can
    decide to overwrite).

    If ``config.is_enabled()`` is False, returns ``(wav_path, False)``
    unchanged.

    Two strategies are tried in order:
      1. Forced alignment via MMS_FA — exact positions, requires
         ~1 GB model download from a CDN that may throttle heavily.
      2. Proportional fallback — distribute the punctuation positions
         evenly across the audio by char index. Imprecise but works
         without any model.
    """
    if not config.is_enabled():
        return wav_path, False

    if not wav_path.exists():
        logger.warning("insert_pauses: wav not found: %s", wav_path)
        return wav_path, False

    # 1) Locate punctuation positions in text
    punct_positions: list[tuple[int, int, str]] = []  # (char_idx, ms, ch)
    # Walk the text and consume "..." as one ellipsis (three ASCII dots).
    i = 0
    while i < len(text):
        ch = text[i]
        # 3-character ASCII ellipsis
        if ch == "." and text[i:i + 3] == "...":
            ms = config.duration_for_char("…")
            if ms and ms > 0:
                punct_positions.append((i + 2, ms, "…"))
            i += 3
            continue
        # Single-char ellipsis
        if ch == "…":
            ms = config.duration_for_char("…")
            if ms and ms > 0:
                punct_positions.append((i, ms, "…"))
            i += 1
            continue
        # Regular punctuation
        ms = config.duration_for_char(ch)
        if ms and ms > 0:
            punct_positions.append((i, ms, ch))
        i += 1

    if not punct_positions:
        return wav_path, False

    # 2) Try forced alignment; fall back to proportional on any failure
    insert_points: list[tuple[float, int]] = []
    degraded = False
    try:
        char_times = _align_text(wav_path, text)
        for char_idx, ms, _ch in punct_positions:
            # Walk back to the previous alphanum char
            j = char_idx - 1
            while j >= 0 and j not in char_times:
                j -= 1
            if j < 0:
                continue
            _, end_t = char_times[j]
            insert_points.append((end_t, ms))
    except Exception as e:
        logger.warning(
            "insert_pauses: forced aligner unavailable (%s: %s) — "
            "using proportional fallback",
            type(e).__name__, e,
        )
        degraded = True
        insert_points = _proportional_punct_timestamps(text, punct_positions, wav_path)

    if not insert_points:
        return wav_path, degraded

    # 3) Splice silences into the wav
    try:
        wav, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    except Exception as e:
        logger.warning("insert_pauses: cannot read wav: %s", e)
        return wav_path, degraded

    if wav.ndim == 1:
        channels = 1
    else:
        channels = wav.shape[1]

    # Sort inserts by time (must be ascending to compute cumulative offset)
    insert_points.sort(key=lambda x: x[0])
    # Cap each pause: never insert silence beyond the END of the audio
    total_sec = wav.shape[0] / sr
    insert_points = [(t, ms) for t, ms in insert_points if t < total_sec - 0.01]

    if not insert_points:
        return wav_path, degraded

    # Build a new sample array by interleaving segments
    samples_per_ms = sr / 1000.0
    out_segments: list[np.ndarray] = []
    cursor_samples = 0
    for t_sec, ms in insert_points:
        end_samples = int(round(t_sec * sr))
        if end_samples <= cursor_samples:
            continue  # punctuation positions collided after rounding
        out_segments.append(wav[cursor_samples:end_samples])
        silence_samples = int(round(ms * samples_per_ms))
        silence = np.zeros((silence_samples, channels) if channels > 1
                           else (silence_samples,), dtype=np.float32)
        out_segments.append(silence)
        cursor_samples = end_samples
    out_segments.append(wav[cursor_samples:])

    new_wav = np.concatenate(out_segments, axis=0)

    # Overwrite in place — caller passed an output_path that they own
    sf.write(str(wav_path), new_wav, sr, subtype="FLOAT")
    new_dur = new_wav.shape[0] / sr
    mode = "proportional" if degraded else "aligned"
    logger.info(
        "insert_pauses [%s]: %d pauses inserted, %.2fs → %.2fs (+%.0fms total)",
        mode, len(insert_points), total_sec, new_dur, int((new_dur - total_sec) * 1000),
    )
    return wav_path, degraded


def _proportional_punct_timestamps(
    text: str,
    punct_positions: list[tuple[int, int, str]],
    wav_path: Path,
) -> list[tuple[float, int]]:
    """Distribute punctuation positions evenly across the audio duration.

    Used when the forced aligner can't be loaded. Each punctuation's
    timestamp is set to ``char_idx / len(text) * total_sec`` — coarse
    but better than no pauses at all.

    Returns ``[(time_sec, pause_ms), ...]``.
    """
    try:
        info = sf.info(str(wav_path))
        total_sec = float(info.frames) / float(info.samplerate)
    except Exception as e:
        logger.warning("proportional fallback: cannot read wav: %s", e)
        return []
    n = max(1, len(text))
    return [((idx / n) * total_sec, ms) for idx, ms, _ in punct_positions]
