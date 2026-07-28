"""Audio processing: podcast mixing and subtitle generation.

Phase 4 (LTV-inspired):

- ``mix`` — FFmpeg-based podcast mixing with ducking, fades, loudnorm.
- ``subtitles`` — SRT and ASS subtitle generation from Whisper timestamps.

Public API::

    from russian_tts_studio.audio import mix_podcast, MixConfig
    from russian_tts_studio.audio import write_srt, write_ass, words_from_whisper
"""

from __future__ import annotations

from .mix import MixConfig, get_default_music, list_background_music, mix_podcast
from .subtitles import (
    SubtitleCue,
    WordTimestamp,
    words_from_timestamps_list,
    words_from_whisper,
    write_ass,
    write_srt,
)

__all__ = [
    "mix_podcast",
    "MixConfig",
    "list_background_music",
    "get_default_music",
    "write_srt",
    "write_ass",
    "WordTimestamp",
    "SubtitleCue",
    "words_from_whisper",
    "words_from_timestamps_list",
]