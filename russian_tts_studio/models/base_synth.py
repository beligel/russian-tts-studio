"""Base dataclasses shared by all TTS engines (XTTS, Silero).

Extracted from the old cosyvoice_synth.py so that the wrapper package
no longer depends on the Russian TTS Studio upstream module at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SynthesisRequest:
    """Single TTS synthesis request."""

    text: str
    reference_audio: str | Path
    reference_text: Optional[str] = None
    instruct: Optional[str] = None
    speed: float = 1.0
    output_path: str | Path | None = None
    speaker_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class SynthesisResult:
    """TTS synthesis result with metadata."""

    audio_path: Path
    duration_sec: float
    generation_time_sec: float
    rtf: float
    model: str
    text: str
    reference: Optional[Path] = None
    metadata: dict = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None
