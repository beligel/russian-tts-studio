"""Common base for TTS engines used in comparison tests."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TTSOutput:
    """Unified TTS output across engines."""

    audio_path: Path
    duration_sec: float
    generation_time_sec: float
    rtf: float
    engine: str
    text: str
    success: bool = True
    error: Optional[str] = None
    extra: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.extra is None:
            self.extra = {}


class TTSEngine(ABC):
    """Abstract base class for TTS engines used in comparison."""

    name: str = "base"
    supports_cloning: bool = False
    license: str = "unknown"

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def synthesize(
        self,
        text: str,
        reference_audio: Optional[Path] = None,
        output_path: Optional[Path] = None,
        **kwargs,
    ) -> TTSOutput:
        ...

    def cleanup(self) -> None:
        pass
