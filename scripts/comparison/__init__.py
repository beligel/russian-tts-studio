"""__init__ for comparison engines.

Engines shipped in this build:
- silero (always available, no cloning)
- xtts (Coqui XTTS-v2, zero-shot cloning, Russian)
- voxcpm (OpenBMB VoxCPM2, zero-shot cloning, 30 langs incl. Russian,
  48 kHz. Apache-2.0. Must run in .venv-voxcpm.)
"""

from .base import TTSEngine, TTSOutput
from .silero_engine import SileroEngine
from .xtts_engine import XTTSEngine
from .voxcpm_engine import VoxCPMEngine

__all__ = ["TTSEngine", "TTSOutput", "SileroEngine", "XTTSEngine", "VoxCPMEngine"]


def get_engine(name: str, **kwargs) -> TTSEngine:
    """Lazy-load a TTS engine by name.

    Engines with optional dependencies (xtts, voxcpm) are only
    imported if requested.
    """
    name = name.lower()
    if name in ("silero",):
        return SileroEngine(**kwargs)
    if name in ("xtts", "xtts-v2", "xtts_v2", "xttsv2"):
        from .xtts_engine import XTTSEngine
        return XTTSEngine(**kwargs)
    if name in ("voxcpm", "voxcpm-2", "voxcpm2", "voxcpm_v2"):
        from .voxcpm_engine import VoxCPMEngine
        return VoxCPMEngine(**kwargs)
    raise ValueError(
        f"Unknown engine: {name}. Available: silero, xtts, voxcpm"
    )
