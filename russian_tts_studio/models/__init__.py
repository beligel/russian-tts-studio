"""Models package — TTS engine wrappers.

Currently shipped:
- SileroSynthesizer (fast Russian fallback, no voice cloning)
- XTTSSynthesizer (Coqui XTTS-v2, zero-shot voice cloning, Russian)
- VoxCPMSynthesizer (OpenBMB VoxCPM2, zero-shot voice cloning, 30 langs
  incl. Russian, 48 kHz output. Lives in .venv-voxcpm — incompatible
  torch versions with the XTTS venv.)
"""

from .base_synth import SynthesisRequest, SynthesisResult
from .silero_synth import SileroSynthesizer
from .xtts_synth import XTTSSynthesizer
from .voxcpm_synth import VoxCPMSynthesizer

__all__ = [
    "SynthesisRequest",
    "SynthesisResult",
    "SileroSynthesizer",
    "XTTSSynthesizer",
    "VoxCPMSynthesizer",
]
