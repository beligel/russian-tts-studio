"""Models package — TTS engine wrappers.

Currently shipped:
- VoxCPMSynthesizer (OpenBMB VoxCPM2, zero-shot voice cloning, 30 langs
  incl. Russian, 48 kHz output. Primary engine.)
- HiggsAudioSynthesizer (Boson AI Higgs Audio v2, 100+ langs, smart
  voice + multi-speaker + sound events + voice profiles, 24 kHz
  output. Apache-2.0. Requires the upstream ``boson-ai/higgs-audio``
  repo installed.)
- Qwen3Synthesizer (Qwen3-TTS, voice clone / custom voice / voice design,
  10 langs incl. Russian, 24 kHz output. Optional second engine.)
- SileroSynthesizer (fast Russian fallback, no voice cloning)
- voice_profiles: text-described voices via YAML (``profile:<name>``
  syntax, inspired by Higgs Audio).
"""

from .base_synth import SynthesisRequest, SynthesisResult
from .silero_synth import SileroSynthesizer
from .voxcpm_synth import VoxCPMSynthesizer

__all__ = [
    "SynthesisRequest",
    "SynthesisResult",
    "SileroSynthesizer",
    "VoxCPMSynthesizer",
]


def get_higgs_synthesizer(**kwargs):
    """Lazy import for HiggsAudioSynthesizer.

    Requires the upstream ``boson-ai/higgs-audio`` repo installed::

        git clone https://github.com/boson-ai/higgs-audio.git
        cd higgs-audio && pip install -r requirements.txt && pip install -e .
    """
    from .higgs_synth import HiggsAudioSynthesizer
    return HiggsAudioSynthesizer(**kwargs)


def get_qwen3_synthesizer(**kwargs):
    """Lazy import for Qwen3Synthesizer (requires ``pip install qwen-tts``)."""
    from .qwen3_synth import Qwen3Synthesizer
    return Qwen3Synthesizer(**kwargs)
