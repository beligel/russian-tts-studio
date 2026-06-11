"""Post-process audio: trim silence, normalize loudness, denoise.

Usage:
    python scripts/postprocess/process.py \
        --input output/samples/test.wav \
        --output output/samples/test_clean.wav \
        --target-dbfs -20
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from russian_tts_studio.utils.audio_utils import (  # noqa: E402
    get_duration, load_audio, normalize_loudness, trim_silence,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("postprocess")


def process(
    input_path: Path,
    output_path: Path,
    target_dbfs: float = -20.0,
    trim: bool = True,
    normalize: bool = True,
    denoise: bool = False,
    target_sr: int = 22050,
) -> Path:
    """Apply post-processing pipeline to audio."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    waveform = load_audio(input_path, target_sr=target_sr, mono=True)
    logger.info("Loaded: %s (%.2fs)", input_path.name, get_duration(waveform, target_sr))

    if trim:
        waveform = trim_silence(waveform, threshold=0.01)
        logger.info("After trim: %.2fs", get_duration(waveform, target_sr))

    if normalize:
        waveform = normalize_loudness(waveform, target_dbfs=target_dbfs, sample_rate=target_sr)
        logger.info("Normalized to %.1f dBFS", target_dbfs)

    if denoise:
        try:
            from df.enhance import enhance, init_df, load_audio as df_load
            model, df_state, _ = init_df()
            wav_np = waveform.numpy()
            enhanced = enhance(model, df_state, torch.from_numpy(wav_np).unsqueeze(0))
            waveform = enhanced.squeeze(0)
            logger.info("Denoised with DeepFilterNet")
        except ImportError:
            logger.warning("DeepFilterNet not installed; skipping denoise")

    torchaudio.save(str(output_path), waveform.unsqueeze(0), target_sr)
    logger.info("Saved: %s (%.2fs)", output_path.name, get_duration(waveform, target_sr))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-process audio")
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--target-dbfs", type=float, default=-20.0)
    parser.add_argument("--no-trim", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--denoise", action="store_true")
    parser.add_argument("--target-sr", type=int, default=22050)
    args = parser.parse_args()

    process(
        input_path=args.input,
        output_path=args.output,
        target_dbfs=args.target_dbfs,
        trim=not args.no_trim,
        normalize=not args.no_normalize,
        denoise=args.denoise,
        target_sr=args.target_sr,
    )


if __name__ == "__main__":
    main()
