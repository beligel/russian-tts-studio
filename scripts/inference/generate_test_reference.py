"""Generate a synthetic reference voice for testing the TTS pipeline.

If you don't have a real Russian reference audio, this script uses
Silero to synthesize one with a clean neutral voice. Useful for
smoke tests, not for production-quality evaluation.

Usage:
    python scripts/inference/generate_test_reference.py \
        --output output/reference/ru_voice.wav \
        --text "Привет, меня зовут Анна. Я расскажу вам о нашей компании." \
        --speaker xenia
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("gen_ref")


DEFAULT_REFERENCE_TEXT = (
    "Привет, меня зовут Анна. Сегодня я расскажу вам о нашей компании. "
    "Мы работаем на рынке с две тысячи пятнадцатого года и предоставляем "
    "качественные услуги. Наша команда состоит из опытных специалистов, "
    "готовых решать самые сложные задачи."
)


def generate_reference(
    output: Path,
    text: str | None = None,
    speaker: str = "xenia",
) -> Path:
    """Generate a synthetic Russian reference voice using Silero."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not text or not text.strip():
        text = DEFAULT_REFERENCE_TEXT
        logger.info("Using default reference text (%d chars)", len(text))

    logger.info("Loading Silero...")
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_tts",
        language="ru",
        speaker="v3_1_ru",
        trust_repo=True,
    )

    logger.info("Synthesizing reference: '%s'...", text[:60])
    model.save_wav(
        text=text,
        speaker=speaker,
        sample_rate=48000,
        audio_path=str(output),
    )
    logger.info("Saved: %s", output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate test reference voice")
    parser.add_argument("--output", "-o", type=Path,
                        default=PROJECT_ROOT / "output" / "reference" / "ru_voice.wav")
    parser.add_argument("--text", "-t", type=str, default=None,
                        help="Text to synthesize (default: built-in reference text)")
    parser.add_argument("--speaker", "-s", default="xenia",
                        choices=["aidar", "baya", "kseniya", "xenia", "eugene"])
    args = parser.parse_args()

    generate_reference(args.output, args.text, args.speaker)


if __name__ == "__main__":
    main()
