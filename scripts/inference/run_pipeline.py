"""CLI for the production TTS pipeline (XTTS-v2 + Silero fallback).

Usage:
    python scripts/inference/run_pipeline.py \
        --text "Привет, это тестовая фраза." \
        --reference output/reference/ru_voice.wav \
        --output output/samples/test.wav

Without a reference, will use Silero directly:
    python scripts/inference/run_pipeline.py \
        --text "Привет, это тест." \
        --output output/samples/test.wav \
        --speaker xenia
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from russian_tts_studio.pipeline import PipelineConfig, TTSPipeline  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_pipeline")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TTS pipeline")
    parser.add_argument("--text", "-t", required=True, help="Text to synthesize")
    parser.add_argument("--reference", "-r", type=Path, default=None,
                        help="Reference audio for voice cloning")
    parser.add_argument("--reference-text", default=None)
    parser.add_argument("--instruct", "-i", default=None,
                        help="Style instruction (e.g. 'Говори медленно и спокойно')")
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument("--speaker", default="xenia",
                        help="Silero speaker for fallback")
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--no-quality-check", action="store_true")
    parser.add_argument("--no-postprocess", action="store_true")
    parser.add_argument("--wer-threshold", type=float, default=0.20)
    parser.add_argument("--sim-threshold", type=float, default=0.50)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = PipelineConfig(
        enable_fallback=not args.no_fallback,
        enable_quality_check=not args.no_quality_check,
        enable_postprocess=not args.no_postprocess,
        wer_threshold=args.wer_threshold,
        sim_threshold=args.sim_threshold,
        device=args.device,
    )

    pipeline = TTSPipeline(config)
    try:
        result = pipeline.synthesize(
            text=args.text,
            reference_audio=args.reference,
            reference_text=args.reference_text,
            instruct=args.instruct,
            output_path=args.output,
            speaker_fallback=args.speaker,
        )
        outcome = result["outcome"].value
        path = result["final_path"]
        duration = result["result"].duration_sec
        gen_time = result["result"].generation_time_sec

        report: dict = {
            "outcome": outcome,
            "audio_path": str(path),
            "duration_sec": round(duration, 3),
            "generation_time_sec": round(gen_time, 2),
            "rtf": round(result["result"].rtf, 3),
            "model": result["result"].model,
        }
        if result["metrics"]:
            report["metrics"] = result["metrics"].to_dict()
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()
