"""Compare XTTS-v2 and Silero on identical Russian phrases.

Usage:
    python scripts/comparison/compare_engines.py \
        --reference output/reference/ru_voice.wav \
        --engines silero,xtts \
        --output-dir output/comparison

Outputs:
    - output/comparison/<engine>/<phrase>.wav — audio from each engine
    - output/comparison/comparison_<timestamp>.json — full results
    - output/comparison/comparison_<timestamp>.md — readable report
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.comparison import get_engine  # noqa: E402
from russian_tts_studio.utils.audio_utils import load_audio  # noqa: E402
from russian_tts_studio.utils.metrics import (  # noqa: E402
    SpeakerSimilarityCalculator, Transcriber,
    calculate_cer, calculate_wer, normalize_text_for_wer,
    TTSQualityMetrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("compare_engines")


TEST_PHRASES: dict[str, list[str]] = {
    "easy": [
        "Привет, как дела?",
        "Сегодня хорошая погода.",
    ],
    "medium": [
        "В 2024 году компания выручила полтора миллиарда рублей.",
        "API, Python и GPT — модель знает эти термины?",
    ],
    "hard": [
        "Ёжик в тумане — культовый советский мультфильм.",
    ],
}


def cleanup_engine(engine) -> None:
    try:
        engine.cleanup()
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_comparison(
    reference: Path,
    engines: list[str],
    output_dir: Path,
    reference_text: str | None = None,
    skip_engines_requiring_cloning_without_ref: bool = True,
) -> dict:
    """Run a comparison across all specified engines."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Loading transcriber + similarity calculator...")
    transcriber = Transcriber(model_size="base", device="cpu")
    sim_calc = SpeakerSimilarityCalculator(model_name="wavlm")

    ref_wav = load_audio(reference, target_sr=16000, mono=True) if reference.exists() else None

    all_results: list[dict] = []

    for engine_name in engines:
        logger.info("=" * 70)
        logger.info("Engine: %s", engine_name)
        try:
            engine = get_engine(engine_name)
        except ImportError as e:
            logger.warning("Engine %s unavailable: %s", engine_name, e)
            all_results.append({"engine": engine_name, "error": f"import failed: {e}"})
            continue
        except Exception as e:
            logger.error("Failed to init engine %s: %s", engine_name, e)
            all_results.append({"engine": engine_name, "error": str(e)})
            continue

        engine_dir = output_dir / engine.name
        engine_dir.mkdir(parents=True, exist_ok=True)

        try:
            engine.load()
            for category, phrases in TEST_PHRASES.items():
                for i, phrase in enumerate(phrases, 1):
                    safe = "".join(c if c.isalnum() else "_" for c in phrase[:30])
                    out_path = engine_dir / f"{category}_{i}_{safe}.wav"

                    if engine.supports_cloning and reference.exists():
                        output = engine.synthesize(
                            text=phrase,
                            reference_audio=reference,
                            output_path=out_path,
                            reference_text=reference_text,
                            language="ru",
                        )
                    elif not engine.supports_cloning:
                        output = engine.synthesize(
                            text=phrase,
                            reference_audio=None,
                            output_path=out_path,
                        )
                    else:
                        logger.warning("Skipping %s for %s — needs reference", engine_name, phrase)
                        continue

                    if not output.success:
                        all_results.append({
                            "engine": engine_name,
                            "phrase": phrase, "category": category,
                            "error": output.error,
                        })
                        continue

                    metrics = TTSQualityMetrics()
                    try:
                        synth = load_audio(output.audio_path, target_sr=16000, mono=True)
                        metrics.transcript = transcriber.transcribe(synth, language="ru")
                        if metrics.transcript:
                            ref_norm = normalize_text_for_wer(phrase)
                            hyp_norm = normalize_text_for_wer(metrics.transcript)
                            metrics.wer = calculate_wer(ref_norm, hyp_norm)
                            metrics.cer = calculate_cer(
                                ref_norm.replace(" ", ""), hyp_norm.replace(" ", ""),
                            )
                        if ref_wav is not None and sim_calc._try_load_model():
                            metrics.speaker_similarity = sim_calc.similarity(
                                ref_wav=ref_wav, synth_wav=synth,
                            )
                    except Exception as e:
                        logger.warning("Metrics failed for %s: %s", output.audio_path, e)

                    all_results.append({
                        "engine": engine_name,
                        "category": category,
                        "phrase": phrase,
                        "audio_path": str(output.audio_path.relative_to(PROJECT_ROOT)),
                        "duration_sec": round(output.duration_sec, 3),
                        "rtf": round(output.rtf, 3),
                        "generation_time_sec": round(output.generation_time_sec, 2),
                        "wer": round(metrics.wer, 4),
                        "cer": round(metrics.cer, 4),
                        "speaker_similarity": round(metrics.speaker_similarity, 3),
                        "transcript": metrics.transcript,
                    })
                    logger.info(
                        "  %s/%s: WER=%.1f%% CER=%.1f%% SIM=%.2f RTF=%.3f",
                        category, i, metrics.wer * 100, metrics.cer * 100,
                        metrics.speaker_similarity, output.rtf,
                    )
        except Exception as e:
            logger.exception("Engine %s failed: %s", engine_name, e)
            all_results.append({"engine": engine_name, "error": str(e)})
        finally:
            cleanup_engine(engine)

    summary: dict = {
        "per_result": all_results,
        "by_engine": {},
        "meta": {
            "engines": engines,
            "reference": str(reference),
            "timestamp": datetime.now().isoformat(),
        },
    }
    by_engine: dict[str, list[dict]] = {}
    for r in all_results:
        if "wer" in r:
            by_engine.setdefault(r["engine"], []).append(r)
    for eng, results in by_engine.items():
        if not results:
            continue
        summary["by_engine"][eng] = {
            "n_phrases": len(results),
            "wer_mean": sum(r["wer"] for r in results) / len(results),
            "cer_mean": sum(r["cer"] for r in results) / len(results),
            "sim_mean": sum(r["speaker_similarity"] for r in results) / len(results),
            "rtf_mean": sum(r["rtf"] for r in results) / len(results),
        }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"comparison_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("JSON: %s", json_path)

    md_path = output_dir / f"comparison_{timestamp}.md"
    write_report(md_path, summary)
    logger.info("MD: %s", md_path)

    return summary


def write_report(path: Path, summary: dict) -> None:
    lines: list[str] = ["# TTS Engine Comparison — Russian\n"]
    meta = summary.get("meta", {})
    lines.append(f"- **Timestamp:** {meta.get('timestamp', '—')}")
    lines.append(f"- **Reference:** `{meta.get('reference', '—')}`")
    lines.append(f"- **Engines:** {', '.join(meta.get('engines', []))}\n")

    lines.append("## Aggregate by engine\n")
    lines.append("| Engine | N | WER | CER | SIM | RTF |")
    lines.append("|---|---|---|---|---|---|")
    for eng, stats in summary.get("by_engine", {}).items():
        lines.append(
            f"| {eng} | {stats['n_phrases']} | "
            f"{stats['wer_mean']:.1%} | {stats['cer_mean']:.1%} | "
            f"{stats['sim_mean']:.3f} | {stats['rtf_mean']:.3f} |"
        )
    lines.append("")

    lines.append("## Per-phrase results\n")
    lines.append("| Engine | Category | Phrase | WER | SIM | RTF |")
    lines.append("|---|---|---|---|---|---|")
    for r in summary.get("per_result", []):
        if "wer" not in r:
            continue
        lines.append(
            f"| {r['engine']} | {r['category']} | {r['phrase'][:35]} | "
            f"{r['wer']:.1%} | {r['speaker_similarity']:.2f} | {r['rtf']:.3f} |"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare TTS engines on Russian")
    parser.add_argument("--reference", "-r", type=Path, required=True)
    parser.add_argument("--engines", "-e",
                        default="silero,xtts",
                        help="Comma-separated: silero,xtts")
    parser.add_argument("--output-dir", "-o", type=Path,
                        default=PROJECT_ROOT / "output" / "comparison")
    parser.add_argument("--reference-text", default=None)
    args = parser.parse_args()

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    run_comparison(
        reference=args.reference,
        engines=engines,
        output_dir=args.output_dir,
        reference_text=args.reference_text,
    )


if __name__ == "__main__":
    main()
