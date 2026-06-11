"""Bridge between Russian TTS Studio pipeline and ComfyUI plugin.

This script:
- Discovers ComfyUI + plugin installation
- Allows using ComfyUI's saved speaker presets
- Lets the pipeline save results in ComfyUI-compatible format
- Exposes pipeline as a node-callable function (if needed)

Usage:
    # 1. Discover ComfyUI/plugin
    python scripts/integration/comfyui_bridge.py --discover

    # 2. List available speakers
    python scripts/integration/comfyui_bridge.py --list-speakers

    # 3. Convert pipeline output → ComfyUI speaker preset
    python scripts/integration/comfyui_bridge.py \
        --export-speaker --audio output/samples/test.wav \
        --name "my_voice" --text "Reference text"

    # 4. Synthesize with a ComfyUI speaker preset
    python scripts/integration/comfyui_bridge.py \
        --synthesize --text "Привет!" --speaker "my_voice" \
        --output output/samples/synth.wav

    # 5. Install the plugin (if not yet)
    python scripts/integration/comfyui_bridge.py --install
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from russian_tts_studio.integrations import ComfyUIConfig  # noqa: E402
from russian_tts_studio.integrations.comfyui import (  # noqa: E402
    export_pipeline_result_to_speaker,
    find_comfyui,
    find_plugin,
    import_comfyui_speakers_to_pipeline,
    install_plugin,
    load_speaker_preset,
    save_speaker_preset,
)
from russian_tts_studio.pipeline import TTSPipeline, PipelineConfig  # noqa: E402
from russian_tts_studio.utils.audio_utils import (  # noqa: E402
    load_audio, get_duration,
)
from russian_tts_studio.utils.metrics import Transcriber  # noqa: E402
import torchaudio  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("comfyui_bridge")


def cmd_discover() -> int:
    config = ComfyUIConfig.discover()
    if not config:
        print("❌ ComfyUI or plugin not found")
        print("\nTo install:  python scripts/integration/comfyui_bridge.py --install")
        return 1
    print(f"✅ ComfyUI:  {config.comfyui_path}")
    print(f"✅ Plugin:   {config.plugin_path}")
    print(f"✅ Models:   {config.models_path}")
    print(f"✅ Speakers: {config.speakers_path}")
    print(f"\nModels: {len(config.list_models())} | Speakers: {len(config.list_speakers())}")
    return 0


def cmd_list_speakers() -> int:
    config = ComfyUIConfig.discover()
    if not config:
        print("❌ ComfyUI not found")
        return 1
    speakers = config.list_speakers()
    if not speakers:
        print("No speaker presets found")
        print(f"Path: {config.speakers_path}")
        return 0
    print(f"Found {len(speakers)} speaker presets:\n")
    for sp in speakers:
        try:
            data = load_speaker_preset(config, sp.stem)
            audio = data.get("audio")
            sr = data.get("sample_rate", 16000)
            text = data.get("text", "")
            dur = audio.shape[-1] / sr if audio is not None else 0
            print(f"  • {sp.stem}")
            print(f"    duration: {dur:.2f}s | text: {text[:60]!r}")
        except Exception as e:
            print(f"  • {sp.stem} (failed to load: {e})")
    return 0


def cmd_export_speaker(args: argparse.Namespace) -> int:
    config = ComfyUIConfig.discover()
    if not config:
        print("❌ ComfyUI not found")
        return 1

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"❌ Audio not found: {audio_path}")
        return 1

    text = args.text or ""
    if not text and not args.no_auto_transcribe:
        print("Auto-transcribing reference audio...")
        try:
            transcriber = Transcriber(model_size="base", device="cpu")
            wav = load_audio(audio_path, target_sr=16000, mono=True)
            text = transcriber.transcribe(wav, language="ru")
            print(f"Transcript: {text!r}")
        except Exception as e:
            print(f"⚠️  Auto-transcribe failed: {e}")
            text = ""

    out = save_speaker_preset(
        config=config,
        name=args.name,
        reference_audio=audio_path,
        reference_text=text,
    )
    print(f"✅ Exported speaker preset: {out}")
    return 0


def cmd_synthesize(args: argparse.Namespace) -> int:
    config = ComfyUIConfig.discover()
    if not config:
        print("❌ ComfyUI not found")
        return 1

    try:
        preset = load_speaker_preset(config, args.speaker)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 1

    ref_audio_data = preset.get("audio")
    sr = preset.get("sample_rate", 16000)
    ref_text = preset.get("text", "")
    if ref_audio_data is None:
        print("❌ Speaker preset has no audio data")
        return 1

    tmp_ref = PROJECT_ROOT / "output" / "samples" / f"_ref_{args.speaker}.wav"
    tmp_ref.parent.mkdir(parents=True, exist_ok=True)
    if ref_audio_data.dim() == 1:
        ref_audio_data = ref_audio_data.unsqueeze(0)
    torchaudio.save(str(tmp_ref), ref_audio_data, sr)
    print(f"Using ComfyUI speaker preset: {args.speaker} ({get_duration(ref_audio_data.squeeze(0), sr):.2f}s ref)")

    pipeline = TTSPipeline(PipelineConfig(device=args.device))
    try:
        out_path = Path(args.output) if args.output else \
            PROJECT_ROOT / "output" / "samples" / f"synth_{args.speaker}.wav"
        result = pipeline.synthesize(
            text=args.text,
            reference_audio=tmp_ref,
            reference_text=ref_text,
            output_path=out_path,
        )
        print(f"✅ Synthesized: {result['final_path']}")
        print(f"   Model: {result['result'].model} | "
              f"Duration: {result['result'].duration_sec:.2f}s | "
              f"Outcome: {result['outcome'].value}")
    finally:
        pipeline.cleanup()
    return 0


def cmd_install(args: argparse.Namespace) -> int:
    comfy = find_comfyui()
    if not comfy:
        print("❌ ComfyUI not found. Set COMFYUI_PATH env var.")
        return 1
    if find_plugin(comfy):
        print("✅ Plugin already installed")
        return 0
    install_plugin(comfy)
    print("✅ Plugin installed. Restart ComfyUI to enable.")
    return 0


def cmd_import_speakers(args: argparse.Namespace) -> int:
    config = ComfyUIConfig.discover()
    if not config:
        print("❌ ComfyUI not found")
        return 1
    target = Path(args.target)
    copied = import_comfyui_speakers_to_pipeline(config, target)
    print(f"✅ Imported {len(copied)} speakers to {target}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Russian TTS Studio pipeline ↔ ComfyUI plugin bridge")
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--list-speakers", action="store_true")
    parser.add_argument("--export-speaker", action="store_true")
    parser.add_argument("--synthesize", action="store_true")
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--import-speakers", action="store_true")
    parser.add_argument("--audio", type=Path, default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--text", "-t", default=None)
    parser.add_argument("--no-auto-transcribe", action="store_true")
    parser.add_argument("--speaker", default=None)
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument("--target", type=Path,
                        default=PROJECT_ROOT / "output" / "reference")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    rc = 0
    if args.discover:
        rc = cmd_discover()
    elif args.list_speakers:
        rc = cmd_list_speakers()
    elif args.export_speaker:
        if not args.audio or not args.name:
            parser.error("--export-speaker requires --audio and --name")
        rc = cmd_export_speaker(args)
    elif args.synthesize:
        if not args.text or not args.speaker:
            parser.error("--synthesize requires --text and --speaker")
        rc = cmd_synthesize(args)
    elif args.install:
        rc = cmd_install(args)
    elif args.import_speakers:
        rc = cmd_import_speakers(args)
    else:
        parser.print_help()
        rc = 0
    sys.exit(rc)


if __name__ == "__main__":
    main()
