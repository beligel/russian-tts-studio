"""Adapter layer for using the ComfyUI_FL-Russian TTS Studio3 plugin's speakers/models.

The plugin stores:
- Models in `ComfyUI/models/cosyvoice/`
- Speaker presets in `ComfyUI/models/cosyvoice/speaker/<name>.pt`

This module provides utilities to:
- Locate ComfyUI installation
- Load/save speaker presets
- Convert presets to/from our pipeline format
- Share models between ComfyUI and our pipeline
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


COMFYUI_SEARCH_PATHS: list[Path] = [
    Path.home() / "ComfyUI",
    Path("/opt/ComfyUI"),
    Path("/usr/local/ComfyUI"),
    Path.home() / "Documents" / "ComfyUI",
    Path("/workspace/ComfyUI"),
    Path("C:/ComfyUI"),
    Path(os.environ.get("COMFYUI_PATH", "")),
]

COMFYUI_PLUGIN_NAME = "ComfyUI_FL-Russian TTS Studio3"


def find_comfyui(start: Path | None = None) -> Optional[Path]:
    """Locate ComfyUI installation by searching common paths."""
    start = start or Path.cwd()
    for candidate in COMFYUI_SEARCH_PATHS:
        if not str(candidate) or candidate == Path(""):
            continue
        if (candidate / "main.py").exists() or (candidate / "nodes.py").exists():
            logger.info("Found ComfyUI at: %s", candidate)
            return candidate
    try:
        result = subprocess.run(
            ["find", "/", "-name", "main.py", "-path", "*ComfyUI*"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().split("\n")[:1]:
                path = Path(line).parent
                if (path / "custom_nodes").exists():
                    logger.info("Found ComfyUI via filesystem search: %s", path)
                    return path
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    logger.warning("ComfyUI installation not found in standard paths")
    return None


def find_plugin(comfyui: Path) -> Optional[Path]:
    """Find the ComfyUI_FL-Russian TTS Studio3 plugin inside custom_nodes."""
    candidates = [
        comfyui / "custom_nodes" / COMFYUI_PLUGIN_NAME,
        comfyui / "custom_nodes" / "ComfyUI_FL-Russian TTS Studio3-main",
        comfyui / "custom_nodes" / "filliptm-ComfyUI_FL-Russian TTS Studio3",
    ]
    for c in candidates:
        if c.exists() and (c / "nodes" / "__init__.py").exists():
            logger.info("Found plugin at: %s", c)
            return c
    custom_nodes = comfyui / "custom_nodes"
    if custom_nodes.exists():
        for sub in custom_nodes.iterdir():
            if "cosyvoice" in sub.name.lower() and (sub / "nodes").exists():
                logger.info("Found cosyvoice-related plugin: %s", sub)
                return sub
    return None


@dataclass
class ComfyUIConfig:
    """Configuration for integration with ComfyUI plugin."""

    comfyui_path: Path
    plugin_path: Path
    models_path: Path
    speakers_path: Path

    @classmethod
    def discover(cls) -> Optional["ComfyUIConfig"]:
        """Auto-discover ComfyUI + plugin installation."""
        comfy = find_comfyui()
        if not comfy:
            return None
        plugin = find_plugin(comfy)
        if not plugin:
            logger.warning("Plugin %s not installed in custom_nodes", COMFYUI_PLUGIN_NAME)
            return None
        return cls(
            comfyui_path=comfy,
            plugin_path=plugin,
            models_path=comfy / "models" / "cosyvoice",
            speakers_path=comfy / "models" / "cosyvoice" / "speaker",
        )

    def list_models(self) -> list[Path]:
        """List available model files in ComfyUI's cosyvoice directory."""
        if not self.models_path.exists():
            return []
        return sorted([
            p for p in self.models_path.iterdir()
            if p.is_dir() or p.suffix in {".pt", ".bin", ".safetensors"}
        ])

    def list_speakers(self) -> list[Path]:
        """List saved speaker presets."""
        if not self.speakers_path.exists():
            return []
        return sorted(self.speakers_path.glob("*.pt"))


def save_speaker_preset(
    config: ComfyUIConfig,
    name: str,
    reference_audio: Path,
    reference_text: str = "",
    instruct: str = "",
    metadata: Optional[dict] = None,
) -> Path:
    """Save a speaker preset compatible with the ComfyUI plugin.

    The plugin uses torch.save with the following structure:
        {
            "audio": tensor,
            "text": str,
            "instruct": str,
            "metadata": dict
        }
    """
    import torch
    import torchaudio

    config.speakers_path.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    out_path = config.speakers_path / f"{safe_name}.pt"

    waveform, sr = torchaudio.load(str(reference_audio))
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    payload = {
        "audio": waveform.squeeze(0),
        "sample_rate": 16000,
        "text": reference_text,
        "instruct": instruct,
        "metadata": metadata or {},
        "format_version": 1,
    }
    torch.save(payload, out_path)
    logger.info("Saved speaker preset: %s (%.2fs audio)", out_path.name, waveform.shape[-1] / 16000)
    return out_path


def load_speaker_preset(
    config: ComfyUIConfig,
    name: str,
) -> dict:
    """Load a speaker preset saved by the ComfyUI plugin."""
    import torch
    candidates = [
        config.speakers_path / f"{name}.pt",
        config.speakers_path / name,
    ]
    for p in candidates:
        if p.exists():
            data = torch.load(p, map_location="cpu", weights_only=False)
            logger.info("Loaded speaker preset: %s", p.name)
            return data
    raise FileNotFoundError(f"Speaker preset '{name}' not found in {config.speakers_path}")


def export_pipeline_result_to_speaker(
    config: ComfyUIConfig,
    name: str,
    audio_path: Path,
    text: str = "",
) -> Path:
    """Export a synthesized audio to a ComfyUI-compatible speaker preset."""
    return save_speaker_preset(
        config=config,
        name=name,
        reference_audio=audio_path,
        reference_text=text,
    )


def import_comfyui_speakers_to_pipeline(
    config: ComfyUIConfig,
    target_dir: Path,
) -> list[Path]:
    """Copy ComfyUI speaker presets into our pipeline's reference directory."""
    target_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for sp in config.list_speakers():
        target = target_dir / sp.name
        if not target.exists():
            shutil.copy2(sp, target)
            copied.append(target)
            logger.info("Imported speaker preset: %s", sp.name)
    return copied


def install_plugin(comfyui: Path, branch: str = "main") -> Path:
    """Clone and install the Russian TTS Studio3 plugin into ComfyUI."""
    import subprocess
    target = comfyui / "custom_nodes" / COMFYUI_PLUGIN_NAME
    if target.exists():
        logger.info("Plugin already exists at: %s", target)
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/filliptm/{COMFYUI_PLUGIN_NAME}.git"
    logger.info("Cloning %s into %s", url, target)
    subprocess.run(
        ["git", "clone", "-b", branch, url, str(target)],
        check=True,
    )

    req = target / "requirements.txt"
    if req.exists():
        logger.info("Installing plugin requirements...")
        subprocess.run(
            ["pip", "install", "-r", str(req)],
            check=False,
        )
    return target


def main() -> None:
    """CLI: discover and print ComfyUI plugin config."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("Searching for ComfyUI + plugin...")
    config = ComfyUIConfig.discover()
    if not config:
        print("❌ ComfyUI or plugin not found.")
        print("   Set COMFYUI_PATH env var or install:")
        print("   https://github.com/filliptm/ComfyUI_FL-Russian TTS Studio3")
        return
    print(f"✅ ComfyUI:  {config.comfyui_path}")
    print(f"✅ Plugin:   {config.plugin_path}")
    print(f"✅ Models:   {config.models_path}")
    print(f"✅ Speakers: {config.speakers_path}")
    print(f"\nAvailable models: {len(config.list_models())}")
    print(f"Available speakers: {len(config.list_speakers())}")


if __name__ == "__main__":
    main()
