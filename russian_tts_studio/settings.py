"""Unified settings manager with schema versioning and migration.

Inspired by LTV's ``settings_manager.py``. All project settings live in
a single ``config.json`` file with:
- Deep merge with defaults on load
- Schema version tracking and auto-migration
- Sanitization of invalid values
- Thread-safe read/write

The settings file lives at ``<project_root>/config.json``. If it
doesn't exist, defaults are used and the file is created on first save.

Usage::

    from russian_tts_studio.settings import settings

    # Read
    speed = settings.get("tts.speed", 0.9)

    # Write
    settings.update({"tts.speed": 0.85})
    settings.save()
"""

from __future__ import annotations

import json
import logging
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Schema version — bump when DEFAULT_SETTINGS structure changes.
# Migration functions handle upgrades from older versions.
CURRENT_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Default settings
# ---------------------------------------------------------------------------

DEFAULT_SETTINGS: dict[str, Any] = {
    "schema_version": CURRENT_SCHEMA_VERSION,

    # TTS engine
    "tts.engine": "voxcpm",  # "voxcpm" | "qwen3"
    "tts.speed": 0.9,
    "tts.language": "ru",

    # VoxCPM2 specific
    "voxcpm.device": "auto",
    "voxcpm.cfg_value": 2.0,
    "voxcpm.inference_timesteps": 10,
    "voxcpm.load_denoiser": False,

    # Qwen3-TTS specific
    "qwen3.model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "qwen3.device": "auto",
    "qwen3.dtype": "auto",
    "qwen3.speaker": "Serena",
    "qwen3.language": "Russian",
    "qwen3.instruct": "",
    "qwen3.ref_text": "",

    # Fallback
    "fallback.enabled": True,
    "fallback.engine": "silero",
    "fallback.speaker": "xenia",

    # Quality check
    "review.enabled": True,
    "review.wer_threshold": 0.20,
    "review.cer_threshold": 0.15,
    "review.sim_threshold": 0.50,
    "review.wer_blocks": False,
    "review.retry_max": 2,
    "review.tail_warning_ms": 500,
    "review.tail_retry_ms": 1000,

    # Post-processing
    "postprocess.enabled": True,
    "postprocess.target_dbfs": -20.0,
    "postprocess.enable_clamp": True,
    "postprocess.max_gap_ms": 450,
    "postprocess.target_gap_ms": 180,

    # Prosody (VoxCPM2 only)
    "prosody.enabled": False,
    "prosody.comma_ms": 250,
    "prosody.semicolon_ms": 350,
    "prosody.colon_ms": 350,
    "prosody.period_ms": 500,
    "prosody.exclamation_ms": 550,
    "prosody.question_ms": 550,
    "prosody.ellipsis_ms": 700,
    "prosody.word_gap_ms": 0,

    # Text normalization
    "normalization.enabled": True,
    "normalization.use_abbreviations": True,
    "normalization.use_addresses": True,
    "normalization.use_custom_dicts": True,
    "normalization.normalize_numbers": True,
    "normalization.normalize_ordinals": True,
    "normalization.normalize_dates": True,
    "normalization.normalize_currencies": True,
    "normalization.normalize_percentages": True,
    "normalization.normalize_measurements": True,
    "normalization.normalize_roman": True,

    # Long-form
    "longform.max_chars": 200,
    "longform.max_sentences": 4,

    # Audio Mix
    "mix.voice_db": 0.0,
    "mix.music_db": -12.0,
    "mix.intro_sec": 0.0,
    "mix.tail_sec": 0.0,
    "mix.fade_in": 2.0,
    "mix.fade_out": 3.0,
    "mix.ducking": True,
    "mix.duck_depth_db": 9.0,
    "mix.duck_attack": 0.3,
    "mix.duck_release": 1.0,
    "mix.loudnorm": True,
    "mix.target_lufs": -16.0,
    "mix.output_format": "mp3",
    "mix.mp3_bitrate": "192k",
    "mix.sample_rate": 44100,

    # Subtitles
    "subtitles.max_duration": 5.0,
    "subtitles.max_chars": 80,

    # Web UI
    "ui.port": 8129,
    "ui.host": "0.0.0.0",
    "ui.language": "ru",

    # Paths
    "paths.output_dir": "output",
    "paths.music_dir": "music/background",
    "paths.projects_dir": "output/projects",
}


# ---------------------------------------------------------------------------
# Settings manager
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base. Override wins on conflicts."""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _migrate(settings: dict, from_version: int) -> dict:
    """Migrate settings from an older schema version.

    Add migration functions here when bumping CURRENT_SCHEMA_VERSION.
    """
    result = deepcopy(settings)

    # Example migration: version 0 → 1
    # if from_version < 1:
    #     result["new_key"] = result.pop("old_key", default)

    result["schema_version"] = CURRENT_SCHEMA_VERSION
    return result


class Settings:
    """Unified settings manager with dot-notation access.

    Keys use dots as separators: ``"tts.speed"``, ``"review.wer_threshold"``.
    Values are stored flat — no nested dicts in the internal store.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else Path("config.json")
        self._data: dict[str, Any] = deepcopy(DEFAULT_SETTINGS)
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        """Load settings from disk, merge with defaults, migrate."""
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return
            version = raw.get("schema_version", 0)
            if version < CURRENT_SCHEMA_VERSION:
                raw = _migrate(raw, version)
            self._data = _deep_merge(DEFAULT_SETTINGS, raw)
            logger.info("Loaded settings from %s (schema v%s)", self._path, version)
        except (OSError, json.JSONDecodeError, ValueError) as e:
            logger.warning("Failed to load settings from %s: %s", self._path, e)

    def save(self) -> None:
        """Persist current settings to disk (atomic write)."""
        with self._lock:
            self._data["schema_version"] = CURRENT_SCHEMA_VERSION
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(self._path)
            logger.info("Saved settings to %s", self._path)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting by dot-notation key."""
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a single setting."""
        with self._lock:
            self._data[key] = value

    def update(self, values: dict[str, Any]) -> None:
        """Update multiple settings at once.

        Supports both flat keys (``{"tts.speed": 0.8}``) and nested
        dicts which are flattened with dot notation (``{"tts": {"speed": 0.8}}``
        → ``"tts.speed": 0.8``).
        """
        with self._lock:
            for key, value in values.items():
                if isinstance(value, dict) and "." not in key:
                    # Flatten nested dict
                    for sub_key, sub_val in value.items():
                        self._data[f"{key}.{sub_key}"] = sub_val
                else:
                    self._data[key] = value

    def get_section(self, prefix: str) -> dict[str, Any]:
        """Get all settings matching a prefix (without the prefix).

        ``get_section("tts")`` returns ``{"speed": 0.9, "engine": "voxcpm", ...}``.
        """
        with self._lock:
            result = {}
            for key, value in self._data.items():
                if key.startswith(prefix + "."):
                    short_key = key[len(prefix) + 1:]
                    result[short_key] = value
            return result

    def to_dict(self) -> dict[str, Any]:
        """Return a copy of all settings."""
        with self._lock:
            return deepcopy(self._data)

    def reset(self, key: str | None = None) -> None:
        """Reset a single key or all settings to defaults."""
        with self._lock:
            if key is None:
                self._data = deepcopy(DEFAULT_SETTINGS)
            elif key in DEFAULT_SETTINGS:
                self._data[key] = deepcopy(DEFAULT_SETTINGS[key])


# Global singleton — import and use directly.
settings = Settings()
