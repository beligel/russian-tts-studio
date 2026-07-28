"""Voice profiles — text-described voices instead of audio references.

Inspired by Higgs Audio's ``voice_prompts/profile.yaml``
(``profile:male_en_british`` syntax in ``examples/generation.py``). A
voice profile is a natural-language description of a voice ("male,
British accent, calm tone") that a capable engine renders without a
reference audio clip. Higgs Audio v2 supports this natively; VoxCPM2
does not (it always needs a reference WAV).

Profiles live in a single YAML file (default
``output/reference/profiles.yaml``) with the structure::

    profiles:
      male_en: "Male, American accent, modern speaking rate, moderate-pitch, friendly tone."
      female_ru_calm: "Женский голос, спокойная интонация, умеренный темп, ясная дикция."
      male_en_british: "He speaks with a clear British accent and a conversational, inquisitive tone."

The file is user-editable — the UI can add/remove profiles at runtime.
On load, the file is parsed once and cached; callers can force a
reload via :func:`reload_profiles`.

Engine integration:
- **Higgs Audio**: ``profile:<name>`` is resolved to the description
  text and passed in the system message's speaker description slot
  (see ``higgs_synth._build_context`` — the ``profile:`` prefix is
  detected there).
- **VoxCPM2 / Silero**: voice profiles are not supported (these
  engines need a reference audio file). The pipeline falls back to
  Silero's built-in speakers when a profile is requested but the
  engine can't honour it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default location for the profiles YAML. Kept next to the reference
# audio files (``output/reference/``) so users find it naturally.
DEFAULT_PROFILES_PATH = Path("output/reference/profiles.yaml")

# Built-in starter profiles. Mirrors the upstream Higgs Audio
# ``voice_prompts/profile.yaml`` plus Russian-language additions
# (RTTS is a Russian TTS studio). Users can override by editing the
# YAML file — these are only written on first run.
_STARTER_PROFILES = """\
# Voice profiles — describe a voice in natural language instead of
# supplying a reference audio clip. Engines that support text-described
# voices (Higgs Audio v2) render these directly; engines that don't
# (VoxCPM2) fall back to their default voice.
#
# Edit this file freely — it's re-read on each synthesis request.
# Add a profile: add a key under `profiles:` with a text description.
profiles:
  # English voices (from upstream Higgs Audio examples)
  male_en: "Male, American accent, modern speaking rate, moderate-pitch, friendly tone, and very clear audio."
  female_en_story: "She speaks with a calm, gentle, and informative tone at a measured pace, with excellent articulation and very clear audio. She naturally brings storytelling to life with an articulate, genuine, and personable vocal style."
  male_en_british: "He speaks with a clear British accent and a conversational, inquisitive tone. His delivery is articulate and at a moderate pace, and very clear audio."
  female_en_british: "A female voice with a clear British accent speaking at a modern rate with a moderate-pitch in an expressive and friendly tone and very clear audio."
  # Russian voices (RTTS additions)
  male_ru_calm: "Мужской голос, спокойная и уверенная интонация, умеренный темп, чёткая дикция, мягкий тембр."
  female_ru_warm: "Женский голос, тёплый и дружелюбный тембр, выразительная интонация, умеренный темп, очень чёткая дикция."
  male_ru_narrator: "Мужской голос рассказчика, размеренная подача, чуть пониженный тон, спокойная интонация, подходит для аудиокниг."
  female_ru_narrator: "Женский голос рассказчицы, ровная подача, средний темп, ясная дикция, подходит для длительного повествования."
"""


@dataclass
class VoiceProfile:
    """One text-described voice profile.

    ``name`` is the profile key (e.g. ``male_en_british``). ``description``
    is the natural-language voice description passed to capable engines.
    """

    name: str
    description: str


@dataclass
class ProfileRegistry:
    """In-memory cache of loaded voice profiles."""

    profiles: dict[str, VoiceProfile] = field(default_factory=dict)
    source_path: Optional[Path] = None
    source_mtime: float = 0.0

    def get(self, name: str) -> Optional[VoiceProfile]:
        return self.profiles.get(name)

    def list_names(self) -> list[str]:
        return sorted(self.profiles.keys())

    def is_stale(self) -> bool:
        """True if the source file's mtime changed since last load."""
        if self.source_path is None or not self.source_path.exists():
            return False
        try:
            return self.source_path.stat().st_mtime > self.source_mtime
        except OSError:
            return False


# Module-level singleton cache. Reloaded on demand (when the file
# changes on disk, or via explicit ``reload_profiles()``).
_registry = ProfileRegistry()


def _ensure_starter_file(path: Path) -> None:
    """Write the starter profiles YAML if it doesn't exist yet.

    Idempotent — never overwrites user edits. Only writes on first run
    (or if the user deletes the file).
    """
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_STARTER_PROFILES, encoding="utf-8")
    logger.info("Wrote starter voice profiles to %s", path)


def load_profiles(path: Optional[str | Path] = None, *, force: bool = False) -> ProfileRegistry:
    """Load (or return cached) voice profiles from the YAML file.

    The cache is invalidated automatically when the file's mtime
    changes. Pass ``force=True`` to bypass the cache check and reload
    unconditionally.

    Returns the :class:`ProfileRegistry` singleton. Callers should
    use :func:`get_profile` or :func:`list_profiles` rather than
    touching the registry directly.
    """
    global _registry
    profiles_path = Path(path) if path else DEFAULT_PROFILES_PATH

    # Cache hit: file unchanged since last load.
    if not force and _registry.source_path == profiles_path and not _registry.is_stale():
        return _registry

    # Ensure the file exists (write starter on first run).
    _ensure_starter_file(profiles_path)
    if not profiles_path.exists():
        # Starter write failed (read-only FS?) → empty registry.
        _registry = ProfileRegistry(source_path=profiles_path)
        return _registry

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import 'yaml' (PyYAML). Install with:\n"
            "  pip install pyyaml\n"
            f"Underlying error: {exc}"
        ) from exc

    try:
        data = yaml.safe_load(profiles_path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        logger.error("Failed to parse %s: %s — using empty profiles", profiles_path, e)
        _registry = ProfileRegistry(source_path=profiles_path)
        return _registry

    raw_profiles = data.get("profiles") if isinstance(data, dict) else None
    profiles: dict[str, VoiceProfile] = {}
    if isinstance(raw_profiles, dict):
        for name, desc in raw_profiles.items():
            if isinstance(name, str) and isinstance(desc, str) and desc.strip():
                profiles[name] = VoiceProfile(name=name, description=desc.strip())

    _registry = ProfileRegistry(
        profiles=profiles,
        source_path=profiles_path,
        source_mtime=profiles_path.stat().st_mtime,
    )
    logger.info("Loaded %d voice profiles from %s", len(profiles), profiles_path)
    return _registry


def reload_profiles(path: Optional[str | Path] = None) -> ProfileRegistry:
    """Force-reload the profiles YAML. Convenience wrapper."""
    return load_profiles(path, force=True)


def get_profile(name: str, path: Optional[str | Path] = None) -> Optional[VoiceProfile]:
    """Look up a single profile by name. Returns ``None`` if not found."""
    reg = load_profiles(path)
    return reg.get(name)


def list_profiles(path: Optional[str | Path] = None) -> list[VoiceProfile]:
    """Return all loaded profiles, sorted by name."""
    reg = load_profiles(path)
    return [reg.profiles[n] for n in reg.list_names()]


def is_profile_reference(ref: str) -> bool:
    """True if ``ref`` is a ``profile:<name>`` voice profile reference
    (rather than a file path).

    Mirrors the upstream Higgs Audio ``profile:male_en_british`` syntax
    from ``examples/generation.py``.
    """
    return isinstance(ref, str) and ref.startswith("profile:")


def resolve_reference(ref: str, profiles_path: Optional[str | Path] = None) -> Optional[str]:
    """Resolve a ``profile:<name>`` reference to its description text.

    Returns the description string if the profile exists, ``None``
    otherwise. For non-profile references (file paths), returns ``None``
    — the caller should treat the input as a file path.
    """
    if not is_profile_reference(ref):
        return None
    name = ref[len("profile:"):].strip()
    profile = get_profile(name, profiles_path)
    return profile.description if profile else None


def add_profile(name: str, description: str, path: Optional[str | Path] = None) -> None:
    """Add or update a profile in the YAML file (persists to disk).

    Used by the UI's "save profile" flow.
    """
    profiles_path = Path(path) if path else DEFAULT_PROFILES_PATH
    _ensure_starter_file(profiles_path)

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML required for add_profile") from exc

    # Load existing file content (preserve user edits + comments).
    try:
        data = yaml.safe_load(profiles_path.read_text(encoding="utf-8")) or {}
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    raw = data.get("profiles")
    if not isinstance(raw, dict):
        raw = {}
    raw[name] = description
    data["profiles"] = raw
    # Write back (note: this drops comments — PyYAML limitation. The
    # starter file keeps comments only on first creation; subsequent
    # add_profile calls rewrite without them. Acceptable tradeoff for
    # a UI-driven flow; users editing by hand should edit the file
    # directly rather than via the API.)
    profiles_path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    reload_profiles(profiles_path)


def delete_profile(name: str, path: Optional[str | Path] = None) -> bool:
    """Remove a profile from the YAML file. Returns True if removed."""
    profiles_path = Path(path) if path else DEFAULT_PROFILES_PATH
    if not profiles_path.exists():
        return False
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML required for delete_profile") from exc
    try:
        data = yaml.safe_load(profiles_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return False
    raw = data.get("profiles") if isinstance(data, dict) else None
    if not isinstance(raw, dict) or name not in raw:
        return False
    del raw[name]
    data["profiles"] = raw
    profiles_path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    reload_profiles(profiles_path)
    return True