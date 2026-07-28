"""Audio mixing for podcast-style narration + background music.

Uses FFmpeg for all DSP: ducking (sidechain compress), fades, loudnorm,
and MP3 encoding. No Python audio libraries for the mix itself — FFmpeg
is faster and handles format conversion automatically.

Typical workflow::

    from russian_tts_studio.audio.mix import mix_podcast, MixConfig

    cfg = MixConfig(
        voice_db=0.0,
        music_db=-12.0,
        intro_sec=3.0,
        fade_in=2.0,
        fade_out=3.0,
        ducking=True,
        loudnorm=True,
    )
    out = mix_podcast(narration="narration.wav", music="bg.mp3", config=cfg)
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class MixConfig:
    """Configuration for podcast-style audio mixing.

    All durations are in seconds. All gains are in dB. A value of
    ``None`` means "use the FFmpeg default" / "don't apply".
    """

    # Voice gain (dB). 0.0 = unchanged.
    voice_db: float = 0.0
    # Music gain (dB). -12.0 is a common starting point so music sits
    # under narration without being inaudible.
    music_db: float = -12.0

    # Music-only intro before narration starts (seconds).
    intro_sec: float = 0.0
    # Music tail after narration ends (seconds). 0 = music ends with voice.
    tail_sec: float = 0.0

    # Fade-in / fade-out durations for the music track (seconds).
    fade_in: float = 2.0
    fade_out: float = 3.0

    # Enable sidechain-ducking: music volume drops while voice is active.
    ducking: bool = True
    # Ducking depth in dB when voice is active (music is attenuated by
    # this amount). Typical: 6-12 dB.
    duck_depth_db: float = 9.0
    # Ducking attack/release in seconds (how fast music reacts to voice).
    duck_attack: float = 0.3
    duck_release: float = 1.0
    # Ducking strength preset — overrides duck_depth/attack/release
    # when set to "low", "medium", or "high". Empty string = use raw params.
    ducking_strength: str = ""  # "" | "low" | "medium" | "high"

    # Loop background music when it's shorter than the narration.
    # Uses FFmpeg ``-stream_loop -1``. Essential for audiobooks with
    # short music clips.
    loop_music: bool = True

    # Normalize final mix to podcast-friendly loudness (EBU R128).
    loudnorm: bool = True
    # Target integrated loudness (LUFS). -16 is podcast standard.
    target_lufs: float = -16.0

    # Output format: "wav" or "mp3". MP3 requires libmp3lame.
    output_format: str = "mp3"
    # MP3 bitrate (only used when output_format="mp3").
    mp3_bitrate: str = "192k"

    # Sample rate for the final mix.
    sample_rate: int = 44100

    # Mute switches — set to True to silence a track entirely.
    voice_muted: bool = False
    music_muted: bool = False

    # MP3 metadata tags (embedded in output file).
    meta_title: str = ""
    meta_artist: str = ""
    meta_album: str = "Russian TTS Studio"

    # Preview: render only a segment (for UI waveform).
    # When set, the output contains only [preview_start, preview_start+preview_duration]
    # seconds of the full mix. None = render the full mix.
    preview_start: Optional[float] = None
    preview_duration: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def get_ducking_params(self) -> tuple[float, float, float]:
        """Return (threshold, ratio, attack, release) for sidechaincompress.

        When ``ducking_strength`` is set to a preset, it overrides
        the raw duck_depth_db/duck_attack/duck_release values.
        """
        presets = {
            "low": (0.035, 3.0, 0.030, 0.350),
            "medium": (0.025, 6.0, 0.020, 0.500),
            "high": (0.018, 10.0, 0.015, 0.650),
        }
        if self.ducking_strength in presets:
            return presets[self.ducking_strength]
        # Compute threshold from duck_depth_db
        ratio = 4.0
        threshold_db = -self.duck_depth_db / (1.0 - 1.0 / ratio)
        threshold = max(0.001, min(1.0, 10 ** (threshold_db / 20)))
        return (threshold, ratio, self.duck_attack, self.duck_release)


def _run_ffmpeg(args: list[str], desc: str = "ffmpeg") -> subprocess.CompletedProcess:
    """Run FFmpeg with error handling."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning"] + args
    logger.info("Running %s: %s", desc, " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"{desc} failed (rc={result.returncode}): {result.stderr}")
    return result


def _get_duration(path: str | Path) -> float:
    """Get audio duration in seconds via FFprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def mix_podcast(
    narration: str | Path,
    music: str | Path,
    config: MixConfig | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Mix narration with background music into a podcast-style output.

    Pipeline:
    1. Apply voice gain (``-af volume=XdB``).
    2. Apply music gain + fades + intro/tail padding.
    3. If ducking: use ``sidechaincompress`` to duck music under voice.
    4. Mix voice + music streams.
    5. If loudnorm: apply EBU R128 normalization.
    6. Encode to output format.

    Returns the path to the mixed output file.
    """
    narration = Path(narration)
    music = Path(music)
    cfg = config or MixConfig()

    if not narration.exists():
        raise FileNotFoundError(f"Narration not found: {narration}")
    if not music.exists():
        raise FileNotFoundError(f"Music not found: {music}")

    # Determine output path.
    if output_path is None:
        ext = "mp3" if cfg.output_format == "mp3" else "wav"
        output_path = narration.parent / f"{narration.stem}_mix.{ext}"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    voice_dur = _get_duration(narration)
    music_dur = _get_duration(music)
    total_dur = voice_dur + cfg.intro_sec + cfg.tail_sec

    # --- Build FFmpeg inputs ---
    # Music loop: when music is shorter than narration, loop it.
    music_loop_args: list[str] = []
    if cfg.loop_music and music_dur < total_dur:
        music_loop_args = ["-stream_loop", "-1"]
    inputs: list[str] = ["-i", str(narration)] + music_loop_args + ["-i", str(music)]

    # --- Build the FFmpeg filter graph ---

    filters: list[str] = []

    # Step 1: Prepare the voice track (input 0).
    voice_filters: list[str] = []
    if cfg.voice_muted:
        voice_filters.append("volume=-96dB")
    elif cfg.voice_db != 0.0:
        voice_filters.append(f"volume={cfg.voice_db}dB")

    if cfg.intro_sec > 0:
        delay_ms = int(cfg.intro_sec * 1000)
        voice_filters.append(f"adelay={delay_ms}|{delay_ms}")

    voice_label = "voice"
    if voice_filters:
        filters.append(f"[0:a]{','.join(voice_filters)}[{voice_label}]")
    else:
        filters.append(f"[0:a]acopy[{voice_label}]")

    # When ducking is enabled, voice is needed as BOTH a sidechain
    # control signal AND an audio input to amix. FFmpeg labels are
    # consumed on use, so we split the voice stream. When ducking is
    # disabled, voice only goes to amix — no split needed.
    voice_sidechain = voice_label
    if cfg.ducking:
        voice_sidechain = "voice_sc"
        voice_mix = "voice_mx"
        filters.append(f"[{voice_label}]asplit=2[{voice_sidechain}][{voice_mix}]")
    else:
        voice_mix = voice_label

    # Step 2: Prepare the music track (input 1).
    music_filters: list[str] = []

    # Gain
    if cfg.music_muted:
        music_filters.append("volume=-96dB")
    elif cfg.music_db != 0.0:
        music_filters.append(f"volume={cfg.music_db}dB")

    # Fades on music
    if cfg.fade_in > 0:
        music_filters.append(f"afade=t=in:st=0:d={cfg.fade_in}")
    if cfg.fade_out > 0:
        fade_start = max(0, total_dur - cfg.fade_out)
        music_filters.append(f"afade=t=out:st={fade_start}:d={cfg.fade_out}")

    # Trim music to total_dur
    music_filters.append(f"atrim=0:{total_dur}")
    music_filters.append("asetpts=PTS-STARTPTS")

    music_label = "music_prepared"
    if music_filters:
        filters.append(f"[1:a]{','.join(music_filters)}[{music_label}]")
    else:
        filters.append(f"[1:a]acopy[{music_label}]")

    # Step 3: Ducking — sidechaincompress on music using voice as sidechain.
    duck_label = "music_ducked"
    if cfg.ducking:
        threshold, ratio, attack, release = cfg.get_ducking_params()
        filters.append(
            f"[{music_label}][{voice_sidechain}]sidechaincompress="
            f"threshold={threshold:.6f}:ratio={ratio:.1f}:attack={attack:.3f}:"
            f"release={release:.3f}:level_sc=1[{duck_label}]"
        )
        music_for_mix = duck_label
    else:
        music_for_mix = music_label

    # Step 4: Mix voice + music.
    mix_label = "mixed"
    filters.append(
        f"[{voice_mix}][{music_for_mix}]amix=inputs=2:"
        f"duration=longest:dropout_transition=0[{mix_label}]"
    )

    # Step 5: Preview window (trim to segment)
    window_label = mix_label
    if cfg.preview_start is not None and cfg.preview_duration is not None:
        window_label = "mix_window"
        pstart = cfg.preview_start
        pdur = cfg.preview_duration
        filters.append(
            f"[{mix_label}]atrim={pstart:.3f}:{pstart + pdur:.3f},"
            f"asetpts=PTS-STARTPTS[{window_label}]"
        )

    # Step 6: Loudnorm
    final_label = "final"
    if cfg.loudnorm:
        filters.append(
            f"[{window_label}]loudnorm=I={cfg.target_lufs}:TP=-1.5:LRA=11[{final_label}]"
        )
    else:
        filters.append(f"[{window_label}]acopy[{final_label}]")

    # Build the complete command.
    filter_graph = ";\n".join(filters)

    cmd = inputs + [
        "-filter_complex", filter_graph,
        "-map", f"[{final_label}]",
        "-ar", str(cfg.sample_rate),
    ]

    if cfg.output_format == "mp3":
        cmd += ["-codec:a", "libmp3lame", "-b:a", cfg.mp3_bitrate]
        # MP3 metadata tags
        if cfg.meta_title:
            cmd += ["-metadata", f"title={cfg.meta_title}"]
        if cfg.meta_artist:
            cmd += ["-metadata", f"artist={cfg.meta_artist}"]
        if cfg.meta_album:
            cmd += ["-metadata", f"album={cfg.meta_album}"]
    else:
        cmd += ["-codec:a", "pcm_s16le"]

    cmd.append(str(output_path))

    _run_ffmpeg(cmd, desc="mix_podcast")

    out_dur = _get_duration(output_path)
    logger.info(
        "Mixed podcast: voice=%.1fs, music=%.1fs (loop=%s), total=%.1fs → %s",
        voice_dur, music_dur, cfg.loop_music and music_dur < total_dur, out_dur, output_path,
    )
    return output_path


# ---------------------------------------------------------------------------
# Music library helpers
# ---------------------------------------------------------------------------


def list_background_music(music_dir: str | Path = "music/background") -> list[dict]:
    """List available background music files.

    Returns ``[{name, path, duration_sec}, ...]`` for each audio file
    in ``music_dir``.
    """
    music_dir = Path(music_dir)
    if not music_dir.exists():
        return []

    results: list[dict] = []
    for p in sorted(music_dir.iterdir()):
        if p.suffix.lower() in (".mp3", ".wav", ".flac", ".ogg", ".m4a"):
            try:
                dur = _get_duration(p)
                results.append({
                    "name": p.name,
                    "path": str(p),
                    "duration_sec": round(dur, 2),
                })
            except Exception as e:
                logger.warning("Could not read duration for %s: %s", p, e)
                results.append({"name": p.name, "path": str(p), "duration_sec": None})
    return results


def get_default_music(music_dir: str | Path = "music/background") -> Optional[Path]:
    """Return the first music file in the library, or None."""
    items = list_background_music(music_dir)
    if items:
        return Path(items[0]["path"])
    return None