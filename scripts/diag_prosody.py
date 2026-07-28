"""Diagnostic: synthesise the in-UI demo text twice (off / conservative),
then print an RMS profile so we can see *where* the wav actually has
silence vs energy, regardless of what ``insert_pauses`` was supposed
to do.

Output files (NOT overwritten on rerun unless you delete them):
    /tmp/diag_off.wav
    /tmp/diag_conservative.wav
    /tmp/diag_off.png
    /tmp/diag_conservative.png
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = Path("/home/che/projects/russian-tts-studio")
sys.path.insert(0, str(PROJECT_ROOT))

DEMO_TEXT = (
    "Привет! Это тестовая фраза на русском языке. "
    "Russian TTS Studio3 умеет клонировать голос по короткому референсу."
)
REF = PROJECT_ROOT / "output" / "reference" / "ru_voice.wav"
OUT = Path("/tmp")


def rms_envelope(wav: np.ndarray, win_ms: int = 20, sr: int = 48000) -> np.ndarray:
    """Return per-window RMS values. win_ms controls time resolution."""
    win = int(sr * win_ms / 1000)
    if win <= 0:
        return np.array([0.0])
    n = len(wav) // win
    if n == 0:
        return np.array([0.0])
    trimmed = wav[: n * win].reshape(n, win)
    return np.sqrt(np.mean(trimmed ** 2, axis=1))


def find_pauses(env: np.ndarray, thr: float = 0.005, min_ms: int = 100, win_ms: int = 20) -> list[tuple[float, float]]:
    """Return list of (start_sec, end_sec) for windows where RMS<thr."""
    sil = env < thr
    pauses = []
    i = 0
    while i < len(sil):
        if sil[i]:
            j = i
            while j < len(sil) and sil[j]:
                j += 1
            dur_ms = (j - i) * win_ms
            if dur_ms >= min_ms:
                pauses.append((i * win_ms / 1000.0, j * win_ms / 1000.0))
            i = j
        else:
            i += 1
    return pauses


def analyse(wav_path: Path, label: str) -> dict:
    wav, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    dur = len(wav) / sr
    env = rms_envelope(wav, win_ms=20, sr=sr)
    pauses = find_pauses(env, thr=0.005, min_ms=100, win_ms=20)
    print(f"\n=== {label}  ({wav_path}) ===")
    print(f"  duration: {dur:.2f}s   sample_rate: {sr}   samples: {len(wav)}")
    print(f"  pauses (RMS<0.005, >=100ms): {len(pauses)}")
    for i, (s, e) in enumerate(pauses):
        print(f"    [{i+1}]  {s:5.2f}s → {e:5.2f}s   len={e-s:.2f}s")
    return {"wav": wav, "sr": sr, "env": env, "pauses": pauses, "dur": dur}


def main():
    from russian_tts_studio.models.voxcpm_synth import VoxCPMSynthesizer
    from russian_tts_studio.models.base_synth import SynthesisRequest
    from russian_tts_studio.utils.prosody import DEFAULT_PAUSE_MS

    print(f"Reference: {REF.name}  ({'OK' if REF.exists() else 'MISSING'})")
    print(f"Text: {DEMO_TEXT!r}")
    print(f"DEFAULT_PAUSE_MS: {DEFAULT_PAUSE_MS}")

    synth = VoxCPMSynthesizer()
    print("Loading VoxCPM2 (this takes ~30-60s)...")
    t0 = time.time()
    synth.load()
    print(f"  loaded in {time.time() - t0:.1f}s")

    # --- Pass 1: prosody OFF (all pause_ms_* = 0) ---
    out_off = OUT / "diag_off.wav"
    if out_off.exists():
        out_off.unlink()
    req_off = SynthesisRequest(
        text=DEMO_TEXT,
        output_path=out_off,
        reference_audio=str(REF),
        metadata={f"pause_ms_{k}": 0 for k in DEFAULT_PAUSE_MS},
    )
    print("\n[A] Synthesising with all pauses=0 (no prosody)...")
    t0 = time.time()
    r_off = synth.synthesize(req_off)
    print(f"  synth in {time.time() - t0:.1f}s, dur={r_off.duration_sec:.2f}s, "
          f"success={r_off.success}, error={r_off.error}")
    off = analyse(out_off, "PASS 1: prosody OFF (pause_ms_*=0)")

    # --- Pass 2: prosody ON with current defaults ---
    out_c = OUT / "diag_conservative.wav"
    if out_c.exists():
        out_c.unlink()
    req_c = SynthesisRequest(
        text=DEMO_TEXT,
        output_path=out_c,
        reference_audio=str(REF),
        metadata=dict(DEFAULT_PAUSE_MS),
    )
    print("\n[B] Synthesising with current conservative defaults...")
    t0 = time.time()
    r_c = synth.synthesize(req_c)
    print(f"  synth in {time.time() - t0:.1f}s, dur={r_c.duration_sec:.2f}s, "
          f"success={r_c.success}, error={r_c.error}")
    cons = analyse(out_c, "PASS 2: prosody ON (conservative defaults)")

    # --- Compare delta ---
    delta = r_c.duration_sec - r_off.duration_sec
    print(f"\n=== Duration delta (conservative − off): {delta:+.2f}s ===")
    expected = (sum(DEFAULT_PAUSE_MS.values())) / 1000.0
    print(f"=== Sum of all DEFAULT_PAUSE_MS: {expected:.2f}s ===")
    print(f"=== Number of punct marks in text: 4 ('!', '.', '.', '.') ===")
    print(f"=== Expected delta from punct: "
          f"(700+600+600+600)/1000 = {(700+600+600+600)/1000:.2f}s ===")

    # --- Plot RMS envelopes side-by-side ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=False)
        for ax, data, title in [
            (axes[0], off,   "Prosody OFF (pause_ms_*=0)"),
            (axes[1], cons,  "Prosody ON (conservative defaults)"),
        ]:
            t = np.arange(len(data["env"])) * 0.020
            ax.fill_between(t, data["env"], color="steelblue", alpha=0.7)
            for s, e in data["pauses"]:
                ax.axvspan(s, e, color="red", alpha=0.25)
            ax.set_ylim(0, max(0.1, data["env"].max() * 1.1))
            ax.set_ylabel("RMS")
            ax.set_title(f"{title}  —  {data['dur']:.2f}s, "
                         f"{len(data['pauses'])} pauses ≥100ms")
            ax.grid(True, alpha=0.3)
        axes[1].set_xlabel("seconds")
        fig.suptitle("RMS envelope (20ms windows). Red bands = auto-detected pauses.")
        fig.tight_layout()
        plot_path = OUT / "diag_prosody.png"
        fig.savefig(plot_path, dpi=110)
        print(f"\nPlot saved to: {plot_path}")
    except Exception as e:
        print(f"\n(matplotlib failed: {e})")


if __name__ == "__main__":
    main()
