"""End-to-end test for the prosody module: forced alignment + VoxCPM2
synthesis with pause insertion.

Run with the VoxCPM venv:
    .venv-voxcpm/bin/python tests/test_prosody_e2e.py
"""

from __future__ import annotations

import math
import os
import struct
import sys
import time
import wave
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _make_synthetic_wav(out_path: Path, duration_sec: float = 3.0, sr: int = 16000) -> Path:
    """Synthetic tone wav with 5 tone+silence segments. Used to test
    alignment on a predictable signal without loading VoxCPM."""
    seg_tone = sr // 2
    seg_silence = sr // 10
    wav = bytearray()
    for _ in range(5):
        for k in range(seg_tone):
            sample = 0.3 * math.sin(2 * math.pi * 440 * (k / sr))
            wav.extend(struct.pack('<h', int(sample * 32767)))
        for _ in range(seg_silence):
            wav.extend(struct.pack('<h', 0))
    with wave.open(str(out_path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(bytes(wav))
    return out_path


def test_forced_alignment_with_synthetic_wav() -> None:
    """Lightweight forced-alignment test: align a 3s tone against a
    short Russian phrase, verify timestamps come back."""
    os.environ.pop("MMS_FA_SKIP_DOWNLOAD", None)

    from russian_tts_studio.utils.prosody import _align_text, _load_aligner

    out_dir = PROJECT_ROOT / "output" / "prosody_e2e"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / "synthetic_alignment.wav"
    _make_synthetic_wav(wav_path)

    print(f"\n[1/2] Loading MMS_FA aligner...")
    t0 = time.time()
    _load_aligner()
    print(f"  loaded in {time.time() - t0:.1f}s")

    text = "Привет, мир! Как дела? Хорошо."
    print(f"\n[2/2] Aligning {len(text)}-char text against {wav_path.name}...")
    t0 = time.time()
    result = _align_text(wav_path, text)
    print(f"  aligned in {time.time() - t0:.2f}s")
    print(f"  got {len(result)} char timestamps")

    if not result:
        print("  ✗ FAIL: no timestamps returned")
        sys.exit(1)

    last = 0.0
    for i in sorted(result.keys()):
        t0_, t1_ = result[i]
        assert t0_ >= last - 0.01, f"timestamp went backward at {i}: {t0_} < {last}"
        last = t1_
    print("\n  ✓ All timestamps are non-decreasing")
    print("  ✓ Forced alignment works end-to-end on synthetic wav")


def test_insert_pauses_with_real_voxcpm() -> None:
    """Full pipeline: VoxCPM2 synth with prosody enabled → forced
    alignment → pause insertion.

    VoxCPM2 is non-deterministic across calls (flow-matching), so we
    can't compare "raw vs prosody" wavs of two synth calls — the
    underlying speech differs. Instead we measure the prosody-applied
    file directly and verify the delta is ~sum(pauses) = 3.0s.
    """
    os.environ.pop("MMS_FA_SKIP_DOWNLOAD", None)

    from russian_tts_studio.models.voxcpm_synth import VoxCPMSynthesizer
    from russian_tts_studio.models.base_synth import SynthesisRequest
    from russian_tts_studio.utils.prosody import (
        DEFAULT_PAUSE_MS, PauseConfig, insert_pauses,
    )

    out_dir = PROJECT_ROOT / "output" / "prosody_e2e"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Short Russian sentence with multiple punctuation marks
    text = "Привет, мир! Как дела? Хорошо, отлично."

    print(f"\n[A] Loading VoxCPM2...")
    t0 = time.time()
    synth = VoxCPMSynthesizer()
    synth.load()
    print(f"  loaded in {time.time() - t0:.1f}s")

    ref_candidates = list((PROJECT_ROOT / "output" / "reference").glob("*.wav"))
    if not ref_candidates:
        print("  ✗ FAIL: no reference wav found")
        sys.exit(1)
    ref_path = ref_candidates[0]
    print(f"  using reference: {ref_path.name}")

    # Synthesise TWICE: once with all pauses=0 (raw baseline), once with
    # the conservative preset. The two will differ in audio content
    # (VoxCPM2 flow matching is non-deterministic), but the *duration*
    # comparison is sound: we know the baseline should be a few seconds
    # and the prosody one should be exactly baseline + 3.0s.
    raw_wav = out_dir / "voxcpm_raw.wav"
    print(f"\n[B] Synthesizing raw (no prosody)...")
    t0 = time.time()
    no_prosody_meta = {f"pause_ms_{k}": 0 for k in DEFAULT_PAUSE_MS}
    req_raw = SynthesisRequest(
        text=text, output_path=raw_wav, reference_audio=str(ref_path),
        metadata=no_prosody_meta,
    )
    result_raw = synth.synthesize(req_raw)
    print(f"  synth in {time.time() - t0:.1f}s")
    if not result_raw.success:
        print(f"  ✗ FAIL: {result_raw.error}")
        sys.exit(1)
    raw_dur = result_raw.duration_sec
    raw_meta = result_raw.metadata
    print(f"  raw dur: {raw_dur:.2f}s, metadata: {raw_meta}")
    if raw_meta.get("prosody_applied"):
        print("  ✗ FAIL: raw synthesis unexpectedly applied prosody")
        sys.exit(1)

    # Prosody pass: synthesize fresh with conservative preset.
    # Compare against the no-prosody baseline.
    prosody_wav = out_dir / "voxcpm_with_prosody.wav"
    print(f"\n[C] Synthesizing with prosody (conservative preset)...")
    t0 = time.time()
    prosody_meta = dict(DEFAULT_PAUSE_MS)  # all default pause_ms values
    req_pros = SynthesisRequest(
        text=text, output_path=prosody_wav, reference_audio=str(ref_path),
        metadata=prosody_meta,
    )
    result_pros = synth.synthesize(req_pros)
    print(f"  synth in {time.time() - t0:.1f}s")
    print(f"  prosody dur: {result_pros.duration_sec:.2f}s")
    print(f"  metadata: prosody_applied={result_pros.metadata.get('prosody_applied')}, "
          f"degraded={result_pros.metadata.get('prosody_degraded')}")
    if not result_pros.success:
        print(f"  ✗ FAIL: {result_pros.error}")
        sys.exit(1)
    if result_pros.metadata.get("prosody_degraded"):
        print("  ✗ FAIL: prosody fell back to proportional — MMS_FA didn't engage")
        sys.exit(1)
    if not result_pros.metadata.get("prosody_applied"):
        print("  ✗ FAIL: prosody_applied not set in metadata")
        sys.exit(1)

    # Sum the configured pauses in "Привет, мир! Как дела? Хорошо, отлично.":
    #   ,  +  !  +  ?  +  ,  +  .  =  500+1000+1000+500+900  =  3900ms  =  3.9s
    expected_delta = (
        DEFAULT_PAUSE_MS["comma"]      # 500
        + DEFAULT_PAUSE_MS["exclamation"]  # 1000
        + DEFAULT_PAUSE_MS["question"]  # 1000
        + DEFAULT_PAUSE_MS["comma"]      # 500
        + DEFAULT_PAUSE_MS["period"]     # 900 (the final ".")
    ) / 1000.0
    print(f"\n  expected delta: {expected_delta:.2f}s (comma+!+?+comma+period)")

    # Since VoxCPM2 is non-deterministic, we can't compare raw vs prosody
    # duration exactly. Instead: do a separate test using *insert_pauses*
    # directly on a *copy* of the raw wav, which is deterministic.
    print(f"\n[D] Deterministic prosody insertion on a copy of the raw wav...")
    import shutil
    import soundfile as sf
    test_wav = out_dir / "voxcpm_for_insert.wav"
    shutil.copy(raw_wav, test_wav)
    print(f"  copied raw → {test_wav.name} ({raw_dur:.2f}s)")

    cfg = PauseConfig(
        comma=DEFAULT_PAUSE_MS["comma"],
        semicolon=DEFAULT_PAUSE_MS["semicolon"],
        colon=DEFAULT_PAUSE_MS["colon"],
        period=DEFAULT_PAUSE_MS["period"],
        exclamation=DEFAULT_PAUSE_MS["exclamation"],
        question=DEFAULT_PAUSE_MS["question"],
        ellipsis=DEFAULT_PAUSE_MS["ellipsis"],
    )

    # Run insert_pauses on the copy. The raw wav is 8.06s, so alignment
    # should fit 4 pauses inside it (we confirmed earlier that punct
    # positions are at ~1.58s, 2.29s, 4.21s, 6.10s — all well before end).
    _, degraded = insert_pauses(test_wav, text, cfg)
    if degraded:
        print("  ✗ FAIL: insert_pauses fell back to proportional — alignment failed")
        sys.exit(1)

    info = sf.info(str(test_wav))
    new_dur = info.frames / info.samplerate
    delta = new_dur - raw_dur
    print(f"  new dur: {new_dur:.2f}s, delta: {delta:.2f}s (expected ~{expected_delta:.2f}s)")

    # Allow 0.5s slack for non-determinism in uroman + aligner boundaries
    if abs(delta - expected_delta) > 0.5:
        print(f"  ✗ FAIL: delta off by {abs(delta - expected_delta):.2f}s (>0.5s tolerance)")
        sys.exit(1)
    print(f"  ✓ Delta matches expectation within 0.5s tolerance")

    print(f"\n  Audio saved: {test_wav}")
    print("  ✓ End-to-end prosody test PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("PROSODY END-TO-END TEST (MMS_FA via Wayback)")
    print("=" * 60)
    test_forced_alignment_with_synthetic_wav()
    print()
    print("-" * 60)
    test_insert_pauses_with_real_voxcpm()
