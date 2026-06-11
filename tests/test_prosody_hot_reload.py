"""Unit test for the MMS_FA hot-reload hook in prosody.py.

Simulates the "user dropped the model into ~/.cache/torch/hub/checkpoints/
model.pt *after* the web server was already started" scenario and
verifies that ``_load_aligner()`` re-checks the disk and lifts the
auto-set skip flag — without ever overriding a flag the user set
explicitly.

Run with either venv:
    .venv/bin/python tests/test_prosody_hot_reload.py
    .venv-voxcpm/bin/python tests/test_prosody_hot_reload.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Minimum size used by both web/app.py and prosody.py — keep in sync.
_FAKE_MODEL_BYTES = 200_000_000  # 200 MB > 100 MB floor


def _make_fake_model(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\0" * _FAKE_MODEL_BYTES)


def _reset_aligner_state() -> None:
    """Drop the cached aligner so each test starts from a clean state."""
    from russian_tts_studio.utils import prosody
    prosody._aligner = None
    prosody._tokenizer = None
    prosody._model = None


def test_file_appears_after_import_clears_flag() -> None:
    """Simulate: import-time auto-detect set the flag (file was missing).
    Then the user drops a 200 MB file in place. The next
    ``_load_aligner()`` call must (a) lift the auto-set flag, (b) reset
    the cached singleton, and (c) NOT call the real model (we just
    check the env / state — the model load is a separate path)."""
    from russian_tts_studio.utils import prosody

    # 1. Simulate import-time auto-detect.
    os.environ["MMS_FA_SKIP_DOWNLOAD"] = "1"
    os.environ["MMS_FA_SKIP_DOWNLOAD_AUTO"] = "1"
    _reset_aligner_state()

    # 2. Simulate "user dropped in the model" by writing a 200 MB
    #    file at the path the helper will look at.
    model_path = prosody._aligner_model_path()
    assert model_path is not None, "torch is not importable in this env"
    fake = Path(model_path)
    if not fake.exists() or fake.stat().st_size < _FAKE_MODEL_BYTES:
        _make_fake_model(fake)
        made_file = True
    else:
        made_file = False  # real model already in place from a prior session
    try:
        # 3. The hook should lift the auto-set flag.
        lifted = prosody._maybe_unset_auto_skip()
        if not made_file:
            # Real model was already there — the hook should lift.
            assert lifted, (
                "hook didn't lift the auto-set flag even though model "
                f"({fake.stat().st_size} bytes) is on disk"
            )
        assert os.environ.get("MMS_FA_SKIP_DOWNLOAD") is None, (
            f"hook should have popped MMS_FA_SKIP_DOWNLOAD, "
            f"still: {os.environ.get('MMS_FA_SKIP_DOWNLOAD')!r}"
        )
        assert os.environ.get("MMS_FA_SKIP_DOWNLOAD_AUTO") is None, (
            "hook should have popped the auto marker"
        )
        print("  ✓ _maybe_unset_auto_skip() lifted the auto-set flag")
        print("  ✓ MMS_FA_SKIP_DOWNLOAD popped from env")
        print("  ✓ MMS_FA_SKIP_DOWNLOAD_AUTO popped from env")
    finally:
        if made_file:
            fake.unlink(missing_ok=True)

    print("  ✓ Hot-reload: file-appears-after-import scenario works")


def test_user_set_flag_is_never_touched() -> None:
    """If the user set MMS_FA_SKIP_DOWNLOAD=1 themselves (no AUTO
    marker), the hook must NOT lift it — even if the file is on
    disk. User intent wins."""
    from russian_tts_studio.utils import prosody

    # User-set flag: no AUTO marker
    os.environ["MMS_FA_SKIP_DOWNLOAD"] = "1"
    os.environ.pop("MMS_FA_SKIP_DOWNLOAD_AUTO", None)
    _reset_aligner_state()

    # Make sure the file is there (real one or fake)
    model_path = prosody._aligner_model_path()
    assert model_path is not None
    fake = Path(model_path)
    made_file = False
    if not fake.exists() or fake.stat().st_size < _FAKE_MODEL_BYTES:
        _make_fake_model(fake)
        made_file = True
    try:
        lifted = prosody._maybe_unset_auto_skip()
        assert lifted is False, (
            "hook lifted a USER-set flag — must never do that"
        )
        assert os.environ.get("MMS_FA_SKIP_DOWNLOAD") == "1", (
            f"user-set flag was removed: {os.environ.get('MMS_FA_SKIP_DOWNLOAD')!r}"
        )
        print("  ✓ user-set flag preserved (no auto marker → no lift)")
    finally:
        if made_file:
            fake.unlink(missing_ok=True)

    print("  ✓ Hot-reload: user-set flag honoured")


def test_file_still_missing_leaves_flag_in_place() -> None:
    """File still missing → hook is a no-op."""
    from russian_tts_studio.utils import prosody

    os.environ["MMS_FA_SKIP_DOWNLOAD"] = "1"
    os.environ["MMS_FA_SKIP_DOWNLOAD_AUTO"] = "1"
    _reset_aligner_state()

    # Point the lookup at a *non-existent* path by stubbing the helper.
    orig = prosody._aligner_model_path
    prosody._aligner_model_path = lambda: "/nonexistent/never/created/model.pt"  # type: ignore[assignment]
    try:
        lifted = prosody._maybe_unset_auto_skip()
        assert lifted is False
        assert os.environ.get("MMS_FA_SKIP_DOWNLOAD") == "1"
        assert os.environ.get("MMS_FA_SKIP_DOWNLOAD_AUTO") == "1"
        print("  ✓ no-op when file is missing")
    finally:
        prosody._aligner_model_path = orig  # type: ignore[assignment]
        os.environ.pop("MMS_FA_SKIP_DOWNLOAD", None)
        os.environ.pop("MMS_FA_SKIP_DOWNLOAD_AUTO", None)


def test_file_too_small_leaves_flag_in_place() -> None:
    """File present but < 100 MB → probably a partial download, hook
    is a no-op. To test this we have to *temporarily* replace the
    real model (if present) with a tiny stub, then restore it."""
    import shutil
    import tempfile

    from russian_tts_studio.utils import prosody

    os.environ["MMS_FA_SKIP_DOWNLOAD"] = "1"
    os.environ["MMS_FA_SKIP_DOWNLOAD_AUTO"] = "1"
    _reset_aligner_state()

    # Back up the real model (if any) and put a tiny stub in its place.
    model_path = Path(prosody._aligner_model_path())
    backup_path = None
    backup_tmp = tempfile.NamedTemporaryFile(delete=False)
    backup_tmp.close()
    backup_tmp_path = Path(backup_tmp.name)
    if model_path.exists():
        shutil.move(str(model_path), str(backup_tmp_path))
        backup_path = backup_tmp_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"\0" * 1024)  # 1 KB < 100 MB floor
    try:
        lifted = prosody._maybe_unset_auto_skip()
        assert lifted is False, (
            f"hook lifted flag for a 1 KB file: {os.environ.get('MMS_FA_SKIP_DOWNLOAD')!r}"
        )
        assert os.environ.get("MMS_FA_SKIP_DOWNLOAD") == "1"
        print("  ✓ no-op when file is below size floor")
    finally:
        model_path.unlink(missing_ok=True)
        if backup_path is not None:
            shutil.move(str(backup_path), str(model_path))
        backup_tmp_path.unlink(missing_ok=True)
        os.environ.pop("MMS_FA_SKIP_DOWNLOAD", None)
        os.environ.pop("MMS_FA_SKIP_DOWNLOAD_AUTO", None)


if __name__ == "__main__":
    print("=" * 60)
    print("MMS_FA HOT-RELOAD TEST (env + helper logic)")
    print("=" * 60)

    print("\n[1/4] test_file_appears_after_import_clears_flag")
    test_file_appears_after_import_clears_flag()

    print("\n[2/4] test_user_set_flag_is_never_touched")
    test_user_set_flag_is_never_touched()

    print("\n[3/4] test_file_still_missing_leaves_flag_in_place")
    test_file_still_missing_leaves_flag_in_place()

    print("\n[4/4] test_file_too_small_leaves_flag_in_place")
    test_file_too_small_leaves_flag_in_place()

    print("\n" + "=" * 60)
    print("✓ All hot-reload tests PASSED")
    print("=" * 60)
