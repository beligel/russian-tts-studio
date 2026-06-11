"""Unit test for web.desktop graceful shutdown.

The "soft zombie" failure mode: webview.start() crashes, leaves the
process alive with FDs → /dev/null, port not bound, no logs. We
verify that the desktop wrapper distinguishes "clean shutdown" from
"crash" and force-exits via os._exit in the crash path so the
zombie case can't recur.

Run with either venv:
    .venv/bin/python tests/test_desktop_shutdown.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _make_fake_webview_module() -> mock.MagicMock:
    """Build a fake ``webview`` module with ``start`` and
    ``create_window``. Used to monkey-patch ``sys.modules`` so the
    ``import webview`` inside ``desktop.main()`` picks it up."""
    fake = mock.MagicMock(name="fake_webview")
    fake.create_window.return_value = mock.MagicMock(name="fake_window")
    return fake


# A mock for ``os._exit`` that raises SystemExit, simulating the
# real behaviour (terminate the process). Use this in any test that
# expects a call to ``os._exit`` to *stop* execution — a plain
# MagicMock would let the test fall through to the next exit call.
def _fake_exit_raising(code: int) -> None:
    raise SystemExit(code)


def test_clean_shutdown_returns_normally() -> None:
    """When webview.start() returns (user closed the window), the
    main() should return 0 — NOT call os._exit. This preserves
    normal cleanup for tests and any future 'run desktop in a
    subprocess' workflows."""
    from web import desktop

    fake_webview = _make_fake_webview_module()
    fake_webview.start.return_value = None
    with mock.patch.dict(sys.modules, {"webview": fake_webview}), \
         mock.patch.object(desktop.os, "_exit") as mock_exit:
        rc = desktop.main()
        assert rc == 0, f"expected rc=0, got {rc}"
        mock_exit.assert_not_called()
        print("  ✓ clean shutdown → return 0, no os._exit")


def test_webview_crash_force_exits() -> None:
    """When webview.start() raises, main() must call os._exit
    (not sys.exit) to prevent the soft-zombie case."""
    from web import desktop

    fake_webview = _make_fake_webview_module()
    fake_webview.start.side_effect = RuntimeError("GTK init failed")
    with mock.patch.dict(sys.modules, {"webview": fake_webview}), \
         mock.patch.object(desktop.os, "_exit", side_effect=_fake_exit_raising) as mock_exit:
        try:
            desktop.main()
        except SystemExit as e:
            assert e.code == 1, f"SystemExit code should be 1, got {e.code!r}"
        else:
            raise AssertionError("os._exit(1) was not called")
        # If we got here via SystemExit, mock recorded the call.
        mock_exit.assert_called_with(1)
        print("  ✓ webview crash → os._exit(1) (no soft zombie)")


def test_keyboard_interrupt_exits_with_posix_sigint_code() -> None:
    """Ctrl+C in a terminal must use the POSIX SIGINT exit code
    (128 + 2 = 130) so wrapper scripts can distinguish it from
    "process crashed" (1) and "user closed the window" (0)."""
    from web import desktop

    fake_webview = _make_fake_webview_module()
    fake_webview.start.side_effect = KeyboardInterrupt()
    with mock.patch.dict(sys.modules, {"webview": fake_webview}), \
         mock.patch.object(desktop.os, "_exit", side_effect=_fake_exit_raising) as mock_exit:
        try:
            desktop.main()
        except SystemExit as e:
            assert e.code == 130, f"SystemExit code should be 130, got {e.code!r}"
        else:
            raise AssertionError("os._exit(130) was not called")
        mock_exit.assert_called_with(130)
        print("  ✓ KeyboardInterrupt → os._exit(130) (POSIX SIGINT code)")


if __name__ == "__main__":
    print("=" * 60)
    print("DESKTOP SHUTDOWN TEST (clean vs crash vs Ctrl+C)")
    print("=" * 60)

    print("\n[1/3] test_clean_shutdown_returns_normally")
    test_clean_shutdown_returns_normally()

    print("\n[2/3] test_webview_crash_force_exits")
    test_webview_crash_force_exits()

    print("\n[3/3] test_keyboard_interrupt_exits_with_posix_sigint_code")
    test_keyboard_interrupt_exits_with_posix_sigint_code()

    print("\n" + "=" * 60)
    print("✓ All desktop shutdown tests PASSED")
    print("=" * 60)
