"""Unit test for web.desktop graceful shutdown.

The "soft zombie" failure mode: webview.start() crashes or the GTK
main loop wedges, leaving the process alive with FDs → /dev/null,
port not bound, no logs. We verify that the desktop wrapper
distinguishes "clean shutdown" from "crash" / "hang" and force-exits
via os._exit so the zombie case can't recur.

The watchdog plumbing is also tested in isolation: a dedicated
``_start_watchdog_thread`` function spawns a daemon thread that
os._exit(1)s the process if the heartbeat event stays cleared for
``timeout`` seconds. We mock this out for the integration tests
(so the real watchdog doesn't kill the test runner) and exercise
it directly with a 0.1s timeout for the watchdog unit test.

Run with either venv:
    .venv/bin/python tests/test_desktop_shutdown.py
"""

from __future__ import annotations

import os
import sys
import threading
import time
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


def _no_op_watchdog(*args, **kwargs) -> mock.MagicMock:
    """Replacement for ``_start_watchdog_thread`` used by the
    integration tests below. The real watchdog would wait 5s
    for a heartbeat and then ``os._exit(1)`` — fine in
    production, fatal in a test runner."""
    return mock.MagicMock(name="fake_watchdog_thread")


def test_clean_shutdown_returns_normally() -> None:
    """When webview.start() returns (user closed the window), the
    main() should return 0 — NOT call os._exit. This preserves
    normal cleanup for tests and any future 'run desktop in a
    subprocess' workflows."""
    from web import desktop

    fake_webview = _make_fake_webview_module()
    fake_webview.start.return_value = None
    with mock.patch.dict(sys.modules, {"webview": fake_webview}), \
         mock.patch.object(desktop, "_start_watchdog_thread", _no_op_watchdog), \
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
         mock.patch.object(desktop, "_start_watchdog_thread", _no_op_watchdog), \
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
         mock.patch.object(desktop, "_start_watchdog_thread", _no_op_watchdog), \
         mock.patch.object(desktop.os, "_exit", side_effect=_fake_exit_raising) as mock_exit:
        try:
            desktop.main()
        except SystemExit as e:
            assert e.code == 130, f"SystemExit code should be 130, got {e.code!r}"
        else:
            raise AssertionError("os._exit(130) was not called")
        mock_exit.assert_called_with(130)
        print("  ✓ KeyboardInterrupt → os._exit(130) (POSIX SIGINT code)")


def test_watchdog_force_exits_on_silent_gtk_loop() -> None:
    """The watchdog's raison d'être: if the GTK main loop goes
    silent (the second soft-zombie mode that the cb96bf8 fix
    didn't cover), the watchdog must os._exit(1) the process
    from its background thread, with no help from main().

    We start a real watchdog with a 0.1s timeout, never ping
    the heartbeat, and verify that the watchdog calls
    ``os._exit(1)`` within a small grace period.

    Note: the actual ``os._exit`` is called from a *background*
    thread, so the SystemExit it would normally raise cannot
    propagate to main() (Python's threading model: unhandled
    exceptions in non-main threads are printed to stderr and
    the thread silently dies). To observe the call from this
    test, we replace ``os._exit`` with a recorder that uses a
    shared ``threading.Event`` instead of raising.
    """
    from web import desktop

    heartbeat = threading.Event()
    shutdown_acknowledged = threading.Event()
    exit_called = threading.Event()
    exit_code_holder: list[int] = []

    def _recording_exit(code: int) -> None:
        exit_code_holder.append(code)
        exit_called.set()

    with mock.patch.object(desktop.os, "_exit", side_effect=_recording_exit):
        desktop._start_watchdog_thread(heartbeat, shutdown_acknowledged, timeout=0.1)
        # Don't ping the heartbeat. The watchdog should fire
        # os._exit(1) within ~100ms.
        fired = exit_called.wait(timeout=1.0)

    assert fired, "watchdog did not call os._exit(1) when heartbeat was silent"
    assert exit_code_holder == [1], f"expected os._exit(1), got {exit_code_holder!r}"
    print("  ✓ silent GTK loop → watchdog os._exit(1) (catches cb96bf8's blind spot)")


def test_watchdog_quiesces_on_clean_shutdown() -> None:
    """When main() sets ``shutdown_acknowledged`` after a clean
    webview.start() return, the watchdog must stop firing even
    if no further heartbeats arrive. (Without this, the watchdog
    would kill the process 5s after the user closes the window.)

    We simulate the production scenario: a heartbeat thread
    pings ``heartbeat`` every 25ms (faster than the 100ms
    timeout), then ``shutdown_acknowledged`` is set. The
    watchdog should exit via the while-loop check, never
    reaching ``os._exit``.
    """
    from web import desktop

    heartbeat = threading.Event()
    shutdown_acknowledged = threading.Event()

    stop_pinging = threading.Event()

    def _pinger() -> None:
        while not stop_pinging.is_set():
            heartbeat.set()
            time.sleep(0.025)  # ping 25ms — well under the 100ms timeout

    pinger_thread = threading.Thread(target=_pinger, name="test-pinger", daemon=True)
    pinger_thread.start()

    try:
        with mock.patch.object(desktop.os, "_exit") as mock_exit:
            desktop._start_watchdog_thread(heartbeat, shutdown_acknowledged, timeout=0.1)
            # Let the watchdog cycle a few times (each iteration
            # is well under 100ms because the pinger keeps
            # refreshing the heartbeat).
            time.sleep(0.25)
            mock_exit.assert_not_called()
            # Now simulate the "user closed the window" event:
            # main() sets shutdown_acknowledged, which the
            # watchdog's while-loop should pick up on the next
            # iteration.
            shutdown_acknowledged.set()
            time.sleep(0.25)
            mock_exit.assert_not_called()
            print("  ✓ clean shutdown acknowledged → watchdog quiesces")
    finally:
        stop_pinging.set()
        pinger_thread.join(timeout=1.0)


if __name__ == "__main__":
    print("=" * 60)
    print("DESKTOP SHUTDOWN TEST (clean vs crash vs Ctrl+C vs hang)")
    print("=" * 60)

    print("\n[1/5] test_clean_shutdown_returns_normally")
    test_clean_shutdown_returns_normally()

    print("\n[2/5] test_webview_crash_force_exits")
    test_webview_crash_force_exits()

    print("\n[3/5] test_keyboard_interrupt_exits_with_posix_sigint_code")
    test_keyboard_interrupt_exits_with_posix_sigint_code()

    print("\n[4/5] test_watchdog_force_exits_on_silent_gtk_loop")
    test_watchdog_force_exits_on_silent_gtk_loop()

    print("\n[5/5] test_watchdog_quiesces_on_clean_shutdown")
    test_watchdog_quiesces_on_clean_shutdown()

    print("\n" + "=" * 60)
    print("✓ All desktop shutdown tests PASSED")
    print("=" * 60)
