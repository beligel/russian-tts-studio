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


def _no_op_uvicorn_watchdog_runner() -> None:
    """Replacement for the uvicorn-watchdog runner threaded in
    the webview branch. In production this would call
    ``_watch_uvicorn_until`` and os._exit(1) on crash. In tests
    we let it run as a no-op daemon thread so the test runner
    doesn't get killed by an unexpected crash of the leftover
    uvicorn thread from a previous test.
    """
    return


# Tests can use this to control what ``_start_uvicorn_in_thread``
# returns — the real one would bind port 8129 and leave a
# background thread running across tests, which makes order
# matter and slows the suite down. We pass a thread that's
# always alive + a sentinel server.
def _fake_start_uvicorn(*args, **kwargs) -> tuple[threading.Thread, mock.MagicMock]:
    """Stub for ``_start_uvicorn_in_thread`` used by integration
    tests. Returns ``(alive_thread, sentinel_server)`` so
    ``_watch_uvicorn_until`` sees a healthy uvicorn and never
    fires the crash path. The thread is a daemon so it dies
    with the test runner regardless."""
    t = threading.Thread(target=lambda: None, daemon=True)
    t.start()
    return t, mock.MagicMock(name="fake_uvicorn_server")


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
         mock.patch.object(desktop, "_start_uvicorn_in_thread", _fake_start_uvicorn), \
         mock.patch.object(desktop, "_wait_for_server", return_value=True), \
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
         mock.patch.object(desktop, "_start_uvicorn_in_thread", _fake_start_uvicorn), \
         mock.patch.object(desktop, "_wait_for_server", return_value=True), \
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
         mock.patch.object(desktop, "_start_uvicorn_in_thread", _fake_start_uvicorn), \
         mock.patch.object(desktop, "_wait_for_server", return_value=True), \
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


def test_uvicorn_watchdog_returns_crashed_when_thread_dies() -> None:
    """The uvicorn watchdog (``_watch_uvicorn_until``) is the
    third soft-zombie detector: it watches the uvicorn worker
    thread and returns ``"crashed"`` when the thread dies
    without ``stop_event`` being set. The webview-branch
    caller then os._exit(1)s the process.

    We don't drive ``main()`` for this test — we exercise
    ``_watch_uvicorn_until`` directly with a thread that
    exits almost immediately, a sentinel server, and a
    stop_event that is *never* set. The watchdog must
    return "crashed" within ~1.5s (1s poll interval + slack).
    """
    from web import desktop

    stop_event = threading.Event()
    server = mock.MagicMock(name="sentinel_uvicorn_server")

    def _short_lived() -> None:
        # Thread target that returns immediately, marking
        # the thread as "not alive" for subsequent checks.
        return

    t = threading.Thread(target=_short_lived, daemon=True)
    t.start()
    t.join(timeout=1.0)  # ensure the thread is fully done

    outcome = desktop._watch_uvicorn_until(server, t, stop_event, poll_interval=0.1)
    assert outcome == "crashed", f"expected 'crashed', got {outcome!r}"
    print("  ✓ uvicorn thread death → _watch_uvicorn_until returns 'crashed'")


def test_uvicorn_watchdog_returns_stopped_when_stop_event_fires() -> None:
    """Symmetric case: if the user closes the window and
    main() sets stop_event, ``_watch_uvicorn_until`` returns
    ``"stopped"`` even though the uvicorn thread is still
    alive. The caller must NOT os._exit(1) on "stopped"."""
    from web import desktop

    stop_event = threading.Event()
    server = mock.MagicMock(name="sentinel_uvicorn_server")

    def _stay_alive() -> None:
        # Block until the test ends. This is a real live
        # thread the watchdog will see as alive.
        stop_event.wait(timeout=5.0)

    t = threading.Thread(target=_stay_alive, daemon=True)
    t.start()

    try:
        # Signal clean shutdown on a short delay. The
        # watchdog's poll loop will see stop_event on its
        # next iteration.
        def _signal_later() -> None:
            time.sleep(0.2)
            stop_event.set()

        threading.Thread(target=_signal_later, daemon=True).start()

        outcome = desktop._watch_uvicorn_until(server, t, stop_event, poll_interval=0.05)
        assert outcome == "stopped", f"expected 'stopped', got {outcome!r}"
        print("  ✓ stop_event set → _watch_uvicorn_until returns 'stopped'")
    finally:
        stop_event.set()
        t.join(timeout=1.0)


def test_blocking_wait_calls_os_exit_on_uvicorn_crash() -> None:
    """The user-facing symptom of the third soft-zombie case:
    the launcher is alive, the FastAPI process is gone, the
    UI says "Сервер не отвечает". ``_blocking_wait_with_uvicorn_watchdog``
    is the layer that turns this into ``os._exit(1)`` instead
    of an indefinite hang. We drive it with a dead thread and
    verify it calls os._exit(1) within ~1.5s."""
    from web import desktop

    stop_event = threading.Event()
    server = mock.MagicMock(name="sentinel_uvicorn_server")
    exit_called = threading.Event()
    exit_code_holder: list[int] = []

    def _recording_exit(code: int) -> None:
        exit_code_holder.append(code)
        exit_called.set()

    def _short_lived() -> None:
        return

    t = threading.Thread(target=_short_lived, daemon=True)
    t.start()
    t.join(timeout=1.0)

    with mock.patch.object(desktop.os, "_exit", side_effect=_recording_exit):
        desktop._blocking_wait_with_uvicorn_watchdog(stop_event, t, server)

    assert exit_called.is_set(), "_blocking_wait_with_uvicorn_watchdog did not call os._exit"
    assert exit_code_holder == [1], f"expected os._exit(1), got {exit_code_holder!r}"
    print("  ✓ uvicorn crash → _blocking_wait_with_uvicorn_watchdog calls os._exit(1)")


if __name__ == "__main__":
    print("=" * 60)
    print("DESKTOP SHUTDOWN TEST (clean vs crash vs Ctrl+C vs hang)")
    print("=" * 60)

    print("\n[1/8] test_clean_shutdown_returns_normally")
    test_clean_shutdown_returns_normally()

    print("\n[2/8] test_webview_crash_force_exits")
    test_webview_crash_force_exits()

    print("\n[3/8] test_keyboard_interrupt_exits_with_posix_sigint_code")
    test_keyboard_interrupt_exits_with_posix_sigint_code()

    print("\n[4/8] test_watchdog_force_exits_on_silent_gtk_loop")
    test_watchdog_force_exits_on_silent_gtk_loop()

    print("\n[5/8] test_watchdog_quiesces_on_clean_shutdown")
    test_watchdog_quiesces_on_clean_shutdown()

    print("\n[6/8] test_uvicorn_watchdog_returns_crashed_when_thread_dies")
    test_uvicorn_watchdog_returns_crashed_when_thread_dies()

    print("\n[7/8] test_uvicorn_watchdog_returns_stopped_when_stop_event_fires")
    test_uvicorn_watchdog_returns_stopped_when_stop_event_fires()

    print("\n[8/8] test_blocking_wait_calls_os_exit_on_uvicorn_crash")
    test_blocking_wait_calls_os_exit_on_uvicorn_crash()

    print("\n" + "=" * 60)
    print("✓ All desktop shutdown tests PASSED")
    print("=" * 60)
