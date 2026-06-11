"""Desktop wrapper: launch the Web UI in a native WebView window.

The FastAPI server is started in a background thread, then pywebview opens
a native window pointing at it. When the user closes the window, the
server thread is shut down gracefully.

Usage:
    python -m web.desktop
    python -m web.desktop --port 8080 --no-window   # server only (debug)
    python -m web.desktop --width 1200 --height 800
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("web.desktop")


def _is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    """Return True if a TCP connection to (host, port) succeeds."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, ConnectionRefusedError):
        return False


def _wait_for_server(host: str, port: int, timeout: float = 30.0) -> bool:
    """Block until the server accepts a TCP connection (or timeout)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _is_port_open(host, port):
            return True
        time.sleep(0.2)
    return False


def _start_watchdog_thread(
    heartbeat: threading.Event,
    shutdown_acknowledged: threading.Event,
    timeout: float = 5.0,
) -> threading.Thread:
    """Spawn a daemon thread that os._exit(1)s the process if
    ``heartbeat`` stops being set for ``timeout`` seconds.

    Exposed at module level so tests can substitute a no-op
    watchdog (otherwise the real watchdog's 5-second grace
    period can race with the test runner and os._exit the
    test itself).
    """
    def _watchdog() -> None:
        while not shutdown_acknowledged.is_set():
            if not heartbeat.wait(timeout=timeout):
                logger.error(
                    "GTK main loop silent for %ss — likely soft-zombie, force-exiting",
                    timeout,
                )
                os._exit(1)
                # Defensive: in production os._exit(1) never returns
                # (it terminates the process), but in tests it's
                # mocked to a no-op. Return explicitly so the
                # watchdog thread can't keep looping against a
                # mocked os._exit and call _exit() repeatedly.
                return
            heartbeat.clear()

    thread = threading.Thread(target=_watchdog, name="desktop-watchdog", daemon=True)
    thread.start()
    return thread


def _start_uvicorn_in_thread(
    host: str,
    port: int,
) -> tuple[threading.Thread, "uvicorn.Server"]:
    """Run uvicorn in a daemon thread so the main thread can host the WebView.

    Returns ``(thread, server)`` so the caller can monitor liveness
    via ``thread.is_alive()`` and inspect ``server.should_exit``.
    The thread is daemon=True; the parent process is responsible
    for exiting it cleanly (via os._exit on crashes, or sys.exit
    on user-driven shutdown).
    """
    import uvicorn

    config = uvicorn.Config(
        "web.app:app",
        host=host,
        port=port,
        log_level="info",
        lifespan="on",
    )
    server = uvicorn.Server(config)

    def _run() -> None:
        try:
            server.run()
        except Exception as e:  # noqa: BLE001
            logger.exception("Uvicorn crashed: %s", e)

    thread = threading.Thread(target=_run, name="uvicorn-desktop", daemon=True)
    thread.start()
    return thread, server


def _watch_uvicorn_until(
    server: "uvicorn.Server",
    thread: threading.Thread,
    stop_event: threading.Event,
    poll_interval: float = 1.0,
) -> str:
    """Block until either ``stop_event`` is set or the uvicorn
    thread dies, and return a tag describing which happened.

    Return values:
        "stopped" — ``stop_event`` fired (clean user shutdown,
                    the main blocking function returned / was
                    interrupted normally). Caller should exit
                    the process normally (sys.exit / return).
        "crashed"  — uvicorn thread died without the stop_event
                    being set. This is the third soft-zombie
                    case: server.run() returned (e.g. due to an
                    unhandled exception in a request handler,
                    a segfault in libtorch during TTS
                    inference, an OOM-kill of the worker, or
                    an os.execv during engine re-exec that
                    somehow leaked the port). The main thread
                    would otherwise sit forever in
                    ``threading.Event().wait()`` thinking
                    everything is fine. Caller must call
                    os._exit(1) to surface the failure to the
                    user instead of pretending the app is
                    healthy.

    The 1-second poll interval is coarse — the uvicorn thread
    could be dead for up to 1s before we notice. That's fine:
    the alternative (faster polling) buys nothing for a human
    user, and the request that triggered the crash has already
    been lost either way.

    If ``thread`` is None (the port was already bound when we
    started, so we don't own the uvicorn process), this just
    blocks on ``stop_event`` and returns "stopped" — the
    third-party uvicorn is someone else's problem.
    """
    if thread is None:
        stop_event.wait()
        return "stopped"
    while not stop_event.is_set():
        if not thread.is_alive():
            return "crashed"
        if stop_event.wait(timeout=poll_interval):
            return "stopped"
    return "stopped"


def _blocking_wait_with_uvicorn_watchdog(
    stop_event: threading.Event,
    uvicorn_thread: threading.Thread | None,
    uvicorn_server: "uvicorn.Server | None",
) -> None:
    """Replace ``threading.Event().wait()`` in the --no-window,
    --browser, and --force-server code paths.

    On clean shutdown the caller sets ``stop_event`` and we
    return normally. If the uvicorn thread dies unexpectedly
    (third soft-zombie case), we ``os._exit(1)`` so the
    failure surfaces to the user — otherwise the process
    would sit forever with a dead server, no port bound, no
    log output, and a UI that says "Сервер не отвечает".
    """
    if uvicorn_thread is None or uvicorn_server is None:
        # No watchdog needed — port was already bound, we're
        # not the owner of the uvicorn process.
        try:
            stop_event.wait()
        except KeyboardInterrupt:
            stop_event.set()
            raise
        return
    outcome = _watch_uvicorn_until(uvicorn_server, uvicorn_thread, stop_event)
    if outcome == "crashed":
        logger.error(
            "Uvicorn thread died unexpectedly — the FastAPI "
            "process is gone but this launcher is still alive. "
            "Force-exiting to surface the failure."
        )
        os._exit(1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch Russian TTS Studio Russian TTS Studio in a native WebView window",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8129, help="Port (default: 8129)")
    parser.add_argument("--width", type=int, default=1280, help="Window width")
    parser.add_argument("--height", type=int, default=820, help="Window height")
    parser.add_argument("--title", default="Russian TTS Studio — Russian TTS Studio")
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Don't open a WebView — just run the server (for debugging).",
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Open in the default system browser instead of a native WebView.",
    )
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"

    # The uvicorn watchdog. We need to know when uvicorn dies
    # without a clean shutdown signal (the third soft-zombie
    # case: server.run() returns due to an unhandled exception
    # in a request handler, a libtorch segfault during TTS
    # inference, an OOM-kill of the worker, or an os.execv
    # that somehow leaks the port). The branches below run
    # their main loop in a child thread and set ``stop_event``
    # when they're done. The main thread waits on either
    # ``stop_event`` (clean exit) or uvicorn thread death
    # (crash) — whichever comes first.
    stop_event = threading.Event()
    uvicorn_thread: threading.Thread | None = None
    uvicorn_server = None  # type: ignore[var-annotated]

    if _is_port_open(args.host, args.port):
        logger.info("Port %d already in use — assuming server is already running", args.port)
    else:
        logger.info("Starting uvicorn on %s …", url)
        uvicorn_thread, uvicorn_server = _start_uvicorn_in_thread(args.host, args.port)
        if not _wait_for_server(args.host, args.port, timeout=60.0):
            logger.error("Server failed to start within 60s")
            return 1
        logger.info("Server is up at %s", url)

    if args.no_window:
        logger.info("Running in --no-window mode. Press Ctrl+C to stop.")
        try:
            _blocking_wait_with_uvicorn_watchdog(
                stop_event,
                uvicorn_thread,
                uvicorn_server,
            )
        except KeyboardInterrupt:
            logger.info("Stopping…")
            stop_event.set()
        return 0

    if args.browser:
        logger.info("Opening %s in the default browser", url)
        webbrowser.open(url)
        try:
            _blocking_wait_with_uvicorn_watchdog(
                stop_event,
                uvicorn_thread,
                uvicorn_server,
            )
        except KeyboardInterrupt:
            logger.info("Stopping…")
            stop_event.set()
        return 0

    try:
        import webview  # pywebview
    except ImportError:
        logger.error(
            "pywebview is not installed. Run:\n"
            "    pip install pywebview\n"
            "or use --browser to open in the default browser instead.",
        )
        return 1

    logger.info("Opening native WebView window: %dx%d — %s", args.width, args.height, url)
    # On Linux+WebKit, webview.start() *should* return when the
    # window is closed by the user, but we've seen TWO failure modes
    # that leave the Python process alive with FDs → /dev/null and
    # no port bound (a "soft zombie"):
    #
    #  1. webview.start() raises an exception (GTK crash, missing
    #     display, WebKit segfault). The try/except below handles
    #     this with os._exit(1).
    #  2. webview.start() never returns at all — the GTK main loop
    #     spins forever even though the user closed the window
    #     (seen after X11 session changes, wmctrl/xdotool kills of
    #     the window, GPU driver resets). In this case the try block
    #     never falls through, so the os._exit(1) above is never
    #     reached. We need an *external* watchdog.
    #
    # The watchdog below is a background thread that pings an event
    # while the GTK main loop is alive. The ``func=_heartbeat``
    # callback runs inside GTK's main loop (pywebview docs:
    # ``start(func=...)`` calls func() repeatedly while the GUI
    # thread is responsive). If the callback stops being called for
    # ``WATCHDOG_TIMEOUT`` seconds — meaning the main loop is
    # stuck or the window was destroyed but the loop didn't notice
    # — we os._exit(1) from the watchdog thread. os._exit kills the
    # whole process immediately, regardless of which thread calls it.
    #
    # On a *clean* shutdown webview.start() returns, we set
    # ``shutdown_acknowledged`` and call os._exit(0) ourselves, so
    # the watchdog never has to fire.
    clean_shutdown = False

    window = webview.create_window(
        title=args.title,
        url=url,
        width=args.width,
        height=args.height,
        min_size=(800, 600),
        resizable=True,
        text_select=True,
    )

    # Watchdog plumbing. The watchdog is a separate function
    # (see _start_watchdog_thread above) so tests can replace
    # it with a no-op.
    WATCHDOG_TIMEOUT = 5.0  # seconds without a heartbeat → force-exit
    heartbeat = threading.Event()
    shutdown_acknowledged = threading.Event()

    def _heartbeat() -> None:
        """Called repeatedly by pywebview while the GTK main loop runs.

        Each call signals the watchdog that the GUI thread is alive.
        Returning immediately is fine — pywebview just calls us again
        on the next iteration of its loop. We use a per-call ping
        rather than a per-call wait so the GTK loop isn't blocked.
        """
        heartbeat.set()
        # Don't .clear() here — the watchdog thread is the only
        # consumer and it sets a deadline. If we cleared on every
        # tick, the watchdog's "no heartbeat for N seconds" check
        # would race with our own set().

    _start_watchdog_thread(heartbeat, shutdown_acknowledged, WATCHDOG_TIMEOUT)

    # Prime the heartbeat so the watchdog doesn't trip during the
    # gap between the thread starting and the first _heartbeat
    # callback firing (typically a few hundred ms, but it can be
    # longer on a busy X server or while WebKit is spinning up).
    heartbeat.set()

    # The uvicorn watchdog for the webview branch. The GTK/webkit
    # watchdog above only covers the case where the GUI thread
    # itself is stuck — it doesn't notice if uvicorn dies (e.g.
    # unhandled exception in a request handler, libtorch
    # segfault during TTS inference, OOM-kill, engine reexec
    # mishap). While the main thread is blocked in webview.start()
    # we have no way to react from the main thread, so we spawn
    # a *separate* daemon thread whose only job is to call
    # os._exit(1) if uvicorn thread dies. On a clean shutdown
    # (webview.start() returns) we set stop_event so the watchdog
    # can exit normally. The watchdog thread cannot suppress a
    # main-thread return — it only acts when uvicorn dies first.
    if uvicorn_thread is not None and uvicorn_server is not None:

        def _uvicorn_watchdog_runner() -> None:
            outcome = _watch_uvicorn_until(
                uvicorn_server, uvicorn_thread, stop_event,
            )
            if outcome == "crashed":
                logger.error(
                    "Uvicorn thread died during webview session — "
                    "force-exiting to surface the failure",
                )
                os._exit(1)

        uvicorn_watchdog = threading.Thread(
            target=_uvicorn_watchdog_runner,
            name="uvicorn-watchdog",
            daemon=True,
        )
        uvicorn_watchdog.start()
    else:
        # Port was already bound by an external uvicorn — we
        # don't own the process, so we can't watch it. If it
        # dies, the user will see the "Сервер не отвечает"
        # error from app.js instead, which is honest about
        # what's happening.
        uvicorn_watchdog = None

    try:
        webview.start(_heartbeat, gui=None)
        clean_shutdown = True
    except KeyboardInterrupt:
        # POSIX exit code for SIGINT is 128+2 = 130. Use it so
        # terminal-driven workflows (e.g. a wrapper script) can
        # distinguish "user pressed Ctrl+C" from "process crashed".
        logger.info("Window closed by user (Ctrl+C)")
        stop_event.set()
        shutdown_acknowledged.set()
        os._exit(130)
    except Exception:  # noqa: BLE001
        # GTK/WebKit crash, missing display, etc. — log and force-exit
        # to make sure we don't leak a soft-zombie python process.
        logger.exception("webview.start() crashed — force-exiting")
        stop_event.set()
        shutdown_acknowledged.set()
        os._exit(1)
    # Clean path: webview.start() returned. Stop both watchdogs
    # and hand off to the normal sys.exit chain (preserves
    # subprocess test workflows that import this module).
    stop_event.set()
    shutdown_acknowledged.set()
    if clean_shutdown:
        return 0
    # Crashed path: uvicorn thread is daemon=True, dies with us.
    # os._exit (not sys.exit) to skip any stuck atexit/finalisers.
    os._exit(1)


if __name__ == "__main__":
    sys.exit(main())
