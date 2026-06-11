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


def _start_uvicorn_in_thread(host: str, port: int) -> threading.Thread:
    """Run uvicorn in a daemon thread so the main thread can host the WebView."""
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
    return thread


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

    if _is_port_open(args.host, args.port):
        logger.info("Port %d already in use — assuming server is already running", args.port)
    else:
        logger.info("Starting uvicorn on %s …", url)
        _start_uvicorn_in_thread(args.host, args.port)
        if not _wait_for_server(args.host, args.port, timeout=60.0):
            logger.error("Server failed to start within 60s")
            return 1
        logger.info("Server is up at %s", url)

    if args.no_window:
        logger.info("Running in --no-window mode. Press Ctrl+C to stop.")
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            logger.info("Stopping…")
        return 0

    if args.browser:
        logger.info("Opening %s in the default browser", url)
        webbrowser.open(url)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            logger.info("Stopping…")
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

    try:
        webview.start(_heartbeat, gui=None)
        clean_shutdown = True
    except KeyboardInterrupt:
        # POSIX exit code for SIGINT is 128+2 = 130. Use it so
        # terminal-driven workflows (e.g. a wrapper script) can
        # distinguish "user pressed Ctrl+C" from "process crashed".
        logger.info("Window closed by user (Ctrl+C)")
        shutdown_acknowledged.set()
        os._exit(130)
    except Exception:  # noqa: BLE001
        # GTK/WebKit crash, missing display, etc. — log and force-exit
        # to make sure we don't leak a soft-zombie python process.
        logger.exception("webview.start() crashed — force-exiting")
        shutdown_acknowledged.set()
        os._exit(1)
    # Clean path: webview.start() returned. Stop the watchdog,
    # hand off to the normal sys.exit chain (preserves subprocess
    # test workflows that import this module).
    shutdown_acknowledged.set()
    if clean_shutdown:
        return 0
    # Crashed path: uvicorn thread is daemon=True, dies with us.
    # os._exit (not sys.exit) to skip any stuck atexit/finalisers.
    os._exit(1)


if __name__ == "__main__":
    sys.exit(main())
