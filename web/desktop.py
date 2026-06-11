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
    window = webview.create_window(
        title=args.title,
        url=url,
        width=args.width,
        height=args.height,
        min_size=(800, 600),
        resizable=True,
        text_select=True,
    )
    try:
        webview.start()
    except KeyboardInterrupt:
        logger.info("Window closed by user")
    return 0


if __name__ == "__main__":
    sys.exit(main())
