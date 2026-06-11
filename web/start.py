"""Smart launcher: pick the best mode automatically and just go.

Decision tree:
    1. Try pywebview (native window, no browser needed)
    2. Fall back to webbrowser (system default browser)
    3. As a last resort, run as plain server (no UI)

Usage:
    python -m web.start                 # auto-pick mode
    python -m web.start --port 9000     # custom port
    python -m web.start --width 1400 --height 900
    python -m web.start --force-server  # skip the GUI, just run uvicorn
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("web.start")


def _try_import_webview() -> tuple[bool, str]:
    """Return (available, reason). 'reason' is empty on success."""
    try:
        import webview  # noqa: F401
    except ImportError:
        return False, "pywebview is not installed"
    try:
        import gi  # noqa: F401
    except ImportError:
        return False, "PyGObject (python3-gi) is not installed (Linux needs: sudo apt install python3-gobject)"
    try:
        gi.require_version("Gtk", "3.0")
        gi.require_version("WebKit2", "4.1")
    except Exception as e:  # noqa: BLE001
        return False, f"GTK/WebKit introspection not available: {e}"
    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Russian TTS Studio Studio — smart launcher (auto-picks native window or browser)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8129, help="Port (default: 8129)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=820)
    parser.add_argument("--title", default="Russian TTS Studio — Russian TTS Studio")
    parser.add_argument(
        "--force-browser",
        action="store_true",
        help="Skip the native window, open the system default browser.",
    )
    parser.add_argument(
        "--force-server",
        action="store_true",
        help="Don't open any UI — just run the server (Ctrl+C to stop).",
    )
    args = parser.parse_args()

    # If --force-server, skip the GUI branch entirely
    if args.force_server:
        from web.desktop import _start_uvicorn_in_thread, _wait_for_server
        url = f"http://{args.host}:{args.port}"
        logger.info("Server-only mode. Will be available at %s", url)
        _start_uvicorn_in_thread(args.host, args.port)
        if not _wait_for_server(args.host, args.port, timeout=60.0):
            logger.error("Server failed to start within 60s")
            return 1
        logger.info("Server is up. Open %s in your browser. Press Ctrl+C to stop.", url)
        try:
            import threading
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
        return 0

    # Decide between native window and browser
    if args.force_browser:
        use_webview, reason = False, "--force-browser was set"
    else:
        use_webview, reason = _try_import_webview()

    if use_webview:
        logger.info("Mode: native window (pywebview)")
        sys.argv = [
            sys.argv[0],
            "--host", args.host,
            "--port", str(args.port),
            "--width", str(args.width),
            "--height", str(args.height),
            "--title", args.title,
        ]
        from web.desktop import main as desktop_main
        return desktop_main()
    else:
        logger.info("Mode: system browser (reason: %s)", reason)
        sys.argv = [
            sys.argv[0],
            "--host", args.host,
            "--port", str(args.port),
            "--browser",
        ]
        from web.desktop import main as desktop_main
        return desktop_main()


if __name__ == "__main__":
    sys.exit(main())
