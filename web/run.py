"""Convenience launcher for the Web UI.

Usage:
    python -m web.run
    python -m web.run --port 8080 --no-reload
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Russian TTS Studio Russian TTS Studio")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8129, help="Port (default: 8129; override with PORT env or --port)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    parser.add_argument("--no-reload", dest="reload", action="store_false")
    parser.set_defaults(reload=True)
    args = parser.parse_args()

    import uvicorn
    print(f"\n  🎙️  Russian TTS Studio Russian TTS Studio")
    print(f"  → http://localhost:{args.port}\n")
    uvicorn.run(
        "web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
