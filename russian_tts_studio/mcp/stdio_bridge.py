"""MCP stdio bridge for Claude Desktop and other MCP clients.

Reads JSON-RPC messages from stdin, dispatches to the MCP server,
and writes responses to stdout. This is the entry point for
``claude_desktop_config.json``.

Usage::

    python -m russian_tts_studio.mcp.stdio_bridge

Or in ``claude_desktop_config.json``::

    {
        "mcpServers": {
            "russian-tts-studio": {
                "command": "python",
                "args": ["-m", "russian_tts_studio.mcp.stdio_bridge"],
                "cwd": "/path/to/russian-tts-studio"
            }
        }
    }

For Codex / ``~/.codex/config.toml``::

    [mcp_servers.russian-tts-studio]
    command = 'python'
    args = ['-m', 'russian_tts_studio.mcp.stdio_bridge']
    cwd = '/path/to/russian-tts-studio'
"""

from __future__ import annotations

import json
import sys
import logging

# Set up logging to stderr (stdout is reserved for JSON-RPC).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("mcp.stdio")

# Add project root to path for imports.
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    """Main loop: read JSON-RPC from stdin, write responses to stdout."""
    from russian_tts_studio.mcp.server import handle_mcp_request

    logger.info("MCP stdio bridge started")
    logger.info("Reading JSON-RPC from stdin, writing to stdout")

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Invalid JSON: %s", e)
                response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {e}"},
                }
                _write_response(response)
                continue

            # Handle batch requests
            if isinstance(request, list):
                for req in request:
                    response = handle_mcp_request(req)
                    if response:  # notifications return {}
                        _write_response(response)
            else:
                response = handle_mcp_request(request)
                if response:
                    _write_response(response)

    except KeyboardInterrupt:
        logger.info("MCP stdio bridge stopped")
    except EOFError:
        logger.info("stdin closed")
    except Exception as e:
        logger.exception("Unexpected error: %s", e)


def _write_response(response: dict) -> None:
    """Write a JSON-RPC response to stdout."""
    if not response:
        return
    line = json.dumps(response, ensure_ascii=False)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
