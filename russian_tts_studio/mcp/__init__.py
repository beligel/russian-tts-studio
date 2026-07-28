"""MCP (Model Context Protocol) server for RTTS.

Public API::

    from russian_tts_studio.mcp import handle_mcp_request, generate_config

Phase 7 (LTV-inspired):
- JSON-RPC MCP server with 15 tools
- stdio bridge for Claude Desktop
- HTTP endpoint for other clients
- Config generator for Claude Desktop / Codex
"""

from __future__ import annotations

from pathlib import Path

from .server import TOOLS, handle_mcp_request


def generate_config(
    python_path: str = "python",
    project_path: str | None = None,
) -> dict:
    """Generate MCP config for Claude Desktop and Codex.

    Returns a dict with ``claude_desktop`` and ``codex`` keys.
    """
    if project_path is None:
        project_path = str(Path(__file__).resolve().parent.parent.parent)

    args = ["-m", "russian_tts_studio.mcp.stdio_bridge"]

    claude_desktop = {
        "mcpServers": {
            "russian-tts-studio": {
                "command": python_path,
                "args": [str(Path(project_path) / a) if not a.startswith("-") else a for a in args],
                "cwd": project_path,
            }
        }
    }

    codex = {
        "mcp_servers": {
            "russian-tts-studio": {
                "command": python_path,
                "args": args,
                "cwd": project_path,
            }
        }
    }

    return {"claude_desktop": claude_desktop, "codex": codex}


__all__ = ["handle_mcp_request", "generate_config", "TOOLS"]
