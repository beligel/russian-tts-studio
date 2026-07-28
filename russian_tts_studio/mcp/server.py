"""MCP (Model Context Protocol) server for RTTS.

Exposes RTTS functionality as MCP tools that AI agents (Claude Desktop,
Codex, Cursor) can call programmatically. Implements the MCP JSON-RPC
protocol over stdio (for Claude Desktop) and HTTP (for other clients).

Tools exposed:
- ``server_info`` — server metadata and version
- ``list_engines`` — available TTS engines
- ``list_references`` — saved reference voices
- ``create_audiobook`` — create project from text
- ``generate_audio`` — synthesise text (single segment)
- ``get_projects`` — list all projects
- ``get_project`` — load project with segments
- ``get_project_source`` — read project source text (paginated)
- ``edit_project_source`` — edit source text (with SHA-256 concurrency)
- ``get_segments`` — list segments for a project
- ``approve_segment`` — mark segment approved
- ``discard_segment`` — mark segment for retry
- ``regenerate_segment`` — re-synthesise one segment
- ``rebuild_audiobook`` — concat approved segments
- ``get_markup_help`` — markup command reference

Usage::

    # As a module (for HTTP endpoint in web/app.py)
    from russian_tts_studio.mcp import handle_mcp_request

    # As stdio bridge (for Claude Desktop)
    python -m russian_tts_studio.mcp.stdio_bridge
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool definitions (MCP schema)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "server_info",
        "description": "Get server metadata, version, and available features.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_engines",
        "description": "List available TTS engines (voxcpm, qwen3, silero).",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_references",
        "description": "List saved reference voice files.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "create_audiobook",
        "description": "Create a TTS project from source text. Detects chapters, chunks text, creates segments. Returns project ID and segment count.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Project name"},
                "source_text": {"type": "string", "description": "Full source text (supports Markdown headings, {{chapter}} markup)"},
                "max_chars": {"type": "integer", "description": "Max chars per segment (default 200)", "default": 200},
            },
            "required": ["name", "source_text"],
        },
    },
    {
        "name": "generate_audio",
        "description": "Synthesise a single text snippet. Returns audio path and duration.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to synthesise"},
                "reference_path": {"type": "string", "description": "Path to reference voice WAV (for voice cloning)"},
                "speed": {"type": "number", "description": "Speed multiplier (default 0.9)", "default": 0.9},
                "output_name": {"type": "string", "description": "Output filename (without extension)"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "get_projects",
        "description": "List all TTS projects with summary info.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_project",
        "description": "Load a project with all chapters and segments.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_id": {"type": "string", "description": "Project ID"},
            },
            "required": ["project_id"],
        },
    },
    {
        "name": "get_project_source",
        "description": "Read project source text (paginated for large texts).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "page": {"type": "integer", "description": "Page number (0-based)", "default": 0},
                "page_size": {"type": "integer", "description": "Chars per page (default 10000)", "default": 10000},
            },
            "required": ["project_id"],
        },
    },
    {
        "name": "edit_project_source",
        "description": "Edit project source text with SHA-256 concurrency check.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "new_text": {"type": "string", "description": "New source text"},
                "expected_sha256": {"type": "string", "description": "SHA-256 of the text you read (prevents concurrent overwrite)"},
            },
            "required": ["project_id", "new_text", "expected_sha256"],
        },
    },
    {
        "name": "get_segments",
        "description": "List all segments for a project.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
            },
            "required": ["project_id"],
        },
    },
    {
        "name": "approve_segment",
        "description": "Mark a segment as approved.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "segment_id": {"type": "string"},
            },
            "required": ["project_id", "segment_id"],
        },
    },
    {
        "name": "discard_segment",
        "description": "Mark a segment for retry.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "segment_id": {"type": "string"},
            },
            "required": ["project_id", "segment_id"],
        },
    },
    {
        "name": "regenerate_segment",
        "description": "Re-synthesise a single segment with optional parameter overrides.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "segment_id": {"type": "string"},
                "speed": {"type": "number", "description": "Speed override"},
                "instruct": {"type": "string", "description": "Instruction for Qwen3-TTS"},
            },
            "required": ["project_id", "segment_id"],
        },
    },
    {
        "name": "rebuild_audiobook",
        "description": "Concatenate all approved segments into a single audio file.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "output_name": {"type": "string", "description": "Output filename (default: audiobook)"},
            },
            "required": ["project_id"],
        },
    },
    {
        "name": "get_markup_help",
        "description": "Get reference for RTTS markup commands ({{pause}}, {{speed}}, {{volume}}, {{chapter}}, {{alias}}, {{reset}}).",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tool_server_info() -> dict:
    return {
        "name": "Russian TTS Studio",
        "version": "0.3.0",
        "engines": ["voxcpm", "qwen3", "silero"],
        "features": ["markup", "projects", "audio_mix", "subtitles", "normalization", "review"],
    }


def _tool_list_engines() -> dict:
    return {
        "engines": [
            {"id": "voxcpm", "name": "VoxCPM2", "languages": ["ru", "+30"], "voice_clone": True},
            {"id": "qwen3", "name": "Qwen3-TTS", "languages": ["ru", "+9"], "voice_clone": True, "voice_design": True, "instruct": True},
            {"id": "silero", "name": "Silero", "languages": ["ru"], "voice_clone": False},
        ],
    }


def _tool_list_references() -> dict:
    refs_dir = Path("output/reference")
    if not refs_dir.exists():
        return {"references": []}
    refs = []
    for p in sorted(refs_dir.glob("*.wav")) + sorted(refs_dir.glob("*.mp3")):
        refs.append({"name": p.name, "path": str(p)})
    return {"references": refs}


def _tool_create_audiobook(name: str, source_text: str, max_chars: int = 200) -> dict:
    from ..projects import create_project

    project = create_project(name, source_text, max_chars=max_chars)
    return {
        "project_id": project.id,
        "name": project.name,
        "total_segments": project.total_segments,
        "chapters": len(project.chapters),
    }


def _tool_generate_audio(
    text: str,
    reference_path: str | None = None,
    speed: float = 0.9,
    output_name: str | None = None,
) -> dict:
    # Lazy import to avoid loading heavy models at startup
    from ..pipeline.tts_pipeline import TTSPipeline, PipelineConfig

    pipeline = TTSPipeline(PipelineConfig(device="auto"))
    pipeline.initialize()

    ref = Path(reference_path) if reference_path else None
    out = Path(f"output/samples/{output_name}.wav") if output_name else None

    result = pipeline.synthesize(
        text=text,
        reference_audio=ref,
        output_path=out,
        speed=speed,
    )
    final = result["final_path"]
    return {
        "audio_path": str(final),
        "duration_sec": round(result["result"].duration_sec, 3),
        "rtf": round(result["result"].rtf, 3),
    }


def _tool_get_projects() -> dict:
    from ..projects import list_projects

    projects = list_projects()
    return {"projects": [p.summary() for p in projects]}


def _tool_get_project(project_id: str) -> dict:
    from ..projects import get_project_or_404

    try:
        project = get_project_or_404(project_id)
        return project.to_dict()
    except ValueError as e:
        return {"error": str(e)}


def _tool_get_project_source(project_id: str, page: int = 0, page_size: int = 10000) -> dict:
    from ..projects import get_project_or_404

    try:
        project = get_project_or_404(project_id)
    except ValueError as e:
        return {"error": str(e)}

    text = project.source_text
    total_chars = len(text)
    start = page * page_size
    end = min(start + page_size, total_chars)
    chunk = text[start:end]

    return {
        "text": chunk,
        "page": page,
        "page_size": page_size,
        "total_chars": total_chars,
        "total_pages": (total_chars + page_size - 1) // page_size,
        "sha256": _sha256(text),
    }


def _tool_edit_project_source(project_id: str, new_text: str, expected_sha256: str) -> dict:
    import russian_tts_studio.projects.store as store_mod

    from ..projects import get_project_or_404

    try:
        project = get_project_or_404(project_id)
    except ValueError as e:
        return {"error": str(e)}

    # Concurrency check
    current_sha = _sha256(project.source_text)
    if current_sha != expected_sha256:
        return {
            "error": "SHA-256 mismatch — source text was modified since you read it",
            "expected": expected_sha256,
            "current": current_sha,
        }

    # Update source text
    conn = store_mod.get_db()
    conn.execute(
        "UPDATE projects SET source_text = ? WHERE id = ?",
        (new_text, project_id),
    )
    conn.commit()

    return {
        "ok": True,
        "sha256": _sha256(new_text),
        "render_required": True,
    }


def _tool_get_segments(project_id: str) -> dict:
    from ..projects import get_project_or_404

    try:
        project = get_project_or_404(project_id)
    except ValueError as e:
        return {"error": str(e)}

    return {
        "segments": [
            {
                "id": s.id,
                "idx": s.idx,
                "chapter_title": s.chapter_title,
                "text": s.text[:100] + ("..." if len(s.text) > 100 else ""),
                "status": s.status,
                "duration_sec": s.duration_sec,
            }
            for s in project.segments
        ],
    }


def _tool_approve_segment(project_id: str, segment_id: str) -> dict:
    from ..projects import approve_segment

    try:
        seg = approve_segment(project_id, segment_id)
        return {"id": seg.id, "status": seg.status}
    except ValueError as e:
        return {"error": str(e)}


def _tool_discard_segment(project_id: str, segment_id: str) -> dict:
    from ..projects import discard_segment

    try:
        seg = discard_segment(project_id, segment_id)
        return {"id": seg.id, "status": seg.status}
    except ValueError as e:
        return {"error": str(e)}


def _tool_regenerate_segment(
    project_id: str,
    segment_id: str,
    speed: float | None = None,
    instruct: str | None = None,
) -> dict:
    from ..projects import get_project_or_404, record_synthesis, mark_error, update_config
    from ..projects.store import update_segment_status

    try:
        project = get_project_or_404(project_id)
    except ValueError as e:
        return {"error": str(e)}

    seg = next((s for s in project.segments if s.id == segment_id), None)
    if seg is None:
        return {"error": f"Segment not found: {segment_id}"}

    # Update config
    import json
    config = json.loads(seg.config_json) if seg.config_json else {}
    if speed is not None:
        config["speed"] = speed
    if instruct:
        config["instruct"] = instruct
    update_config(project_id, segment_id, json.dumps(config))

    # Synthesise
    from ..pipeline.tts_pipeline import TTSPipeline, PipelineConfig

    pipeline = TTSPipeline(PipelineConfig(device="auto"))
    pipeline.initialize()

    audio_dir = Path("output") / "projects" / project_id / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    seg_out = audio_dir / f"seg_{seg.idx:03d}.wav"

    try:
        result = pipeline.synthesize(
            text=seg.text,
            output_path=seg_out,
            speed=config.get("speed", 0.9),
        )
        final = result["final_path"]
        metrics = result["metrics"].to_dict() if result["metrics"] else {}
        record_synthesis(
            project_id, segment_id,
            audio_path=str(final),
            wav_path=str(final),
            duration_sec=result["result"].duration_sec,
            rtf=result["result"].rtf,
            metrics=metrics,
            status="approved",
        )
        return {"id": segment_id, "status": "approved", "audio_path": str(final)}
    except Exception as e:
        mark_error(project_id, segment_id, str(e))
        return {"error": str(e)}


def _tool_rebuild_audiobook(project_id: str, output_name: str = "audiobook") -> dict:
    from ..projects import rebuild_audiobook

    try:
        out = rebuild_audiobook(project_id, output_name=output_name)
        return {"audio_path": str(out)}
    except ValueError as e:
        return {"error": str(e)}


def _tool_get_markup_help() -> dict:
    return {
        "markup_commands": {
            "{{pause 700ms}}": "Insert 700ms silence after current segment",
            "{{pause.short}}": "300ms pause preset",
            "{{pause.medium}}": "700ms pause preset",
            "{{pause.long}}": "1200ms pause preset",
            "{{pause random 500 1200}}": "Random pause between 500-1200ms",
            "{{speed 0.9}}": "Set speech speed for following text",
            "{{speed.slow}}": "0.85x speed preset",
            "{{speed.fast}}": "1.15x speed preset",
            "{{volume -3db}}": "Set volume gain in dB",
            "{{volume 80%}}": "Set volume as percentage",
            "{{volume.normalize -16}}": "Normalize to -16 LUFS",
            "{{volume.normal}}": "Reset volume to default",
            '{{chapter "Title"}}': "Mark chapter boundary",
            '{{alias "GPT" "gee pee tee"}}': "Text replacement before TTS",
            "{{reset}}": "Reset all state to defaults",
            "{{reset.audio}}": "Reset speed + volume only",
        },
        "notes": [
            "Commands are case-insensitive",
            "Smart quotes and unicode dashes are normalized",
            "Unknown commands produce warnings but don't stop generation",
            "Markup is optional — without {{...}} everything works as before",
        ],
    }


# Tool dispatcher
_TOOL_DISPATCH = {
    "server_info": lambda args: _tool_server_info(),
    "list_engines": lambda args: _tool_list_engines(),
    "list_references": lambda args: _tool_list_references(),
    "create_audiobook": lambda args: _tool_create_audiobook(
        args["name"], args["source_text"], args.get("max_chars", 200),
    ),
    "generate_audio": lambda args: _tool_generate_audio(
        args["text"], args.get("reference_path"), args.get("speed", 0.9), args.get("output_name"),
    ),
    "get_projects": lambda args: _tool_get_projects(),
    "get_project": lambda args: _tool_get_project(args["project_id"]),
    "get_project_source": lambda args: _tool_get_project_source(
        args["project_id"], args.get("page", 0), args.get("page_size", 10000),
    ),
    "edit_project_source": lambda args: _tool_edit_project_source(
        args["project_id"], args["new_text"], args["expected_sha256"],
    ),
    "get_segments": lambda args: _tool_get_segments(args["project_id"]),
    "approve_segment": lambda args: _tool_approve_segment(args["project_id"], args["segment_id"]),
    "discard_segment": lambda args: _tool_discard_segment(args["project_id"], args["segment_id"]),
    "regenerate_segment": lambda args: _tool_regenerate_segment(
        args["project_id"], args["segment_id"], args.get("speed"), args.get("instruct"),
    ),
    "rebuild_audiobook": lambda args: _tool_rebuild_audiobook(
        args["project_id"], args.get("output_name", "audiobook"),
    ),
    "get_markup_help": lambda args: _tool_get_markup_help(),
}


# ---------------------------------------------------------------------------
# MCP JSON-RPC handler
# ---------------------------------------------------------------------------

def handle_mcp_request(request: dict) -> dict:
    """Handle a single MCP JSON-RPC request.

    Supports:
    - ``initialize`` — return server capabilities
    - ``tools/list`` — return tool definitions
    - ``tools/call`` — dispatch to tool implementation

    Returns a JSON-RPC response dict.
    """
    method = request.get("method", "")
    req_id = request.get("id")
    params = request.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {
                    "name": "russian-tts-studio",
                    "version": "0.3.0",
                },
            },
        }

    if method == "notifications/initialized":
        # Notification — no response needed
        return {}

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": TOOLS},
        }

    if method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})

        dispatcher = _TOOL_DISPATCH.get(tool_name)
        if dispatcher is None:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
            }

        try:
            result = dispatcher(tool_args)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}],
                    "isError": "error" in result,
                },
            }
        except Exception as e:
            logger.exception("Tool %s failed: %s", tool_name, e)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps({"error": str(e)}, ensure_ascii=False)}],
                    "isError": True,
                },
            }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Unknown method: {method}"},
    }
