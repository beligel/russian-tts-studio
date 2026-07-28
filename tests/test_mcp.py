"""Tests for MCP server, stdio bridge, and config generator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestMCPServer:
    """Tests for the MCP JSON-RPC handler."""

    def test_initialize(self):
        from russian_tts_studio.mcp.server import handle_mcp_request

        resp = handle_mcp_request({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 1
        assert "serverInfo" in resp["result"]
        assert resp["result"]["serverInfo"]["name"] == "russian-tts-studio"

    def test_tools_list(self):
        from russian_tts_studio.mcp.server import handle_mcp_request, TOOLS

        resp = handle_mcp_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        assert "tools" in resp["result"]
        assert len(resp["result"]["tools"]) == len(TOOLS)

    def test_server_info_tool(self):
        from russian_tts_studio.mcp.server import handle_mcp_request

        resp = handle_mcp_request({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "server_info", "arguments": {}},
        })
        content = resp["result"]["content"][0]["text"]
        data = json.loads(content)
        assert data["name"] == "Russian TTS Studio"
        assert "voxcpm" in data["engines"]

    def test_list_engines_tool(self):
        from russian_tts_studio.mcp.server import handle_mcp_request

        resp = handle_mcp_request({
            "jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": {"name": "list_engines", "arguments": {}},
        })
        content = resp["result"]["content"][0]["text"]
        data = json.loads(content)
        assert len(data["engines"]) == 3

    def test_list_references_tool(self):
        from russian_tts_studio.mcp.server import handle_mcp_request

        resp = handle_mcp_request({
            "jsonrpc": "2.0", "id": 5, "method": "tools/call",
            "params": {"name": "list_references", "arguments": {}},
        })
        assert resp["result"]["content"][0]["type"] == "text"

    def test_get_markup_help_tool(self):
        from russian_tts_studio.mcp.server import handle_mcp_request

        resp = handle_mcp_request({
            "jsonrpc": "2.0", "id": 6, "method": "tools/call",
            "params": {"name": "get_markup_help", "arguments": {}},
        })
        content = resp["result"]["content"][0]["text"]
        data = json.loads(content)
        assert "{{pause 700ms}}" in data["markup_commands"]

    def test_get_projects_tool(self):
        from russian_tts_studio.mcp.server import handle_mcp_request

        resp = handle_mcp_request({
            "jsonrpc": "2.0", "id": 7, "method": "tools/call",
            "params": {"name": "get_projects", "arguments": {}},
        })
        content = resp["result"]["content"][0]["text"]
        data = json.loads(content)
        assert "projects" in data

    def test_unknown_tool(self):
        from russian_tts_studio.mcp.server import handle_mcp_request

        resp = handle_mcp_request({
            "jsonrpc": "2.0", "id": 8, "method": "tools/call",
            "params": {"name": "nonexistent_tool", "arguments": {}},
        })
        assert "error" in resp

    def test_unknown_method(self):
        from russian_tts_studio.mcp.server import handle_mcp_request

        resp = handle_mcp_request({"jsonrpc": "2.0", "id": 9, "method": "unknown/method", "params": {}})
        assert "error" in resp

    def test_notifications_initialized(self):
        from russian_tts_studio.mcp.server import handle_mcp_request

        resp = handle_mcp_request({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        assert resp == {}

    def test_create_audiobook_tool(self):
        from russian_tts_studio.mcp.server import handle_mcp_request

        resp = handle_mcp_request({
            "jsonrpc": "2.0", "id": 10, "method": "tools/call",
            "params": {
                "name": "create_audiobook",
                "arguments": {"name": "MCP Test", "source_text": "# Глава 1\nТекст."},
            },
        })
        content = resp["result"]["content"][0]["text"]
        data = json.loads(content)
        assert "project_id" in data
        assert data["total_segments"] >= 1

    def test_get_project_source_tool(self):
        from russian_tts_studio.mcp.server import handle_mcp_request, _TOOL_DISPATCH

        # Create a project first
        create_resp = handle_mcp_request({
            "jsonrpc": "2.0", "id": 11, "method": "tools/call",
            "params": {
                "name": "create_audiobook",
                "arguments": {"name": "Source Test", "source_text": "Hello world test text."},
            },
        })
        project_id = json.loads(create_resp["result"]["content"][0]["text"])["project_id"]

        # Get source
        resp = handle_mcp_request({
            "jsonrpc": "2.0", "id": 12, "method": "tools/call",
            "params": {
                "name": "get_project_source",
                "arguments": {"project_id": project_id},
            },
        })
        content = resp["result"]["content"][0]["text"]
        data = json.loads(content)
        assert "Hello world" in data["text"]
        assert "sha256" in data

    def test_edit_project_source_sha_mismatch(self):
        from russian_tts_studio.mcp.server import handle_mcp_request

        # Create a project
        create_resp = handle_mcp_request({
            "jsonrpc": "2.0", "id": 13, "method": "tools/call",
            "params": {
                "name": "create_audiobook",
                "arguments": {"name": "SHA Test", "source_text": "Original text."},
            },
        })
        project_id = json.loads(create_resp["result"]["content"][0]["text"])["project_id"]

        # Edit with wrong SHA
        resp = handle_mcp_request({
            "jsonrpc": "2.0", "id": 14, "method": "tools/call",
            "params": {
                "name": "edit_project_source",
                "arguments": {
                    "project_id": project_id,
                    "new_text": "Modified text.",
                    "expected_sha256": "wrong_sha",
                },
            },
        })
        content = resp["result"]["content"][0]["text"]
        data = json.loads(content)
        assert "error" in data
        assert "SHA-256 mismatch" in data["error"]

    def test_approve_segment_tool(self):
        from russian_tts_studio.mcp.server import handle_mcp_request

        # Create project
        create_resp = handle_mcp_request({
            "jsonrpc": "2.0", "id": 15, "method": "tools/call",
            "params": {
                "name": "create_audiobook",
                "arguments": {"name": "Approve Test", "source_text": "Test text."},
            },
        })
        project_id = json.loads(create_resp["result"]["content"][0]["text"])["project_id"]

        # Get segments
        seg_resp = handle_mcp_request({
            "jsonrpc": "2.0", "id": 16, "method": "tools/call",
            "params": {"name": "get_segments", "arguments": {"project_id": project_id}},
        })
        segments = json.loads(seg_resp["result"]["content"][0]["text"])["segments"]
        seg_id = segments[0]["id"]

        # Approve
        resp = handle_mcp_request({
            "jsonrpc": "2.0", "id": 17, "method": "tools/call",
            "params": {
                "name": "approve_segment",
                "arguments": {"project_id": project_id, "segment_id": seg_id},
            },
        })
        content = resp["result"]["content"][0]["text"]
        data = json.loads(content)
        assert data["status"] == "approved"


class TestMCPConfig:
    """Tests for config generator."""

    def test_generate_config(self):
        from russian_tts_studio.mcp import generate_config

        config = generate_config()
        assert "claude_desktop" in config
        assert "codex" in config
        assert "mcpServers" in config["claude_desktop"]
        assert "russian-tts-studio" in config["claude_desktop"]["mcpServers"]

    def test_generate_config_custom_path(self):
        from russian_tts_studio.mcp import generate_config

        config = generate_config(project_path="/custom/path")
        server = config["claude_desktop"]["mcpServers"]["russian-tts-studio"]
        assert server["cwd"] == "/custom/path"


class TestMCPAPI:
    """Tests for /mcp HTTP endpoint."""

    def test_mcp_initialize(self):
        from fastapi.testclient import TestClient
        from web.app import app

        client = TestClient(app)
        resp = client.post("/mcp", json={
            "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {},
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["result"]["serverInfo"]["name"] == "russian-tts-studio"

    def test_mcp_tools_list(self):
        from fastapi.testclient import TestClient
        from web.app import app

        client = TestClient(app)
        resp = client.post("/mcp", json={
            "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {},
        })
        assert resp.status_code == 200
        assert len(resp.json()["result"]["tools"]) >= 10

    def test_mcp_tools_call(self):
        from fastapi.testclient import TestClient
        from web.app import app

        client = TestClient(app)
        resp = client.post("/mcp", json={
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "server_info", "arguments": {}},
        })
        assert resp.status_code == 200
        content = resp.json()["result"]["content"][0]["text"]
        assert "Russian TTS Studio" in content

    def test_mcp_config_endpoint(self):
        from fastapi.testclient import TestClient
        from web.app import app

        client = TestClient(app)
        resp = client.get("/mcp/config")
        assert resp.status_code == 200
        body = resp.json()
        assert "claude_desktop" in body
        assert "codex" in body