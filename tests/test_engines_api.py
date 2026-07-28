"""Tests for the ``/api/engines`` endpoint.

Verifies that the endpoint lists both VoxCPM2 and Higgs Audio, and
that Higgs is marked unavailable when the ``boson_multimodal`` package
isn't importable (the common case in environments without the upstream
repo installed).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestEnginesEndpoint:
    """``/api/engines`` — lists available TTS engines."""

    def test_lists_voxcpm(self):
        from fastapi.testclient import TestClient
        from web.app import app

        client = TestClient(app)
        resp = client.get("/api/engines")
        assert resp.status_code == 200
        body = resp.json()
        ids = [e["id"] for e in body["engines"]]
        assert "voxcpm" in ids

    def test_lists_higgs(self):
        from fastapi.testclient import TestClient
        from web.app import app

        client = TestClient(app)
        resp = client.get("/api/engines")
        body = resp.json()
        ids = [e["id"] for e in body["engines"]]
        assert "higgs" in ids

    def test_voxcpm_always_available(self):
        from fastapi.testclient import TestClient
        from web.app import app

        client = TestClient(app)
        resp = client.get("/api/engines")
        body = resp.json()
        voxcpm = next(e for e in body["engines"] if e["id"] == "voxcpm")
        # VoxCPM2 doesn't have an "available" field (default True) OR
        # it's explicitly True — either way, it's selectable.
        assert voxcpm.get("available", True) is True

    def test_higgs_availability_reflects_import(self, monkeypatch):
        """When ``boson_multimodal`` can't be imported, Higgs is marked
        ``available=False`` so the UI greys it out."""
        import builtins

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "boson_multimodal":
                raise ImportError("No module named 'boson_multimodal'")
            if name == "importlib":
                # importlib itself must import normally; we only block
                # the boson_multimodal lookup that the endpoint does via
                # importlib.import_module.
                return real_import(name, *args, **kwargs)
            return real_import(name, *args, **kwargs)

        # Patch importlib.import_module to raise for boson_multimodal.
        import importlib
        real_import_module = importlib.import_module

        def _fake_import_module(name, package=None):
            if name == "boson_multimodal":
                raise ImportError("No module named 'boson_multimodal'")
            return real_import_module(name, package)

        monkeypatch.setattr(importlib, "import_module", _fake_import_module)

        from fastapi.testclient import TestClient
        from web.app import app

        client = TestClient(app)
        resp = client.get("/api/engines")
        body = resp.json()
        higgs = next(e for e in body["engines"] if e["id"] == "higgs")
        assert higgs.get("available") is False

    def test_engines_have_labels_and_descriptions(self):
        from fastapi.testclient import TestClient
        from web.app import app

        client = TestClient(app)
        resp = client.get("/api/engines")
        body = resp.json()
        for eng in body["engines"]:
            assert eng.get("label"), f"engine {eng['id']} missing label"
            assert eng.get("description"), f"engine {eng['id']} missing description"

    def test_active_engine_in_response(self):
        from fastapi.testclient import TestClient
        from web.app import app

        client = TestClient(app)
        resp = client.get("/api/engines")
        body = resp.json()
        # active is the cached pipeline engine (defaults to voxcpm)
        assert body["active"] in ("voxcpm", "higgs")
        assert body["default"] == "voxcpm"