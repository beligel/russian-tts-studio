"""Tests for the projects package: store, manager, and API endpoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestProjectModels:
    """Unit tests for Project/Segment/ChapterMarker dataclasses."""

    def test_project_summary(self):
        from russian_tts_studio.projects.models import Project, Segment

        p = Project(id="test1", name="Test", source_text="Hello")
        p.segments = [
            Segment(id="s1", status="approved"),
            Segment(id="s2", status="pending"),
            Segment(id="s3", status="error"),
        ]
        s = p.summary()
        assert s["total_segments"] == 3
        assert s["approved"] == 1
        assert s["pending"] == 1
        assert s["errors"] == 1

    def test_project_to_dict(self):
        from russian_tts_studio.projects.models import Project, Segment, ChapterMarker

        p = Project(id="test2", name="Book", source_text="Text")
        p.chapters = [ChapterMarker(title="Ch1", idx=0)]
        p.segments = [Segment(id="s1", idx=0, text="Hello", status="approved")]
        d = p.to_dict()
        assert d["name"] == "Book"
        assert len(d["chapters"]) == 1
        assert len(d["segments"]) == 1
        assert d["segments"][0]["text"] == "Hello"

    def test_new_id_is_unique(self):
        from russian_tts_studio.projects.models import _new_id

        ids = {_new_id() for _ in range(100)}
        assert len(ids) == 100


class TestProjectStore:
    """Integration tests for the SQLite store layer (uses temp DB)."""

    @pytest.fixture(autouse=True)
    def _tmp_db(self, tmp_path):
        """Create a fresh in-memory DB for each test."""
        import russian_tts_studio.projects.store as store_mod

        self._db_path = tmp_path / "test.db"
        self._conn = None

        # Patch get_db to return a fresh connection to tmp_path
        import sqlite3

        self._conn = sqlite3.connect(str(self._db_path), timeout=10, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(store_mod._SCHEMA)
        self._conn.executescript(store_mod._TRIGGERS)

        # Monkey-patch get_db for this test
        orig = store_mod.get_db
        store_mod.get_db = lambda *a, **kw: self._conn
        yield
        store_mod.get_db = orig
        self._conn.close()

    def test_create_and_load_project(self):
        from russian_tts_studio.projects.models import Project
        import russian_tts_studio.projects.store as store_mod

        p = Project(id="proj1", name="My Book", source_text="Hello world")
        store_mod.create_project(p, db=self._conn)
        loaded = store_mod.get_project("proj1", db=self._conn)
        assert loaded is not None
        assert loaded.name == "My Book"
        assert loaded.source_text == "Hello world"

    def test_list_projects(self):
        from russian_tts_studio.projects.models import Project
        import russian_tts_studio.projects.store as store_mod

        store_mod.create_project(Project(id="p1", name="A"), db=self._conn)
        store_mod.create_project(Project(id="p2", name="B"), db=self._conn)
        projects = store_mod.list_projects(db=self._conn)
        assert len(projects) == 2

    def test_delete_project_cascades(self):
        from russian_tts_studio.projects.models import (
            Project, Segment, ChapterMarker,
        )
        import russian_tts_studio.projects.store as store_mod

        p = Project(id="del1", name="Delete Me")
        store_mod.create_project(p, db=self._conn)
        store_mod.insert_segments(
            [Segment(id="s1", project_id="del1", idx=0, text="Hi")],
            db=self._conn,
        )
        store_mod.insert_chapters(
            [ChapterMarker(id="c1", project_id="del1", title="Ch1", idx=0)],
            db=self._conn,
        )
        assert store_mod.delete_project("del1", db=self._conn) is True
        assert store_mod.get_project("del1", db=self._conn) is None
        assert store_mod.get_segments("del1", db=self._conn) == []
        assert store_mod.get_chapters("del1", db=self._conn) == []

    def test_segment_crud(self):
        from russian_tts_studio.projects.models import Project, Segment
        import russian_tts_studio.projects.store as store_mod

        store_mod.create_project(Project(id="p1", name="X"), db=self._conn)
        seg = Segment(id="s1", project_id="p1", idx=0, text="Test")
        store_mod.insert_segments([seg], db=self._conn)

        loaded = store_mod.get_segment("s1", db=self._conn)
        assert loaded is not None
        assert loaded.text == "Test"
        assert loaded.status == "pending"

        store_mod.update_segment_status("s1", status="approved", db=self._conn)
        loaded = store_mod.get_segment("s1", db=self._conn)
        assert loaded.status == "approved"

        store_mod.update_segment_audio(
            "s1", audio_path="/tmp/out.wav", duration_sec=1.5, rtf=0.3,
            metrics_json='{"wer": 0.05}', db=self._conn,
        )
        loaded = store_mod.get_segment("s1", db=self._conn)
        assert loaded.audio_path == "/tmp/out.wav"
        assert loaded.duration_sec == 1.5
        assert json.loads(loaded.metrics_json)["wer"] == 0.05

    def test_load_project_full(self):
        from russian_tts_studio.projects.models import (
            Project, Segment, ChapterMarker,
        )
        import russian_tts_studio.projects.store as store_mod

        store_mod.create_project(Project(id="full1", name="Full"), db=self._conn)
        store_mod.insert_chapters(
            [ChapterMarker(id="c1", project_id="full1", title="Ch1", idx=0)],
            db=self._conn,
        )
        store_mod.insert_segments(
            [Segment(id="s1", project_id="full1", idx=0, text="Text")],
            db=self._conn,
        )
        full = store_mod.load_project_full("full1", db=self._conn)
        assert full is not None
        assert len(full.chapters) == 1
        assert len(full.segments) == 1
        assert full.chapters[0].title == "Ch1"
        assert full.segments[0].text == "Text"


class TestProjectManager:
    """Integration tests for manager operations (uses temp dir)."""

    @pytest.fixture(autouse=True)
    def _tmp_projects(self, tmp_path, monkeypatch):
        """Redirect project storage to a temp directory."""
        import russian_tts_studio.projects.manager as mgr
        import russian_tts_studio.projects.store as store_mod
        import sqlite3

        # Create a fresh DB in tmp_path
        db_path = tmp_path / "projects.db"
        conn = sqlite3.connect(str(db_path), timeout=10, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(store_mod._SCHEMA)
        conn.executescript(store_mod._TRIGGERS)

        orig_get_db = store_mod.get_db
        store_mod.get_db = lambda *a, **kw: conn
        monkeypatch.setattr(mgr, "_PROJECTS_DIR", tmp_path / "projects")
        monkeypatch.setattr(mgr, "_OUTPUT_DIR", tmp_path)

        self.conn = conn
        yield
        store_mod.get_db = orig_get_db
        conn.close()

    def test_create_project_with_chapters(self):
        from russian_tts_studio.projects.manager import create_project

        text = (
            "# Глава 1\n"
            "Первый абзац. Второе предложение.\n\n"
            "# Глава 2\n"
            "Текст второй главы."
        )
        project = create_project("My Book", text, max_chars=200)
        assert project.total_segments >= 2
        assert len(project.chapters) == 2
        assert project.chapters[0].title == "Глава 1"
        assert project.chapters[1].title == "Глава 2"
        # All segments start as pending
        assert all(s.status == "pending" for s in project.segments)

    def test_approve_and_discard(self):
        from russian_tts_studio.projects.manager import (
            create_project, approve_segment, discard_segment,
        )

        project = create_project("Test", "Hello world.")
        seg_id = project.segments[0].id
        seg = approve_segment(project.id, seg_id)
        assert seg.status == "approved"
        seg = discard_segment(project.id, seg_id)
        assert seg.status == "needs_retry"

    def test_record_synthesis(self):
        from russian_tts_studio.projects.manager import (
            create_project, record_synthesis,
        )

        project = create_project("Test", "Hello.")
        seg_id = project.segments[0].id
        seg = record_synthesis(
            project.id, seg_id,
            audio_path="/tmp/test.wav",
            duration_sec=1.2,
            rtf=0.3,
            metrics={"wer": 0.05, "cer": 0.02},
        )
        assert seg.status == "approved"
        assert seg.audio_path == "/tmp/test.wav"
        assert seg.duration_sec == 1.2

    def test_list_projects(self):
        from russian_tts_studio.projects.manager import create_project, list_projects

        create_project("Book 1", "Text 1.")
        create_project("Book 2", "Text 2.")
        projects = list_projects()
        assert len(projects) == 2

    def test_rebuild_no_segments_raises(self):
        from russian_tts_studio.projects.manager import create_project, rebuild_audiobook

        project = create_project("Empty", "Hello.")
        with pytest.raises(ValueError, match="No approved"):
            rebuild_audiobook(project.id)


class TestProjectsAPI:
    """Integration tests for /api/projects endpoints."""

    @pytest.fixture(autouse=True)
    def _tmp_projects(self, tmp_path, monkeypatch):
        """Redirect project storage to a temp directory."""
        import russian_tts_studio.projects.manager as mgr
        import russian_tts_studio.projects.store as store_mod
        import sqlite3

        db_path = tmp_path / "projects.db"
        conn = sqlite3.connect(str(db_path), timeout=10, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(store_mod._SCHEMA)
        conn.executescript(store_mod._TRIGGERS)

        orig_get_db = store_mod.get_db
        store_mod.get_db = lambda *a, **kw: conn
        monkeypatch.setattr(mgr, "_PROJECTS_DIR", tmp_path / "projects")
        monkeypatch.setattr(mgr, "_OUTPUT_DIR", tmp_path)

        yield
        store_mod.get_db = orig_get_db
        conn.close()

    def test_create_and_get_project(self):
        from fastapi.testclient import TestClient
        from web.app import app

        client = TestClient(app)
        resp = client.post(
            "/api/projects",
            data={"name": "My Book", "source_text": "# Глава 1\nТекст.", "max_chars": "200"},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["name"] == "My Book"
        pid = body["id"]

        resp = client.get(f"/api/projects/{pid}")
        assert resp.status_code == 200
        assert resp.json()["id"] == pid
        assert len(resp.json()["segments"]) >= 1

    def test_list_projects(self):
        from fastapi.testclient import TestClient
        from web.app import app

        client = TestClient(app)
        client.post("/api/projects", data={"name": "A", "source_text": "Text A."})
        client.post("/api/projects", data={"name": "B", "source_text": "Text B."})
        resp = client.get("/api/projects")
        assert resp.status_code == 200
        assert len(resp.json()["projects"]) == 2

    def test_delete_project(self):
        from fastapi.testclient import TestClient
        from web.app import app

        client = TestClient(app)
        resp = client.post("/api/projects", data={"name": "Del", "source_text": "Bye."})
        pid = resp.json()["id"]
        resp = client.delete(f"/api/projects/{pid}")
        assert resp.status_code == 200
        resp = client.get(f"/api/projects/{pid}")
        assert resp.status_code == 404

    def test_approve_and_discard_segment(self):
        from fastapi.testclient import TestClient
        from web.app import app

        client = TestClient(app)
        resp = client.post("/api/projects", data={"name": "Seg", "source_text": "Hello."})
        pid = resp.json()["id"]
        seg_id = resp.json()["segments"][0]["id"]

        resp = client.post(f"/api/projects/{pid}/segments/{seg_id}/approve")
        assert resp.status_code == 200
        assert resp.json()["status"] == "approved"

        resp = client.post(f"/api/projects/{pid}/segments/{seg_id}/discard")
        assert resp.status_code == 200
        assert resp.json()["status"] == "needs_retry"

    def test_get_segments(self):
        from fastapi.testclient import TestClient
        from web.app import app

        client = TestClient(app)
        resp = client.post("/api/projects", data={"name": "Segs", "source_text": "Text."})
        pid = resp.json()["id"]
        resp = client.get(f"/api/projects/{pid}/segments")
        assert resp.status_code == 200
        assert len(resp.json()["segments"]) >= 1

    def test_nonexistent_project_404(self):
        from fastapi.testclient import TestClient
        from web.app import app

        client = TestClient(app)
        resp = client.get("/api/projects/nonexistent")
        assert resp.status_code == 404

    def test_empty_source_text_rejected(self):
        """Empty source text should be rejected (400 or 422)."""
        from fastapi.testclient import TestClient
        from web.app import app

        client = TestClient(app)
        resp = client.post("/api/projects", data={"name": "Empty", "source_text": ""})
        assert resp.status_code >= 400