"""Tests for file importers and long-form API endpoints."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from russian_tts_studio.text import (  # noqa: E402
    detect_chapters,
    import_file,
    import_md,
    import_txt,
)


class TestImporters:
    """Tests for .txt / .md / .docx import."""

    def test_import_txt_utf8(self, tmp_path):
        p = tmp_path / "test.txt"
        p.write_text("Привет, мир! Тест на русском.", encoding="utf-8")
        text, fmt = import_txt(p)
        assert "Привет" in text
        assert fmt == "txt"

    def test_import_txt_cp1251_fallback(self, tmp_path):
        p = tmp_path / "legacy.txt"
        p.write_bytes("Привет из Windows".encode("cp1251"))
        text, fmt = import_txt(p)
        assert "Привет" in text
        assert fmt == "txt"

    def test_import_md(self, tmp_path):
        p = tmp_path / "test.md"
        p.write_text("# Глава 1\n\nТекст урока.\n", encoding="utf-8")
        text, fmt = import_md(p)
        assert "# Глава 1" in text
        assert fmt == "md"

    def test_import_file_dispatch_txt(self, tmp_path):
        p = tmp_path / "x.txt"
        p.write_text("текст", encoding="utf-8")
        text, fmt = import_file(p)
        assert fmt == "txt"

    def test_import_file_dispatch_md(self, tmp_path):
        p = tmp_path / "x.md"
        p.write_text("# H\nтекст", encoding="utf-8")
        text, fmt = import_file(p)
        assert fmt == "md"

    def test_import_file_dispatch_markdown_ext(self, tmp_path):
        p = tmp_path / "x.markdown"
        p.write_text("# H", encoding="utf-8")
        text, fmt = import_file(p)
        assert fmt == "md"

    def test_import_file_unsupported_ext(self, tmp_path):
        p = tmp_path / "x.pdf"
        p.write_bytes(b"%PDF-1.4")
        with pytest.raises(ValueError, match="Unsupported"):
            import_file(p)

    def test_import_docx_missing_package(self, tmp_path):
        """If python-docx is not installed, raise ImportError with a hint."""
        # Create a minimal fake .docx (zip) — the importer fails at the
        # import level, not at file parsing.
        p = tmp_path / "fake.docx"
        p.write_bytes(b"PK\x03\x04")  # zip magic bytes
        try:
            import docx  # noqa: F401
            pytest.skip("python-docx is installed — can't test the missing-package path")
        except ImportError:
            with pytest.raises(ImportError, match="python-docx"):
                import_file(p)

    def test_import_then_detect_chapters(self, tmp_path):
        """Round-trip: import .md → detect chapters."""
        p = tmp_path / "book.md"
        p.write_text(
            "# Глава 1\nТекст первой главы.\n\n# Глава 2\nТекст второй.\n",
            encoding="utf-8",
        )
        text, _ = import_file(p)
        chapters = detect_chapters(text)
        assert len(chapters) == 2
        assert chapters[0].title == "Глава 1"
        assert chapters[1].title == "Глава 2"


class TestLongFormAPI:
    """Tests for /api/import and /api/longform/chunk endpoints."""

    def test_import_endpoint_txt(self, tmp_path):
        from fastapi.testclient import TestClient

        from web.app import app

        client = TestClient(app)
        # Create a temp txt file to upload
        txt_content = "# Урок 1\n\nПривет, мир! Это тест.\n"
        resp = client.post(
            "/api/import",
            files={"file": ("test.txt", txt_content.encode("utf-8"), "text/plain")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["source_format"] == "txt"
        assert "Привет" in body["text"]
        assert len(body["chapters"]) >= 1

    def test_import_endpoint_md(self):
        from fastapi.testclient import TestClient

        from web.app import app

        client = TestClient(app)
        md = "# Глава 1\n\nТекст.\n\n# Глава 2\n\nЕщё текст.\n"
        resp = client.post(
            "/api/import",
            files={"file": ("book.md", md.encode("utf-8"), "text/markdown")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["source_format"] == "md"
        assert len(body["chapters"]) == 2
        assert body["chapters"][0]["title"] == "Глава 1"

    def test_import_endpoint_unsupported_ext(self):
        from fastapi.testclient import TestClient

        from web.app import app

        client = TestClient(app)
        resp = client.post(
            "/api/import",
            files={"file": ("x.pdf", b"%PDF-1.4", "application/pdf")},
        )
        assert resp.status_code == 400
        assert "Unsupported" in resp.json()["detail"]

    def test_longform_chunk_endpoint(self):
        from fastapi.testclient import TestClient

        from web.app import app

        client = TestClient(app)
        text = "# Глава 1\nПервый абзац. Второе предложение.\n\nВторой абзац.\n\n# Глава 2\nТекст второй главы."
        resp = client.post(
            "/api/longform/chunk",
            data={"text": text, "max_chars": "200", "max_sentences": "4"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_chunks"] >= 2
        titles = {c["chapter_title"] for c in body["chunks"] if c["chapter_title"]}
        assert "Глава 1" in titles
        assert "Глава 2" in titles

    def test_longform_chunk_endpoint_plain_text(self):
        from fastapi.testclient import TestClient

        from web.app import app

        client = TestClient(app)
        resp = client.post(
            "/api/longform/chunk",
            data={"text": "Просто короткий текст без глав."},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_chunks"] == 1
        assert body["chunks"][0]["chapter_title"] == ""