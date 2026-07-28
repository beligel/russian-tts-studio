"""SQLite persistence layer for TTS projects.

Schema is auto-created on first use via ``get_db()`` / ``_ensure_schema``.
All writes go through helper functions that commit immediately — there's
no long-lived transaction or ORM session to leak.

The database lives at ``<output_dir>/projects.db`` (default:
``output/projects.db``). Each project's audio files live under
``<output_dir>/<project_id>/`` alongside the database.

Design notes:
- No ORM (sqlite3 stdlib only) — the schema is 3 tables; an ORM adds
  import weight and hides the queries.
- TEXT primary keys (12-char hex) — avoids integer-ID enumeration and
  makes project IDs opaque in URLs.
- ``updated_at`` is set on every write via a trigger.
- Foreign keys are ON DELETE CASCADE so ``DELETE FROM projects`` cleans
  up segments + chapters in one shot.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .models import ChapterMarker, Project, Segment, _now

logger = logging.getLogger(__name__)

_DB_LOCK = threading.Lock()

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    source_text TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);

CREATE TABLE IF NOT EXISTS segments (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL DEFAULT 0,
    chapter_title TEXT NOT NULL DEFAULT '',
    text TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    audio_path TEXT,
    wav_path TEXT,
    duration_sec REAL,
    rtf REAL,
    metrics_json TEXT NOT NULL DEFAULT '{}',
    config_json TEXT NOT NULL DEFAULT '{}',
    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    UNIQUE(project_id, idx)
);

CREATE TABLE IF NOT EXISTS chapters (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    title TEXT NOT NULL DEFAULT '',
    char_start INTEGER NOT NULL DEFAULT 0,
    char_end INTEGER NOT NULL DEFAULT 0,
    idx INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_segments_project ON segments(project_id);
CREATE INDEX IF NOT EXISTS idx_chapters_project ON chapters(project_id);
"""

# Trigger: auto-touch updated_at on projects and segments.
_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS trg_projects_updated
AFTER UPDATE ON projects
BEGIN UPDATE projects SET updated_at = datetime('now', 'localtime') WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_segments_updated
AFTER UPDATE ON segments
BEGIN UPDATE segments SET updated_at = datetime('now', 'localtime') WHERE id = NEW.id;
END;
"""


def get_db_path(output_dir: str | Path = "output") -> Path:
    return Path(output_dir) / "projects.db"


def get_db(output_dir: str | Path = "output") -> sqlite3.Connection:
    """Return a thread-safe connection to the projects database.

    The connection has WAL mode and foreign keys enabled. Callers should
    NOT call ``conn.close()`` — the connection is cached per-thread.
    """
    db_path = get_db_path(output_dir)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Thread-local connection cache.
    thread_id = threading.current_thread().ident
    key = (str(db_path), thread_id)
    conn = _thread_conns.get(key)
    if conn is not None:
        return conn

    with _DB_LOCK:
        conn = sqlite3.connect(str(db_path), timeout=10, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(_SCHEMA)
        conn.executescript(_TRIGGERS)
        _thread_conns[key] = conn
        logger.info("Opened projects DB: %s", db_path)
        return conn


_thread_conns: dict[tuple[str, int], sqlite3.Connection] = {}


# ---------------------------------------------------------------------------
# Row ↔ dataclass helpers
# ---------------------------------------------------------------------------

def _row_to_segment(row: sqlite3.Row) -> Segment:
    return Segment(
        id=row["id"],
        project_id=row["project_id"],
        idx=row["idx"],
        chapter_title=row["chapter_title"],
        text=row["text"],
        status=row["status"],
        audio_path=row["audio_path"],
        wav_path=row["wav_path"],
        duration_sec=row["duration_sec"],
        rtf=row["rtf"],
        metrics_json=row["metrics_json"] or "{}",
        config_json=row["config_json"] or "{}",
        error_message=row["error_message"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_chapter(row: sqlite3.Row) -> ChapterMarker:
    return ChapterMarker(
        id=row["id"],
        project_id=row["project_id"],
        title=row["title"],
        char_start=row["char_start"],
        char_end=row["char_end"],
        idx=row["idx"],
    )


def _row_to_project(row: sqlite3.Row) -> Project:
    return Project(
        id=row["id"],
        name=row["name"],
        source_text=row["source_text"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


# ---------------------------------------------------------------------------
# CRUD — Projects
# ---------------------------------------------------------------------------

def create_project(project: Project, db: sqlite3.Connection | None = None) -> Project:
    """Insert a new project row. Returns the project with timestamps."""
    conn = db or get_db()
    project.created_at = _now()
    project.updated_at = project.created_at
    conn.execute(
        "INSERT INTO projects (id, name, source_text, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (project.id, project.name, project.source_text,
         project.created_at, project.updated_at),
    )
    conn.commit()
    return project


def get_project(project_id: str, db: sqlite3.Connection | None = None) -> Optional[Project]:
    """Load a project by ID (without segments/chapters — call
    ``load_project_full`` for the complete picture)."""
    conn = db or get_db()
    row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    if row is None:
        return None
    return _row_to_project(row)


def list_projects(db: sqlite3.Connection | None = None) -> list[Project]:
    """List all projects, ordered by most-recently-updated."""
    conn = db or get_db()
    rows = conn.execute(
        "SELECT * FROM projects ORDER BY updated_at DESC"
    ).fetchall()
    return [_row_to_project(r) for r in rows]


def delete_project(project_id: str, db: sqlite3.Connection | None = None) -> bool:
    """Delete a project and all its segments/chapters (CASCADE).
    Returns True if a row was deleted."""
    conn = db or get_db()
    cur = conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    conn.commit()
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# CRUD — Segments
# ---------------------------------------------------------------------------

def insert_segments(segments: list[Segment], db: sqlite3.Connection | None = None) -> None:
    """Bulk-insert segments (all belong to the same project)."""
    conn = db or get_db()
    conn.executemany(
        "INSERT INTO segments "
        "(id, project_id, idx, chapter_title, text, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (s.id, s.project_id, s.idx, s.chapter_title, s.text,
             s.status, s.created_at or _now(), s.updated_at or _now())
            for s in segments
        ],
    )
    conn.commit()


def get_segments(project_id: str, db: sqlite3.Connection | None = None) -> list[Segment]:
    """Load all segments for a project, ordered by idx."""
    conn = db or get_db()
    rows = conn.execute(
        "SELECT * FROM segments WHERE project_id = ? ORDER BY idx",
        (project_id,),
    ).fetchall()
    return [_row_to_segment(r) for r in rows]


def get_segment(segment_id: str, db: sqlite3.Connection | None = None) -> Optional[Segment]:
    """Load a single segment by ID."""
    conn = db or get_db()
    row = conn.execute("SELECT * FROM segments WHERE id = ?", (segment_id,)).fetchone()
    return _row_to_segment(row) if row else None


def update_segment_status(
    segment_id: str,
    status: str,
    error_message: str | None = None,
    db: sqlite3.Connection | None = None,
) -> None:
    """Update a segment's status (and optional error message)."""
    conn = db or get_db()
    conn.execute(
        "UPDATE segments SET status = ?, error_message = ? WHERE id = ?",
        (status, error_message, segment_id),
    )
    conn.commit()


def update_segment_audio(
    segment_id: str,
    audio_path: str | None = None,
    wav_path: str | None = None,
    duration_sec: float | None = None,
    rtf: float | None = None,
    metrics_json: str | None = None,
    status: str | None = None,
    db: sqlite3.Connection | None = None,
) -> None:
    """Update a segment's audio output and metrics after synthesis."""
    conn = db or get_db()
    sets: list[str] = []
    vals: list = []
    if audio_path is not None:
        sets.append("audio_path = ?")
        vals.append(audio_path)
    if wav_path is not None:
        sets.append("wav_path = ?")
        vals.append(wav_path)
    if duration_sec is not None:
        sets.append("duration_sec = ?")
        vals.append(duration_sec)
    if rtf is not None:
        sets.append("rtf = ?")
        vals.append(rtf)
    if metrics_json is not None:
        sets.append("metrics_json = ?")
        vals.append(metrics_json)
    if status is not None:
        sets.append("status = ?")
        vals.append(status)
    if not sets:
        return
    vals.append(segment_id)
    conn.execute(f"UPDATE segments SET {', '.join(sets)} WHERE id = ?", vals)
    conn.commit()


def update_segment_config(
    segment_id: str,
    config_json: str,
    db: sqlite3.Connection | None = None,
) -> None:
    """Update a segment's regeneration config."""
    conn = db or get_db()
    conn.execute(
        "UPDATE segments SET config_json = ? WHERE id = ?",
        (config_json, segment_id),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# CRUD — Chapters
# ---------------------------------------------------------------------------

def insert_chapters(chapters: list[ChapterMarker], db: sqlite3.Connection | None = None) -> None:
    """Bulk-insert chapter markers."""
    conn = db or get_db()
    conn.executemany(
        "INSERT INTO chapters (id, project_id, title, char_start, char_end, idx) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            (c.id, c.project_id, c.title, c.char_start, c.char_end, c.idx)
            for c in chapters
        ],
    )
    conn.commit()


def get_chapters(project_id: str, db: sqlite3.Connection | None = None) -> list[ChapterMarker]:
    """Load all chapters for a project, ordered by idx."""
    conn = db or get_db()
    rows = conn.execute(
        "SELECT * FROM chapters WHERE project_id = ? ORDER BY idx",
        (project_id,),
    ).fetchall()
    return [_row_to_chapter(r) for r in rows]


# ---------------------------------------------------------------------------
# Full project loader
# ---------------------------------------------------------------------------

def load_project_full(project_id: str, db: sqlite3.Connection | None = None) -> Optional[Project]:
    """Load a project with its chapters and segments populated."""
    project = get_project(project_id, db)
    if project is None:
        return None
    conn = db or get_db()
    project.chapters = get_chapters(project_id, conn)
    project.segments = get_segments(project_id, conn)
    return project
