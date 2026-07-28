"""Project data models — dataclasses for projects, segments, chapters.

These are pure data containers with no database logic. The SQLite layer
in ``store.py`` reads/writes these; the manager in ``manager.py``
orchestrates business logic on top.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class ChapterMarker:
    """A chapter detected in the project source text."""

    id: str = field(default_factory=_new_id)
    project_id: str = ""
    title: str = ""
    char_start: int = 0
    char_end: int = 0
    idx: int = 0


@dataclass
class Segment:
    """One TTS-synthesised segment within a project.

    ``status`` lifecycle:
        pending → approved | needs_review | error
        needs_review → approved | needs_retry | error
        needs_retry → pending (re-queued for synthesis)
    """

    id: str = field(default_factory=_new_id)
    project_id: str = ""
    idx: int = 0
    chapter_title: str = ""
    text: str = ""
    status: str = "pending"  # pending|approved|needs_review|needs_retry|error
    audio_path: Optional[str] = None  # final output (WAV, possibly post-processed)
    wav_path: Optional[str] = None  # raw WAV before post-processing
    duration_sec: Optional[float] = None
    rtf: Optional[float] = None
    metrics_json: str = "{}"  # WER, CER, SpkSim, silence_ratio
    config_json: str = "{}"  # speed, volume, instruct, reference for regeneration
    error_message: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""


@dataclass
class Project:
    """A TTS project — source text, chapters, segments, and metadata."""

    id: str = field(default_factory=_new_id)
    name: str = ""
    source_text: str = ""
    created_at: str = ""
    updated_at: str = ""
    chapters: list[ChapterMarker] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)

    @property
    def total_segments(self) -> int:
        return len(self.segments)

    @property
    def approved_count(self) -> int:
        return sum(1 for s in self.segments if s.status == "approved")

    @property
    def pending_count(self) -> int:
        return sum(1 for s in self.segments if s.status in ("pending", "needs_retry"))

    @property
    def error_count(self) -> int:
        return sum(1 for s in self.segments if s.status == "error")

    @property
    def total_duration(self) -> float:
        return sum(s.duration_sec or 0 for s in self.segments if s.status == "approved")

    def summary(self) -> dict:
        """Lightweight summary for list endpoints."""
        return {
            "id": self.id,
            "name": self.name,
            "total_segments": self.total_segments,
            "approved": self.approved_count,
            "pending": self.pending_count,
            "errors": self.error_count,
            "total_duration_sec": round(self.total_duration, 2),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_dict(self) -> dict:
        """Full serialisation for API responses."""
        return {
            **self.summary(),
            "source_text": self.source_text,
            "chapters": [
                {
                    "id": c.id,
                    "title": c.title,
                    "char_start": c.char_start,
                    "char_end": c.char_end,
                    "idx": c.idx,
                }
                for c in self.chapters
            ],
            "segments": [
                {
                    "id": s.id,
                    "idx": s.idx,
                    "chapter_title": s.chapter_title,
                    "text": s.text,
                    "status": s.status,
                    "audio_path": s.audio_path,
                    "wav_path": s.wav_path,
                    "duration_sec": s.duration_sec,
                    "rtf": s.rtf,
                    "metrics_json": s.metrics_json,
                    "config_json": s.config_json,
                    "error_message": s.error_message,
                    "created_at": s.created_at,
                    "updated_at": s.updated_at,
                }
                for s in self.segments
            ],
        }
