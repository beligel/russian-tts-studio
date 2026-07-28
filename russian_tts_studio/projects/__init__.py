"""TTS projects — SQLite-backed persistent projects with per-segment management.

Public API::

    from russian_tts_studio.projects import (
        create_project, get_project_or_404, list_projects, delete_project,
        approve_segment, discard_segment, rebuild_audiobook,
        Project, Segment, ChapterMarker,
    )

Phase 3 (LTV-inspired):
- Projects store source text, chapters, and segments in SQLite.
- Each segment has audio output, metrics, and regeneration config.
- Segments can be individually approved / discarded / re-synthesised.
- ``rebuild_audiobook`` concatenates approved segments into a final file.
"""

from __future__ import annotations

from .manager import (
    approve_segment,
    create_and_synthesize,
    create_project,
    delete_project_files,
    discard_segment,
    get_project_or_404,
    list_projects,
    mark_error,
    record_synthesis,
    rebuild_audiobook,
    update_config,
)
from .models import ChapterMarker, Project, Segment

__all__ = [
    "create_project",
    "get_project_or_404",
    "list_projects",
    "delete_project_files",
    "approve_segment",
    "discard_segment",
    "mark_error",
    "record_synthesis",
    "update_config",
    "rebuild_audiobook",
    "create_and_synthesize",
    "Project",
    "Segment",
    "ChapterMarker",
]