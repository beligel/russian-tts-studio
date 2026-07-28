"""Project manager — high-level operations on TTS projects.

Sits between the API layer (``web/app.py``) and the low-level store
(``store.py``). Handles:

- **create**: detect chapters → chunk → insert project/segments/chapters
- **regenerate**: re-synthesise a single segment with config overrides
- **approve / discard**: update segment status
- **rebuild**: concatenate all approved segments into a final MP3/WAV
- **delete**: remove project + audio files

All file I/O goes through ``store`` + ``Path`` helpers. The pipeline
is instantiated lazily via ``_State.get_pipeline`` — the manager
never holds a long-lived reference.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Optional

from .models import ChapterMarker, Project, Segment, _new_id
from .store import (
    create_project as _store_create,
    delete_project as _store_delete,
    get_segments,
    get_segment,
    get_chapters,
    insert_chapters,
    insert_segments,
    load_project_full,
    update_segment_audio,
    update_segment_config,
    update_segment_status,
    get_db,
    _now,
)

logger = logging.getLogger(__name__)

# Output root — relative to PROJECT_ROOT (set at import time by app.py
# or by tests). The projects store database lives alongside audio files.
_OUTPUT_DIR = Path("output")
_PROJECTS_DIR = _OUTPUT_DIR / "projects"


def _project_dir(project_id: str) -> Path:
    """Per-project directory for audio files."""
    d = _PROJECTS_DIR / project_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _project_audio_dir(project_id: str) -> Path:
    d = _project_dir(project_id) / "audio"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------

def create_project(
    name: str,
    source_text: str,
    max_chars: int = 200,
    max_sentences: int = 4,
) -> Project:
    """Create a new project from source text.

    1. Detect chapters via ``text.longform.detect_chapters``.
    2. Chunk each chapter via ``text.longform.chunk_document``.
    3. Insert project, chapters, and segments into SQLite.
    4. Return the fully-populated :class:`Project`.

    Audio is NOT synthesised here — the caller should iterate segments
    and call ``pipeline.synthesize`` for each, or use ``regenerate_segment``
    for one-at-a-time generation.
    """
    from ..text import chunk_document, detect_chapters

    project = Project(id=_new_id(), name=name, source_text=source_text)
    project = _store_create(project)

    # Detect chapters in source text.
    raw_chapters = detect_chapters(source_text)
    chapter_markers: list[ChapterMarker] = []
    for i, ch in enumerate(raw_chapters):
        cm = ChapterMarker(
            id=_new_id(),
            project_id=project.id,
            title=ch.title,
            char_start=ch.char_start,
            char_end=ch.char_end,
            idx=i,
        )
        chapter_markers.append(cm)
    project.chapters = chapter_markers
    insert_chapters(chapter_markers)

    # Chunk the source text.
    chunks = chunk_document(source_text, max_chars=max_chars, max_sentences=max_sentences)

    # Create segments from chunks.
    segments: list[Segment] = []
    for i, chunk in enumerate(chunks):
        seg = Segment(
            id=_new_id(),
            project_id=project.id,
            idx=i,
            chapter_title=chunk.chapter_title,
            text=chunk.text,
            status="pending",
            config_json=json.dumps({
                "max_chars": max_chars,
                "max_sentences": max_sentences,
            }),
            created_at=_now(),
            updated_at=_now(),
        )
        segments.append(seg)
    project.segments = segments
    insert_segments(segments)

    # Create per-project audio directory.
    _project_audio_dir(project.id)

    logger.info(
        "Created project %s: %d chapters, %d segments, %d chars",
        project.id, len(chapter_markers), len(segments), len(source_text),
    )
    return project


# ---------------------------------------------------------------------------
# Load / List / Delete
# ---------------------------------------------------------------------------

def get_project_or_404(project_id: str) -> Project:
    """Load a full project or raise ValueError."""
    project = load_project_full(project_id)
    if project is None:
        raise ValueError(f"Project not found: {project_id}")
    return project


def list_projects() -> list[Project]:
    """List all projects (without full segment data)."""
    return list_projects_from_store()


def list_projects_from_store() -> list[Project]:
    from .store import list_projects as _list
    return _list()


def delete_project_files(project_id: str) -> None:
    """Delete a project's audio directory from disk."""
    proj_dir = _PROJECTS_DIR / project_id
    if proj_dir.exists():
        shutil.rmtree(proj_dir, ignore_errors=True)
        logger.info("Deleted project directory: %s", proj_dir)


# ---------------------------------------------------------------------------
# Segment operations
# ---------------------------------------------------------------------------

def approve_segment(project_id: str, segment_id: str) -> Segment:
    """Mark a segment as approved."""
    seg = get_segment(segment_id)
    if seg is None or seg.project_id != project_id:
        raise ValueError(f"Segment not found: {segment_id}")
    update_segment_status(segment_id, status="approved")
    return get_segment(segment_id)


def discard_segment(project_id: str, segment_id: str) -> Segment:
    """Mark a segment as needs_retry (will be re-queued)."""
    seg = get_segment(segment_id)
    if seg is None or seg.project_id != project_id:
        raise ValueError(f"Segment not found: {segment_id}")
    update_segment_status(segment_id, status="needs_retry")
    return get_segment(segment_id)


def mark_error(project_id: str, segment_id: str, error: str) -> Segment:
    """Mark a segment as error with a message."""
    seg = get_segment(segment_id)
    if seg is None or seg.project_id != project_id:
        raise ValueError(f"Segment not found: {segment_id}")
    update_segment_status(segment_id, status="error", error_message=error)
    return get_segment(segment_id)


def record_synthesis(
    project_id: str,
    segment_id: str,
    audio_path: str,
    wav_path: str | None = None,
    duration_sec: float | None = None,
    rtf: float | None = None,
    metrics: dict | None = None,
    status: str = "approved",
) -> Segment:
    """Record synthesis output for a segment after TTS generation."""
    seg = get_segment(segment_id)
    if seg is None or seg.project_id != project_id:
        raise ValueError(f"Segment not found: {segment_id}")
    update_segment_audio(
        segment_id,
        audio_path=audio_path,
        wav_path=wav_path,
        duration_sec=duration_sec,
        rtf=rtf,
        metrics_json=json.dumps(metrics) if metrics else None,
        status=status,
    )
    return get_segment(segment_id)


def update_config(project_id: str, segment_id: str, config: dict) -> Segment:
    """Update a segment's regeneration config."""
    seg = get_segment(segment_id)
    if seg is None or seg.project_id != project_id:
        raise ValueError(f"Segment not found: {segment_id}")
    update_segment_config(segment_id, json.dumps(config))
    return get_segment(segment_id)


# ---------------------------------------------------------------------------
# Rebuild audiobook
# ---------------------------------------------------------------------------

def rebuild_audiobook(
    project_id: str,
    output_name: str = "audiobook",
    format: str = "wav",
) -> Path:
    """Concatenate all approved segments into a single output file.

    Returns the path to the final audio file. Only segments with
    ``status="approved"`` are included; others are skipped silently.
    """
    import numpy as np
    import soundfile as sf

    project = get_project_or_404(project_id)
    approved = [s for s in project.segments if s.status == "approved" and s.audio_path]
    if not approved:
        raise ValueError("No approved segments to rebuild")

    audio_dir = _project_audio_dir(project_id)
    out_path = audio_dir / f"{output_name}.{format}"

    chunks: list[np.ndarray] = []
    target_sr = 22050

    for seg in approved:
        p = Path(seg.audio_path)
        if not p.exists():
            # Try relative to project audio dir
            p = audio_dir / Path(seg.audio_path).name
        if not p.exists():
            logger.warning("Segment %s audio missing: %s", seg.id, seg.audio_path)
            continue
        wav, sr = sf.read(str(p), dtype="float32", always_2d=False)
        if sr != target_sr:
            import torch
            import torchaudio

            t = torch.from_numpy(wav).float()
            if t.dim() == 1:
                t = t.unsqueeze(0)
            t = torchaudio.functional.resample(t, sr, target_sr)
            wav = t.squeeze(0).numpy()
        chunks.append(wav)

    if not chunks:
        raise ValueError("No audio files found for approved segments")

    final = np.concatenate(chunks, axis=0)
    sf.write(str(out_path), final, target_sr, subtype="FLOAT")
    logger.info(
        "Rebuilt audiobook %s: %d segments, %.2fs → %s",
        project_id, len(chunks), len(final) / target_sr, out_path,
    )
    return out_path


# ---------------------------------------------------------------------------
# Convenience: create + synthesise all segments
# ---------------------------------------------------------------------------

def create_and_synthesize(
    name: str,
    source_text: str,
    pipeline,  # TTSPipeline instance
    reference_audio: str | Path | None = None,
    reference_text: str | None = None,
    max_chars: int = 200,
    max_sentences: int = 4,
    speed: float = 0.9,
    on_progress=None,  # callable(idx, total, segment)
) -> Project:
    """Create a project and synthesise every segment.

    This is the "one-shot" convenience for API callers that want a
    project created and fully generated in a single request. For
    interactive workflows, use ``create_project`` + individual
    ``record_synthesis`` calls.
    """
    project = create_project(name, source_text, max_chars, max_sentences)
    total = len(project.segments)

    for i, seg in enumerate(project.segments):
        if on_progress:
            on_progress(i + 1, total, seg)
        seg_out = _project_audio_dir(project.id) / f"seg_{seg.idx:03d}.wav"
        try:
            # Parse config overrides from the segment.
            config = json.loads(seg.config_json) if seg.config_json else {}
            seg_speed = config.get("speed", speed)

            result = pipeline.synthesize(
                text=seg.text,
                reference_audio=reference_audio,
                reference_text=reference_text,
                output_path=seg_out,
                speed=seg_speed,
            )
            final = result["final_path"]
            metrics = result["metrics"].to_dict() if result["metrics"] else {}

            record_synthesis(
                project.id, seg.id,
                audio_path=str(final),
                wav_path=str(final),
                duration_sec=result["result"].duration_sec,
                rtf=result["result"].rtf,
                metrics=metrics,
                status="approved",
            )
        except Exception as e:
            logger.exception("Segment %d synthesis failed: %s", i, e)
            mark_error(project.id, seg.id, str(e))

    # Refresh from DB to pick up recorded audio paths.
    return get_project_or_404(project.id)
