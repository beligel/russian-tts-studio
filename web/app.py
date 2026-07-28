"""FastAPI web UI for Russian TTS Studio Russian TTS pipeline.

Single-page app: upload reference, type text, get audio + metrics.
Replaces the CLI workflow in scripts/inference/run_pipeline.py etc.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from fastapi import (  # noqa: E402
    FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from russian_tts_studio.integrations import ComfyUIConfig  # noqa: E402
from russian_tts_studio.integrations.comfyui import (  # noqa: E402
    find_comfyui, find_plugin, install_plugin, load_speaker_preset, save_speaker_preset,
)
from russian_tts_studio.pipeline import PipelineConfig, TTSPipeline  # noqa: E402
from russian_tts_studio.utils.audio_utils import (  # noqa: E402
    get_duration, load_audio, normalize_loudness, trim_silence,
)
from russian_tts_studio.utils.metrics import (  # noqa: E402
    SpeakerSimilarityCalculator, Transcriber, calculate_cer, calculate_silence_ratio,
    calculate_wer, normalize_text_for_wer,
)

LOGS_DIR = PROJECT_ROOT / "output" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# Also write to a daily-rotated file under output/logs/ so the user can
# grab the full traceback if anything goes wrong (request/response, modelscope
# downloads, model load failures, ASR transcripts, etc.).
_file_handler = logging.FileHandler(
    LOGS_DIR / f"web-{time.strftime('%Y%m%d')}.log", encoding="utf-8"
)
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)
logging.getLogger().addHandler(_file_handler)
logger = logging.getLogger("web.api")
logger.info("=== web.app started; log file: %s ===", _file_handler.baseFilename)

UPLOAD_DIR = PROJECT_ROOT / "output" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR = PROJECT_ROOT / "output" / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
REFERENCES_DIR = PROJECT_ROOT / "output" / "reference"
REFERENCES_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus"}
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB


def _auto_disable_mms_fa_download() -> None:
    """If the MMS_FA model weights are missing and the user hasn't
    explicitly opted in to a download, set ``MMS_FA_SKIP_DOWNLOAD=1`` so
    the prosody post-processor falls back to proportional placement
    instead of hanging for hours on a throttled CDN.

    Runs at module import — same logic as ``web.start`` — so it covers
    both ``uvicorn web.app:app`` and ``python -m web.start`` launch
    paths. Honours the user's intent: never *unset* the var, and never
    set it if the user has already set it to a value.

    We also set ``MMS_FA_SKIP_DOWNLOAD_AUTO=1`` as a marker so the
    prosody loader (which only runs at request time) can tell the
    flag came from us, not from the user. If the file later appears
    on disk — e.g. the user dropped in a 1.3 GB model after the
    web server was already running — the loader's hot-reload hook
    will lift the flag and trigger a real load. User-set flags
    are never touched (no marker → no auto-revert).
    """
    if os.environ.get("MMS_FA_SKIP_DOWNLOAD") is not None:
        return
    try:
        import torch  # type: ignore[import-not-found]
        cache_dir = torch.hub.get_dir()
    except Exception:
        return
    model_path = os.path.join(cache_dir, "checkpoints", "model.pt")
    if os.path.exists(model_path) and os.path.getsize(model_path) > 100_000_000:
        return
    os.environ["MMS_FA_SKIP_DOWNLOAD"] = "1"
    os.environ["MMS_FA_SKIP_DOWNLOAD_AUTO"] = "1"
    logging.getLogger(__name__).warning(
        "MMS_FA aligner weights not found at %s — setting "
        "MMS_FA_SKIP_DOWNLOAD=1 to use proportional pause placement. "
        "If you download the model later, the loader will hot-reload "
        "on the next request. Override by pre-downloading the model "
        "or unsetting the var.",
        model_path,
    )


_auto_disable_mms_fa_download()


app = FastAPI(
    title="Russian TTS Studio (VoxCPM2 + Silero)",
    description="Web UI for VoxCPM2 voice cloning + Silero fallback",
    version="0.2.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _active_request_tracker(request, call_next):
    """Track active HTTP requests so the heartbeat watchdog doesn't
    kill the server during long synthesis operations."""
    with _State._req_lock:
        _State.active_requests += 1
    try:
        response = await call_next(request)
        return response
    finally:
        with _State._req_lock:
            _State.active_requests -= 1

WEB_DIR = Path(__file__).resolve().parent
# Force no-cache on every static file so the browser cannot serve stale HTML/JS/CSS
# (this is what made the engine-switch button look like a hidden <select>).
class _NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp

app.mount("/static", _NoCacheStaticFiles(directory=str(WEB_DIR / "static")), name="static")


# ---------------------------------------------------------------------------
# Lazy-loaded singletons
# ---------------------------------------------------------------------------


class _State:
    pipeline: TTSPipeline | None = None
    pipeline_engine: str | None = None  # engine of the cached pipeline
    pipeline_device: str | None = None  # device of the cached pipeline
    transcriber: Transcriber | None = None
    sim_calc: SpeakerSimilarityCalculator | None = None
    comfyui_config: ComfyUIConfig | None | object = None  # cached discover() result
    comfyui_resolved: bool = False  # whether we've tried to discover yet
    last_heartbeat: float = 0.0  # updated by /api/status and /api/heartbeat
    # Active request counter — incremented by middleware on every
    # incoming request, decremented on response. The heartbeat watchdog
    # checks this to avoid killing the server during long synthesis.
    active_requests: int = 0
    _req_lock = threading.Lock()
    lock = threading.Lock()

    @classmethod
    def get_pipeline(
        cls, device: str = "auto", engine: str = "voxcpm",
    ) -> TTSPipeline:
        """Return a TTSPipeline configured for ``engine``/``device``.

        The pipeline is rebuilt whenever ``engine`` or ``device``
        changes. Accepted engines: ``"voxcpm"`` (default, OpenBMB
        VoxCPM2) and ``"higgs"`` (Boson AI Higgs Audio v2 — requires
        the upstream ``boson-ai/higgs-audio`` repo installed).
        """
        with cls.lock:
            if (
                cls.pipeline is None
                or cls.pipeline_engine != engine
                or cls.pipeline_device != device
            ):
                if cls.pipeline is not None:
                    logger.info(
                        "Engine/device changed (was %s/%s, now %s/%s) — "
                        "rebuilding pipeline",
                        cls.pipeline_engine, cls.pipeline_device,
                        engine, device,
                    )
                    cls.pipeline.cleanup()
                cls.pipeline = TTSPipeline(
                    PipelineConfig(device=device), engine=engine,
                )
                cls.pipeline.initialize()
                cls.pipeline_engine = engine
                cls.pipeline_device = device
            return cls.pipeline

    @classmethod
    def get_transcriber(cls) -> Transcriber:
        with cls.lock:
            if cls.transcriber is None:
                cls.transcriber = Transcriber(model_size="base", device="cpu")
            return cls.transcriber

    @classmethod
    def get_sim_calc(cls) -> SpeakerSimilarityCalculator:
        with cls.lock:
            if cls.sim_calc is None:
                cls.sim_calc = SpeakerSimilarityCalculator(model_name="wavlm")
            return cls.sim_calc

    @classmethod
    def get_comfyui_config(cls) -> ComfyUIConfig | None:
        """Cache ComfyUIConfig.discover() — the underlying filesystem search
        can take 5+ seconds on a cold miss (spawns `find /` subprocess).
        """
        with cls.lock:
            if not cls.comfyui_resolved:
                cls.comfyui_config = ComfyUIConfig.discover()
                cls.comfyui_resolved = True
            return cls.comfyui_config


def _validate_audio_upload(file: UploadFile) -> Path:
    """Save uploaded audio to disk and return its path."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename")
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {ext}. Allowed: {ALLOWED_AUDIO_EXTENSIONS}",
        )

    timestamp = int(time.time() * 1000)
    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in file.filename)
    dest = UPLOAD_DIR / f"{timestamp}_{safe_name}"
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)
    if dest.stat().st_size > MAX_UPLOAD_SIZE:
        dest.unlink()
        raise HTTPException(status_code=413, detail="File too large (max 50 MB)")
    return dest


# ---------------------------------------------------------------------------
# Routes — pages
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the main SPA."""
    return FileResponse(str(WEB_DIR / "templates" / "index.html"))


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "pipeline_loaded": _State.pipeline is not None and _State.pipeline._initialized,
        "engine": _State.pipeline_engine or "voxcpm",
        "version": "0.2.0",
    }


@app.get("/api/engines")
async def engines() -> dict:
    """List available TTS engines and which one is currently active.

    VoxCPM2 is the default primary. Higgs Audio v2 is available when
    the upstream ``boson-ai/higgs-audio`` repo is installed (the
    pipeline falls back to VoxCPM2 if the package is missing). Silero
    is always available as a fallback (kicks in automatically when the
    primary engine's QC fails or no reference audio is given).
    """
    # Detect whether the Higgs engine is actually importable so the
    # UI can grey it out / show an install hint when it's not.
    higgs_available = True
    try:
        import importlib
        importlib.import_module("boson_multimodal")
    except ImportError:
        higgs_available = False

    engine_list = [
        {
            "id": "voxcpm",
            "label": "VoxCPM2 (русский + 30 языков, рекомендуется)",
            "description": (
                "OpenBMB VoxCPM2 — 2B-параметров, диффузионно-авторегрессионный. "
                "Zero-shot клонирование, 48 кГц. "
                "✅ Лицензия Apache-2.0."
            ),
        },
        {
            "id": "higgs",
            "label": "Higgs Audio v2 (100+ языков, multi-speaker, sound events)",
            "description": (
                "Boson AI Higgs Audio v2 — text-audio foundation model. "
                "Zero-shot клонирование, multi-speaker диалоги, smart voice, "
                "звуковые события ([laugh]/[music]), ударения (U+0301). 24 кГц. "
                "✅ Лицензия Apache-2.0 (v2 3B). "
                + ("✅ Установлен." if higgs_available
                   else "⚠️ Не установлен: pip install -e /path/to/higgs-audio")
            ),
            "available": higgs_available,
        },
    ]
    return {
        "default": "voxcpm",
        "active": _State.pipeline_engine or "voxcpm",
        "engines": engine_list,
    }


# ---------------------------------------------------------------------------
# Routes — file import + long-form
# ---------------------------------------------------------------------------


@app.post("/api/import")
async def import_file_endpoint(file: UploadFile = File(...)) -> JSONResponse:
    """Import a .txt / .md / .docx file and return text + chapter structure.

    The UI can then feed the returned ``text`` into ``/api/markup/parse``
    or ``/api/synthesize`` (with ``enable_markup=true``).
    """
    from russian_tts_studio.text import detect_chapters, import_file

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename")
    ext = Path(file.filename).suffix.lower()
    if ext not in (".txt", ".md", ".markdown", ".docx"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {ext}. Allowed: .txt, .md, .markdown, .docx",
        )

    # Save upload to a temp file, then import.
    timestamp = int(time.time() * 1000)
    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in file.filename)
    dest = UPLOAD_DIR / f"{timestamp}_{safe_name}"
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        text, source_format = import_file(dest)
    except ImportError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception("Import failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

    chapters = detect_chapters(text)
    return JSONResponse({
        "text": text,
        "source_format": source_format,
        "char_count": len(text),
        "chapters": [
            {
                "title": c.title,
                "char_start": c.char_start,
                "char_end": c.char_end,
                "is_preamble": c.is_preamble,
            }
            for c in chapters
        ],
    })


@app.post("/api/longform/chunk")
async def longform_chunk(
    text: str = Form(...),
    max_chars: int = Form(200),
    max_sentences: int = Form(4),
) -> JSONResponse:
    """Chunk long-form text into TTS-safe segments with chapter metadata.

    Returns a flat list of chunks, each with ``text``, ``chapter_title``,
    ``is_chapter_start``, and ``char_start``. The UI can use this to
    show a segment tree before synthesis.
    """
    from russian_tts_studio.text import chunk_document

    chunks = chunk_document(text, max_chars=max_chars, max_sentences=max_sentences)
    return JSONResponse({
        "chunks": [
            {
                "text": c.text,
                "chapter_title": c.chapter_title,
                "is_chapter_start": c.is_chapter_start,
                "char_start": c.char_start,
            }
            for c in chunks
        ],
        "total_chunks": len(chunks),
    })


# ---------------------------------------------------------------------------
# Routes — normalization dictionaries (Phase 5)
# ---------------------------------------------------------------------------


@app.get("/api/normalization")
async def list_dictionaries_endpoint() -> JSONResponse:
    """List all normalization dictionaries."""
    from russian_tts_studio.text.normalization import list_dictionaries

    dicts = list_dictionaries()
    return JSONResponse({"dictionaries": dicts})


@app.get("/api/normalization/{dict_id}")
async def get_dictionary_endpoint(dict_id: str) -> JSONResponse:
    """Load a dictionary with all its entries."""
    from russian_tts_studio.text.normalization import get_dictionary

    d = get_dictionary(dict_id)
    if d is None:
        raise HTTPException(status_code=404, detail=f"Dictionary not found: {dict_id}")
    return JSONResponse(d)


@app.post("/api/normalization")
async def create_dictionary_endpoint(
    id: str = Form(...),
    name: str = Form(...),
    description: str = Form(""),
) -> JSONResponse:
    """Create a new empty dictionary."""
    from russian_tts_studio.text.normalization import create_dictionary

    try:
        d = create_dictionary(id, name, description)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return JSONResponse(d, status_code=201)


@app.delete("/api/normalization/{dict_id}")
async def delete_dictionary_endpoint(dict_id: str) -> JSONResponse:
    """Delete a dictionary and all its entries."""
    from russian_tts_studio.text.normalization import delete_dictionary

    if delete_dictionary(dict_id):
        return JSONResponse({"deleted": dict_id})
    raise HTTPException(status_code=404, detail=f"Dictionary not found: {dict_id}")


@app.post("/api/normalization/{dict_id}/entries")
async def add_entry_endpoint(
    dict_id: str,
    source: str = Form(...),
    replacement: str = Form(...),
) -> JSONResponse:
    """Add an entry to a dictionary."""
    from russian_tts_studio.text.normalization import add_entry

    try:
        entry_id = add_entry(dict_id, source, replacement)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return JSONResponse({"id": entry_id, "source": source, "replacement": replacement}, status_code=201)


@app.put("/api/normalization/entries/{entry_id}")
async def update_entry_endpoint(
    entry_id: int,
    source: str | None = Form(None),
    replacement: str | None = Form(None),
    enabled: bool | None = Form(None),
) -> JSONResponse:
    """Update an entry (source, replacement, or enabled state)."""
    from russian_tts_studio.text.normalization import update_entry

    update_entry(entry_id, source=source, replacement=replacement, enabled=enabled)
    return JSONResponse({"updated": entry_id})


@app.delete("/api/normalization/entries/{entry_id}")
async def delete_entry_endpoint(entry_id: int) -> JSONResponse:
    """Delete an entry."""
    from russian_tts_studio.text.normalization import delete_entry

    if delete_entry(entry_id):
        return JSONResponse({"deleted": entry_id})
    raise HTTPException(status_code=404, detail=f"Entry not found: {entry_id}")


@app.get("/api/normalization/{dict_id}/export")
async def export_dictionary_endpoint(dict_id: str) -> JSONResponse:
    """Export a dictionary as JSON."""
    from russian_tts_studio.text.normalization import export_dictionary_json

    json_str = export_dictionary_json(dict_id)
    return JSONResponse(json.loads(json_str))


@app.post("/api/normalization/{dict_id}/import")
async def import_dictionary_endpoint(
    dict_id: str,
    data: str = Form(...),
) -> JSONResponse:
    """Import entries from JSON."""
    from russian_tts_studio.text.normalization import import_dictionary_json

    count = import_dictionary_json(dict_id, data)
    return JSONResponse({"imported": count})


@app.post("/api/normalization/preview")
async def normalization_preview_endpoint(
    text: str = Form(...),
    use_abbreviations: bool = Form(True),
    use_addresses: bool = Form(True),
    use_custom_dicts: bool = Form(True),
    normalize_numbers: bool = Form(True),
    normalize_ordinals: bool = Form(True),
    normalize_dates: bool = Form(True),
    normalize_currencies: bool = Form(True),
    normalize_percentages: bool = Form(True),
    normalize_measurements: bool = Form(True),
) -> JSONResponse:
    """Preview normalization diff without modifying anything."""
    from russian_tts_studio.text.normalization import (
        NormalizationConfig, get_normalization_diff,
    )

    cfg = NormalizationConfig(
        use_abbreviations=use_abbreviations,
        use_addresses=use_addresses,
        use_custom_dicts=use_custom_dicts,
        normalize_numbers=normalize_numbers,
        normalize_ordinals=normalize_ordinals,
        normalize_dates=normalize_dates,
        normalize_currencies=normalize_currencies,
        normalize_percentages=normalize_percentages,
        normalize_measurements=normalize_measurements,
    )
    diff = get_normalization_diff(text, cfg)
    return JSONResponse(diff)


# ---------------------------------------------------------------------------
# Routes — settings (unified config)
# ---------------------------------------------------------------------------


@app.get("/api/settings")
async def get_settings() -> JSONResponse:
    """Get all settings."""
    from russian_tts_studio.settings import settings

    return JSONResponse(settings.to_dict())


@app.put("/api/settings")
async def update_settings(data: dict) -> JSONResponse:
    """Update settings (partial update, deep merge)."""
    from russian_tts_studio.settings import settings

    settings.update(data)
    settings.save()
    return JSONResponse({"ok": True})


@app.post("/api/settings/reset")
async def reset_settings(key: str | None = None) -> JSONResponse:
    """Reset a key or all settings to defaults."""
    from russian_tts_studio.settings import settings

    settings.reset(key)
    settings.save()
    return JSONResponse({"ok": True, "reset": key or "all"})


# ---------------------------------------------------------------------------
# Routes — projects (Phase 3: SQLite persistence)
# ---------------------------------------------------------------------------


@app.get("/api/projects")
async def list_projects_endpoint() -> JSONResponse:
    """List all projects with summary info."""
    from russian_tts_studio.projects import list_projects

    projects = list_projects()
    return JSONResponse({"projects": [p.summary() for p in projects]})


@app.post("/api/projects")
async def create_project_endpoint(
    name: str = Form(...),
    source_text: str = Form(...),
    max_chars: int = Form(200),
    max_sentences: int = Form(4),
) -> JSONResponse:
    """Create a new project from source text.

    Detects chapters, chunks the text, and inserts segments into SQLite.
    Audio is NOT synthesised — use ``POST /api/projects/{id}/segments/{seg}/regenerate``
    or ``POST /api/projects/{id}/synthesize-all`` afterwards.
    """
    from russian_tts_studio.projects import create_project

    if not source_text.strip():
        raise HTTPException(status_code=400, detail="source_text cannot be empty")

    try:
        project = create_project(
            name=name,
            source_text=source_text,
            max_chars=max_chars,
            max_sentences=max_sentences,
        )
    except Exception as e:
        logger.exception("Failed to create project: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

    return JSONResponse(project.to_dict(), status_code=201)


@app.get("/api/projects/{project_id}")
async def get_project_endpoint(project_id: str) -> JSONResponse:
    """Load a full project with chapters and segments."""
    from russian_tts_studio.projects import get_project_or_404

    try:
        project = get_project_or_404(project_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    return JSONResponse(project.to_dict())


@app.delete("/api/projects/{project_id}")
async def delete_project_endpoint(project_id: str) -> JSONResponse:
    """Delete a project and its audio files."""
    from russian_tts_studio.projects import delete_project_files, get_project_or_404
    from russian_tts_studio.projects.store import delete_project as _store_delete

    try:
        get_project_or_404(project_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    delete_project_files(project_id)
    _store_delete(project_id)
    return JSONResponse({"deleted": project_id})


@app.get("/api/projects/{project_id}/segments")
async def list_segments_endpoint(project_id: str) -> JSONResponse:
    """List all segments for a project."""
    from russian_tts_studio.projects import get_project_or_404

    try:
        project = get_project_or_404(project_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    return JSONResponse({
        "project_id": project_id,
        "segments": [
            {
                "id": s.id,
                "idx": s.idx,
                "chapter_title": s.chapter_title,
                "text": s.text,
                "status": s.status,
                "audio_path": s.audio_path,
                "duration_sec": s.duration_sec,
                "rtf": s.rtf,
                "metrics_json": s.metrics_json,
                "error_message": s.error_message,
            }
            for s in project.segments
        ],
    })


@app.post("/api/projects/{project_id}/segments/{segment_id}/approve")
async def approve_segment_endpoint(project_id: str, segment_id: str) -> JSONResponse:
    """Mark a segment as approved."""
    from russian_tts_studio.projects import approve_segment

    try:
        seg = approve_segment(project_id, segment_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    return JSONResponse({"id": seg.id, "status": seg.status})


@app.post("/api/projects/{project_id}/segments/{segment_id}/discard")
async def discard_segment_endpoint(project_id: str, segment_id: str) -> JSONResponse:
    """Mark a segment as needs_retry (will be re-queued)."""
    from russian_tts_studio.projects import discard_segment

    try:
        seg = discard_segment(project_id, segment_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    return JSONResponse({"id": seg.id, "status": seg.status})


@app.post("/api/projects/{project_id}/segments/{segment_id}/regenerate")
async def regenerate_segment_endpoint(
    project_id: str,
    segment_id: str,
    speed: float = Form(0.9),
    instruct: str | None = Form(None),
    reference_path: str | None = Form(None),
) -> JSONResponse:
    """Re-synthesise a single segment with optional config overrides.

    The segment's status is set to ``pending`` while synthesising, then
    ``approved`` on success or ``error`` on failure.
    """
    from russian_tts_studio.projects import (
        get_project_or_404,
        record_synthesis,
        update_config,
        mark_error,
    )
    from russian_tts_studio.projects.store import update_segment_status

    try:
        project = get_project_or_404(project_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    seg = next((s for s in project.segments if s.id == segment_id), None)
    if seg is None:
        raise HTTPException(status_code=404, detail=f"Segment not found: {segment_id}")

    # Update config with new overrides.
    config = json.loads(seg.config_json) if seg.config_json else {}
    if speed != 0.9:
        config["speed"] = speed
    if instruct:
        config["instruct"] = instruct
    if reference_path:
        config["reference_path"] = reference_path
    update_config(project_id, segment_id, config)

    # Mark as pending during synthesis.
    update_segment_status(segment_id, status="pending")

    # Synthesize.
    pipeline = _State.get_pipeline()
    audio_dir = Path("output") / "projects" / project_id / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    seg_out = audio_dir / f"seg_{seg.idx:03d}.wav"

    ref_path = config.get("reference_path") or reference_path
    ref_audio = Path(ref_path) if ref_path and Path(ref_path).exists() else None

    try:
        result = pipeline.synthesize(
            text=seg.text,
            reference_audio=ref_audio,
            instruct=config.get("instruct"),
            output_path=seg_out,
            speed=config.get("speed", speed),
        )
        final = result["final_path"]
        metrics = result["metrics"].to_dict() if result["metrics"] else {}

        seg = record_synthesis(
            project_id, segment_id,
            audio_path=str(final),
            wav_path=str(final),
            duration_sec=result["result"].duration_sec,
            rtf=result["result"].rtf,
            metrics=metrics,
            status="approved",
        )
        return JSONResponse({
            "id": seg.id,
            "status": seg.status,
            "audio_url": f"/api/audio/{Path(final).name}",
            "duration_sec": round(result["result"].duration_sec, 3),
        })
    except Exception as e:
        logger.exception("Segment regeneration failed: %s", e)
        seg = mark_error(project_id, segment_id, str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/projects/{project_id}/rebuild")
async def rebuild_endpoint(project_id: str) -> JSONResponse:
    """Concatenate all approved segments into a single audio file."""
    from russian_tts_studio.projects import rebuild_audiobook

    try:
        out_path = rebuild_audiobook(project_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Rebuild failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

    return JSONResponse({
        "audio_url": f"/api/audio/{out_path.name}",
        "audio_path": str(out_path),
    })


@app.post("/api/projects/{project_id}/synthesize-all")
async def synthesize_all_endpoint(project_id: str) -> JSONResponse:
    """Synthesise all pending segments in a project.

    This is a long-running endpoint — it blocks until all segments are
    done. For interactive use, the WebSocket ``/ws/synthesize`` is
    preferred; this endpoint is for batch generation.
    """
    from russian_tts_studio.projects import get_project_or_404, record_synthesis, mark_error
    from russian_tts_studio.projects.store import update_segment_status

    try:
        project = get_project_or_404(project_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    pipeline = _State.get_pipeline()
    pending = [s for s in project.segments if s.status in ("pending", "needs_retry")]
    audio_dir = Path("output") / "projects" / project_id / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for seg in pending:
        update_segment_status(seg.id, status="pending")
        seg_out = audio_dir / f"seg_{seg.idx:03d}.wav"
        config = json.loads(seg.config_json) if seg.config_json else {}
        ref_path = config.get("reference_path")
        ref_audio = Path(ref_path) if ref_path and Path(ref_path).exists() else None

        try:
            result = pipeline.synthesize(
                text=seg.text,
                reference_audio=ref_audio,
                instruct=config.get("instruct"),
                output_path=seg_out,
                speed=config.get("speed", 0.9),
            )
            final = result["final_path"]
            metrics = result["metrics"].to_dict() if result["metrics"] else {}
            record_synthesis(
                project_id, seg.id,
                audio_path=str(final),
                wav_path=str(final),
                duration_sec=result["result"].duration_sec,
                rtf=result["result"].rtf,
                metrics=metrics,
                status="approved",
            )
            results.append({"segment_id": seg.id, "status": "approved"})
        except Exception as e:
            logger.exception("Segment %s synthesis failed: %s", seg.id, e)
            mark_error(project_id, seg.id, str(e))
            results.append({"segment_id": seg.id, "status": "error", "error": str(e)})

    return JSONResponse({
        "project_id": project_id,
        "processed": len(results),
        "results": results,
    })


# ---------------------------------------------------------------------------
# Routes — audio mix + music library (Phase 4)
# ---------------------------------------------------------------------------

MUSIC_DIR = PROJECT_ROOT / "music" / "background"
MUSIC_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/api/music")
async def list_music() -> JSONResponse:
    """List available background music files."""
    from russian_tts_studio.audio import list_background_music

    items = list_background_music(MUSIC_DIR)
    return JSONResponse({"music": items, "dir": str(MUSIC_DIR)})


@app.post("/api/music/upload")
async def upload_music(file: UploadFile = File(...)) -> JSONResponse:
    """Upload a music file to the background music library."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename")
    ext = Path(file.filename).suffix.lower()
    if ext not in (".mp3", ".wav", ".flac", ".ogg", ".m4a"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {ext}. Allowed: .mp3, .wav, .flac, .ogg, .m4a",
        )
    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in file.filename)
    dest = MUSIC_DIR / safe_name
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)
    return JSONResponse({"name": dest.name, "path": str(dest)})


@app.delete("/api/music/{filename}")
async def delete_music(filename: str) -> JSONResponse:
    """Remove a music file from the library."""
    if ".." in filename or "/" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    target = MUSIC_DIR / filename
    if not target.exists():
        raise HTTPException(status_code=404, detail="Music not found")
    target.unlink()
    return JSONResponse({"deleted": filename})


@app.post("/api/mix")
async def mix_endpoint(
    narration_path: str = Form(...),
    music_path: str = Form(...),
    voice_db: float = Form(0.0),
    music_db: float = Form(-12.0),
    intro_sec: float = Form(0.0),
    tail_sec: float = Form(0.0),
    fade_in: float = Form(2.0),
    fade_out: float = Form(3.0),
    ducking: bool = Form(True),
    duck_depth_db: float = Form(9.0),
    loudnorm: bool = True,
    target_lufs: float = Form(-16.0),
    output_format: str = Form("mp3"),
) -> JSONResponse:
    """Mix narration with background music.

    Returns the path to the mixed output file. The narration and music
    files must already exist (uploaded or generated by synthesis).
    """
    from russian_tts_studio.audio import MixConfig, mix_podcast

    # Resolve narration path — check common directories.
    narr_path = Path(narration_path)
    if not narr_path.is_absolute():
        for base in (SAMPLES_DIR, UPLOAD_DIR, REFERENCES_DIR, Path("output") / "projects"):
            candidate = base / narr_path.name
            if candidate.exists():
                narr_path = candidate
                break

    music_p = Path(music_path)
    if not music_p.is_absolute():
        # Check music library and common directories.
        for base in (MUSIC_DIR, Path("music"), SAMPLES_DIR, UPLOAD_DIR):
            candidate = base / music_p.name
            if candidate.exists():
                music_p = candidate
                break

    if not narr_path.exists():
        raise HTTPException(status_code=404, detail=f"Narration not found: {narration_path}")
    if not music_p.exists():
        raise HTTPException(status_code=404, detail=f"Music not found: {music_path}")

    cfg = MixConfig(
        voice_db=voice_db,
        music_db=music_db,
        intro_sec=intro_sec,
        tail_sec=tail_sec,
        fade_in=fade_in,
        fade_out=fade_out,
        ducking=ducking,
        duck_depth_db=duck_depth_db,
        loudnorm=loudnorm,
        target_lufs=target_lufs,
        output_format=output_format,
    )

    try:
        out = mix_podcast(narr_path, music_p, config=cfg)
    except Exception as e:
        logger.exception("Mix failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

    return JSONResponse({
        "audio_url": f"/api/audio/{out.name}",
        "audio_path": str(out),
    })


@app.get("/api/subtitles/preview")
async def subtitles_preview(
    text: str,
    duration_sec: float = 5.0,
    max_duration: float = 5.0,
) -> JSONResponse:
    """Preview subtitle cues from text (without real timestamps).

    Generates word timestamps by distributing words evenly across the
    given duration. Useful for UI preview before actual synthesis.
    """
    from russian_tts_studio.audio.subtitles import WordTimestamp, _group_cues

    words = text.split()
    if not words:
        return JSONResponse({"cues": [], "total_words": 0})

    interval = duration_sec / len(words)
    timestamps = [
        WordTimestamp(word=w, start=i * interval, end=(i + 1) * interval)
        for i, w in enumerate(words)
    ]
    cues = _group_cues(timestamps, max_duration=max_duration)
    return JSONResponse({
        "cues": [
            {"text": c.text, "start": round(c.start, 3), "end": round(c.end, 3)}
            for c in cues
        ],
        "total_words": len(words),
    })


# ---------------------------------------------------------------------------
# Routes — MCP (Model Context Protocol)
# ---------------------------------------------------------------------------


@app.post("/mcp")
async def mcp_endpoint(request: dict) -> JSONResponse:
    """MCP JSON-RPC endpoint.

    Accepts JSON-RPC requests and returns JSON-RPC responses.
    Compatible with MCP protocol version 2024-11-05.
    """
    from russian_tts_studio.mcp import handle_mcp_request

    response = handle_mcp_request(request)
    if not response:
        return JSONResponse({}, status_code=202)
    return JSONResponse(response)


@app.get("/mcp/config")
async def mcp_config_endpoint() -> JSONResponse:
    """Generate MCP configuration for Claude Desktop and Codex."""
    from russian_tts_studio.mcp import generate_config

    config = generate_config()
    return JSONResponse(config)


# ---------------------------------------------------------------------------
# Routes — synthesis
# ---------------------------------------------------------------------------


@app.post("/api/markup/parse")
async def markup_parse(text: str = Form(...)) -> JSONResponse:
    """Parse ``{{...}}`` markup without synthesising.

    Returns the segment list, chapters, and warnings so the UI can
    preview the structure before committing to a (slow) generation.
    """
    from russian_tts_studio.markup import parse as parse_markup

    doc = parse_markup(text)
    return JSONResponse({
        "has_markup": doc.has_markup,
        "segments": [
            {
                "text": s.text,
                "speed": s.state.speed,
                "volume": {
                    "gain_db": s.state.volume.gain_db,
                    "multiplier": s.state.volume.multiplier,
                    "normalize_lufs": s.state.volume.normalize_lufs,
                } if s.state.volume.is_enabled() else None,
                "chapter": s.state.chapter,
                "pause_after_ms": s.pause_after.sample_ms() if s.pause_after and s.pause_after.is_enabled() else None,
                "pause_random": s.pause_after.random if s.pause_after else False,
                "aliases": [{"target": a.target, "replacement": a.replacement} for a in s.state.aliases],
                "stresses": [
                    {"target": st.target, "stressed": st.stressed, "hint_vowel": st.hint_vowel}
                    for st in s.state.stresses
                ],
                "active_span": (
                    {
                        "name": s.state.active_span.name,
                        "token": s.state.active_span.token,
                        "label": s.state.active_span.label,
                    }
                    if s.state.active_span else None
                ),
                "source_start": s.source_start,
            }
            for s in doc.segments
        ],
        "chapters": [{"title": c.title, "source_offset": c.source_offset} for c in doc.chapters],
        "warnings": doc.warnings,
    })


@app.post("/api/synthesize")
async def synthesize(
    text: str = Form(...),
    reference: UploadFile | None = File(None),
    reference_path: str | None = Form(None),
    reference_text: str | None = Form(None),
    instruct: str | None = Form(None),
    speaker_fallback: str = Form("xenia"),
    speed: float = Form(0.9),
    enable_fallback: bool = Form(True),
    enable_postprocess: bool = Form(True),
    enable_clamp: bool = Form(True),
    enable_quality_check: bool = Form(True),
    engine: str = Form("voxcpm"),
    # Markup: when True, parse ``{{...}}`` commands in the text and
    # synthesise segment-by-segment (Phase 1 LTV-inspired markup).
    # Off by default — without markup the endpoint behaves exactly as
    # before.
    enable_markup: bool = Form(False),
    # VoxCPM-only prosody: per-punctuation silence durations in ms.
    # ``enable_prosody`` is a master switch (off by default). Each
    # ``pause_ms_*`` is read by ``utils.prosody.PauseConfig.from_metadata``.
    enable_prosody: bool = Form(False),
    pause_ms_comma: int = Form(0),
    pause_ms_semicolon: int = Form(0),
    pause_ms_colon: int = Form(0),
    pause_ms_period: int = Form(0),
    pause_ms_exclamation: int = Form(0),
    pause_ms_question: int = Form(0),
    pause_ms_ellipsis: int = Form(0),
    pause_ms_word_gap: int = Form(0),
) -> JSONResponse:
    """Synthesize text using uploaded reference audio (or Silero fallback)."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if len(text) > 2000:
        raise HTTPException(status_code=400, detail="Text too long (max 2000 chars)")

    # Normalise the engine string.
    #   voxcpm / voxcpm-2 / voxcpm2 / voxcpm_v2  → "voxcpm"
    #   higgs / higgs-v2 / higgs-v2.5            → "higgs"
    engine_norm = (engine or "voxcpm").strip().lower()
    if engine_norm in ("voxcpm-2", "voxcpm2", "voxcpm_v2"):
        engine_norm = "voxcpm"
    if engine_norm in ("higgs-v2", "higgs-v2.5", "higgs-v2-3b", "higgs-audio", "higgs2"):
        engine_norm = "higgs"
    if engine_norm not in ("voxcpm", "higgs"):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown engine {engine!r}; expected 'voxcpm' or 'higgs'",
        )

    ref_path: Path | None = None
    if reference is not None:
        ref_path = _validate_audio_upload(reference)
    elif reference_path:
        candidate = Path(reference_path)
        if candidate.exists() and candidate.is_file():
            ref_path = candidate

    # Auto-load reference_text from sidecar JSON if user didn't pass one.
    # Russian TTS Studio3 needs the transcript of the reference audio for proper
    # voice cloning — without it, the LLM gets a degenerate prompt and
    # generates a generic (often female) timbre.
    ref_text_resolved: str | None = reference_text
    if ref_path is not None and not (ref_text_resolved and ref_text_resolved.strip()):
        sidecar = ref_path.with_suffix(".json")
        if sidecar.exists():
            try:
                meta = json.loads(sidecar.read_text(encoding="utf-8"))
                t = meta.get("transcript") or meta.get("text") or ""
                if t.strip():
                    ref_text_resolved = t.strip()
                    logger.info(
                        "[req preview] loaded reference_text from sidecar %s "
                        "(%d chars)", sidecar.name, len(ref_text_resolved),
                    )
            except Exception as e:
                logger.warning("Failed to read sidecar %s: %s", sidecar, e)

    config = PipelineConfig(
        enable_fallback=enable_fallback,
        enable_postprocess=enable_postprocess,
        enable_quality_check=enable_quality_check,
        enable_clamp=enable_clamp,
    )
    pipeline = _State.get_pipeline(engine=engine_norm)
    pipeline.config = config
    request_id = time.strftime("%H%M%S") + f"-{int(time.time() * 1000) % 100000:05d}"
    logger.info(
        "[req %s] synthesize(engine=%s, text=%r, ref=%s, ref_text=%s chars, "
        "instruct=%r, fallback=%s, postproc=%s, qc=%s, prosody=%s)",
        request_id, engine_norm, text[:80], ref_path,
        len(ref_text_resolved) if ref_text_resolved else 0, instruct,
        enable_fallback, enable_postprocess, enable_quality_check,
        enable_prosody,
    )

    # Build the prosody metadata dict. Only constructed when explicitly
    # enabled AND a VoxCPM engine is selected — prosody is a no-op for
    # VoxCPM2. Silero has no prosody support. ``enable_prosody=False``
    # (the default) means
    # ``prosody_meta = {}`` and the pipeline sees no pause overrides.
    prosody_meta: dict = {}
    if enable_prosody and engine_norm == "voxcpm":
        raw = {
            "comma": pause_ms_comma,
            "semicolon": pause_ms_semicolon,
            "colon": pause_ms_colon,
            "period": pause_ms_period,
            "exclamation": pause_ms_exclamation,
            "question": pause_ms_question,
            "ellipsis": pause_ms_ellipsis,
            "word_gap": pause_ms_word_gap,
        }
        # Clamp to [0, 5000] ms (5 s — sanity limit) and drop 0s to
        # keep the dict small and the log line readable.
        for name, ms in raw.items():
            ms_i = max(0, min(int(ms or 0), 5000))
            if ms_i > 0:
                prosody_meta[f"pause_ms_{name}"] = ms_i
    elif enable_prosody and engine_norm != "voxcpm":
        logger.info(
            "[req %s] enable_prosody=True but engine=%s — prosody is "
            "VoxCPM-only, ignoring", request_id, engine_norm,
        )

    try:
        if enable_markup and "{{" in text:
            # Markup path: parse ``{{...}}`` and synthesise per segment.
            from russian_tts_studio.markup import parse as parse_markup

            doc = parse_markup(text)
            logger.info(
                "[req %s] markup enabled: %d segments, %d chapters, %d warnings",
                request_id, len(doc.segments), len(doc.chapters), len(doc.warnings),
            )
            out_name = f"markup_{request_id}.wav"
            out_path = SAMPLES_DIR / out_name
            result = pipeline.synthesize_markup(
                doc=doc,
                reference_audio=ref_path,
                reference_text=ref_text_resolved,
                instruct=instruct,
                output_path=out_path,
                speaker_fallback=speaker_fallback,
                quality_check=enable_quality_check,
                base_speed=speed,
                prosody=prosody_meta or None,
            )
            final_path = result["final_path"]
            audio_url = f"/api/audio/{Path(final_path).name}"
            return JSONResponse({
                "request_id": request_id,
                "audio_url": audio_url,
                "audio_path": str(final_path),
                "markup": {
                    "segments": len(doc.segments),
                    "chapters": result["chapters"],
                    "warnings": result["warnings"],
                },
                "segment_results": [
                    {"outcome": r.get("outcome", "error").value if hasattr(r.get("outcome"), "value") else r.get("outcome", "error")}
                    for r in result["segments"]
                ],
            })

        result = pipeline.synthesize(
            text=text,
            reference_audio=ref_path,
            reference_text=ref_text_resolved,
            instruct=instruct,
            speaker_fallback=speaker_fallback,
            speed=speed,
            prosody=prosody_meta or None,
        )
    except Exception as e:
        logger.exception("[req %s] Synthesis failed: %s", request_id, e)
        log_path = _file_handler.baseFilename
        raise HTTPException(
            status_code=500,
            detail=(
                f"{e}\n\n"
                f"(request_id={request_id}; full traceback in {log_path})"
            ),
        ) from e

    final_path = result["final_path"]
    res = result["result"]
    metrics_dict = result["metrics"].to_dict() if result["metrics"] else {}

    # ``final_path`` is built by the engine wrapper (VoxCPM2 / Silero)
    # as a *relative* path like "output/samples/...wav". We need an
    # absolute path to compare against PROJECT_ROOT, otherwise
    # ``Path.relative_to`` raises ValueError ("'foo' is not in the
    # subpath of '/abs/path'"). Make it absolute relative to
    # PROJECT_ROOT first, then compute the relative form safely.
    final_path_abs = final_path if final_path.is_absolute() else (PROJECT_ROOT / final_path)
    try:
        audio_path_rel = str(final_path_abs.relative_to(PROJECT_ROOT))
    except ValueError:
        audio_path_rel = str(final_path_abs)

    logger.info(
        "[req %s] synth response: outcome=%s, final_path=%s, audio_url=%s, "
        "duration=%.3fs, rtf=%.3f",
        request_id, result["outcome"].value, final_path_abs,
        final_path_abs.name, res.duration_sec, res.rtf,
    )

    return JSONResponse({
        "request_id": request_id,
        "audio_url": f"/api/audio/{final_path_abs.name}",
        "audio_path": audio_path_rel,
        "duration_sec": round(res.duration_sec, 3),
        "generation_time_sec": round(res.generation_time_sec, 2),
        "rtf": round(res.rtf, 3),
        "model": res.model,
        "outcome": result["outcome"].value,
        "metrics": metrics_dict,
        "transcript": metrics_dict.get("transcript", ""),
        "prosody_degraded": bool(result.get("prosody_degraded", False)),
    })


@app.get("/api/audio/{filename}")
async def get_audio(filename: str) -> FileResponse:
    """Stream a generated audio file."""
    if ".." in filename or "/" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    for base in (SAMPLES_DIR, UPLOAD_DIR, REFERENCES_DIR):
        candidate = base / filename
        if candidate.exists() and candidate.is_file():
            return FileResponse(str(candidate), media_type="audio/wav")
    raise HTTPException(status_code=404, detail="Audio not found")


@app.get("/api/references")
async def list_references() -> JSONResponse:
    """List uploaded/saved reference voices."""
    refs: list[dict] = []
    for p in sorted(REFERENCES_DIR.glob("*.wav")) + sorted(REFERENCES_DIR.glob("*.mp3")):
        try:
            wav = load_audio(p, target_sr=16000, mono=True)
            dur = get_duration(wav, 16000)
            refs.append({
                "name": p.name,
                "path": str(p),
                "duration_sec": round(dur, 2),
                "url": f"/api/audio/{p.name}",
            })
        except Exception as e:
            logger.warning("Could not read %s: %s", p, e)
    for p in sorted(UPLOAD_DIR.glob("*.wav"))[-10:]:
        try:
            wav = load_audio(p, target_sr=16000, mono=True)
            dur = get_duration(wav, 16000)
            refs.append({
                "name": p.name,
                "path": str(p),
                "duration_sec": round(dur, 2),
                "url": f"/api/audio/{p.name}",
                "uploaded": True,
            })
        except Exception:
            pass
    return JSONResponse({"references": refs})


@app.post("/api/references/upload")
async def upload_reference(file: UploadFile = File(...), name: str | None = Form(None)) -> JSONResponse:
    """Save an uploaded audio file as a reusable reference voice."""
    dest = _validate_audio_upload(file)
    final_name = name or dest.name
    final_path = REFERENCES_DIR / Path(final_name).name
    final_path = final_path.with_suffix(".wav")
    shutil.copy2(dest, final_path)
    wav = load_audio(final_path, target_sr=16000, mono=True)
    dur = get_duration(wav, 16000)
    return JSONResponse({
        "name": final_path.name,
        "path": str(final_path),
        "duration_sec": round(dur, 2),
        "url": f"/api/audio/{final_path.name}",
    })


@app.delete("/api/references/{filename}")
async def delete_reference(filename: str) -> JSONResponse:
    """Delete a reference voice."""
    if ".." in filename or "/" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    target = REFERENCES_DIR / filename
    if not target.exists():
        raise HTTPException(status_code=404, detail="Reference not found")
    target.unlink()
    return JSONResponse({"deleted": filename})


# ---------------------------------------------------------------------------
# Routes — voice profiles (text-described voices via YAML)
# ---------------------------------------------------------------------------

@app.get("/api/voice-profiles")
async def list_voice_profiles() -> JSONResponse:
    """List all voice profiles from ``output/reference/profiles.yaml``.

    Voice profiles are text-described voices (``profile:<name>``
    syntax, inspired by Higgs Audio). Engines that support them
    (Higgs) render the description directly; engines that don't
    (VoxCPM2) fall back to their default voice.
    """
    from russian_tts_studio.models.voice_profiles import list_profiles
    profiles = list_profiles()
    return JSONResponse({
        "profiles": [
            {"name": p.name, "description": p.description}
            for p in profiles
        ],
        "path": str(__import__("russian_tts_studio.models.voice_profiles", fromlist=["DEFAULT_PROFILES_PATH"]).DEFAULT_PROFILES_PATH),
    })


@app.post("/api/voice-profiles")
async def add_voice_profile(
    name: str = Form(...),
    description: str = Form(...),
) -> JSONResponse:
    """Add or update a voice profile. Persists to ``profiles.yaml``."""
    from russian_tts_studio.models.voice_profiles import add_profile, get_profile
    name = name.strip()
    description = description.strip()
    if not name or not description:
        raise HTTPException(status_code=400, detail="name and description required")
    add_profile(name, description)
    return JSONResponse({"name": name, "description": description})


@app.delete("/api/voice-profiles/{name}")
async def delete_voice_profile(name: str) -> JSONResponse:
    """Delete a voice profile from ``profiles.yaml``."""
    from russian_tts_studio.models.voice_profiles import delete_profile
    if delete_profile(name):
        return JSONResponse({"deleted": name})
    raise HTTPException(status_code=404, detail="Profile not found")


# ---------------------------------------------------------------------------
# Routes — quality analysis
# ---------------------------------------------------------------------------


@app.post("/api/evaluate")
async def evaluate_audio(
    file: UploadFile = File(...),
    reference_text: str = Form(""),
) -> JSONResponse:
    """Evaluate an audio file: WER + speaker similarity against itself (if text given)."""
    dest = _validate_audio_upload(file)
    try:
        wav = load_audio(dest, target_sr=16000, mono=True)
        transcriber = _State.get_transcriber()
        transcript = transcriber.transcribe(wav, language="ru")

        result = {
            "audio_url": f"/api/audio/{dest.name}",
            "duration_sec": round(get_duration(wav, 16000), 2),
            "transcript": transcript,
            "silence_ratio": round(calculate_silence_ratio(wav), 4),
        }

        if reference_text:
            ref_norm = normalize_text_for_wer(reference_text)
            hyp_norm = normalize_text_for_wer(transcript) if transcript else ""
            result["wer"] = round(calculate_wer(ref_norm, hyp_norm), 4)
            result["cer"] = round(calculate_cer(
                ref_norm.replace(" ", ""), hyp_norm.replace(" ", "")
            ), 4)
            result["reference_text"] = reference_text
        return JSONResponse(result)
    except Exception as e:
        logger.exception("Evaluation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/speaker-similarity")
async def speaker_similarity(
    reference: UploadFile = File(...),
    synthesized: UploadFile = File(...),
) -> JSONResponse:
    """Compute speaker similarity between two audio files."""
    ref_path = _validate_audio_upload(reference)
    synth_path = _validate_audio_upload(synthesized)
    try:
        sim_calc = _State.get_sim_calc()
        ref_wav = load_audio(ref_path, target_sr=16000, mono=True)
        synth_wav = load_audio(synth_path, target_sr=16000, mono=True)
        sim = sim_calc.similarity(ref_waveform=ref_wav, synth_waveform=synth_wav)
        return JSONResponse({
            "speaker_similarity": round(sim, 4),
            "reference_duration": round(get_duration(ref_wav, 16000), 2),
            "synthesized_duration": round(get_duration(synth_wav, 16000), 2),
        })
    except Exception as e:
        logger.exception("Similarity failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


# ---------------------------------------------------------------------------
# Routes — ComfyUI integration
# ---------------------------------------------------------------------------


@app.get("/api/comfyui/status")
async def comfyui_status() -> JSONResponse:
    """Check if ComfyUI + plugin are available."""
    config = _State.get_comfyui_config()
    if not config:
        return JSONResponse({
            "available": False,
            "message": "ComfyUI not found. Set COMFYUI_PATH or install plugin.",
        })
    return JSONResponse({
        "available": True,
        "comfyui_path": str(config.comfyui_path),
        "plugin_path": str(config.plugin_path),
        "models_count": len(config.list_models()),
        "speakers_count": len(config.list_speakers()),
    })


@app.get("/api/comfyui/speakers")
async def comfyui_speakers() -> JSONResponse:
    """List ComfyUI plugin's saved speaker presets."""
    config = _State.get_comfyui_config()
    if not config:
        raise HTTPException(status_code=404, detail="ComfyUI not found")
    speakers: list[dict] = []
    for sp in config.list_speakers():
        try:
            data = load_speaker_preset(config, sp.stem)
            audio = data.get("audio")
            sr = data.get("sample_rate", 16000)
            dur = audio.shape[-1] / sr if audio is not None else 0
            speakers.append({
                "name": sp.stem,
                "duration_sec": round(dur, 2),
                "text": data.get("text", ""),
                "instruct": data.get("instruct", ""),
            })
        except Exception as e:
            speakers.append({"name": sp.stem, "error": str(e)})
    return JSONResponse({"speakers": speakers})


@app.post("/api/comfyui/synthesize")
async def comfyui_synthesize(
    text: str = Form(...),
    speaker_name: str = Form(...),
    output_name: str | None = Form(None),
) -> JSONResponse:
    """Synthesize using a ComfyUI speaker preset."""
    config = _State.get_comfyui_config()
    if not config:
        raise HTTPException(status_code=404, detail="ComfyUI not found")
    try:
        preset = load_speaker_preset(config, speaker_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    import torchaudio
    ref_audio = preset.get("audio")
    sr = preset.get("sample_rate", 16000)
    ref_text = preset.get("text", "")
    if ref_audio is None:
        raise HTTPException(status_code=400, detail="Speaker preset has no audio")

    tmp_ref = SAMPLES_DIR / f"_ref_{speaker_name}.wav"
    if ref_audio.dim() == 1:
        ref_audio = ref_audio.unsqueeze(0)
    torchaudio.save(str(tmp_ref), ref_audio, sr)

    pipeline = _State.get_pipeline()
    out_name = output_name or f"comfyui_{speaker_name}_{int(time.time())}.wav"
    out_path = SAMPLES_DIR / out_name
    result = pipeline.synthesize(
        text=text,
        reference_audio=tmp_ref,
        reference_text=ref_text,
        output_path=out_path,
    )
    return JSONResponse({
        "audio_url": f"/api/audio/{result['final_path'].name}",
        "outcome": result["outcome"].value,
        "model": result["result"].model,
        "duration_sec": round(result["result"].duration_sec, 3),
        "rtf": round(result["result"].rtf, 3),
    })


@app.post("/api/comfyui/install")
async def comfyui_install() -> JSONResponse:
    """Install the Russian TTS Studio3 ComfyUI plugin (if ComfyUI is present)."""
    comfy = find_comfyui()
    if not comfy:
        raise HTTPException(status_code=404, detail="ComfyUI not found. Set COMFYUI_PATH env var.")
    if find_plugin(comfy):
        return JSONResponse({"status": "already_installed", "path": str(find_plugin(comfy))})
    try:
        path = install_plugin(comfy)
        return JSONResponse({"status": "installed", "path": str(path)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/comfyui/export-speaker")
async def comfyui_export_speaker(
    audio: UploadFile = File(...),
    name: str = Form(...),
    text: str = Form(""),
    auto_transcribe: bool = Form(True),
) -> JSONResponse:
    """Save an audio file as a ComfyUI-compatible speaker preset."""
    config = _State.get_comfyui_config()
    if not config:
        raise HTTPException(status_code=404, detail="ComfyUI not found")
    audio_path = _validate_audio_upload(audio)

    if auto_transcribe and not text:
        try:
            transcriber = _State.get_transcriber()
            wav = load_audio(audio_path, target_sr=16000, mono=True)
            text = transcriber.transcribe(wav, language="ru")
        except Exception as e:
            logger.warning("Auto-transcribe failed: %s", e)

    out = save_speaker_preset(
        config=config,
        name=name,
        reference_audio=audio_path,
        reference_text=text,
    )
    return JSONResponse({
        "name": out.stem,
        "path": str(out),
        "transcript": text,
    })


# ---------------------------------------------------------------------------
# Routes — pipeline status
# ---------------------------------------------------------------------------


@app.get("/api/status")
async def status() -> JSONResponse:
    """Full pipeline + environment status."""
    import torch  # noqa: PLC0415  - cheap reimport (already loaded transitively)
    _State.last_heartbeat = time.time()
    return JSONResponse({
        "device": (
            "cuda" if torch.cuda.is_available()
            else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        ),
        "cuda_available": torch.cuda.is_available(),
        "pipeline_loaded": _State.pipeline is not None and _State.pipeline._initialized,
        "transcriber_loaded": _State.transcriber is not None and _State.transcriber._loaded,
        "comfyui": _State.get_comfyui_config() is not None,
    })


@app.post("/api/heartbeat")
async def heartbeat() -> JSONResponse:
    """Ping from the browser UI. Used in --browser mode to auto-shutdown
    the server when the user closes the tab/window."""
    _State.last_heartbeat = time.time()
    return JSONResponse({"ok": True})


@app.post("/api/postprocess")
async def postprocess(
    file: UploadFile = File(...),
    target_dbfs: float = Form(-20.0),
    trim: bool = Form(True),
    normalize: bool = Form(True),
) -> JSONResponse:
    """Apply post-processing to an audio file."""
    dest = _validate_audio_upload(file)
    try:
        wav = load_audio(dest, target_sr=22050, mono=True)
        if trim:
            wav = trim_silence(wav, threshold=0.01)
        if normalize:
            wav = normalize_loudness(wav, target_dbfs=target_dbfs)
        out_path = SAMPLES_DIR / f"processed_{dest.stem}.wav"
        import torchaudio
        torchaudio.save(str(out_path), wav.unsqueeze(0), 22050)
        return JSONResponse({
            "audio_url": f"/api/audio/{out_path.name}",
            "duration_sec": round(get_duration(wav, 22050), 2),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# ---------------------------------------------------------------------------
# WebSocket — streaming synthesis (for long texts)
# ---------------------------------------------------------------------------


@app.websocket("/ws/synthesize")
async def ws_synthesize(ws: WebSocket) -> None:
    """Stream synthesis progress for long texts.

    Client sends JSON: {text, reference_audio_url, instruct, ...}
    Server sends: {type: 'progress', chunk, idx, total}
                   {type: 'done', audio_url, metrics}
                   {type: 'error', message}
    """
    await ws.accept()
    try:
        data = await ws.receive_json()
        text = data.get("text", "").strip()
        if not text:
            await ws.send_json({"type": "error", "message": "Empty text"})
            return

        reference_url = data.get("reference_audio_url")
        ref_path: Path | None = None
        if reference_url:
            filename = Path(reference_url).name
            for base in (UPLOAD_DIR, REFERENCES_DIR, SAMPLES_DIR):
                cand = base / filename
                if cand.exists():
                    ref_path = cand
                    break

        pipeline = _State.get_pipeline()

        # Markup path: if the text contains ``{{...}}`` commands, parse
        # and synthesise segment-by-segment via ``synthesize_markup``.
        # This streams per-segment progress with chapter info.
        if "{{" in text:
            from russian_tts_studio.markup import parse as parse_markup

            doc = parse_markup(text)
            non_empty = [s for s in doc.segments if s.text.strip()]
            await ws.send_json({
                "type": "started",
                "total_chunks": len(non_empty),
                "chapters": [{"title": c.title, "source_offset": c.source_offset} for c in doc.chapters],
                "warnings": doc.warnings,
                "mode": "markup",
            })

            out_name = f"markup_ws_{int(time.time() * 1000)}.wav"
            out_path = SAMPLES_DIR / out_name

            progress_state = {"idx": 0}

            def _on_progress(idx: int, total: int, segment) -> None:
                progress_state["idx"] = idx

            # We can't easily make synthesize_markup async-send to the ws
            # from a sync method, so we poll: run synthesize_markup in a
            # thread and send progress events as idx advances. For Phase 1
            # simplicity, we run it synchronously and send a single done
            # event — the per-segment progress is available via the
            # returned ``segments`` list. A future iteration can make
            # this truly streaming.
            try:
                result = pipeline.synthesize_markup(
                    doc=doc,
                    reference_audio=ref_path,
                    instruct=data.get("instruct"),
                    output_path=out_path,
                    on_progress=_on_progress,
                )
                final_path = result["final_path"]
                await ws.send_json({
                    "type": "done",
                    "audio_url": f"/api/audio/{Path(final_path).name}",
                    "segment_count": len(result["segments"]),
                    "chapters": result["chapters"],
                    "warnings": result["warnings"],
                })
            except Exception as e:
                logger.exception("[ws markup] failed: %s", e)
                await ws.send_json({"type": "error", "message": str(e)})
            return

        chunks: list[str] = []
        if len(text) > 180:
            from russian_tts_studio.utils.text_utils import chunk_text_for_tts
            chunks = chunk_text_for_tts(text)
        else:
            chunks = [text]

        await ws.send_json({"type": "started", "total_chunks": len(chunks), "mode": "plain"})

        for i, chunk in enumerate(chunks):
            await ws.send_json({"type": "progress", "chunk": i + 1, "total": len(chunks)})
            try:
                result = pipeline.synthesize(
                    text=chunk,
                    reference_audio=ref_path,
                    instruct=data.get("instruct"),
                )
                await ws.send_json({
                    "type": "chunk_done",
                    "chunk": i + 1,
                    "audio_url": f"/api/audio/{result['final_path'].name}",
                    "outcome": result["outcome"].value,
                })
            except Exception as e:
                await ws.send_json({"type": "chunk_error", "chunk": i + 1, "message": str(e)})
                return

        await ws.send_json({"type": "done"})
    except WebSocketDisconnect:
        logger.info("Client disconnected during synthesis")
    except Exception as e:
        logger.exception("WebSocket error: %s", e)
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except RuntimeError:
            pass


def run() -> None:
    """Entry point for `python -m web.app`."""
    import uvicorn
    uvicorn.run(
        "web.app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8129")),
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    run()
