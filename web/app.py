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


def _reexec_for_engine(engine: str) -> None:
    """Replace the current process with the venv that hosts the requested engine.

    Two engines, two venvs (torch 2.3.1 + Coqui in .venv, torch 2.12 + voxcpm
    in .venv-voxcpm — they can't coexist). On each /api/synthesize we re-exec
    the uvicorn worker into the right interpreter. The in-flight HTTP
    request is closed and the client retries against the new process.

    Safe to call: no-op if we're already in the right venv, if the target
    venv doesn't exist, or if RUSSIAN_TTS_NO_AUTO_REEXEC=1.
    """
    if os.environ.get("RUSSIAN_TTS_NO_AUTO_REEXEC") == "1":
        return
    # If the user opted out of MMS_FA downloads (env var MMS_FA_SKIP_DOWNLOAD=1
    # was set before startup), propagate it into the child so the new venv
    # sees the same flag. We never *set* this automatically — only pass
    # through the user's intent.
    venvs = {
        "xtts":   PROJECT_ROOT / ".venv"        / "bin" / "python",
        "voxcpm": PROJECT_ROOT / ".venv-voxcpm" / "bin" / "python",
    }
    target = venvs.get(engine)
    if target is None or not target.exists():
        return
    # NB: we intentionally do NOT use .resolve() here. Both venvs ship
    # .venv/bin/python and .venv-voxcpm/bin/python as symlinks to a shared
    # system python3 (e.g. /usr/bin/python3.12), so resolving collapses
    # them onto the same path and the "already in the right venv" check
    # would always be True — reexec would never happen. Comparing the
    # symlink path strings as-is distinguishes the two venvs reliably.
    try:
        already = Path(sys.executable).absolute() == target.absolute()
    except OSError:
        already = False
    if already:
        return
    print(
        f"\n  ⟳  engine={engine!r} требует {target} — "
        f"перезапускаю сервер под нужный venv\n",
        file=sys.stderr,
        flush=True,
    )
    # Preserve PYTHONPATH and sys.argv[0] so the new interpreter can find
    # this package. `os.execv` resets sys.path from scratch; without
    # PROJECT_ROOT on it, `import web.app` raises ModuleNotFoundError
    # (we hit this in .venv-voxcpm because `web` isn't pip-installed —
    # it's discovered only via sys.path[0] when running `python -m`).
    existing_pp = os.environ.get("PYTHONPATH", "")
    parts = [str(PROJECT_ROOT)]
    if existing_pp:
        parts.append(existing_pp)
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)
    # Pass sys.argv[0] as an absolute path so the new python inserts
    # the script's directory (web/) into sys.path[0] — same effect as
    # `python -m web.run` had in the parent process.
    argv = list(sys.argv)
    if argv and not Path(argv[0]).is_absolute():
        argv[0] = str(Path(argv[0]).resolve())
    # Strip the no-auto-reexec one-shot guard from the environment we hand
    # to the new process. Setting `os.environ[...]` here mutates the
    # current process so the same request can't loop back into us; but
    # `os.execv` inherits os.environ, so the new process would also see
    # "1" and refuse any future reexec (defeating the whole feature).
    # Pass an env explicitly without the guard so reexec remains
    # bidirectional.
    child_env = {k: v for k, v in os.environ.items() if k != "RUSSIAN_TTS_NO_AUTO_REEXEC"}
    os.execve(str(target), [str(target), *argv], child_env)


# Backwards-compatible alias for older callers.
def _reexec_into_voxcpm_venv() -> None:
    """Deprecated: use _reexec_for_engine('voxcpm')."""
    _reexec_for_engine("voxcpm")


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
    logging.getLogger(__name__).warning(
        "MMS_FA aligner weights not found at %s — setting "
        "MMS_FA_SKIP_DOWNLOAD=1 to use proportional pause placement. "
        "Override by pre-downloading the model or unsetting the var.",
        model_path,
    )


_auto_disable_mms_fa_download()


app = FastAPI(
    title="XTTS Russian TTS",
    description="Web UI for XTTS-v2 voice cloning + Silero fallback",
    version="0.2.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    lock = threading.Lock()

    @classmethod
    def get_pipeline(
        cls, device: str = "auto", engine: str = "xtts",
    ) -> TTSPipeline:
        """Return a TTSPipeline configured for ``engine``/``device``.

        Different engines load different model weights (F5-TTS vs
        Russian TTS Studio3 vs XTTS-v2), so we cannot share a pipeline across
        engines — when the requested engine changes we tear down the
        cached pipeline and build a new one. Switching engine
        mid-process is therefore slow (one full model load) but the UI
        only requests one engine per page load in practice.
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
        "engine": _State.pipeline_engine or "xtts",
        "version": "0.2.0",
    }


@app.get("/api/engines")
async def engines() -> dict:
    """List available TTS engines and which one is currently active.

    XTTS-v2 (default) and VoxCPM2 are selectable as primaries. Silero
    is always available as a fallback but is not selectable as a
    primary "engine" (it's invoked automatically when XTTS QC fails
    or no reference is given).

    Note: VoxCPM2 lives in a separate venv (.venv-voxcpm). The web
    server can be started from either venv; if you started from
    .venv-voxcpm the import will succeed and the engine will be
    selectable, otherwise selecting it will fail-fast with a clear
    "Cannot import voxcpm" error.
    """
    return {
        "default": "xtts",
        "active": _State.pipeline_engine or "xtts",
        "engines": [
            {
                "id": "xtts",
                "label": "XTTS v2 (русский)",
                "description": (
                    "Coqui XTTS v2 — мультиязычный, включая русский. "
                    "Zero-shot клонирование по 6-10 с референса. "
                    "⚠️ Лицензия CPML (некоммерческая)."
                ),
            },
            {
                "id": "voxcpm",
                "label": "VoxCPM2 (русский + 30 языков)",
                "description": (
                    "OpenBMB VoxCPM2 — 2B-параметров, диффузионно-авторегрессионный. "
                    "Zero-shot клонирование, 48 кГц, лучше XTTS на длинных "
                    "русских фразах и без CAPS-ограничений словаря. "
                    "✅ Лицензия Apache-2.0. ⚠️ Требует .venv-voxcpm."
                ),
            },
        ],
    }


# ---------------------------------------------------------------------------
# Routes — synthesis
# ---------------------------------------------------------------------------


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
    enable_quality_check: bool = Form(True),
    engine: str = Form("xtts"),
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
) -> JSONResponse:
    """Synthesize text using uploaded reference audio (or Silero fallback)."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if len(text) > 2000:
        raise HTTPException(status_code=400, detail="Text too long (max 2000 chars)")

    # Normalise the engine string.
    #   - "xtts-v2"/"xtts_v2"/"xttsv2" → "xtts"
    #   - "voxcpm-2"/"voxcpm2"           → "voxcpm"
    # The pipeline supports "xtts" and "voxcpm"; anything else is 400.
    engine_norm = (engine or "xtts").strip().lower()
    if engine_norm in ("xtts-v2", "xtts_v2", "xttsv2"):
        engine_norm = "xtts"
    elif engine_norm in ("voxcpm-2", "voxcpm2", "voxcpm_v2"):
        engine_norm = "voxcpm"
    if engine_norm not in ("xtts", "voxcpm"):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown engine {engine!r}; expected 'xtts' or 'voxcpm'",
        )

    # If the user picked an engine whose dependencies live in a different
    # venv (XTTS in .venv, VoxCPM2 in .venv-voxcpm — torch versions don't
    # coexist), transparently re-exec the worker into the right interpreter.
    # The execv replaces the current process, so the request never completes
    # in the wrong interpreter; the browser/curl retries against the new one.
    if engine_norm in ("xtts", "voxcpm"):
        # uvicorn --reload spawns a watcher that doesn't tolerate the worker
        # process replacing its binary under it. In that case, refuse and
        # tell the user how to start the right way.
        if "--reload" in sys.argv:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Движок {engine_norm!r} требует запуск из нужного venv. "
                    "Перезапустите сервер без --reload, например: "
                    ".venv/bin/python -m web.run --no-reload --port 8129"
                ),
            )
        _reexec_for_engine(engine_norm)

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
    # XTTS/Silero. ``enable_prosody=False`` (the default) means
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

    # ``final_path`` is built by the engine wrapper (XTTS / Silero) as a
    # *relative* path like "output/samples/...wav". We need an absolute
    # path to compare against PROJECT_ROOT, otherwise ``Path.relative_to``
    # raises ValueError ("'foo' is not in the subpath of '/abs/path'").
    # Make it absolute relative to PROJECT_ROOT first, then compute the
    # relative form safely.
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
        chunks: list[str] = []
        if len(text) > 180:
            from russian_tts_studio.utils.text_utils import chunk_text_for_tts
            chunks = chunk_text_for_tts(text)
        else:
            chunks = [text]

        await ws.send_json({"type": "started", "total_chunks": len(chunks)})

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
