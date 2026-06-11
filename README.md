# Russian TTS Studio

Production-ready TTS pipeline for Russian with **XTTS-v2** voice cloning, **Silero** fallback, **VoxCPM2** alternative engine, and **ComfyUI** integration.

## 🚀 Quick start

There are **two engines** and **two venvs**. Pick one.

### A. XTTS-v2 (default, what you've been using)

```bash
cd /home/che/projects/russian-tts-studio
./start.sh                          # opens native window or browser
# or headless server on http://127.0.0.1:8129:
.venv/bin/python -m web.start --force-server --port 8129
```

### B. VoxCPM2 (newer, better Russian prosody, Apache-2.0)

```bash
cd /home/che/projects/russian-tts-studio
.venv-voxcpm/bin/python -m web.start --force-server --port 8129
# then open http://127.0.0.1:8129 and pick engine="voxcpm" in the UI
```

**You only need to start the server once.** When a request comes in for an engine whose
venv differs from the running one, `web/app.py` re-execs into the right venv automatically
(see "Auto-reexec" below). The example above is only useful if you want to skip the
reexec-on-first-request and pre-warm the voxcpm model at startup.

| | XTTS-v2 | VoxCPM2 |
|---|---|---|
| Venv | `.venv/` (xtts) | `.venv-voxcpm/` (voxcpm) |
| Default port | 8129 (both venvs share the port — `web.app` re-execs into the right venv on first request) | 8129 |
| Russian | ✅ | ✅ (better prosody) |
| Voice cloning | ✅ (6-10 s) | ✅ (6-25 s) |
| Sample rate | 24 kHz | 48 kHz |
| License | ⚠️ CPML (non-commercial) | ✅ Apache-2.0 |
| CPU speed | ~RTF 0.5 | ~RTF 12 (GPU: 0.5-2) |
| Stress marks | ❌ vocab can't take them | ❌ autoprosoody only |
| Explicit `speed=` | ✅ | ❌ (use `instruct="(slow)"`) |

## 🎯 Features

- **XTTS-v2** — Coqui's multilingual TTS, zero-shot Russian voice cloning (6–10 sec reference audio)
- **VoxCPM2** — OpenBMB's 2B diffusion-autoregressive TTS, zero-shot cloning, 30 languages, 48 kHz
- **Silero TTS** — lightweight Russian fallback, no cloning, CPU real-time
- **Auto-fallback** — quality checks (WER + speaker similarity) trigger Silero if primary engine fails
- **Post-processing** — silence trimming, loudness normalization, optional denoising
- **ComfyUI bridge** — discover a ComfyUI installation, list/install the plugin, reuse saved speakers
- **Evaluation suite** — automated WER (Whisper) and speaker similarity (WavLM) metrics
- **Engine comparison** — head-to-head benchmarks against Silero, XTTS, VoxCPM2

> **Note on CosyVoice3:** Earlier versions of this project shipped a CosyVoice3 backend.
> It has been removed in favour of XTTS-v2; CosyVoice3 is no longer supported and is
> not installable from this repo. The ComfyUI plugin name `ComfyUI_FL-CosyVoice3`
> still appears in the integration code as it is the upstream plugin identifier.

## 🎯 Features

- **XTTS-v2** — Coqui's multilingual TTS, zero-shot Russian voice cloning (6–10 sec reference audio)
- **VoxCPM2** — OpenBMB's 2B-param TTS, zero-shot cloning, 30 languages, 48 kHz (see Quick start above)
- **Silero TTS** — lightweight Russian fallback, no cloning, CPU real-time
- **Auto-fallback** — quality checks (WER + speaker similarity) trigger Silero if primary engine fails
- **Post-processing** — silence trimming, loudness normalization, optional denoising
- **ComfyUI bridge** — discover a ComfyUI installation, list/install the plugin, reuse saved speakers
- **Evaluation suite** — automated WER (Whisper) and speaker similarity (WavLM) metrics
- **Engine comparison** — head-to-head benchmarks against Silero, XTTS, VoxCPM2

> **Note on CosyVoice3:** Earlier versions of this project shipped a CosyVoice3 backend.
> It has been removed in favour of XTTS-v2; CosyVoice3 is no longer supported and is
> not installable from this repo. The ComfyUI plugin name `ComfyUI_FL-CosyVoice3`
> still appears in the integration code as it is the upstream plugin identifier.

## 📦 Installation

### Minimal (Silero only)
```bash
pip install -r requirements-minimal.txt
```

### Full (XTTS-v2 + evaluation)
```bash
pip install -r requirements.txt
```

### With comparison engines (optional)
```bash
pip install f5-tts      # F5-TTS
pip install fish-speech # Fish-Speech
```

## 🖥️ CLI scripts (alternative to the web UI)

If you'd rather not use the web UI, the same pipeline is available as CLI commands:

### 1. Generate a test reference voice
```bash
.venv/bin/python scripts/inference/generate_test_reference.py \
    --output output/reference/ru_voice.wav \
    --speaker xenia
```

### 2. Run the production pipeline
```bash
# With voice cloning (XTTS-v2 + Silero fallback)
.venv/bin/python scripts/inference/run_pipeline.py \
    --text "Привет, это тестовая фраза на русском языке." \
    --reference output/reference/ru_voice.wav \
    --output output/samples/test.wav

# Without reference (Silero only)
.venv/bin/python scripts/inference/run_pipeline.py \
    --text "Привет, это тестовая фраза на русском языке." \
    --output output/samples/test.wav
```

### 3. Compare engines
```bash
.venv/bin/python scripts/comparison/compare_engines.py \
    --reference output/reference/ru_voice.wav \
    --engines silero,xtts,voxcpm \
    --output-dir output/comparison
```

## 🌐 Web UI

If you'd rather not touch the terminal, launch the FastAPI web app:

```bash
# Either of these works:
make web
python -m web.run

# Or with custom port:
python -m web.run --port 8080
```

### 🪟 Desktop wrapper (native window)

For a native window experience (no browser required):

```bash
pip install pywebview            # ~30 KB pure-Python wrapper
make desktop                     # or: python -m web.desktop
```

The launcher starts uvicorn in a background thread, then opens a `pywebview` window
(uses GTK WebKit on Linux, system WebKit on macOS, WebView2 on Windows). Useful flags:

```bash
python -m web.desktop --port 8080 --width 1200 --height 800
python -m web.desktop --browser         # fall back to system default browser
python -m web.desktop --no-window       # server only, no GUI (for debugging)
```

**Platform notes for `pywebview`:**
- **Linux:** `sudo apt install python3-gobject gtk-3` (in addition to the pip package)
- **macOS:** system WebKit is used automatically
- **Windows:** system WebView2 is used automatically

Open **http://localhost:8129** in your browser (or the desktop window). The UI provides:

- **🎤 Синтез** — type/paste Russian text, optionally upload a reference, get audio + WER/CER/SIM metrics
- **📚 Референсы** — upload/drag-drop reference voices, listen, delete
- **🔌 ComfyUI** — see plugin status, list saved speakers, synthesize via a ComfyUI preset, export new presets
- **🔧 Инструменты** — speaker similarity, quality evaluation, post-processing (trim + loudness)

Endpoints (all under `/api/*`, plus `/ws/synthesize` for streaming):

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/status` | Pipeline + device + ComfyUI status |
| GET | `/api/engines` | List available TTS engines (xtts, voxcpm) |
| POST | `/api/synthesize` | Synthesize with reference (multipart) |
| GET | `/api/audio/{filename}` | Stream an output audio file |
| GET / POST / DELETE | `/api/references` | List / upload / delete references |
| POST | `/api/evaluate` | WER + CER + silence analysis |
| POST | `/api/speaker-similarity` | WavLM similarity between two audios |
| GET / POST | `/api/comfyui/{status,speakers,synthesize,install,export-speaker}` | ComfyUI bridge |
| POST | `/api/postprocess` | Trim silence + loudness normalize |
| WS | `/ws/synthesize` | Stream progress for long texts |

## 🔌 ComfyUI integration

### Discover existing installation
```bash
python scripts/integration/comfyui_bridge.py --discover
python scripts/integration/comfyui_bridge.py --list-speakers
```

### Install the plugin
```bash
python scripts/integration/comfyui_bridge.py --install
```

### Use a ComfyUI speaker preset
```bash
python scripts/integration/comfyui_bridge.py \
    --synthesize --text "Привет!" --speaker "my_voice" \
    --output output/samples/synth.wav
```

### Export pipeline output as ComfyUI speaker
```bash
python scripts/integration/comfyui_bridge.py \
    --export-speaker --audio output/samples/test.wav \
    --name "my_voice"
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Russian TTS Studio                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   [Text] → [Normalize] → [XTTS-v2] ──→ [Quality?]       │
│                                   ↙        ↘           │
│                             OK            FALLBACK       │
│                             ↓                ↓          │
│                       [Postprocess]   [Silero]          │
│                             ↓                ↓          │
│                             └────→ [Output]             │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Decision logic:**
1. Try **XTTS-v2** zero-shot with reference audio
2. Run quality check (Whisper WER + WavLM speaker similarity)
3. If `WER > 20%` or `SIM < 0.5` → fallback to **Silero**
4. Post-process: trim silence → normalize loudness → optional denoise

## 🛠️ Development

```bash
make install      # Install dependencies
make test         # Run tests
make run-compare  # Compare engines
make web          # Launch Web UI on http://localhost:8129 (HTTP only)
make desktop      # Launch Web UI in a native WebView window
make start        # Smart launch (recommended) — auto-picks native window or browser
```

## ⚠️ Known limitations

- **XTTS-v2** is multilingual but Russian is not its primary language — quality varies with reference length/quality
- **CPU inference** is slow: XTTS RTF ~0.5, VoxCPM2 RTF ~12. GPU strongly recommended for VoxCPM2.
- **Long texts** must be chunked (built-in: 180 chars / 3 sentences max per chunk)
- **ComfyUI speaker presets** saved by the CosyVoice3 plugin (`ComfyUI_FL-CosyVoice3`) are not loadable by XTTS-v2 and vice versa
- **CPML license** — XTTS-v2 is Coqui Public Model License, non-commercial. Commercial use requires Coqui's permission
- **Two venvs** — VoxCPM2 and XTTS-v2 are not compatible (torch 2.3.1 vs 2.12.0). Pick the venv at server-start time, can't switch mid-process. See Quick start above.
- **No stress marks** — neither engine can take explicit stress; both use autoprosoody. If you need to disambiguate омографы (зАмок/замОк), pre-transliterate to IPA before passing `text=`.

## 📁 Project structure

```
russian-tts-studio/
├── .venv/                       # XTTS-v2 venv (torch 2.3.1+cu121)
├── .venv-voxcpm/                # VoxCPM2 venv (torch 2.12.0+cu130) — optional
├── russian_tts_studio/          # Main package
│   ├── pipeline/                # Production TTS pipeline
│   ├── models/                  # TTS engine wrappers (xtts, silero, voxcpm)
│   ├── integrations/            # ComfyUI bridge
│   └── utils/                   # Audio, text, metrics utilities
├── scripts/
│   ├── comparison/              # Engine benchmarks (silero, xtts, voxcpm adapters)
│   ├── inference/               # Pipeline CLI
│   ├── integration/             # ComfyUI bridge CLI
│   └── postprocess/             # Audio post-processing
├── web/                         # FastAPI Web UI (single-page app)
│   ├── app.py                   # FastAPI backend (REST + WebSocket)
│   ├── start.py                 # Smart launcher: native window / browser / server
│   ├── desktop.py               # pywebview native window wrapper
│   ├── templates/               # index.html
│   └── static/                  # app.js, styles.css
├── tests/                       # Unit tests (43 passing)
├── output/
│   ├── reference/               # Reference voices
│   ├── samples/                 # Generated audio
│   ├── comparison/              # Comparison reports
│   └── reports/                 # Evaluation reports
├── Makefile
├── pyproject.toml
├── requirements.txt
└── README.md
```

## 📜 License

Apache 2.0 (this project). Note: bundled engines have their own licenses —
XTTS-v2 = CPML (non-commercial), Silero = MIT, F5-TTS = MIT, Fish-Speech = Apache 2.0.
