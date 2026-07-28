# Russian TTS Studio

Production-ready TTS pipeline for Russian with **VoxCPM2** voice cloning, **Higgs Audio v2** (multi-speaker + sound events), **Silero** fallback, and **ComfyUI** integration.

## 🚀 Quick start

There is **one engine** and **one venv**.

### VoxCPM2 (default, Apache-2.0)

```bash
cd /home/che/projects/russian-tts-studio
.venv/bin/python -m web.start --force-server --port 8129
# or open native window / browser:
./start.sh
# then open http://127.0.0.1:8129 and pick engine="voxcpm" in the UI
```

| | VoxCPM2 | Higgs Audio v2 |
|---|---|---|
| Venv | `.venv/` (voxcpm) | `.venv/` (higgs — upstream repo install) |
| Default port | 8129 | 8129 |
| Russian | ✅ (good prosody) | ✅ (100+ langs) |
| Voice cloning | ✅ (6-25 s) | ✅ (zero-shot) |
| Multi-speaker dialog | ❌ | ✅ ([SPEAKER0]/[SPEAKER1] tags) |
| Smart voice (no ref) | ❌ | ✅ |
| Sound events (`[laugh]`/`[music]`) | ❌ (stripped) | ✅ (native) |
| Stress marks (U+0301) | ❌ (stripped) | ✅ (honoured) |
| Sample rate | 48 kHz | 24 kHz |
| License | ✅ Apache-2.0 | ✅ Apache-2.0 (v2 3B) |
| CPU speed | ~RTF 12 (GPU: 0.5-2) | GPU recommended (≥24 GB) |
| Stress marks | ❌ autoprosoody only | ✅ U+0301 honoured |
| Explicit `speed=` | ❌ (use `instruct="(slow)"`) | via markup `{{speed 0.9}}` |

### Higgs Audio v2 (optional, Apache-2.0)

Higgs Audio v2 — text-audio foundation model from Boson AI. Adds
multi-speaker dialogs, smart-voice (no reference), and native sound
events. Requires the upstream repo installed in the same venv:

```bash
cd /home/che/projects/russian-tts-studio
git clone https://github.com/boson-ai/higgs-audio.git /tmp/higgs-audio
cd /tmp/higgs-audio
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
# then start the server and pick engine="higgs" in the UI
./start.sh
```

> **Note:** Higgs Audio **v3** is non-commercial-only and is NOT
> supported here. This wrapper uses v2 / v2.5 (Apache-2.0). v3 doesn't
> need this repo at all — it's served via SGLang-Omni or the Boson API.

## 🎯 Features

- **VoxCPM2** — OpenBMB's 2B-param TTS, zero-shot cloning, 30 languages, 48 kHz
- **Silero TTS** — lightweight Russian fallback, no cloning, CPU real-time
- **Markup-управление нарративом** — `{{pause 700ms}}`, `{{speed 0.9}}`, `{{volume -3db}}`, `{{chapter "Урок 1"}}`, `{{alias "GPT" "gee pee tee"}}`, `{{reset}}`, `{{laugh}}`, `{{bgm start}}…{{bgm end}}`, `{{stress "за́мок"}}` прямо в тексте
- **Long-form: главы + чанкинг** — импорт `.txt`/`.md`/`.docx`, детект глав (Markdown, uppercase, `{{chapter}}`), абзацно-осознанный чанкинг с паузами между абзацами
- **Нормализация текста** — SQLite-словари аббревиатур (`и т.д.` → `и так далее`), адресов (`ул.` → `улица`), авто-правила (числа, даты, валюты, проценты, измерения, римские цифры)
- **Whisper retry** — per-segment QC с WER/CER/SpkSim, tail-detection, авто-retry с выбором лучшего кандидата
- **Auto-fallback** — quality checks (WER + speaker similarity) trigger Silero if VoxCPM2 fails
- **Post-processing** — silence trimming, loudness normalization, optional denoising
- **ComfyUI bridge** — discover a ComfyUI installation, list/install the plugin, reuse saved speakers
- **Evaluation suite** — automated WER (Whisper) and speaker similarity (WavLM) metrics
- **Engine comparison** — head-to-head benchmarks against Silero and VoxCPM2

> **Note on CosyVoice3 / XTTS-v2:** Earlier versions of this project shipped CosyVoice3
> and XTTS-v2 backends. Both have been removed; only VoxCPM2 is supported. The ComfyUI
> plugin name `ComfyUI_FL-CosyVoice3` still appears in the integration code as it is
> the upstream plugin identifier.

## 📦 Installation

### Minimal (Silero only)
```bash
pip install -r requirements-minimal.txt
```

### Full (VoxCPM2 + evaluation)
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
# With voice cloning (VoxCPM2 + Silero fallback)
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
    --engines silero,voxcpm \
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
| GET | `/api/engines` | List available TTS engines (voxcpm) |
| POST | `/api/synthesize` | Synthesize with reference (multipart) |
| GET | `/api/audio/{filename}` | Stream an output audio file |
| GET / POST / DELETE | `/api/references` | List / upload / delete references |
| POST | `/api/evaluate` | WER + CER + silence analysis |
| POST | `/api/speaker-similarity` | WavLM similarity between two audios |
| GET / POST | `/api/comfyui/{status,speakers,synthesize,install,export-speaker}` | ComfyUI bridge |
| POST | `/api/postprocess` | Trim silence + loudness normalize |
| POST | `/api/markup/parse` | Parse `{{...}}` markup without synthesis (preview) |
| POST | `/api/import` | Import .txt/.md/.docx → text + chapter structure |
| POST | `/api/longform/chunk` | Chunk long-form text into TTS-safe segments |
| GET | `/api/projects` | List all projects |
| POST | `/api/projects` | Create project from source text |
| GET | `/api/projects/{id}` | Load project with chapters + segments |
| DELETE | `/api/projects/{id}` | Delete project + audio files |
| GET | `/api/projects/{id}/segments` | List segments |
| POST | `/api/projects/{id}/segments/{seg}/approve` | Mark segment approved |
| POST | `/api/projects/{id}/segments/{seg}/discard` | Mark segment needs_retry |
| POST | `/api/projects/{id}/segments/{seg}/regenerate` | Re-synthesise one segment |
| POST | `/api/projects/{id}/rebuild` | Concat approved segments → final audio |
| POST | `/api/projects/{id}/synthesize-all` | Batch-synthesise all pending segments |
| GET | `/api/music` | List background music library |
| POST | `/api/music/upload` | Upload music file |
| DELETE | `/api/music/{file}` | Remove music file |
| POST | `/api/mix` | Mix narration + music (ducking, fades, loudnorm) |
| GET | `/api/subtitles/preview` | Preview subtitle cues from text |
| GET | `/api/normalization` | List normalization dictionaries |
| GET | `/api/normalization/{id}` | Load dictionary with entries |
| POST | `/api/normalization` | Create new dictionary |
| DELETE | `/api/normalization/{id}` | Delete dictionary |
| POST | `/api/normalization/{id}/entries` | Add entry to dictionary |
| PUT | `/api/normalization/entries/{id}` | Update entry |
| DELETE | `/api/normalization/entries/{id}` | Delete entry |
| GET | `/api/normalization/{id}/export` | Export dictionary as JSON |
| POST | `/api/normalization/{id}/import` | Import entries from JSON |
| POST | `/api/normalization/preview` | Preview normalization diff |
| WS | `/ws/synthesize` | Stream progress for long texts |

## ✏️ Markup (inline narration control)

RTTS поддерживает inline-управление нарративом через `{{...}}` команды в тексте.
Markup опционален — без `{{...}}` всё работает как раньше. Включается флагом
`enable_markup=true` в `/api/synthesize` или автоматически в `/ws/synthesize`
при наличии `{{` в тексте.

```text
{{chapter "Урок 1"}}
Привет, мир!

{{pause 700ms}}
{{speed 0.9}}
Теперь медленнее и тише.

{{volume -3db}}
Тихая часть.

{{alias "GPT" "gee pee tee"}}
GPT работает хорошо.

{{laugh}}
Это смешно.

{{bgm start}}
Музыка на фоне.
{{bgm end}}

{{stress "за́мок"}}
замок большой

{{reset}}
Обычная речь снова.
```

### Команды

| Команда | Пример | Описание |
|---------|--------|----------|
| `{{pause ...}}` | `{{pause 700ms}}`, `{{pause 0.7s}}`, `{{pause.short}}`, `{{pause random 500 1200}}` | Пауза после сегмента |
| `{{speed ...}}` | `{{speed 0.9}}`, `{{speed.slow}}` (0.85), `{{speed.fast}}` (1.15) | Скорость следующего сегмента |
| `{{volume ...}}` | `{{volume -3db}}`, `{{volume 80%}}`, `{{volume.normalize -16}}` | Громкость следующего сегмента |
| `{{chapter "..."}}` | `{{chapter "Глава 1"}}` | Метка главы (для навигации) |
| `{{alias "A" "B"}}` | `{{alias "GPT" "gee pee tee"}}` | Замена текста перед TTS |
| `{{reset}}` | `{{reset}}`, `{{reset.audio}}` | Сброс state к умолчаниям |
| `{{laugh}}`, `{{cough}}`, `{{sigh}}` ... | `{{laugh}}`, `{{cough soft}}`, `{{sigh}}` | Inline звуковые события (см. ниже) |
| `{{bgm start}}` ... `{{bgm end}}` | `{{bgm start}} ... {{bgm end}}` | Фоновая музыка (span event) |
| `{{hum start}}` ... `{{hum end}}` | `{{hum start}} ... {{hum end}}` | Напевание (span event) |
| `{{stress "..."}}` | `{{stress "за́мок"}}`, `{{stress "замок" "о"}}` | Ударение в слове (см. ниже) |

Пресеты pause: `.short` (300ms), `.medium` (700ms), `.long` (1200ms).
Пресеты speed: `.slow` (0.85), `.normal` (1.0), `.fast` (1.15).

Case-insensitive, умные кавычки и unicode-тире нормализуются автоматически.
Неизвестные команды → warning в лог, генерация продолжается.

### Звуковые события

Inline-события вставляют токен в позицию команды; span-события оборачивают
регион. Список встроенных событий (вдохновлён схемой Higgs Audio):

| Категория | Команды |
|-----------|---------|
| Inline | `{{laugh}}`, `{{chuckle}}`, `{{giggle}}`, `{{cough}}`, `{{sigh}}`, `{{gasp}}`, `{{cry}}`, `{{sniffle}}`, `{{sneeze}}`, `{{yawn}}`, `{{applause}}`, `{{cheer}}` |
| Span | `{{bgm start}}…{{bgm end}}` (фоновая музыка), `{{hum start}}…{{hum end}}` (напевание), `{{sing start}}…{{sing end}}` (пение) |

Inline-события принимают необязательный модификатор: `{{laugh soft}}`,
`{{cough x2}}`. Движки, которые не понимают модификатор, его игнорируют.

**Совместимость движков:** VoxCPM2 не поддерживает звуковые события — pipeline
автоматически стрипает токены (`[laugh]`, `[music]`, ...) перед вызовом движка,
так что VoxCPM2 читает чистый текст. При подключении движка с поддержкой
(Higgs Audio и др.) токены будут передаваться как есть.

### Ударения

Две формы команды `{{stress ...}}` (алиас: `{{accent ...}}`):

1. **Слово с готовым ударением** — пользователь сам поставил знак:
   ```text
   {{stress "за́мок"}}
   ```
   Парсер подставляет это слово в текст вместо оригинала.

2. **Слово + гласная под ударением** — автодетекция позиции:
   ```text
   {{stress "замок" "о"}}    → замо́к
   {{stress "замок" "а"}}    → за́мок
   ```
   Паркер находит первое вхождение указанной гласной и вставляет
   combining acute accent (U+0301). Если гласная не найдена —
   откатывается на первую гласную в слове и добавляет warning.

Ударения накапливаются (как `{{alias}}`) и применяются ко всем последующим
сегментам. Не сбрасываются `{{reset}}` — это правило произношения, а не
параметр нарратива.

**Совместимость движков:** VoxCPM2 игнорирует U+0301 (autoprosoody) —
pipeline стрипает combining acute перед вызовом. Silero (через eSpeak)
и Higgs Audio получают текст с ударениями как есть.

**Preview без синтеза:** `POST /api/markup/parse` с текстом возвращает структуру
сегментов, главы, stresses, sound events и warnings — удобно для UI-подсветки
перед генерацией.

### Voice profiles (текстовые описания голосов)

Вдохновлён `voice_prompts/profile.yaml` из Higgs Audio. Вместо аудио-референса
пользователь описывает голос текстом — движок с поддержкой (Higgs) рендерит
описание напрямую:

```bash
# Список профилей
curl http://localhost:8129/api/voice-profiles

# Добавить профиль
curl -X POST http://localhost:8129/api/voice-profiles \
  -d "name=male_ru_calm" \
  -d "description=Мужской голос, спокойная интонация, умеренный темп."

# Использовать профиль в синтезе (Higgs)
curl -X POST http://localhost:8129/api/synthesize \
  -d "text=Привет, мир!" \
  -d "engine=higgs" \
  -d "reference_path=profile:male_ru_calm"
```

Профили хранятся в `output/reference/profiles.yaml` (редактируется вручную
или через API). При первом запуске создаётся starter-файл с 8 профилями
(4 английских из upstream Higgs + 4 русских).

**Совместимость:** VoxCPM2 не поддерживает текстовые описания — при
`profile:name` с движком voxcpm pipeline падает на Silero fallback
(его встроенные голоса). Higgs рендерит описание напрямую.

## 📚 Long-form: импорт и разбиение по главам

RTTS умеет импортировать длинные тексты и разбивать их на главы и абзацы
для осознанного синтеза аудиокниг и курсов.

### Импорт файлов

```bash
# .txt, .md, .markdown, .docx
curl -X POST http://localhost:8129/api/import -F "file=@book.md"
```

Возвращает: `{text, source_format, char_count, chapters: [{title, char_start, char_end, is_preamble}]}`.

### Чанкинг с главами

```bash
curl -X POST http://localhost:8129/api/longform/chunk \
  -d "text=# Глава 1
Первый абзац. Второе предложение.

# Глава 2
Текст второй главы." \
  -d "max_chars=200" -d "max_sentences=4"
```

Возвращает плоский список чанков с метаданными главы: `{chunks: [{text, chapter_title, is_chapter_start, char_start, speaker, turn_index}], total_chunks}`.

### Multi-speaker диалоги

`chunk_document` автоматически определяет метод чанкинга по наличию
тегов `[SPEAKER0]`/`[SPEAKER1]` в тексте. При их обнаружении используется
speaker-чанкинг (вдохновлён `prepare_chunk_text` из Higgs Audio):

```text
[SPEAKER0] Привет, как дела?
[SPEAKER1] Привет, отлично. А у тебя?
[SPEAKER0] Тоже хорошо.
```

Каждая реплика → отдельный чанк с метаданными `speaker` ("SPEAKER0") и
`turn_index` (номер реплики этого спикера). `chunk_max_num_turns>1`
группирует несколько реплик в один чанк (для движков, поддерживающих
multi-voice синтез в одном вызове — Higgs).

Принудительный выбор метода: `chunk_method="speaker"` или
`chunk_method="paragraph"` (по умолчанию `"auto"`).

### Context window (скользящий контекст)

При синтезе длинных текстов через `synthesize_markup` параметр
`context_window=N` передаёт текст предыдущих N сегментов в
`metadata["context_segments"]` каждого нового сегмента. Движки с
поддержкой длинного контекста (Higgs) используют это для связности
просодии между чанками; VoxCPM2 игнорирует поле (каждый вызов
независим). По умолчанию 0 (выключено, backward-compatible).

### Детект глав (сигналы, по приоритету)

1. `{{chapter "..."}}` markup
2. Markdown ATX: `#`, `##`, `###`
3. Setext: заголовок + подчёркивание `===` или `---`
4. Uppercase short heading (CAPS строка ≤ 60 символов + пустая строка после)

> **Note:** Строки с multi-speaker тегами (`[SPEAKER0] Привет.`)
> намеренно НЕ определяются как главы — uppercase-детектор их
> пропускает, чтобы speaker-чанкинг корректно их обрабатывал.

### Абзацный чанкинг

- Разделение по пустым строкам → абзацы
- Накопление предложений в чанк до `max_chars` / `max_sentences`
- Граница абзаца = граница чанка (естественная пауза)
- Граница главы = жёсткая граница чанка (две главы не смешиваются в одном)
- Заголовки глав вырезаются из текста чанка (это метаданные, не наррация)

### CLI

```python
from russian_tts_studio.text import import_file, detect_chapters, chunk_document

text, fmt = import_file("book.md")
chapters = detect_chapters(text)
chunks = chunk_document(text, max_chars=200, max_sentences=4)
for c in chunks:
    print(f"[{c.chapter_title}] {c.text[:60]}...")
```

## 📁 Projects (SQLite persistence)

RTTS хранит проекты в SQLite — каждый проект это исходный текст, главы,
сегменты с аудио-выходом, метрики QC и конфиг для перегенерации.

### Создание проекта

```bash
curl -X POST http://localhost:8129/api/projects \
  -d "name=Моя книга" \
  -d "source_text=# Глава 1
Привет, мир!

# Глава 2
Текст второй главы." \
  -d "max_chars=200" -d "max_sentences=4"
```

### Жизненный цикл сегмента

```
pending → approved | needs_review | error
needs_review → approved | needs_retry | error
needs_retry → pending (пере-очередь на синтез)
```

### Перегенерация сегмента

```bash
# Синтез одного сегмента с другими параметрами
curl -X POST http://localhost:8129/api/projects/{id}/segments/{seg_id}/regenerate \
  -F "speed=0.85" -F "instruct=медленно и чётко"
```

### Rebuild audiobook

```bash
# Склеить все approved-сегменты в один файл
curl -X POST http://localhost:8129/api/projects/{id}/rebuild
```

### Батч-синтез

```bash
# Синтезировать все pending-сегменты за раз
curl -X POST http://localhost:8129/api/projects/{id}/synthesize-all
```

### CLI

```python
from russian_tts_studio.projects import (
    create_project, get_project_or_404, approve_segment,
    rebuild_audiobook, list_projects,
)

# Создать проект
project = create_project("Книга", "# Глава 1\nТекст.")

# После синтеза — одобрить сегмент
approve_segment(project.id, project.segments[0].id)

# Склеить в финальный файл
out = rebuild_audiobook(project.id)
print(f"Готово: {out}")
```

## 🎧 Audio Mix (подкаст-микс с музыкой)

RTTS умеет микшировать чистую дикцию с фоновой музыкой в подкаст-стиле:
ducking (автоматическое приглушение музыки при речи), fades, loudnorm.

### Музыкальная библиотека

```bash
# Список треков
curl http://localhost:8129/api/music

# Загрузить трек
curl -X POST http://localhost:8129/api/music/upload -F "file=@bg_music.mp3"

# Удалить трек
curl -X DELETE http://localhost:8129/api/music/bg_music.mp3
```

Файлы хранятся в `music/background/`. Поддерживаемые форматы: `.mp3`, `.wav`, `.flac`, `.ogg`, `.m4a`.

### Микширование

```bash
curl -X POST http://localhost:8129/api/mix \
  -F "narration_path=output/samples/narration.wav" \
  -F "music_path=music/background/bg.mp3" \
  -F "voice_db=0" -F "music_db=-12" \
  -F "intro_sec=3" -F "fade_in=2" -F "fade_out=3" \
  -F "ducking=true" -F "loudnorm=true" \
  -F "output_format=mp3"
```

### Параметры микса

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `voice_db` | 0.0 | Усиление голоса (dB) |
| `music_db` | -12.0 | Усиление музыки (dB) |
| `intro_sec` | 0.0 | Музыкальное вступление до голоса (сек) |
| `tail_sec` | 0.0 | Музыка после конца голоса (сек) |
| `fade_in` | 2.0 | Плавное вступление музыки (сек) |
| `fade_out` | 3.0 | Плавное затухание музыки (сек) |
| `ducking` | true | Автоприглушение музыки при речи |
| `duck_depth_db` | 9.0 | Глубина приглушения (dB) |
| `loudnorm` | true | Нормализация громкости (EBU R128, -16 LUFS) |
| `output_format` | mp3 | Выходной формат: `mp3` или `wav` |

### Субтитры

```python
from russian_tts_studio.audio import write_srt, write_ass, WordTimestamp

words = [
    WordTimestamp("Привет", 0.0, 0.5),
    WordTimestamp("мир", 0.5, 1.0),
]
write_srt(words, "output.srt")
write_ass(words, "output.ass")  # караоке-стиль с word-level timing
```

## 📖 Нормализация текста (словари + авто-правила)

RTTS нормализует текст перед TTS, чтобы диктор произносил правильно.
Оригинальный текст никогда не изменяется — нормализуется копия.

### Встроенные словари

**Аббревиатуры** (SQLite, включены по умолчанию):
- `и т.д.` → `и так далее`
- `т.е.` → `то есть`
- `т.к.` → `так как`
- `руб.` → `рублей`
- `г.` → `год`
- `ул.` → `улица`
- и 50+ других

**Адресные сокращения**:
- `ул.` → `улица`
- `д.` → `дом`
- `корп.` → `корпус`
- `кв.` → `квартира`
- `просп.` → `проспект`
- и другие

### Управление словарями

```bash
# Список словарей
curl http://localhost:8129/api/normalization

# Загрузить словарь с entries
curl http://localhost:8129/api/normalization/abbreviations

# Добавить entry
curl -X POST http://localhost:8129/api/normalization/abbreviations/entries \
  -d "source=КТ" -d "replacement=компьютер"

# Включить/выключить entry
curl -X PUT http://localhost:8129/api/normalization/entries/42 \
  -d "enabled=false"

# Экспорт словаря в JSON
curl http://localhost:8129/api/normalization/abbreviations/export

# Импорт из JSON
curl -X POST http://localhost:8129/api/normalization/abbreviations/import \
  -d 'data={"entries": [{"source": "НТВ", "replacement": "телевидение"}]}'

# Preview нормализации (без изменений)
curl -X POST http://localhost:8129/api/normalization/preview \
  -d "text=Он сказал и т.д. купи 200 г муки"
```

### Авто-правила (переключаемые)

| Правило | Пример | По умолчанию |
|---------|--------|-------------|
| Числа | `200` → `двести` | ✅ |
| Порядковые | `1-й` → `первый` | ✅ |
| Даты | `01.01.2025` → `первое января` | ✅ |
| Валюты | `100 руб.` → `сто рублей` | ✅ |
| Проценты | `50%` → `пятьдесят процентов` | ✅ |
| Измерения | `5 кг` → `пять килограммов` | ✅ |
| Римские | `XVI` → `шестнадцать` | ✅ |

### CLI

```python
from russian_tts_studio.text.normalization import (
    normalize_for_tts, NormalizationConfig, get_normalization_diff,
)

cfg = NormalizationConfig()
result = normalize_for_tts("Он сказал и т.д. купи 200 г муки", cfg)
# → "Он сказал и так далее купи двести граммов муки"

# Preview diff
diff = get_normalization_diff("Он сказал и т.д.", cfg)
for change in diff["changes"]:
    print(f"{change['original']} → {change['normalized']}")
```

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
│   [Text] → [Normalize] → [VoxCPM2] ──→ [Quality?]      │
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
1. Try **VoxCPM2** zero-shot with reference audio
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

- **CPU inference** is slow: VoxCPM2 RTF ~12 (GPU: 0.5-2). GPU strongly recommended.
- **Long texts** must be chunked (built-in: 180 chars / 3 sentences max per chunk)
- **ComfyUI speaker presets** saved by the CosyVoice3 plugin (`ComfyUI_FL-CosyVoice3`) are not loadable by VoxCPM2
- **Single venv** — install everything into `.venv/`; no need to pick a backend at startup
- **No stress marks** — VoxCPM2 cannot take explicit stress; it uses autoprosoody. If you need to disambiguate омографы (зАмок/замОк), pre-transliterate to IPA before passing `text=`.

## 📁 Project structure

```
russian-tts-studio/
├── .venv/                       # Single venv (torch 2.12.0+cu130)
├── russian_tts_studio/          # Main package
│   ├── pipeline/                # Production TTS pipeline
│   ├── models/                  # TTS engine wrappers (silero, voxcpm)
│   ├── integrations/            # ComfyUI bridge
│   └── utils/                   # Audio, text, metrics utilities
├── scripts/
│   ├── comparison/              # Engine benchmarks (silero, voxcpm adapters)
│   ├── inference/               # Pipeline CLI
│   ├── integration/             # ComfyUI bridge CLI
│   └── postprocess/             # Audio post-processing
├── web/                         # FastAPI Web UI (single-page app)
│   ├── app.py                   # FastAPI backend (REST + WebSocket)
│   ├── start.py                 # Smart launcher: native window / browser / server
│   ├── desktop.py               # pywebview native window wrapper
│   ├── templates/               # index.html
│   └── static/                  # app.js, styles.css
├── tests/                       # Unit tests
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
Silero = MIT, VoxCPM2 = Apache 2.0, F5-TTS = MIT, Fish-Speech = Apache 2.0.
