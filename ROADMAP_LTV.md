# Roadmap: перенос идей из LocalText2Voice в Russian TTS Studio

> Источник вдохновения: https://github.com/estebanstifli/LocalText2Voice (MIT, v1.2.1)
> Лицензия LTV — MIT; переносить алгоритмы/паттерны безопасно. Копировать
> чужой исходный код дословно **не нужно** — реимплементируем под свою
> архитектуру (FastAPI + SPA, VoxCPM2/Silero, русский).

## Контекст

RTTS сегодня — single-engine веб-TTS для русского: VoxCPM2 + Silero-фолбэк,
клонирование голоса по референсу, WER/SpkSim QC, пост-обработка аудио,
ComfyUI-мост, WebSocket-стриминг для длинных текстов, launcher с tray-иконкой.

LTV — десктоп-продакшен для аудиокниг/подкастов: мульти-движковый, markup-язык
в тексте, Whisper-ревью с retry, подкаст-микс с музыкой, субтитры, SQLite-
проекты, MCP-сервер, Windows-инсталлер.

**Пересечение:** обе используют Whisper для QC и FFmpeg/torchaudio для пост-
обработки. **Дельта LTV:** продакшен-процессы для длинных форматов.

---

## Принципы переноса

1. **Не трогаем ядро.** VoxCPM2 + Silero + pipeline `tts_pipeline.py` остаются
   как есть. Новое — надстройки и параллельные слои.
2. **Веб, не десктоп.** PySide6 не переносим. Все фичи — через REST/WebSocket
   + SPA-страницы.
3. **Русский-first.** 11 языков UI, голосовая галерея, облачные API —
   пропускаем.
4. **Переиспользуем уже собранное.** У нас уже есть Whisper (Transcriber),
   WavLM (SpeakerSimilarityCalculator), prosody-модуль с forced-alignment
   (MMS_FA), `chunk_text_for_tts`, `normalize_numbers`, `PauseConfig`. Часть
   «новых» фич LTV — это просто экспозиция того, что мы уже считаем.
5. **Обратная совместимость.** Markup опционален; без `{{...}}` всё работает
   как раньше. SQLite-проекты — отдельный режим, не замена `/api/synthesize`.

---

## Фазы и зависимости

```
Фаза 1 (foundation) ──┬─→ Фаза 2 (long-form)  ──→ Фаза 4 (podcast)
   Markup-парсер      │    Главы + чанкинг          Audio Mix + музыка
   Pause-семантика    │    Паузы абзацев            Subtitles SRT/ASS
                      │
                      └─→ Фаза 3 (project)  ──→ Фаза 5 (QC+auto)
                           SQLite-проекты          Whisper retry/tail
                           Per-segment rebuild     Approve/discard
                                                      │
                                                      └─→ Фаза 6 (MCP)
                                                           MCP-сервер
                                                           Automation API

Фаза 7 (text-norm) — независима, можно в любой момент
Фаза 8 (packaging) — опциональна, зависит от целевой платформы
```

---

## Фаза 1 — Foundation: Markup-парсер + pause-семантика

**Цель:** дать пользователю управлять нарративом прямо из текста, не трогая
UI-формы на каждый чанк. Базис для всех последующих фаз.

### Что переносим из LTV
- Синтаксис `{{command}}` / `{{command value}}` / `{{command "q value"}}`.
- Команды: `{{pause ...}}`, `{{pause.short/medium/long}}`, `{{pause random A B}}`,
  `{{speed 0.9}}`, `{{speed.slow/normal/fast}}`, `{{volume ...}}`,
  `{{volume -3db}}`, `{{volume 80%}}`, `{{volume.normalize -16}}`,
  `{{chapter "..."}}`, `{{alias "A" "B"}}`, `{{reset}}`, `{{reset.voice}}`.
- Толерантность: неизвестные команды → warning в лог, не падаем.
- Case-insensitive имена команд; smart quotes/unicode dashes нормализация.

### Что НЕ переносим (пока)
- `{{voice "..."}}` — у нас один движок + клонирование по референсу, голос
  выбирается референс-файлом, а не именем. Можно добавить позже как alias к
  сохранённым референсам из `/api/references`.
- `{{lang ...}}` — VoxCPM2 сам определяет язык.
- `{{play ...}}` / `{{stop ...}}` — это Фаза 4 (Audio Mix), отдельно.
- `{{cmd ...}}` / `{{preset ...}}` — VoxCPM2 SDK 2.0.3 дропнул `instruct`
  (см. `voxcpm_synth.py:236-241`). Нечего прокидывать. Вернётся, если SDK
  вернёт параметр.

### Артефакты
- `russian_tts_studio/markup/__init__.py` — public API: `parse(text) -> ParsedDocument`.
- `russian_tts_studio/markup/parser.py` — regex-парсер `{{...}}` + tokenizer.
- `russian_tts_studio/markup/commands.py` — dataclasses: `Pause`, `Speed`,
  `Volume`, `Chapter`, `Alias`, `Reset`; enum `MarkupCommand`.
- `russian_tts_studio/markup/document.py` — `ParsedDocument`:
  `segments: list[NarrationSegment]`, где `NarrationSegment` = чистый текст +
  активные параметры (speed/volume/pause-config) + chapter-label.
- `tests/test_markup_parser.py` — покрытие синтаксиса + edge cases
  (вложенные кавычки, unicode, неизвестные команды).

### Точки интеграции
- `tts_pipeline.py:TTSPipeline.synthesize` — принимает либо `str`, либо
  `ParsedDocument`. Если `ParsedDocument` — итерирует `segments`, для каждого
  вызывает существующий путь VoxCPM2/Silero, склеивает WAV в конце.
- `web/app.py:/ws/synthesize` — если текст содержит `{{`, прогоняет `parse()`
  и стримит per-segment прогресс; иначе старый путь `chunk_text_for_tts`.
- `/api/synthesize` —新增 поле `enable_markup: bool = Form(False)`. При `True`
  парсит markup и возвращает массив сегментов в ответе.

### Объём
~400 строк кода + ~200 строк тестов. 1-2 дня.

### Критерий готовности
```python
doc = parse("{{chapter \"Урок 1\"}} Привет. {{pause 700ms}} Пока. {{speed 0.9}} Медленно.")
assert len(doc.segments) == 3
assert doc.segments[0].chapter == "Урок 1"
assert doc.segments[1].pause_ms == 700
assert doc.segments[2].speed == 0.9
```

---

## Фаза 2 — Long-form: главы + абзацный чанкинг

**Цель:** превратить RTTS из «пословного синтеза» в инструмент для аудиокниг.

### Что переносим из LTV
- Детект глав: Markdown-заголовки (`#`, `##`), uppercase short headings
  (полная строка в CAPS, ≤ 60 символов), явные `{{chapter "..."}}`.
- Абзацно-осознанный чанкинг: сохранять границы абзацев, не разрезать
  предложение посередине, накапливать до `max_chars` / `max_sentences`.
- Паузы между абзацами со случайным диапазоном
  (`{{pause random 500 1200}}`-стиль): 500-900 мс по умолчанию.
- Импорт `.txt`, `.md`, `.docx` (у нас сейчас только paste в textarea).

### Артефакты
- `russian_tts_studio/text/longform.py` — `detect_chapters(text) -> list[Chapter]`,
  `chunk_with_paragraphs(text, max_chars, max_sentences) -> list[Chunk]`.
- `russian_tts_studio/text/importers.py` — `import_txt`, `import_md`,
  `import_docx` (python-docx, уже в `requirements.txt`? проверить).
- Расширение `web/app.py`: `POST /api/import` — принимает файл, возвращает
  `{text, chapters: [{title, char_start, char_end}]}`.

### Точки интеграции
- Фаза 1 `ParsedDocument` уже даёт `Chapter`-маркеры. Фаза 2 учитывает их
  при чанкинге: глава = граница чанка, никогда не смешиваем главы в одном
  WAV-сегменте.
- WebSocket `/ws/synthesize` репортит `chapter` + `segment_idx/total`.

### Объём
~350 строк кода + ~150 строк тестов. 1 день. Зависит от Фазы 1.

### Критерий готовности
- Вставить `.md` с 3 главами → UI показывает дерево глав, можно выбрать
  «синтезировать только главу 2».
- Длинный текст (10 000 символов) чанкаётся без разрыва предложений.

---

## Фаза 3 — Project: SQLite-персистентность

**Цель:** из «одноразового синтеза» → редактируемый проект аудиокниги.

### Что переносим из LTV
- SQLite-схема: `projects`, `segments`, `segment_audio`, `review_results`,
  `word_timestamps` (JSON).
- Project manifest (портабельный JSON рядом с `.db`).
- Per-segment regenerate / approve / discard / rebuild.
- Хранение исходного текста, конфига на сегмент (voice/lang/speed/volume).

### Артефакты
- `russian_tts_studio/projects/__init__.py` — `Project`, `Segment`,
  `ReviewResult` dataclasses.
- `russian_tts_studio/projects/store.py` — SQLite-слой (sqlite3 stdlib,
  без ORM). Миграции — простые `CREATE TABLE IF NOT EXISTS`.
- `russian_tts_studio/projects/manager.py` — `create_project`,
  `load_project`, `list_projects`, `regenerate_segment`, `rebuild_audiobook`.
- `web/app.py`: `/api/projects` (CRUD), `/api/projects/{id}/segments`,
  `/api/projects/{id}/segments/{seg}/regenerate`, `/api/projects/{id}/rebuild`.
- Новая SPA-страница «📚 Проекты».

### Точки интеграции
- `TTSPipeline.synthesize` теперь может писать результат в `segment_audio`
  таблицу, а не только в `output/samples/`.
- Фаза 1 `ParsedDocument` → `Project` с сегментами 1:1.
- Фаза 2 главы → `Chapter`-группы в `projects` таблице.

### Объём
~600 строк кода + ~300 строк тестов. 2-3 дня. Зависит от Фазы 1-2.

### Критерий готовности
- Создать проект из длинного текста → синтезировать → перегенерировать
  segment #5 с другим `instruct` → rebuild → скачать итоговый MP3.

---

## Фаза 4 — Podcast: Audio Mix + субтитры

**Цель:** чистая дикция → готовый подкаст с музыкой + субтитры для видео.

### 4a. Audio Mix с музыкой
- `music/background/` библиотека (MP3/WAV).
- `POST /api/mix` — принимает `narration_url`, `music_url`, `voice_db`,
  `music_db`, `intro_sec`, `tail_sec`, `fade_in`, `fade_out`, `ducking: bool`,
  `loudnorm: bool`.
- Реализация — FFmpeg `sidechaincompress` (ducking) + `afade` + `loudnorm`.
- UI:波形-preview voice/music/mix, render без повторного TTS.
- `russian_tts_studio/audio/mix.py` — `mix_podcast(narration, music, cfg) -> Path`.

### 4b. Субтитры SRT + ASS
- У нас уже собираются Whisper word-timestamps в `_quality_check` — но
  сейчас `Transcriber.transcribe` возвращает только `text`, без timestamps.
- Расширить `Transcriber`: опционально возвращать word-level timestamps
  (`result["segments"][i]["words"][j]` с `start/end`).
- `russian_tts_studio/subtitles/__init__.py` — `write_srt(segments, path)`,
  `write_ass(segments, path, karaoke=True)`.
- Offset-коррекция для подкаст-микса (narration начинается после intro).
- Chapter-aware SRT: отдельный файл на главу.

### Объём
4a: ~250 строк + FFmpeg-обёртка. 4b: ~200 строк. 1-2 дня. Зависит от Фазы 3.

### Критерий готовности
- Сгенерировать аудиокнигу → нажать «Podcast Mix» → получить MP3 с музыкой,
  ducking, fades, loudnorm + SRT рядом.

---

## Фаза 5 — QC automation: Whisper retry + tail-trim

**Цель:** гарантировать качество каждого сегмента, а не просто «выбрать движок».

### Что переносим из LTV
- Per-segment статус: `approved` / `needs_review` / `needs_retry`.
- Авто-retry loop: при `needs_retry` перегенерировать, выбрать лучший
  кандидат по composite-score (WER + SpkSim + tail-metric).
- Tail-detection: необъяснимый звук после последнего выровненного Whisper-слова.
  Параметры: `safety_ms`, `warning_ms`, `retry_ms`.
- Консервативный trim: если tail > warning → обрезать, пере-ревью, принять
  только если стало лучше.

### Артефакты
- `russian_tts_studio/pipeline/review.py` — `ReviewDecision`,
  `review_segment(audio, text, thresholds) -> ReviewResult`.
- Расширение `PipelineConfig`: `retry_max: int = 2`, `tail_safety_ms`,
  `tail_warning_ms`, `tail_retry_ms`.
- Интеграция с Фазой 3: статус сегмента в SQLite.

### Объём
~400 строк + ~200 строк тестов. 2 дня. Зависит от Фазы 3.

### Критерий готовности
- Сегмент с WER=35% → авто-retry → лучший кандидат WER=12% → `approved`.

---

## Фаза 6 — MCP-сервер автоматизации

**Цель:** AI-агенты (Claude/Codex/Cursor) могут программно гнать тексты на
синтез через MCP.

### Что переносим из LTV
- MCP-инструменты: `server_info`, `list_references` (аналог `list_voices`),
  `create_audiobook`, `generate_audio`, `get_job`, `read_job_source`,
  `search_job_source`, `edit_job_source`, `replace_job_source_text`,
  `cancel_job`, `get_markup_help`.
- Paginated source read с SHA-256 concurrency checks.
- stdio-bridge для Claude Desktop + HTTP MCP-endpoint.

### Артефакты
- `russian_tts_studio/mcp/__init__.py` — MCP-инструменты как FastAPI-роуты
  + stdio-bridge.
- `mcp_stdio_bridge.py` в корне — запускает EngineHost (переиспользуем
  `_State.get_pipeline`).
- Конфиг-генератор для `claude_desktop_config.json` / `~/.codex/config.toml`.

### Объём
~500 строк. 2 дня. Зависит от Фазы 3 (jobs = projects).

### Критерий готовности
- Claude Desktop может: `create_audiobook(text)` → polling `get_job` →
  скачать MP3 → `edit_job_source` → `render_required=true` → rebuild.

---

## Фаза 7 — Text Normalization словари (независимая)

**Цель:** «200 г» → «двести граммов», «1-й» → «первый», «V век» → «пятый век».

### Что переносим из LTV
- SQLite-словарь нормализации с per-entry enable.
- Automatic rules: numbers, ordinals, dates, currencies, percentages,
  measurements, Roman numerals — с global switch и индивидуальными toggles.
- Import/export JSON для внешнего редактирования.
- Не трогаем исходный текст — нормализуется копия перед TTS.

### Артефакты
- `russian_tts_studio/text/normalization.py` — `NormalizationConfig`,
  `normalize_for_tts(text, config, language="ru") -> str`.
- SQLite-таблица `normalization_dict` (per-project или global).
- Расширение `web/app.py`: `/api/normalization` (CRUD правил),
  `/api/normalization/preview` (показать diff до/после).
- Расширение `chunk_text_for_tts`: опционально пропускать через
  `normalize_for_tts` перед разбиением.

### Объём
~450 строк + готовые русские словари-стартеры. 1-2 дня. **Независима** —
можно делать в любой момент.

### Критерий готовности
- «Я родился 5 мая 1990 года» → «Я родился пятого мая тысяча девятьсот
  девяностого года» (VoxCPM2 озвучит правильно).

---

## Фаза 8 — Packaging (опциональная)

**Цель:** Windows-инсталлер для не-Linux аудитории.

### Что переносим из LTV
- Inno Setup + PyInstaller portable build.
- SHA-256 verification + auto-update (check GitHub Release раз в 24ч).
- CPU/GPU setup profiles.

### Решение
**Только если целевая аудитория — Windows.** У тебя сейчас `launcher.py`
(AyatanaAppIndicator + tkinter) и `web/desktop.py` (pywebview) — это Linux-
нативный подход. Если остаёмся на Linux — Фаза 8 пропускается.

### Объём
~1 день на PyInstaller + Inno Setup. Требует Windows-машины для теста.

---

## Рекомендуемый порядок реализации

| Спринт | Фаза | Длительность | Зависимости |
|--------|------|--------------|-------------|
| 1 | Фаза 1 (Markup) | 1-2 дня | — |
| 1 | Фаза 7 (Text Norm) | 1-2 дня | — (параллельно) |
| 2 | Фаза 2 (Long-form) | 1 день | Фаза 1 |
| 3 | Фаза 3 (Projects) | 2-3 дня | Фаза 1-2 |
| 4 | Фаза 4 (Podcast+Subs) | 1-2 дня | Фаза 3 |
| 5 | Фаза 5 (QC auto) | 2 дня | Фаза 3 |
| 6 | Фаза 6 (MCP) | 2 дня | Фаза 3 |
| 7 | Фаза 8 (Packaging) | 1 день | опционально |

**Итого:** ~2 недели на полный продакшен-pipeline уровня LTV, с сохранением
твоего веб-подхода и фокуса на русском.

---

## Что точно НЕ переносим

- ❌ PySide6 desktop UI (у нас веб лучше)
- ❌ External voice gallery / catalog (single-engine + cloning by reference)
- ❌ Облачные API (OpenAI/ElevenLabs/Gemini/Azure) — противоречит offline-first
- ❌ 11 языков UI (русский-first)
- ❌ Мульти-движковую `BaseTTSEngine` с install/validate/remove — у нас
  достаточно текущей `base_synth.py` + voxcpm/silero
- ❌ Persistent Python workers — VoxCPM2 и так грузится один раз в `_State`

---

## Открытые вопросы

1. **Целевая платформа.** Только Linux, или Windows тоже? От этого зависит
   Фаза 8 и подход к打包.
2. **Хранилище.** SQLite в `~/.local/share/russian-tts-studio/` (как launcher
   state) или в `output/projects/` внутри проекта? LTV хранит рядом с
   проектом — портабельно, но засоряет дерево.
3. **Instruct.** VoxCPM2 SDK 2.0.3 дропнул `instruct`. Есть ли смысл ждать
   возврата, или выкинуть `{{cmd}}`/`{{preset}}` навсегда?
4. **Music library.** Свои треки или bundl'ить royalty-free? Лицензии?
5. **Whisper timestamps.** `openai-whisper` vs `faster-whisper`? LTV
   использует faster-whisper (int8 fallback на CPU). У нас openai-whisper.
   Миграция может ускорить QC в 2-3×.