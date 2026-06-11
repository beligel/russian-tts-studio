// Russian TTS Studio Russian TTS Studio — frontend logic

const API = {
  synthesize: '/api/synthesize',
  references: '/api/references',
  refUpload: '/api/references/upload',
  refDelete: (n) => `/api/references/${n}`,
  evaluate: '/api/evaluate',
  similarity: '/api/speaker-similarity',
  comfyStatus: '/api/comfyui/status',
  comfySpeakers: '/api/comfyui/speakers',
  comfySynth: '/api/comfyui/synthesize',
  comfyInstall: '/api/comfyui/install',
  comfyExport: '/api/comfyui/export-speaker',
  engines: '/api/engines',
  status: '/api/status',
  postprocess: '/api/postprocess',
  audio: (n) => `/api/audio/${n}`,
};

const state = {
  references: [],
  comfyAvailable: false,
};

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function $(id) { return document.getElementById(id); }
function $all(sel) { return document.querySelectorAll(sel); }

function showToast(message, type = 'info', duration = 3000) {
  const t = $('toast');
  t.textContent = message;
  t.className = `toast ${type}`;
  setTimeout(() => t.classList.add('hidden'), duration);
  setTimeout(() => t.classList.remove('hidden'), 0);
}

function showLoader(text = 'Работаем…') {
  $('loaderText').textContent = text;
  $('loader').classList.remove('hidden');
}
function hideLoader() {
  $('loader').classList.add('hidden');
}

function fmtSeconds(s) {
  if (s == null || isNaN(s)) return '—';
  return `${Number(s).toFixed(2)}s`;
}
function fmtPct(p) {
  if (p == null || isNaN(p)) return '—';
  return `${(p * 100).toFixed(1)}%`;
}
function classifyMetric(metric, value, thresholds) {
  if (value == null) return '';
  if (value < thresholds.good) return 'good';
  if (value < thresholds.warn) return 'warn';
  return 'bad';
}

async function apiFetch(url, options = {}) {
  const resp = await fetch(url, options);
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || `HTTP ${resp.status}`);
  }
  return resp.json();
}

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------

$all('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    $all('.tab').forEach(t => t.classList.remove('active'));
    $all('.tab-pane').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    $(`tab-${tab.dataset.tab}`).classList.add('active');
  });
});

// ---------------------------------------------------------------------------
// Status / health
// ---------------------------------------------------------------------------

async function refreshStatus() {
  try {
    const status = await apiFetch(API.status);
    const dot = $('statusDot');
    const text = $('statusText');
    dot.classList.add('ok');
    text.textContent = `${status.device.toUpperCase()} • ${status.cuda_available ? 'GPU' : 'CPU'}`;
    if (status.comfyui) text.textContent += ' • ComfyUI ✓';
    $('sysStatus').textContent = JSON.stringify(status, null, 2);
  } catch (e) {
    $('statusDot').classList.add('error');
    $('statusText').textContent = 'Offline';
  }
}

setInterval(refreshStatus, 10000);
refreshStatus();

// ---------------------------------------------------------------------------
// References
// ---------------------------------------------------------------------------

async function loadReferences() {
  try {
    const data = await apiFetch(API.references);
    state.references = data.references;
    const select = $('refSelect');
    select.innerHTML = '<option value="">— Без референса (Silero) —</option>' +
      data.references.map(r =>
        `<option value="${r.path}">${r.name} (${r.duration_sec}s)</option>`
      ).join('');
    renderReferenceList(data.references);
  } catch (e) {
    showToast(`Не удалось загрузить референсы: ${e.message}`, 'error');
  }
}

function renderReferenceList(refs) {
  const list = $('refList');
  if (!refs.length) {
    list.innerHTML = '<p class="muted">Нет референсов. Загрузите аудио для клонирования.</p>';
    return;
  }
  list.innerHTML = refs.map(r => `
    <div class="ref-card">
      <h4>${r.name}</h4>
      <div class="duration">${r.duration_sec}s</div>
      <audio controls src="${API.audio(r.name)}"></audio>
      <div class="actions">
        <button onclick="useReference('${r.path.replace(/'/g, "\\'")}')">🎤 Использовать</button>
        <button onclick="deleteReference('${r.name}')">🗑️</button>
      </div>
    </div>
  `).join('');
}

function useReference(path) {
  $('refSelect').value = path;
  $all('.tab').forEach(t => t.classList.remove('active'));
  $all('.tab-pane').forEach(p => p.classList.remove('active'));
  document.querySelector('[data-tab="synth"]').classList.add('active');
  $('tab-synth').classList.add('active');
  showToast('Референс выбран', 'success');
}

async function deleteReference(filename) {
  if (!confirm(`Удалить ${filename}?`)) return;
  try {
    await apiFetch(API.refDelete(filename), { method: 'DELETE' });
    showToast('Удалено', 'success');
    await loadReferences();
  } catch (e) {
    showToast(`Ошибка: ${e.message}`, 'error');
  }
}

$('refreshRefs').addEventListener('click', loadReferences);
$('refUpload').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  await uploadReferenceFile(file);
  e.target.value = '';
});

async function uploadReferenceFile(file) {
  const form = new FormData();
  form.append('file', file);
  showLoader('Загружаю референс…');
  try {
    const result = await apiFetch(API.refUpload, { method: 'POST', body: form });
    showToast(`Загружено: ${result.name}`, 'success');
    await loadReferences();
  } catch (e) {
    showToast(`Ошибка: ${e.message}`, 'error');
  } finally {
    hideLoader();
  }
}

// Drag-and-drop
const dropZone = $('dropZone');
dropZone.addEventListener('click', () => $('refDropInput').click());
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  if (e.dataTransfer.files.length) uploadReferenceFile(e.dataTransfer.files[0]);
});
$('refDropInput').addEventListener('change', (e) => {
  if (e.target.files[0]) uploadReferenceFile(e.target.files[0]);
});

// ---------------------------------------------------------------------------
// Synthesis
// ---------------------------------------------------------------------------

$('synthBtn').addEventListener('click', async () => {
  const text = $('textInput').value.trim();
  if (!text) { showToast('Введите текст', 'warn'); return; }

  const refPath = $('refSelect').value;
  const form = new FormData();
  form.append('text', text);
  if (refPath) form.append('reference_path', refPath);
  form.append('instruct', $('instructInput').value || '');
  form.append('speaker_fallback', $('speakerFallback').value);
  form.append('speed', $('speedInput').value || '1.0');
  form.append('enable_fallback', $('enableFallback').checked);
  form.append('enable_postprocess', $('enablePostprocess').checked);
  form.append('enable_quality_check', $('enableQualityCheck').checked);
  form.append('engine', $('engineSelect') ? $('engineSelect').value : 'xtts');

  showLoader('Синтезирую…');
  $('synthBtn').disabled = true;
  try {
    const result = await synthesizeWithRetry(form);
    renderResult(result);
  } catch (e) {
    showToast(`Ошибка синтеза: ${e.message}`, 'error');
  } finally {
    hideLoader();
    $('synthBtn').disabled = false;
  }
});

// If the requested engine is in a different venv, the server re-execs
// itself (~5-10s). The current HTTP request is dropped, the new one
// retries against the now-correct process. This wrapper handles that:
// network error / 5xx / 503 → wait 3s, ping /api/engines for health,
// then retry the synthesize POST. Up to 3 attempts.
async function synthesizeWithRetry(form, maxAttempts = 3) {
  let lastErr = null;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await apiFetch(API.synthesize, { method: 'POST', body: form });
    } catch (e) {
      lastErr = e;
      const msg = String(e.message || '');
      const looksLikeReexec = /reexec|перезапуск|503|HTTP 5\d\d|NetworkError|Failed to fetch/i.test(msg);
      if (!looksLikeReexec || attempt === maxAttempts) throw e;
      // Tell the user, wait for the new process to come up, retry.
      showLoader(`Перезапускаю движок (попытка ${attempt + 1}/${maxAttempts})…`);
      await sleep(3000);
      // Block until /api/engines responds (the new uvicorn worker is up).
      const ok = await waitForServer(8);
      if (!ok) throw new Error('Сервер не отвечает после перезапуска движка');
    }
  }
  throw lastErr;
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function waitForServer(maxSeconds) {
  for (let i = 0; i < maxSeconds; i++) {
    try {
      const resp = await fetch(API.engines, { cache: 'no-store' });
      if (resp.ok) return true;
    } catch (_) { /* still booting */ }
    await sleep(1000);
  }
  return false;
}

function renderResult(result) {
  const area = $('resultArea');
  area.classList.remove('result-empty');
  area.innerHTML = `
    <audio class="audio-player" controls src="${result.audio_url}"></audio>
    <p>
      <a class="download-link" href="${result.audio_url}" download>⬇ Скачать .wav</a>
      <span class="muted">(${result.duration_sec}s • ${result.generation_time_sec}s gen • RTF ${result.rtf})</span>
    </p>
  `;

  $('metricsArea').classList.remove('hidden');
  $('metricModel').textContent = result.model || '—';
  $('metricOutcome').textContent = result.outcome || '—';

  const m = result.metrics || {};
  $('metricWER').textContent = fmtPct(m.wer);
  $('metricWER').className = 'metric-value ' + classifyMetric('wer', m.wer, { good: 0.10, warn: 0.20 });
  $('metricCER').textContent = fmtPct(m.cer);
  $('metricCER').className = 'metric-value ' + classifyMetric('cer', m.cer, { good: 0.05, warn: 0.15 });
  $('metricSIM').textContent = m.speaker_similarity ? m.speaker_similarity.toFixed(3) : '—';
  $('metricSIM').className = 'metric-value ' + classifyMetric('sim', m.speaker_similarity, { good: 0.7, warn: 0.5 });
  $('metricRTF').textContent = result.rtf ? result.rtf.toFixed(3) : '—';
  $('metricDuration').textContent = fmtSeconds(result.duration_sec);
  $('metricSilence').textContent = m.silence_ratio != null ? fmtPct(m.silence_ratio) : '—';
  $('transcriptText').textContent = m.transcript || '(нет транскрипции)';
  showToast(`Готово: ${result.outcome}`, result.outcome === 'pass' ? 'success' : 'warn');
}

// ---------------------------------------------------------------------------
// ComfyUI integration
// ---------------------------------------------------------------------------

async function loadComfyStatus() {
  try {
    const status = await apiFetch(API.comfyStatus);
    const el = $('comfyuiStatus');
    if (!status.available) {
      el.className = 'comfyui-status warn';
      el.textContent = `❌ ComfyUI не найден. ${status.message || ''}`;
      state.comfyAvailable = false;
      return;
    }
    el.className = 'comfyui-status ok';
    el.innerHTML = `
      ✅ <strong>ComfyUI:</strong> ${status.comfyui_path}<br>
      ✅ <strong>Плагин:</strong> ${status.plugin_path}<br>
      📦 Моделей: ${status.models_count} • 🎙️ Спикеров: ${status.speakers_count}
    `;
    state.comfyAvailable = true;
    await loadComfySpeakers();
  } catch (e) {
    $('comfyuiStatus').className = 'comfyui-status error';
    $('comfyuiStatus').textContent = `Ошибка: ${e.message}`;
  }
}

async function loadComfySpeakers() {
  if (!state.comfyAvailable) return;
  try {
    const data = await apiFetch(API.comfySpeakers);
    const sel = $('comfyuiSpeakerSelect');
    sel.innerHTML = data.speakers.map(s =>
      `<option value="${s.name}">${s.name} (${s.duration_sec}s)</option>`
    ).join('') || '<option>— Нет сохранённых —</option>';

    const list = $('comfyuiSpeakers');
    if (!data.speakers.length) {
      list.innerHTML = '<p class="muted">Нет сохранённых спикеров. Экспортируйте аудио ниже.</p>';
      return;
    }
    list.innerHTML = data.speakers.map(s => `
      <div class="ref-card">
        <h4>${s.name}</h4>
        <div class="duration">${s.duration_sec}s</div>
        ${s.text ? `<p class="muted" style="font-size: 11px;">"${s.text.slice(0, 60)}…"</p>` : ''}
        <div class="actions">
          <button onclick="useComfySpeaker('${s.name}')">🎤 Использовать</button>
        </div>
      </div>
    `).join('');
  } catch (e) {
    showToast(`Ошибка загрузки спикеров: ${e.message}`, 'error');
  }
}

function useComfySpeaker(name) {
  $('comfyuiSpeakerSelect').value = name;
  $all('.tab').forEach(t => t.classList.remove('active'));
  $all('.tab-pane').forEach(p => p.classList.remove('active'));
  document.querySelector('[data-tab="comfyui"]').classList.add('active');
  $('tab-comfyui').classList.add('active');
  showToast(`Спикер: ${name}`, 'success');
}

$('comfyuiRefresh').addEventListener('click', loadComfyStatus);
$('comfyuiInstall').addEventListener('click', async () => {
  if (!confirm('Установить плагин Russian TTS Studio3 в ComfyUI?')) return;
  showLoader('Устанавливаю плагин…');
  try {
    const result = await apiFetch(API.comfyInstall, { method: 'POST' });
    showToast(`Установлен: ${result.path}`, 'success');
    await loadComfyStatus();
  } catch (e) {
    showToast(`Ошибка: ${e.message}`, 'error');
  } finally {
    hideLoader();
  }
});

$('comfyuiSynthBtn').addEventListener('click', async () => {
  const text = $('comfyuiText').value.trim();
  const speaker = $('comfyuiSpeakerSelect').value;
  if (!text) { showToast('Введите текст', 'warn'); return; }
  if (!speaker) { showToast('Выберите спикера', 'warn'); return; }

  const form = new FormData();
  form.append('text', text);
  form.append('speaker_name', speaker);

  showLoader('Синтез через ComfyUI-спикера…');
  try {
    const result = await apiFetch(API.comfySynth, { method: 'POST', body: form });
    $('comfyuiResult').className = '';
    $('comfyuiResult').innerHTML = `
      <audio class="audio-player" controls src="${result.audio_url}"></audio>
      <p class="muted">${result.model} • ${result.duration_sec}s • ${result.outcome}</p>
    `;
    showToast('Готово', 'success');
  } catch (e) {
    showToast(`Ошибка: ${e.message}`, 'error');
  } finally {
    hideLoader();
  }
});

$('comfyuiExportBtn').addEventListener('click', async () => {
  const file = $('comfyuiExportFile').files[0];
  const name = $('comfyuiExportName').value.trim();
  if (!file) { showToast('Выберите файл', 'warn'); return; }
  if (!name) { showToast('Укажите имя', 'warn'); return; }

  const form = new FormData();
  form.append('audio', file);
  form.append('name', name);
  form.append('auto_transcribe', $('comfyuiAutoTranscribe').checked);

  showLoader('Сохраняю пресет…');
  try {
    const result = await apiFetch(API.comfyExport, { method: 'POST', body: form });
    showToast(`Сохранено: ${result.name}`, 'success');
    await loadComfySpeakers();
  } catch (e) {
    showToast(`Ошибка: ${e.message}`, 'error');
  } finally {
    hideLoader();
  }
});

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

$('simBtn').addEventListener('click', async () => {
  const ref = $('simRef').files[0];
  const synth = $('simSynth').files[0];
  if (!ref || !synth) { showToast('Загрузите оба файла', 'warn'); return; }
  const form = new FormData();
  form.append('reference', ref);
  form.append('synthesized', synth);
  showLoader('Сравниваю…');
  try {
    const result = await apiFetch(API.similarity, { method: 'POST', body: form });
    $('simResult').textContent = result.speaker_similarity.toFixed(4);
    showToast(`SIM: ${result.speaker_similarity.toFixed(3)}`, 'success');
  } catch (e) {
    showToast(`Ошибка: ${e.message}`, 'error');
  } finally {
    hideLoader();
  }
});

$('evalBtn').addEventListener('click', async () => {
  const file = $('evalFile').files[0];
  if (!file) { showToast('Загрузите аудио', 'warn'); return; }
  const form = new FormData();
  form.append('file', file);
  form.append('reference_text', $('evalText').value || '');
  showLoader('Оцениваю…');
  try {
    const result = await apiFetch(API.evaluate, { method: 'POST', body: form });
    $('evalResult').innerHTML = `
      <p><strong>Транскрипция:</strong> <em>${result.transcript || '—'}</em></p>
      <p><strong>Длительность:</strong> ${result.duration_sec}s</p>
      <p><strong>Тишина:</strong> ${(result.silence_ratio * 100).toFixed(1)}%</p>
      ${result.wer != null ? `<p><strong>WER:</strong> ${(result.wer * 100).toFixed(1)}%</p>` : ''}
      ${result.cer != null ? `<p><strong>CER:</strong> ${(result.cer * 100).toFixed(1)}%</p>` : ''}
      <audio class="audio-player" controls src="${result.audio_url}"></audio>
    `;
  } catch (e) {
    showToast(`Ошибка: ${e.message}`, 'error');
  } finally {
    hideLoader();
  }
});

$('ppBtn').addEventListener('click', async () => {
  const file = $('ppFile').files[0];
  if (!file) { showToast('Загрузите файл', 'warn'); return; }
  const form = new FormData();
  form.append('file', file);
  form.append('target_dbfs', $('ppDbfs').value || '-20');
  showLoader('Обрабатываю…');
  try {
    const result = await apiFetch(API.postprocess, { method: 'POST', body: form });
    $('ppResult').innerHTML = `
      <audio class="audio-player" controls src="${result.audio_url}"></audio>
      <p class="muted">Длительность: ${result.duration_sec}s</p>
    `;
  } catch (e) {
    showToast(`Ошибка: ${e.message}`, 'error');
  } finally {
    hideLoader();
  }
});

$('refreshStatus').addEventListener('click', refreshStatus);

// Initial load
loadReferences();
loadComfyStatus();
loadEngines();

// Wire up the engine-switch button group IMMEDIATELY (synchronously) so
// clicks are never lost to a race with the async /api/engines fetch. The
// later `loadEngines()` call may still update aria-pressed to match the
// server's reported "active" engine, but user input is honoured from
// the first click.
(function initEngineSwitch() {
  const sel = $('engineSelect');
  const switchEl = $('engineSwitch');
  if (!switchEl) return;
  switchEl.querySelectorAll('.engine-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const want = btn.dataset.engine;
      // Toggle aria-pressed.
      switchEl.querySelectorAll('.engine-btn').forEach(b =>
        b.setAttribute('aria-pressed', b === btn ? 'true' : 'false')
      );
      // Make sure the <select> has a matching <option> — if loadEngines()
      // hasn't run yet (or the engine isn't in its list), the option
      // would be missing and sel.value would silently keep the old value.
      if (sel && want) {
        let opt = sel.querySelector(`option[value="${CSS.escape(want)}"]`);
        if (!opt) {
          opt = document.createElement('option');
          opt.value = want;
          opt.textContent = want;
          sel.appendChild(opt);
        }
        sel.value = want;
      }
    });
  });
})();

async function loadEngines() {
  // Two UI surfaces show the engine choice: an always-visible button group
  // (.engine-btn) and the legacy <select id="engineSelect"> inside the
  // "Advanced settings" details. Keep both in sync with /api/engines.
  // The click handlers are already wired by initEngineSwitch() above —
  // this function only refreshes the option labels / active highlight
  // from /api/engines, it does NOT re-register the handlers.
  const sel = $('engineSelect');
  const switchEl = $('engineSwitch');
  let engines = [];
  let active = 'xtts';
  try {
    const data = await apiFetch(API.engines);
    if (data && Array.isArray(data.engines) && data.engines.length) {
      engines = data.engines;
      active = data.active || data.default || 'xtts';
    }
  } catch (e) {
    // Fall back to the static options in the template; no toast to avoid
    // alarming the user if the server is just slow to respond.
    console.warn('loadEngines failed:', e);
  }

  // 1) Populate the legacy <select> (still used by synthesize())
  if (sel) {
    if (engines.length) {
      // Preserve the user's current selection across the rebuild.
      const current = sel.value;
      sel.innerHTML = '';
      for (const eng of engines) {
        const opt = document.createElement('option');
        opt.value = eng.id;
        opt.textContent = eng.label || eng.id;
        opt.title = eng.description || '';
        if (eng.id === active) opt.selected = true;
        sel.appendChild(opt);
      }
      if (current && [...sel.options].some(o => o.value === current)) {
        sel.value = current;
      }
    }
  }

  // 2) Sync the visible engine-switch button group. Never override a
  // user choice that disagrees with the server's "active" engine —
  // honour the button's current aria-pressed state.
  if (switchEl) {
    const userPicked = switchEl.querySelector('.engine-btn[aria-pressed="true"]');
    if (!userPicked) {
      switchEl.querySelectorAll('.engine-btn').forEach(btn =>
        btn.setAttribute('aria-pressed', btn.dataset.engine === active ? 'true' : 'false')
      );
    }
  }
}
