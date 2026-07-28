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
  heartbeat: '/api/heartbeat',
  postprocess: '/api/postprocess',
  audio: (n) => `/api/audio/${n}`,
  import: '/api/import',
  voiceProfiles: '/api/voice-profiles',
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

// Heartbeat for browser-mode auto-shutdown: as long as the tab is open,
// we ping the server every 5 seconds. Closing the tab stops the pings,
// and the desktop launcher exits after a short grace period.
async function sendHeartbeat() {
  try {
    await fetch(API.heartbeat, { method: 'POST', cache: 'no-store' });
  } catch {
    // Ignore network errors — the server may already be gone.
  }
}
setInterval(sendHeartbeat, 5000);
sendHeartbeat();

// ---------------------------------------------------------------------------
// References
// ---------------------------------------------------------------------------

async function loadReferences() {
  try {
    const data = await apiFetch(API.references);
    state.references = data.references;
    const select = $('refSelect');
    if (select) {
      select.innerHTML = '<option value="">— Без референса (Silero) —</option>' +
        data.references.map(r =>
          `<option value="${r.path}">${r.name} (${r.duration_sec}s)</option>`
        ).join('');
    }
    renderReferenceList(data.references);
  } catch (e) {
    showToast(`Не удалось загрузить референсы: ${e.message}`, 'error');
  }
  // Also refresh the voice strip (refs are part of it)
  loadVoiceStrip();
}

// --- Voice strip: unified chips for refs + profiles + silero + smart ---

// Silero built-in speakers (static — same as the fallback dropdown).
const SILERO_SPEAKERS = [
  { id: 'xenia', label: 'xenia (жен)', kind: 'silero' },
  { id: 'aidar', label: 'aidar (муж)', kind: 'silero' },
  { id: 'baya', label: 'baya (жен)', kind: 'silero' },
  { id: 'kseniya', label: 'kseniya (жен)', kind: 'silero' },
  { id: 'eugene', label: 'eugene (муж)', kind: 'silero' },
];

// Currently selected voice: { type: 'ref'|'profile'|'silero'|'smart', value: string }
let selectedVoice = { type: 'ref', value: '' };

async function loadVoiceStrip() {
  const container = $('voiceChips');
  if (!container) return;

  const engine = $('engineSelect') ? $('engineSelect').value : 'voxcpm';
  const chips = [];

  // 1. Audio references from /api/references
  try {
    if (!state.references.length) {
      const data = await apiFetch(API.references);
      state.references = data.references || [];
    }
    for (const ref of state.references) {
      chips.push({
        type: 'ref',
        value: ref.path,
        label: ref.name,
        kind: 'ref',
        duration: ref.duration_sec,
      });
    }
  } catch (e) { /* refs not loaded yet — skip */ }

  // 2. Voice profiles from /api/voice-profiles (Higgs only, but show
  //    for all engines — they're just inactive for non-Higgs)
  try {
    const resp = await fetch('/api/voice-profiles');
    if (resp.ok) {
      const data = await resp.json();
      for (const p of (data.profiles || [])) {
        chips.push({
          type: 'profile',
          value: `profile:${p.name}`,
          label: p.name,
          kind: 'profile',
        });
      }
    }
  } catch (e) { /* profiles not available — skip */ }

  // 3. Silero speakers (always available as fallback)
  for (const sp of SILERO_SPEAKERS) {
    chips.push({
      type: 'silero',
      value: sp.id,
      label: sp.label,
      kind: 'silero',
    });
  }

  // 4. Smart voice (Higgs only — picks voice from text, no reference)
  if (engine === 'higgs') {
    chips.push({
      type: 'smart',
      value: '',
      label: '✨ smart voice',
      kind: 'smart',
    });
  }

  // Render chips
  container.innerHTML = '';
  for (const chip of chips) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'voice-chip';
    btn.dataset.voiceType = chip.type;
    btn.dataset.voiceValue = chip.value;
    const kindLabel = chip.kind.toUpperCase();
    const durLabel = chip.duration ? ` · ${chip.duration}s` : '';
    btn.innerHTML = `${chip.label} <span class="kind">${kindLabel}</span>`;
    btn.title = `${chip.kind}: ${chip.label}${durLabel}`;

    // Highlight profiles/smart as disabled for non-Higgs engines
    if (chip.type === 'profile' && engine !== 'higgs') {
      btn.style.opacity = '0.5';
      btn.title += ' (только Higgs)';
    }
    if (chip.type === 'smart' && engine !== 'higgs') {
      btn.style.opacity = '0.5';
    }

    // Restore active state
    if (selectedVoice.type === chip.type && selectedVoice.value === chip.value) {
      btn.classList.add('active');
    }

    btn.addEventListener('click', () => {
      container.querySelectorAll('.voice-chip').forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      selectedVoice = { type: chip.type, value: chip.value };
      // Sync the hidden refSelect for backward compat with synthesize()
      const sel = $('refSelect');
      if (sel) {
        if (chip.type === 'ref') {
          sel.value = chip.value;
        } else if (chip.type === 'silero') {
          // Silero: empty ref + set speaker_fallback
          sel.value = '';
          const fb = $('speakerFallback');
          if (fb) fb.value = chip.value;
        } else if (chip.type === 'profile') {
          // Profile: set refSelect to "profile:name" (Higgs understands it)
          sel.value = chip.value;
        } else if (chip.type === 'smart') {
          // Smart: no ref, no silero — Higgs picks from text
          sel.value = '';
        }
      }
      showToast(`Голос: ${chip.label} (${chip.kind})`, 'info', 1500);
    });
    container.appendChild(btn);
  }

  // "Add" chip — links to the Голоса tab
  const addChip = document.createElement('button');
  addChip.type = 'button';
  addChip.className = 'voice-chip add';
  addChip.textContent = '+ добавить';
  addChip.title = 'Перейти к управлению голосами';
  addChip.addEventListener('click', () => {
    $all('.tab').forEach(t => t.classList.remove('active'));
    $all('.tab-pane').forEach(p => p.classList.remove('active'));
    const refsTab = document.querySelector('[data-tab="refs"]');
    if (refsTab) { refsTab.classList.add('active'); $('tab-refs').classList.add('active'); }
  });
  container.appendChild(addChip);
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
  form.append('enable_clamp', $('enableClamp') ? $('enableClamp').checked : true);
  form.append('enable_quality_check', $('enableQualityCheck').checked);
  form.append('engine', $('engineSelect') ? $('engineSelect').value : 'voxcpm');
  // Prosody (VoxCPM-only) — always sent, ignored by Silero. Backend
  // already gates on engine=='voxcpm', so we just forward the values.
  form.append('enable_prosody', $('enableProsody') ? $('enableProsody').checked : false);
  for (const f of [
    'pauseMsComma', 'pauseMsSemicolon', 'pauseMsColon', 'pauseMsPeriod',
    'pauseMsExclamation', 'pauseMsQuestion', 'pauseMsEllipsis',
    'pauseMsWordGap',
  ]) {
    const el = $(f);
    if (el) form.append(
      f.replace(/^pauseMs/, 'pause_ms_').replace(/[A-Z]/g, c => c.toLowerCase()),
      el.value || '0',
    );
  }

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

// If the server is mid-restart (model load, watchdog cycle, etc.) the
// current HTTP request is dropped, the new one retries against the
// recovered process. This wrapper handles that: network error / 5xx /
// 503 → wait 3s, ping /api/engines for health, then retry the
// synthesize POST. Up to 3 attempts.
async function synthesizeWithRetry(form, maxAttempts = 3) {
  let lastErr = null;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await apiFetch(API.synthesize, { method: 'POST', body: form });
    } catch (e) {
      lastErr = e;
      const msg = String(e.message || '');
      const looksLikeRestart = /перезапуск|503|HTTP 5\d\d|NetworkError|Failed to fetch/i.test(msg);
      if (!looksLikeRestart || attempt === maxAttempts) throw e;
      // Tell the user, wait for the server to come back, retry.
      showLoader(`Сервер перезапускается (попытка ${attempt + 1}/${maxAttempts})…`);
      await sleep(3000);
      // Block until /api/engines responds (the new uvicorn worker is up).
      const ok = await waitForServer(8);
      if (!ok) throw new Error('Сервер не отвечает после перезапуска');
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

// --- Takes history ---
// Each synth / regenerateSelection call adds a take. Takes are kept
// in memory (session-only). Clicking a take loads its player + QC.
let takes = [];
let takeCounter = 0;
let activeTakeId = null;

function addTake(result, context) {
  // context: { type: 'full'|'selection', textPreview: string }
  takeCounter++;
  const take = {
    id: takeCounter,
    result: result,
    context: context,
    timestamp: new Date(),
  };
  takes.unshift(take);  // newest first
  // Keep max 20 takes to avoid memory bloat
  if (takes.length > 20) takes = takes.slice(0, 20);
  activeTakeId = take.id;
  renderTakes();
  renderActiveTake();
}

function renderTakes() {
  const list = $('takesList');
  const area = $('takesArea');
  if (!list || !area) return;
  if (!takes.length) {
    area.classList.add('hidden');
    return;
  }
  area.classList.remove('hidden');
  list.innerHTML = '';
  for (const take of takes) {
    const el = document.createElement('div');
    el.className = 'take' + (take.id === activeTakeId ? ' active' : '');
    const r = take.result;
    const timeStr = take.timestamp.toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const dur = r.duration_sec ? r.duration_sec.toFixed(1) + 's' : '—';
    const typeLabel = take.context.type === 'selection' ? 'выделение' : 'весь текст';
    const preview = take.context.textPreview
      ? take.context.textPreview.substring(0, 40) + (take.context.textPreview.length > 40 ? '…' : '')
      : '';
    el.innerHTML = `
      <span class="take-num">#${take.id}</span>
      <div class="take-info">
        <div>${typeLabel}${preview ? ' · ' + preview : ''}</div>
        <div class="take-meta">${dur} · ${r.outcome || '—'} · ${timeStr}</div>
      </div>
      <div class="take-actions">
        <button title="Слушать" data-action="play">▶</button>
        <button title="Скачать" data-action="download">⬇</button>
      </div>
    `;
    el.addEventListener('click', (e) => {
      const actionBtn = e.target.closest('[data-action]');
      if (actionBtn) {
        e.stopPropagation();
        if (actionBtn.dataset.action === 'download') {
          window.open(r.audio_url, '_blank');
        }
        // play = fall through to selecting the take
      }
      activeTakeId = take.id;
      renderTakes();
      renderActiveTake();
    });
    list.appendChild(el);
  }
}

function renderActiveTake() {
  const take = takes.find(t => t.id === activeTakeId);
  if (!take) return;
  const result = take.result;
  const area = $('resultArea');
  area.classList.remove('result-empty');
  let prosodyBanner = '';
  if (result.prosody_degraded) {
    prosodyBanner = `
      <div class="warning-banner" style="background:#fff3cd;border:1px solid #ffc107;padding:8px 12px;border-radius:6px;margin-bottom:8px;font-size:0.92em;">
        ⚠️ <strong>Просодия в деградированном режиме:</strong> forced-aligner (MMS_FA) недоступен — паузы расставлены пропорционально по длительности аудио, а не по реальным позициям слов. Скачайте <code>model.pt</code> через <a href="https://web.archive.org/web/2024/https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/model.pt" target="_blank">Wayback Machine</a> (прямой CDN троттлится до ~1 КБ/с) и положите в <code>~/.cache/torch/hub/checkpoints/</code>, чтобы включить точное выравнивание.
      </div>
    `;
  }
  area.innerHTML = `
    ${prosodyBanner}
    <audio class="audio-player" controls src="${result.audio_url}" autoplay></audio>
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
  showToast(`Готово: ${result.outcome} (take #${activeTakeId})`, result.outcome === 'pass' ? 'success' : 'warn');
}

function renderResult(result) {
  // Legacy entry point — now adds a take with type 'full'.
  const textPreview = $('textInput') ? $('textInput').value.substring(0, 60) : '';
  addTake(result, { type: 'full', textPreview });
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
  document.querySelector('[data-tab="tools"]').classList.add('active');
  $('tab-tools').classList.add('active');
  // Open the ComfyUI details if collapsed
  const comfyDetails = document.querySelector('#tab-tools > details');
  if (comfyDetails) comfyDetails.open = true;
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

async function loadEngines() {
  // Render engine pills in the topbar from /api/engines. Engines that
  // aren't installed (e.g. Higgs without the upstream repo) are
  // rendered as disabled pills. The hidden <select id="engineSelect">
  // is kept in sync for backward compat with synthesize().
  const pillsContainer = $('enginePills');
  const sel = $('engineSelect');
  let engines = [];
  let active = 'voxcpm';
  try {
    const data = await apiFetch(API.engines);
    if (data && Array.isArray(data.engines) && data.engines.length) {
      engines = data.engines;
      active = data.active || data.default || 'voxcpm';
    }
  } catch (e) {
    console.warn('loadEngines failed:', e);
  }
  lastLoadedEngines = engines;

  // Populate the topbar pills
  if (pillsContainer && engines.length) {
    pillsContainer.innerHTML = '';
    for (const eng of engines) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'engine-pill';
      btn.dataset.engineId = eng.id;
      const available = eng.available !== false;
      // Short label for the pill (first word or short form)
      const shortLabel = (eng.id === 'voxcpm') ? 'VoxCPM2'
                       : (eng.id === 'higgs') ? 'Higgs Audio'
                       : (eng.label || eng.id).split(' ')[0];
      btn.textContent = shortLabel;
      btn.title = eng.description || '';
      if (!available) {
        btn.disabled = true;
        btn.textContent = shortLabel + ' (не установлен)';
      }
      if (eng.id === active && available) btn.classList.add('active');
      btn.addEventListener('click', () => {
        if (btn.disabled) return;
        pillsContainer.querySelectorAll('.engine-pill').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        if (sel) sel.value = eng.id;
        applyProsodyEngineVisibility(eng.id);
        updateEngineHint(eng.id, engines);
        // Reload voice strip — available voice types depend on engine
        loadVoiceStrip();
      });
      pillsContainer.appendChild(btn);
    }
  }

  // Sync the hidden select
  if (sel) sel.value = active;

  // Show/hide the prosody panel based on the active engine.
  applyProsodyEngineVisibility(active);
  updateEngineHint(active, engines);
}

function updateEngineHint(engineId, engines) {
  const hint = $('engineHint');
  if (!hint) return;
  const eng = engines.find(e => e.id === engineId);
  if (eng && eng.description) {
    // Keep the hint short — the full description is in the option tooltip.
    hint.textContent = eng.description.split('. ').slice(0, 1).join('. ') + '.';
  } else {
    hint.textContent = '';
  }
}

// ---------------------------------------------------------------------------
// Prosody panel (VoxCPM-only) — show/hide + preset buttons
// ---------------------------------------------------------------------------

const PROSODY_PRESETS = {
  // Conservative defaults. Kept in sync with DEFAULT_PAUSE_MS in
  // russian_tts_studio/utils/prosody.py — keep both in lockstep.
  // (Shrunk from comma=500, period=900, !?=1000, ellipsis=1300, ;:=700
  //  to remove a perceived stutter on sentence-final syllables.)
  conservative: { comma: 250, semicolon: 350, colon: 350, period: 500, exclamation: 550, question: 550, ellipsis: 700 },
  // Off: all zeros
  off:          { comma: 0, semicolon: 0, colon: 0, period: 0, exclamation: 0, question: 0, ellipsis: 0, wordGap: 0 },
};
// Dramatic: ×1.5 of conservative, derived so the two stay in sync.
PROSODY_PRESETS.dramatic = Object.fromEntries(
  Object.entries(PROSODY_PRESETS.conservative).map(([k, v]) => [k, Math.round(v * 1.5)])
);

function applyProsodyEngineVisibility(engine) {
  const panel = $('prosodyPanel');
  if (!panel) return;
  const wants = (panel.dataset.engineOnly || '').split(',').map(s => s.trim());
  const isRelevant = wants.includes(engine);
  panel.hidden = !isRelevant;
  // If we just hid the panel (engine no longer VoxCPM), also uncheck
  // the master switch so a later re-display starts clean.
  if (!isRelevant) {
    const cb = $('enableProsody');
    if (cb) cb.checked = false;
  }
}

function applyProsodyPreset(name) {
  const p = PROSODY_PRESETS[name];
  if (!p) return;
  const map = {
    comma: 'pauseMsComma', semicolon: 'pauseMsSemicolon', colon: 'pauseMsColon',
    period: 'pauseMsPeriod', exclamation: 'pauseMsExclamation',
    question: 'pauseMsQuestion', ellipsis: 'pauseMsEllipsis',
    wordGap: 'pauseMsWordGap',
  };
  for (const [k, id] of Object.entries(map)) {
    const el = $(id);
    if (el) el.value = p[k];
  }
  if (name !== 'off') {
    const cb = $('enableProsody');
    if (cb) cb.checked = true;
  } else {
    const cb = $('enableProsody');
    if (cb) cb.checked = false;
  }
}

(function initProsodyPanel() {
  // Master checkbox → enable/disable the grid (so the values stay visible
  // but the user can see at a glance whether prosody is active).
  const cb = $('enableProsody');
  const grid = $('prosodyGrid');
  function refresh() {
    if (!grid) return;
    grid.style.opacity = (cb && cb.checked) ? '1' : '0.55';
    grid.style.pointerEvents = (cb && cb.checked) ? '' : 'none';
  }
  if (cb) cb.addEventListener('change', refresh);
  refresh();

  // Preset buttons
  for (const [id, name] of [
    ['prosodyPresetConservative', 'conservative'],
    ['prosodyPresetDramatic', 'dramatic'],
    ['prosodyPresetOff', 'off'],
  ]) {
    const btn = $(id);
    if (btn) btn.addEventListener('click', () => applyProsodyPreset(name));
  }

  // Sync visibility when the engine select changes.
  const sel = $('engineSelect');
  if (sel) {
    sel.addEventListener('change', () => {
      applyProsodyEngineVisibility(sel.value);
      updateEngineHint(sel.value, lastLoadedEngines);
    });
  }
})();

// Cache of the last /api/engines response — used by the change handler
// to update the hint without a re-fetch. Populated by loadEngines().
let lastLoadedEngines = [];

// ---------------------------------------------------------------------------
// Text toolbar: file upload + stress mark button + regenerate selection
// ---------------------------------------------------------------------------

// Combining acute accent (U+0301) — placed AFTER a vowel to mark stress.
const COMBINING_ACUTE = '\u0301';
// Russian + Latin vowels (lowercase + uppercase) for stress validation.
const VOWELS = 'аеёиоуыэюяaeiouyАЕЁИОУЫЭЮЯAEIOUY';

// Style presets — each sets the instruct text + speed value.
// The instruct strings use Russian (VoxCPM2 understands Russian
// voice-design hints). For engines without instruct support (Higgs,
// which uses scene_prompt instead), the speed value still applies.
const STYLE_PRESETS = {
  normal:        { instruct: '',                                           speed: 1.0, label: 'Обычно' },
  slow_solemn:   { instruct: 'Говори медленно, торжественно, с паузами',   speed: 0.7, label: 'Медленно • торжественно' },
  slow:          { instruct: 'Говори медленно и спокойно',                 speed: 0.8, label: 'Медленно' },
  fast:          { instruct: 'Говори быстро, энергично, чётко',            speed: 1.3, label: 'Быстро' },
  cheerful:      { instruct: 'Говори весело, бодро, с радостной интонацией', speed: 1.1, label: 'Весело' },
  sad:           { instruct: 'Говори грустно, печально, медленно, тихо',   speed: 0.75, label: 'Грустно' },
  nervous:       { instruct: 'Говори нервно, тревожно, отрывисто',         speed: 1.2, label: 'Нервно' },
  whisper:       { instruct: 'Говори шёпотом, тихо, интимно',              speed: 0.9, label: 'Шёпот' },
  narrator:      { instruct: 'Говори как рассказчик аудиокниги, размеренно, выразительно', speed: 0.9, label: 'Аудиокнига' },
};

function applyStylePreset(presetName) {
  const p = STYLE_PRESETS[presetName];
  if (!p) return;
  const inst = $('instructInput');
  const spd = $('speedInput');
  const spdVal = $('speedValue');
  if (inst) inst.value = p.instruct;
  if (spd) spd.value = p.speed;
  if (spdVal) spdVal.textContent = parseFloat(p.speed).toFixed(2);
  // Highlight the active preset button
  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.preset === presetName);
  });
  showToast(`Стиль: ${p.label}`, 'info', 1500);
}

async function uploadTextFile(file) {
  // Upload .txt / .md / .docx → /api/import → insert text into textarea.
  const formData = new FormData();
  formData.append('file', file);
  try {
    showToast(`Импорт ${file.name}…`);
    const resp = await fetch(API.import, { method: 'POST', body: formData });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    const ta = $('textInput');
    if (ta && data.text) {
      ta.value = data.text;
      showToast(`Импортирован ${data.source_format}: ${data.char_count} симв.` +
                (data.chapters && data.chapters.length > 1 ? `, ${data.chapters.length} глав` : ''),
                'success');
    }
  } catch (e) {
    showToast(`Ошибка импорта: ${e.message}`, 'error');
  }
}

function applyStressToSelection() {
  // Take the selected text in the textarea. Two cases:
  //  1. Single vowel selected → insert U+0301 right after it.
  //  2. Word/phrase selected → wrap as {{stress "word"}} markup.
  // If nothing is selected, show a hint toast.
  const ta = $('textInput');
  if (!ta) return;
  const start = ta.selectionStart;
  const end = ta.selectionEnd;
  const selected = ta.value.substring(start, end);

  if (!selected) {
    showToast('Выделите гласную букву или слово, затем нажмите «Ударение»', 'error');
    return;
  }

  // Case 1: single character that is a vowel → insert combining acute.
  if (selected.length === 1 && VOWELS.includes(selected)) {
    const newVal = ta.value.substring(0, start) + selected + COMBINING_ACUTE + ta.value.substring(end);
    ta.value = newVal;
    // Place cursor after the vowel + combining mark (2 chars inserted).
    ta.selectionStart = start + 2;
    ta.selectionEnd = start + 2;
    ta.focus();
    showToast(`Ударение поставлено на «${selected}»`, 'success');
    return;
  }

  // Case 2: longer selection → wrap as {{stress "..."}} markup.
  // Strip any existing combining acute from the selection first so
  // we don't double-apply. If the user selected a word that already
  // has a stress mark, the markup form uses the plain word.
  const plainWord = selected.replace(/\u0301/g, '');
  const markup = `{{stress "${plainWord}"}}`;
  const newVal = ta.value.substring(0, start) + markup + ta.value.substring(end);
  ta.value = newVal;
  // Place cursor after the markup.
  ta.selectionStart = start + markup.length;
  ta.selectionEnd = start + markup.length;
  ta.focus();
  showToast(`Вставлено {{stress "${plainWord}"}}`, 'success');
}

async function regenerateSelection() {
  // Take the selected text in the textarea and synthesize ONLY that
  // fragment (with the current reference, style, and engine settings).
  // The full-text synthesis button stays untouched — this is a quick
  // re-render of a portion for A/B comparison.
  const ta = $('textInput');
  if (!ta) return;
  const start = ta.selectionStart;
  const end = ta.selectionEnd;
  const selected = ta.value.substring(start, end).trim();

  if (!selected) {
    showToast('Выделите фрагмент текста для перегенерации', 'error');
    return;
  }

  // Build the form — same as the main synthesize button, but with
  // only the selected text.
  const refPath = $('refSelect').value;
  const form = new FormData();
  form.append('text', selected);
  if (refPath) form.append('reference_path', refPath);
  form.append('instruct', $('instructInput').value || '');
  form.append('speaker_fallback', $('speakerFallback').value);
  form.append('speed', $('speedInput').value || '1.0');
  form.append('enable_fallback', $('enableFallback').checked);
  form.append('enable_postprocess', $('enablePostprocess').checked);
  form.append('enable_clamp', $('enableClamp') ? $('enableClamp').checked : true);
  form.append('enable_quality_check', $('enableQualityCheck').checked);
  form.append('engine', $('engineSelect') ? $('engineSelect').value : 'voxcpm');
  form.append('enable_prosody', $('enableProsody') ? $('enableProsody').checked : false);
  for (const f of [
    'pauseMsComma', 'pauseMsSemicolon', 'pauseMsColon', 'pauseMsPeriod',
    'pauseMsExclamation', 'pauseMsQuestion', 'pauseMsEllipsis',
    'pauseMsWordGap',
  ]) {
    const el = $(f);
    if (el) form.append(
      f.replace(/^pauseMs/, 'pause_ms_').replace(/[A-Z]/g, c => c.toLowerCase()),
      el.value || '0',
    );
  }

  showLoader(`Перегенерирую выделенный фрагмент (${selected.length} симв.)…`);
  try {
    const result = await synthesizeWithRetry(form);
    addTake(result, { type: 'selection', textPreview: selected.substring(0, 60) });
  } catch (e) {
    showToast(`Ошибка перегенерации: ${e.message}`, 'error');
  } finally {
    hideLoader();
  }
}

// ---------------------------------------------------------------------------
// Голоса tab: voice profiles CRUD + display
// ---------------------------------------------------------------------------

async function loadProfilesList() {
  const container = $('profileList');
  if (!container) return;
  try {
    const resp = await fetch('/api/voice-profiles');
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const data = await resp.json();
    const profiles = data.profiles || [];
    if (!profiles.length) {
      container.innerHTML = '<p class="muted">Нет профилей. Добавьте описание голоса ниже.</p>';
      return;
    }
    container.innerHTML = profiles.map(p => `
      <div class="profile-card">
        <div>
          <div class="name">${p.name}</div>
          <div class="desc">${p.description}</div>
        </div>
        <div class="actions">
          <button title="Использовать" data-name="${p.name}" data-action="use">🎤</button>
          <button title="Удалить" data-name="${p.name}" data-action="delete">🗑</button>
        </div>
      </div>
    `).join('');
    // Wire up action buttons
    container.querySelectorAll('[data-action]').forEach(btn => {
      btn.addEventListener('click', async () => {
        const name = btn.dataset.name;
        const action = btn.dataset.action;
        if (action === 'use') {
          // Switch to Studio tab and select the profile chip
          $all('.tab').forEach(t => t.classList.remove('active'));
          $all('.tab-pane').forEach(p => p.classList.remove('active'));
          document.querySelector('[data-tab="synth"]').classList.add('active');
          $('tab-synth').classList.add('active');
          // Find the profile chip and activate it
          const chip = document.querySelector(`.voice-chip[data-voice-value="profile:${name}"]`);
          if (chip) chip.click();
          showToast(`Профиль ${name} выбран`, 'success');
        } else if (action === 'delete') {
          if (!confirm(`Удалить профиль ${name}?`)) return;
          try {
            await fetch(`/api/voice-profiles/${name}`, { method: 'DELETE' });
            showToast('Профиль удалён', 'success');
            loadProfilesList();
            loadVoiceStrip();
          } catch (e) {
            showToast(`Ошибка: ${e.message}`, 'error');
          }
        }
      });
    });
  } catch (e) {
    container.innerHTML = '<p class="muted">Недоступно (нужен /api/voice-profiles)</p>';
  }
}

async function addProfileFromForm() {
  const name = $('newProfileName').value.trim();
  const desc = $('newProfileDesc').value.trim();
  if (!name || !desc) {
    showToast('Заполните имя и описание', 'error');
    return;
  }
  try {
    const resp = await fetch('/api/voice-profiles', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: `name=${encodeURIComponent(name)}&description=${encodeURIComponent(desc)}`,
    });
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    showToast(`Профиль "${name}" сохранён`, 'success');
    $('newProfileName').value = '';
    $('newProfileDesc').value = '';
    loadProfilesList();
    loadVoiceStrip();
  } catch (e) {
    showToast(`Ошибка: ${e.message}`, 'error');
  }
}

// ---------------------------------------------------------------------------
// Проекты tab: long-form project management
// ---------------------------------------------------------------------------

let currentProjectId = null;

async function loadProjectsList() {
  const container = $('projectsList');
  if (!container) return;
  try {
    const resp = await fetch('/api/projects');
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const data = await resp.json();
    const projects = data.projects || [];
    if (!projects.length) {
      container.innerHTML = '<p class="muted">Нет проектов. Создайте новый выше.</p>';
      return;
    }
    container.innerHTML = projects.map(p => `
      <div class="take" style="cursor:pointer;" data-proj-id="${p.id}">
        <span class="take-num">📖</span>
        <div class="take-info">
          <div>${p.name}</div>
          <div class="take-meta">${p.segment_count || 0} сегм. · ${p.chapter_count || 0} глав · ${p.char_count || 0} симв.</div>
        </div>
        <div class="take-actions"><button>открыть →</button></div>
      </div>
    `).join('');
    container.querySelectorAll('[data-proj-id]').forEach(el => {
      el.addEventListener('click', () => loadProjectDetail(el.dataset.projId));
    });
  } catch (e) {
    container.innerHTML = `<p class="muted">Ошибка: ${e.message}</p>`;
  }
}

async function createProject() {
  const name = $('projName').value.trim();
  const source = $('projSource').value.trim();
  if (!name || !source) {
    showToast('Заполните название и текст', 'error');
    return;
  }
  const maxChars = $('projMaxChars') ? $('projMaxChars').value : 200;
  const maxSentences = $('projMaxSentences') ? $('projMaxSentences').value : 4;
  showLoader('Создаю проект…');
  try {
    const resp = await fetch('/api/projects', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: `name=${encodeURIComponent(name)}&source_text=${encodeURIComponent(source)}&max_chars=${maxChars}&max_sentences=${maxSentences}`,
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || 'HTTP ' + resp.status);
    }
    const project = await resp.json();
    showToast(`Проект "${name}" создан: ${project.segments ? project.segments.length : 0} сегм.`, 'success');
    loadProjectsList();
    loadProjectDetail(project.id);
  } catch (e) {
    showToast(`Ошибка: ${e.message}`, 'error');
  } finally {
    hideLoader();
  }
}

async function loadProjectDetail(projectId) {
  currentProjectId = projectId;
  const card = $('projectDetailCard');
  if (!card) return;
  try {
    const resp = await fetch(`/api/projects/${projectId}`);
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const project = await resp.json();
    $('projectDetailTitle').textContent = project.name || projectId;
    card.style.display = '';

    // Render chapters + segments
    const chaptersEl = $('projectChapters');
    const segments = project.segments || [];
    const chapters = project.chapters || [];
    if (!segments.length) {
      chaptersEl.innerHTML = '<p class="project-empty">Нет сегментов</p>';
      return;
    }
    // Group segments by chapter_title
    const byChapter = {};
    for (const seg of segments) {
      const ch = seg.chapter_title || '(без главы)';
      if (!byChapter[ch]) byChapter[ch] = [];
      byChapter[ch].push(seg);
    }
    chaptersEl.innerHTML = '';
    for (const [chTitle, segs] of Object.entries(byChapter)) {
      const chDiv = document.createElement('div');
      chDiv.className = 'project-chapter';
      chDiv.innerHTML = `<h3>${chTitle}</h3>`;
      for (const seg of segs) {
        const segDiv = document.createElement('div');
        segDiv.className = `project-segment ${seg.status || 'pending'}`;
        const dur = seg.duration_sec ? seg.duration_sec.toFixed(1) + 's' : '—';
        const statusLabel = seg.status === 'approved' ? '✅' : seg.status === 'error' ? '❌' : '⏳';
        const preview = (seg.text || '').substring(0, 60) + ((seg.text || '').length > 60 ? '…' : '');
        segDiv.innerHTML = `
          <span class="seg-idx">${seg.idx}</span>
          <div>
            <div class="seg-text">${preview}</div>
            <div class="seg-meta">${statusLabel} ${seg.status || 'pending'} · ${dur}${seg.rtf ? ' · RTF ' + seg.rtf.toFixed(2) : ''}</div>
          </div>
          <div class="seg-actions">
            <button title="Перегенерировать" data-seg-id="${seg.id}" data-action="regen">↻</button>
            <button title="Одобрить" data-seg-id="${seg.id}" data-action="approve">✅</button>
            <button title="Отклонить" data-seg-id="${seg.id}" data-action="discard">❌</button>
          </div>
        `;
        chDiv.appendChild(segDiv);
      }
      chaptersEl.appendChild(chDiv);
    }
    // Wire up segment action buttons
    chaptersEl.querySelectorAll('[data-action]').forEach(btn => {
      btn.addEventListener('click', () => handleSegmentAction(btn.dataset.segId, btn.dataset.action));
    });
  } catch (e) {
    showToast(`Ошибка загрузки проекта: ${e.message}`, 'error');
  }
}

async function handleSegmentAction(segmentId, action) {
  if (!currentProjectId) return;
  if (action === 'regen') {
    showLoader('Перегенерирую сегмент…');
    try {
      const refPath = $('refSelect') ? $('refSelect').value : '';
      const body = new URLSearchParams();
      body.append('speed', $('speedInput') ? $('speedInput').value : '0.9');
      if ($('instructInput') && $('instructInput').value) body.append('instruct', $('instructInput').value);
      if (refPath) body.append('reference_path', refPath);
      const resp = await fetch(`/api/projects/${currentProjectId}/segments/${segmentId}/regenerate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: body.toString(),
      });
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      showToast('Сегмент перегенерирован', 'success');
      loadProjectDetail(currentProjectId);
    } catch (e) {
      showToast(`Ошибка: ${e.message}`, 'error');
    } finally {
      hideLoader();
    }
  } else if (action === 'approve') {
    try {
      await fetch(`/api/projects/${currentProjectId}/segments/${segmentId}/approve`, { method: 'POST' });
      showToast('Одобрено', 'success');
      loadProjectDetail(currentProjectId);
    } catch (e) { showToast(`Ошибка: ${e.message}`, 'error'); }
  } else if (action === 'discard') {
    try {
      await fetch(`/api/projects/${currentProjectId}/segments/${segmentId}/discard`, { method: 'POST' });
      showToast('Отклонено', 'info');
      loadProjectDetail(currentProjectId);
    } catch (e) { showToast(`Ошибка: ${e.message}`, 'error'); }
  }
}

async function synthAllSegments() {
  if (!currentProjectId) return;
  showToast('Синтез всех сегментов — используйте API напрямую', 'info');
  // The backend has /api/projects/{id}/synthesize-all but it's a long
  // operation. For now, we just link to the API.
}

async function rebuildProject() {
  if (!currentProjectId) return;
  showLoader('Собираю WAV…');
  try {
    const resp = await fetch(`/api/projects/${currentProjectId}/rebuild`, { method: 'POST' });
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const data = await resp.json();
    showToast(`Собрано: ${data.audio_path}`, 'success');
  } catch (e) {
    showToast(`Ошибка: ${e.message}`, 'error');
  } finally {
    hideLoader();
  }
}

async function deleteProject() {
  if (!currentProjectId) return;
  if (!confirm('Удалить проект?')) return;
  try {
    await fetch(`/api/projects/${currentProjectId}`, { method: 'DELETE' });
    showToast('Проект удалён', 'success');
    $('projectDetailCard').style.display = 'none';
    loadProjectsList();
  } catch (e) { showToast(`Ошибка: ${e.message}`, 'error'); }
}

// Wire up the toolbar buttons on DOMContentLoaded.
document.addEventListener('DOMContentLoaded', () => {
  // File upload → textarea
  const fileInput = $('textFileInput');
  if (fileInput) {
    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        uploadTextFile(e.target.files[0]);
        e.target.value = '';  // reset so the same file can be re-uploaded
      }
    });
  }

  // Stress mark button
  const stressBtn = $('stressBtn');
  if (stressBtn) {
    stressBtn.addEventListener('click', applyStressToSelection);
  }

  // Regenerate selection button
  const regenBtn = $('regenSelBtn');
  if (regenBtn) {
    regenBtn.addEventListener('click', regenerateSelection);
  }

  // Style preset buttons
  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => applyStylePreset(btn.dataset.preset));
  });

  // Markup quick-insert buttons — insert at cursor position
  document.querySelectorAll('.markup-insert').forEach(btn => {
    btn.addEventListener('click', () => {
      const ta = $('textInput');
      if (!ta) return;
      const insert = btn.dataset.insert || '';
      if (!insert) return;
      const start = ta.selectionStart;
      const end = ta.selectionEnd;
      // Replace selection (or insert at cursor) with the markup text
      const before = ta.value.substring(0, start);
      const after = ta.value.substring(end);
      // Add space before if needed (not at start of line)
      const prefix = (before && !before.endsWith(' ') && !before.endsWith('\n')) ? ' ' : '';
      const suffix = (after && !after.startsWith(' ') && !after.startsWith('\n')) ? ' ' : '';
      const insertion = prefix + insert + suffix;
      ta.value = before + insertion + after;
      // Place cursor after the inserted text
      const newPos = start + insertion.length;
      ta.selectionStart = newPos;
      ta.selectionEnd = newPos;
      ta.focus();
      // Close the dropdown menu if open
      const details = btn.closest('details');
      if (details) details.open = false;
    });
  });

  // Speed slider — update display value on change
  const spd = $('speedInput');
  const spdVal = $('speedValue');
  if (spd && spdVal) {
    spd.addEventListener('input', () => {
      spdVal.textContent = parseFloat(spd.value).toFixed(2);
    });
  }

  // Clear takes button
  const clearTakes = $('clearTakesBtn');
  if (clearTakes) {
    clearTakes.addEventListener('click', () => {
      takes = [];
      activeTakeId = null;
      renderTakes();
      const area = $('resultArea');
      if (area) {
        area.classList.add('result-empty');
        area.innerHTML = '<p class="muted">Аудио появится здесь после синтеза</p>';
      }
      $('metricsArea').classList.add('hidden');
    });
  }

  // Voice profiles (Голоса tab)
  const addProfileBtn = $('addProfileBtn');
  if (addProfileBtn) addProfileBtn.addEventListener('click', addProfileFromForm);

  // Projects (Проекты tab)
  const createProjBtn = $('createProjBtn');
  if (createProjBtn) createProjBtn.addEventListener('click', createProject);
  const synthAllBtn = $('synthAllBtn');
  if (synthAllBtn) synthAllBtn.addEventListener('click', synthAllSegments);
  const rebuildBtn = $('rebuildBtn');
  if (rebuildBtn) rebuildBtn.addEventListener('click', rebuildProject);
  const deleteProjBtn = $('deleteProjBtn');
  if (deleteProjBtn) deleteProjBtn.addEventListener('click', deleteProject);

  // Load Голоса + Проекты data when their tabs are first opened
  let profilesLoaded = false;
  let projectsLoaded = false;
  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
      if (tab.dataset.tab === 'refs' && !profilesLoaded) {
        profilesLoaded = true;
        loadProfilesList();
      }
      if (tab.dataset.tab === 'projects' && !projectsLoaded) {
        projectsLoaded = true;
        loadProjectsList();
      }
    });
  });

  // Keyboard shortcut: Ctrl+Shift+A (A for Accent) on the textarea.
  const ta = $('textInput');
  if (ta) {
    ta.addEventListener('keydown', (e) => {
      if (e.ctrlKey && e.shiftKey && (e.key === 'A' || e.key === 'a' || e.key === 'F')) {
        e.preventDefault();
        applyStressToSelection();
      }
      // Ctrl+Shift+R — regenerate selection
      if (e.ctrlKey && e.shiftKey && (e.key === 'R' || e.key === 'r')) {
        e.preventDefault();
        regenerateSelection();
      }
    });
  }
});
