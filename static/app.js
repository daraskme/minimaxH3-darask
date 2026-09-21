'use strict';

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const ACTIVE_STATES = new Set(['queued', 'running', 'cancelling']);
const TURBO_PATTERN = /(turbo|lightx2v|pdd|fasth3|taomate)/i;
const TURBO_LORAS = {
  turbo8: 'minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors',
  turbo4: 'minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors',
};
const PRESETS = {
  balanced: { steps: 20 },
  quality: { steps: 30 },
  turbo8: { steps: 8, turbo: true },
  turbo4: { steps: 4, turbo: true },
};
const STATUS_LABELS = {
  queued: '待機中', running: '生成中', cancelling: '中止しています',
  cancelled: 'キャンセル済み', failed: '失敗', completed: '完了',
};

const state = {
  connected: false,
  engineReady: false,
  engine: null,
  inventoryLoaded: false,
  inventory: { models: [], loras: [], latent_upscalers: [] },
  generationOptions: {
    native_fps: 24, min_frames: 124, max_frames: 345,
    frame_alignment: 17, frame_remainder: 5, sizes: [], durations: [],
  },
  engineControl: { timer: null, version: 0, controller: null, suspended: false },
  currentMode: 'generate',
  jobs: [],
  sourceJobSignature: '',
  activePreviewJobId: null,
  postprocessCapabilities: null,
  postSources: { upscale: null, interpolate: null },
  videoUploadOps: {
    upscale: { version: 0, controller: null, pending: false, previewUrl: null },
    interpolate: { version: 0, controller: null, pending: false, previewUrl: null },
  },
  preset: 'balanced',
  uploads: { first_frame: null, last_frame: null },
  uploadOps: {
    first_frame: { version: 0, controller: null, pending: false },
    last_frame: { version: 0, controller: null, pending: false },
  },
  previewUrls: { first_frame: null, last_frame: null },
  dragRow: null,
  historyLoading: false,
  submitted: false,
};

async function api(path, options = {}) {
  const response = await fetch(path, { cache: 'no-store', ...options });
  const contentType = response.headers.get('content-type') || '';
  const body = contentType.includes('application/json') ? await response.json() : null;
  if (!response.ok) {
    const detail = body?.detail;
    const message = Array.isArray(detail)
      ? detail.map((item) => item.msg || String(item)).join('\n')
      : detail || `リクエストに失敗しました（HTTP ${response.status}）`;
    throw new Error(message);
  }
  return body;
}

function itemValue(item) { return typeof item === 'string' ? item : (item?.name || item?.path || item?.id || ''); }
function formatBytes(bytes) {
  if (!Number.isFinite(Number(bytes)) || Number(bytes) <= 0) return '';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = Number(bytes); let unit = 0;
  while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit += 1; }
  return `${value.toFixed(unit >= 3 ? 1 : 0)} ${units[unit]}`;
}
function outputUrl(path) {
  return `/api/outputs/${String(path).split('/').map(encodeURIComponent).join('/')}`;
}
function clamp(value, min, max, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.min(max, Math.max(min, number)) : fallback;
}
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function button(label, className, handler) {
  const node = el('button', className, label);
  node.type = 'button'; node.addEventListener('click', handler);
  return node;
}
function toast(message, kind = '') {
  const node = el('div', `toast ${kind}`.trim(), message);
  $('#toasts').append(node);
  window.setTimeout(() => node.remove(), 4200);
}
function setButtonBusy(node, busy, busyLabel) {
  if (!node.dataset.label) node.dataset.label = node.textContent.trim();
  node.disabled = busy;
  node.textContent = busy ? (busyLabel || node.dataset.label) : node.dataset.label;
}

function bindExclusivePlayback(video) {
  if (!video || video.dataset.exclusivePlayback === 'true') return;
  video.dataset.exclusivePlayback = 'true';
  video.addEventListener('play', () => {
    $$('video').forEach((other) => { if (other !== video && !other.paused) other.pause(); });
  });
}

function releaseCardMedia(card) {
  const video = card?.querySelector('video');
  if (!video) return;
  video.pause();
  video.removeAttribute('src');
  video.load();
}

function selectHistoryPreview(job) {
  if (!job?.output_path) return;
  $$('video').forEach((video) => video.pause());
  state.activePreviewJobId = job.id;
  renderHistory(state.jobs);
}

function modeLabel(kind) {
  return { generate: '生成', upscale: 'アップスケール', interpolate: 'フレーム補間' }[kind] || '処理';
}
function postMethodLabel(kind, id) {
  const found = state.postprocessCapabilities?.[kind]?.methods?.find((method) => method.id === id);
  if (found?.label) return found.label;
  return { realesrgan_x2plus: 'Real-ESRGAN x2plus', rife_v4_26: 'RIFE 4.26', lanczos: 'Lanczos' }[id] || id || modeLabel(kind);
}

function activateMode(mode, focus = false) {
  if (!['generate', 'upscale', 'interpolate'].includes(mode)) return;
  state.currentMode = mode;
  $$('.mode-tab').forEach((tab) => {
    const active = tab.dataset.mode === mode;
    tab.classList.toggle('selected', active); tab.setAttribute('aria-selected', String(active));
  });
  $$('[data-mode-panel]').forEach((panel) => { panel.hidden = panel.dataset.modePanel !== mode; });
  $('.control-column').hidden = mode !== 'generate';
  updateHeaderStatus();
  if (focus) {
    window.scrollTo({ top: 0, behavior: 'smooth' });
    const target = mode === 'generate' ? $('#prompt') : $(`#${mode}Source`);
    window.setTimeout(() => target?.focus({ preventScroll: true }), 180);
  }
}

function setConnection(mode, title, detail) {
  const box = $('#engineState');
  box.classList.remove('ready', 'error');
  if (mode) box.classList.add(mode);
  $('#engineStateTitle').textContent = title;
  $('#engineStateDetail').textContent = detail;
}

function updateHeaderStatus() {
  const engine = state.engine || {};
  if (!state.connected) {
    setConnection('error', '接続できません', 'H3 Studioが起動しているか確認してください'); return;
  }
  if (state.currentMode !== 'generate') {
    const mode = state.currentMode;
    const methods = (state.postprocessCapabilities?.[mode]?.methods || []).filter((method) => method.available);
    if (methods.length) setConnection('ready', `${modeLabel(mode)}準備完了`, methods.map((method) => method.label).join(' · '));
    else {
      const reasons = state.postprocessCapabilities?.[mode]?.methods?.map((method) => method.reason).filter(Boolean) || [];
      setConnection('', `${modeLabel(mode)}を準備しています`, reasons[0] || '利用できる方式を確認しています');
    }
    return;
  }
  if (!state.engineReady) {
    setConnection('error', 'エンジンの準備が必要です', engine.error || engine.gpu || '実行環境を確認してください'); return;
  }
  const readyModels = (state.inventory.models || []).filter(modelIsReady);
  if (!state.inventoryLoaded) {
    setConnection('', 'モデルを確認しています', engine.gpu || 'ローカルpipelineを検証中'); return;
  }
  if (!readyModels.length) {
    const candidate = state.inventory.models?.[0];
    setConnection('', 'モデルを準備しています', candidate?.reason || '完全なH3 pipelineを待っています'); return;
  }
  const vram = formatBytes(engine.vram_bytes);
  setConnection('ready', '生成エンジン準備完了', `${engine.gpu || 'GPU'}${vram ? ` · ${vram}` : ''}`);
}

function setRuntime(status) {
  const engine = status?.engine || {};
  state.connected = true;
  state.engineReady = Boolean(engine.ready);
  state.engine = engine;
  const gpu = engine.gpu || 'GPUを検出できません';
  const vram = formatBytes(engine.vram_bytes);
  updateHeaderStatus();

  const fields = [
    ['GPU', gpu],
    ['VRAM', vram || '取得できません'],
    ['Engine', engine.engine || 'standalone-diffusers'],
    ['Precision', String(engine.precision || 'int8').toUpperCase()],
  ];
  const list = $('#runtimeList'); list.replaceChildren();
  fields.forEach(([term, value]) => {
    const row = el('div'); const dt = el('dt', '', term); const dd = el('dd', '', value);
    dd.title = value; row.append(dt, dd); list.append(row);
  });

  const availableAttention = Array.isArray(engine.attention_backends) ? engine.attention_backends : [];
  $('#runtimeSummary').textContent = `${String(engine.precision || 'INT8').toUpperCase()} · ${availableAttention.includes('sage') ? 'Sage' : 'SDPA'}`;
  const attention = $('#attention');
  [...attention.options].forEach((option) => {
    option.disabled = availableAttention.length > 0 && !availableAttention.includes(option.value);
  });
  if (attention.selectedOptions[0]?.disabled) {
    attention.value = availableAttention.includes('sdpa') ? 'sdpa' : availableAttention[0] || 'sdpa';
  }

  const profiles = Array.isArray(engine.memory_profiles) ? engine.memory_profiles : ['auto', 'low_memory'];
  const memory = $('#memoryProfile'); const oldProfile = memory.value;
  memory.replaceChildren();
  profiles.forEach((profile) => memory.add(new Option(profile === 'low_memory' ? 'Low memory' : 'Auto', profile)));
  const recommended = engine.recommended_memory_profile || 'auto';
  memory.value = profiles.includes(oldProfile) ? oldProfile : recommended;

  const tags = $('#featureTags'); tags.replaceChildren();
  const featureData = [
    ['INT8', String(engine.precision || '').toLowerCase() === 'int8'],
    ['BF16 compute', String(engine.compute_dtype || engine.dtype || '').toLowerCase().includes('bf16')],
    ['SageAttention', availableAttention.includes('sage')],
    ['SDPA', availableAttention.includes('sdpa')],
    [`Memory: ${memory.value}`, true],
  ];
  featureData.forEach(([name, active]) => tags.append(el('span', `feature-tag${active ? ' active' : ''}`, name)));

  const vc = status?.vc_attention || {};
  const vcText = $('#vcNote p span');
  vcText.textContent = vc.reason || '研究実装との同等性を確認できないため利用できません。';
  updateGenerateAvailability();
  renderEngineControl();
}

async function refreshStatus() {
  try { setRuntime(await api('/api/status')); }
  catch (error) {
    state.connected = false; state.engineReady = false; state.engine = null;
    updateHeaderStatus();
    updateGenerateAvailability();
  }
}

function selectedModel() {
  const name = $('#model').value;
  return state.inventory.models.find((item) => itemValue(item) === name) || null;
}
function modelIsReady(model) { return Boolean(model && (typeof model === 'string' || model.ready === true)); }
function engineControl() { return state.engine?.runtime || {}; }
function loadedRuntime() { return engineControl().runtime || {}; }
function loadedModelMatchesSelection() {
  const runtime = loadedRuntime();
  return Boolean(
    runtime.loaded && runtime.model === $('#model').value
    && runtime.attention === $('#attention').value
    && runtime.memory_profile === $('#memoryProfile').value
  );
}
function loadedLorasMatchSelection() {
  const loaded = loadedRuntime().loras || [];
  const desired = selectedLoras().filter((item) => item.enabled).map((item) => ({ path: item.path, weight: Number(item.weight) }));
  return JSON.stringify(loaded.map((item) => ({ path: item.path, weight: Number(item.weight) }))) === JSON.stringify(desired);
}
function renderEngineControl() {
  const control = engineControl(); const runtime = loadedRuntime(); const trigger = $('#loadModel');
  if (!trigger) return;
  const ready = modelIsReady(selectedModel());
  const busy = ['queued', 'running'].includes(control.state);
  const loaded = loadedModelMatchesSelection();
  trigger.classList.toggle('loaded', loaded && !busy);
  trigger.classList.toggle('loading', busy);
  trigger.classList.toggle('load-error', control.state === 'error');
  trigger.disabled = !ready || busy;
  const title = $('b', trigger); const hint = $('#loadModelHint');
  if (busy) {
    title.textContent = control.kind === 'loras' ? 'LoRAを適用中' : 'モデルを読み込み中';
    hint.textContent = control.phase || 'GPUキューで準備しています';
  } else if (control.state === 'error') {
    title.textContent = '読み込みを再試行'; hint.textContent = control.error || '読み込みに失敗しました';
  } else if (loaded) {
    title.textContent = '読み込み済み';
    const count = Array.isArray(runtime.loras) ? runtime.loras.length : 0;
    hint.textContent = `${runtime.attention === 'sage' ? 'SageAttention' : 'SDPA'} · LoRA ${count}件`;
  } else {
    title.textContent = 'モデルを読み込む';
    hint.textContent = control.state === 'pending_model' ? control.phase : '選択したモデルをGPU実行環境へ準備';
  }
  const health = $('#modelHealth');
  if (runtime.loaded && !busy && selectedModel()) {
    health.classList.toggle('ready', loaded);
    $('span:last-child', health).textContent = loaded ? 'GPU実行環境に読み込み済み' : '別のモデルが読み込み済み';
  }
  updateGenerateAvailability();
}
function updateModelHealth() {
  const model = selectedModel(); const health = $('#modelHealth'); const text = $('span:last-child', health);
  health.classList.remove('ready', 'error');
  if (!model) {
    health.classList.add('error'); text.textContent = '利用できるH3 pipelineがありません';
  } else if (modelIsReady(model)) {
    health.classList.add('ready'); text.textContent = '必要なファイルを検証済み';
  } else {
    text.textContent = model.reason || (Number.isFinite(model.progress) ? `モデルを取得中 · ${Math.round(model.progress * 100)}%` : '必要なファイルが揃っていません');
  }
  renderEngineControl(); updateGenerateAvailability(); updateHeaderStatus();
}
function fillModels(items) {
  const select = $('#model'); const previous = select.value; select.replaceChildren();
  if (!items.length) {
    select.add(new Option('利用できるモデルがありません', '')); select.disabled = true; updateModelHealth(); return;
  }
  items.forEach((item) => {
    const name = itemValue(item); const ready = modelIsReady(item);
    const option = new Option(`${name}${ready ? '' : ' — 未準備'}`, name);
    option.disabled = !ready; select.add(option);
  });
  select.disabled = false;
  const readyItems = items.filter(modelIsReady);
  if (items.some((item) => itemValue(item) === previous)) {
    select.value = previous; delete select.dataset.missingModel;
  } else if (select.dataset.missingModel && previous) {
    const missing = new Option(`${previous} — 見つかりません`, previous);
    missing.disabled = true; select.add(missing); select.value = previous;
  } else {
    delete select.dataset.missingModel;
    select.value = itemValue(readyItems[0] || items[0]);
  }
  updateModelHealth();
}
async function refreshInventory(showToast = false, silent = false) {
  const refresh = $('#refresh');
  if (!silent) { refresh.classList.add('spinning'); refresh.disabled = true; }
  try {
    const inventory = await api('/api/inventory');
    state.inventoryLoaded = true;
    state.inventory = { ...inventory, models: inventory.models || [], loras: inventory.loras || [], latent_upscalers: inventory.latent_upscalers || [] };
    $('#modelRoot').textContent = inventory.root || '未設定';
    $('#modelRoot').title = inventory.root || '';
    fillModels(state.inventory.models);
    refreshLoraOptions();
    fillLatentUpscalers();
    if (showToast) toast('モデル一覧を更新しました');
  } catch (error) {
    state.inventoryLoaded = false; state.inventory = { models: [], loras: [], latent_upscalers: [] }; fillModels([]); fillLatentUpscalers();
    $('#modelHealth span:last-child').textContent = error.message;
    if (showToast) toast(error.message, 'error');
  } finally { if (!silent) { refresh.classList.remove('spinning'); refresh.disabled = false; } }
}

function loraLabel(item) {
  const name = itemValue(item); const base = name.split('/').pop().replace(/\.safetensors$/i, '');
  const short = base.replace(/^minimax_h3_/i, 'H3 · ').replace(/_comfyui_bf16$/i, '').replace(/_bf16$/i, '');
  const size = formatBytes(item?.size);
  return `${short}${size ? ` · ${size}` : ''}`;
}

function selectedLoras() {
  return $$('.lora-row').map((row) => ({
    path: $('.lora-name', row).value,
    weight: clamp($('.lora-weight', row).value, -4, 4, 1),
    enabled: $('.lora-enabled', row).checked,
  })).filter((item) => item.path);
}
function scheduleLoraApply() {
  if (state.engineControl.suspended) return;
  window.clearTimeout(state.engineControl.timer);
  state.engineControl.timer = window.setTimeout(applyLoras, 420);
}
async function applyLoras() {
  const version = ++state.engineControl.version;
  state.engineControl.controller?.abort();
  const controller = new AbortController(); state.engineControl.controller = controller;
  try {
    const result = await api('/api/engine/loras', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ loras: selectedLoras() }), signal: controller.signal,
    });
    if (version !== state.engineControl.version) return;
    if (state.engine) state.engine.runtime = result;
    renderEngineControl();
  } catch (error) {
    if (error.name !== 'AbortError' && version === state.engineControl.version) toast(`LoRA設定: ${error.message}`, 'error');
  } finally {
    if (version === state.engineControl.version) state.engineControl.controller = null;
  }
}
async function loadSelectedModel() {
  if (!modelIsReady(selectedModel())) return;
  const trigger = $('#loadModel'); trigger.disabled = true;
  try {
    const result = await api('/api/engine/load', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model: $('#model').value, attention: $('#attention').value,
        memory_profile: $('#memoryProfile').value, loras: selectedLoras(),
      }),
    });
    if (state.engine) state.engine.runtime = result;
    renderEngineControl(); toast('モデル読み込みをGPUキューに追加しました');
  } catch (error) { toast(error.message, 'error'); renderEngineControl(); }
}
function loraAvailable(path) {
  return (state.inventory.loras || []).some((item) => itemValue(item) === path);
}
function populateLoraSelect(select, current) {
  select.replaceChildren();
  (state.inventory.loras || []).forEach((item) => {
    const name = itemValue(item);
    select.add(new Option(loraLabel(item), name));
  });
  if (current && !loraAvailable(current)) {
    const missing = new Option(`${current} — 見つかりません`, current);
    missing.dataset.missing = 'true'; select.add(missing);
  }
  if (current) select.value = current;
  select.dataset.missing = String(Boolean(select.value && !loraAvailable(select.value)));
  select.title = select.value;
}
function turboLoraForPreset(preset) {
  const expected = TURBO_LORAS[preset];
  if (!expected) return null;
  return (state.inventory.loras || []).find((item) => itemValue(item).toLowerCase().endsWith(expected)) || null;
}
function hasTurboLora(preset = state.preset) {
  const expected = TURBO_LORAS[preset];
  if (!expected) return selectedLoras().some((item) => item.enabled && TURBO_PATTERN.test(item.path));
  return selectedLoras().some((item) => item.enabled && item.path.toLowerCase().endsWith(expected));
}
function updatePresetCompatibility() {
  const compatible = Object.keys(TURBO_LORAS).filter((preset) => Boolean(turboLoraForPreset(preset)));
  $$('[data-preset^="turbo"]').forEach((node) => { node.disabled = !compatible.includes(node.dataset.preset) && !hasTurboLora(node.dataset.preset); });
  const note = $('#presetNote');
  const acceleration = selectedLoras().filter((item) => item.enabled && TURBO_PATTERN.test(item.path));
  const selectedPreset = Object.keys(TURBO_LORAS).find((preset) => acceleration.length === 1 && acceleration[0].path.toLowerCase().endsWith(TURBO_LORAS[preset]));
  if (selectedPreset && state.preset !== selectedPreset) setPreset(selectedPreset);
  const unverified = acceleration.length === 1 && !selectedPreset;
  note.textContent = unverified ? 'この高速化LoRAは独立エンジンで未検証です。' : compatible.length ? `検証済みの${compatible.map((name) => name === 'turbo4' ? 'Turbo 4' : 'Turbo 8').join(' / ')} LoRAを利用できます。` : 'Turbo 4 / 8には、対応する公式Turbo LoRAが必要です。';
  if (PRESETS[state.preset]?.turbo && !hasTurboLora(state.preset)) setPreset('balanced');
  updateGenerateAvailability();
}
function setPreset(name) {
  if (!PRESETS[name]) return;
  if (!PRESETS[name].turbo) {
    $$('.lora-row').forEach((row) => {
      if (TURBO_PATTERN.test($('.lora-name', row).value)) $('.lora-enabled', row).checked = false;
    });
  }
  if (PRESETS[name].turbo && !hasTurboLora(name)) {
    const item = turboLoraForPreset(name);
    if (!item) return;
    const path = itemValue(item);
    $$('.lora-row').forEach((row) => {
      const currentPath = $('.lora-name', row).value;
      if (currentPath !== path && TURBO_PATTERN.test(currentPath)) $('.lora-enabled', row).checked = false;
    });
    const existing = $$('.lora-row').find((row) => $('.lora-name', row).value === path);
    if (existing) $('.lora-enabled', existing).checked = true;
    else addLora(path, 1, true);
  }
  state.preset = name; $('#steps').value = PRESETS[name].steps;
  $('#steps').readOnly = Boolean(PRESETS[name].turbo);
  $$('#presets button').forEach((node) => {
    const selected = node.dataset.preset === name;
    node.classList.toggle('selected', selected); node.setAttribute('aria-pressed', String(selected));
  });
  scheduleLoraApply();
  updateGenerateAvailability();
}
function refreshLoraOptions() {
  const items = state.inventory.loras || [];
  $$('.lora-name').forEach((select) => {
    populateLoraSelect(select, select.value);
  });
  $('#addLora').disabled = items.length === 0 || $$('.lora-row').length >= 16;
  updateLoraEmpty(); updatePresetCompatibility();
}
function moveLora(row, direction) {
  const sibling = direction < 0 ? row.previousElementSibling : row.nextElementSibling;
  if (!sibling) return;
  if (direction < 0) sibling.before(row); else sibling.after(row);
  $('.lora-handle', row).focus();
  scheduleLoraApply();
}
function addLora(value = '', weight = 1, enabled = true) {
  if ($$('.lora-row').length >= 16) { toast('LoRAは最大16件です', 'error'); return; }
  const items = state.inventory.loras || [];
  if (!items.length && !value) { toast('models/loras にLoRAが見つかりません', 'error'); return; }
  const row = $('#loraTemplate').content.firstElementChild.cloneNode(true);
  const select = $('.lora-name', row);
  populateLoraSelect(select, value);
  if (!value) {
    const used = new Set($$('.lora-name').map((node) => node.value));
    const unused = [...select.options].find((option) => !used.has(option.value));
    if (unused) select.value = unused.value;
  }
  select.dataset.missing = String(Boolean(select.value && !loraAvailable(select.value)));
  select.title = select.value;
  $('.lora-weight', row).value = clamp(weight, -4, 4, 1);
  $('.lora-enabled', row).checked = Boolean(enabled);
  $('.lora-remove', row).addEventListener('click', () => { row.remove(); updateLoraEmpty(); updatePresetCompatibility(); scheduleLoraApply(); });
  select.addEventListener('change', () => {
    select.dataset.missing = String(!loraAvailable(select.value));
    select.title = select.value; updatePresetCompatibility(); scheduleLoraApply();
  });
  $('.lora-enabled', row).addEventListener('change', () => { updatePresetCompatibility(); scheduleLoraApply(); });
  $('.lora-weight', row).addEventListener('input', scheduleLoraApply);
  $('.lora-handle', row).addEventListener('keydown', (event) => {
    if (!event.altKey || !['ArrowUp', 'ArrowDown'].includes(event.key)) return;
    event.preventDefault(); moveLora(row, event.key === 'ArrowUp' ? -1 : 1);
  });
  row.addEventListener('dragstart', (event) => { state.dragRow = row; row.classList.add('dragging'); event.dataTransfer.effectAllowed = 'move'; });
  row.addEventListener('dragend', () => { row.classList.remove('dragging'); $$('.drag-over').forEach((item) => item.classList.remove('drag-over')); state.dragRow = null; });
  row.addEventListener('dragover', (event) => { event.preventDefault(); if (state.dragRow !== row) row.classList.add('drag-over'); });
  row.addEventListener('dragleave', () => row.classList.remove('drag-over'));
  row.addEventListener('drop', (event) => {
    event.preventDefault(); row.classList.remove('drag-over');
    if (!state.dragRow || state.dragRow === row) return;
    const rect = row.getBoundingClientRect();
    if (event.clientY < rect.top + rect.height / 2) row.before(state.dragRow); else row.after(state.dragRow);
    scheduleLoraApply();
  });
  $('#loraList').append(row); updateLoraEmpty(); updatePresetCompatibility(); scheduleLoraApply();
}
function updateLoraEmpty() {
  const count = $$('.lora-row').length;
  $('#loraEmpty').hidden = count > 0;
  $('#addLora').disabled = (state.inventory.loras || []).length === 0 || count >= 16;
}

function fillLatentUpscalers() {
  const select = $('#latentRefineModel'); const toggle = $('#latentRefineEnabled');
  const previous = select.value; const models = state.inventory.latent_upscalers || [];
  select.replaceChildren();
  models.forEach((item) => {
    const option = new Option(`${itemValue(item)}${item.ready ? '' : ` — ${item.reason || '非対応'}`}`, itemValue(item));
    option.disabled = !item.ready; select.add(option);
  });
  const ready = models.filter((item) => item.ready);
  select.value = ready.some((item) => itemValue(item) === previous) ? previous : itemValue(ready[0]);
  select.disabled = ready.length === 0;
  toggle.disabled = ready.length === 0;
  if (!ready.length) toggle.checked = false;
  $('#latentRefineSettings').hidden = !toggle.checked;
  $('#latentRefineNote').textContent = ready.length
    ? '低解像度の24ch潜在表現を学習済みモデルで拡大し、指定サイズで低ノイズ再精製します。初期値はOFFです。'
    : '対応する検証済み24chモデルが未検出のため利用できません。';
  updateGenerateAvailability();
}

function renderGenerationOptions() {
  const sizeHost = $('#sizePresets'); const durationHost = $('#durationPresets');
  sizeHost.replaceChildren(); durationHost.replaceChildren();
  (state.generationOptions.sizes || []).forEach((preset) => {
    const node = button(`${preset.label}  ${preset.width}×${preset.height}`, '', () => {
      $('#width').value = preset.width; $('#height').value = preset.height; normalizedGeometry(true); renderGenerationOptions();
    });
    node.classList.toggle('selected', Number($('#width').value) === preset.width && Number($('#height').value) === preset.height);
    sizeHost.append(node);
  });
  (state.generationOptions.durations || []).forEach((preset) => {
    const isMaximum = Number(preset.frames) === Number(state.generationOptions.max_frames);
    const label = `${isMaximum ? '24 FPS · ' : ''}${preset.label}  ${preset.frames}f · ${Number(preset.seconds).toFixed(2)}秒`;
    const node = button(label, '', () => {
      $('#frames').value = preset.frames; normalizedGeometry(true); renderGenerationOptions();
    });
    node.classList.toggle('selected', Number($('#frames').value) === preset.frames);
    durationHost.append(node);
  });
}
async function refreshGenerationOptions() {
  try {
    const options = await api('/api/generation/options');
    state.generationOptions = { ...state.generationOptions, ...options };
  } catch (_) {
    state.generationOptions.sizes = [
      { label: '横・高速', width: 960, height: 544 }, { label: '縦・高速', width: 544, height: 960 },
      { label: '正方形・高速', width: 704, height: 704 },
    ];
    state.generationOptions.durations = [124, 243, 345].map((frames) => ({ frames, label: frames === 345 ? '最大' : frames === 243 ? '標準' : '短い', seconds: frames / 24 }));
  }
  renderGenerationOptions(); normalizedGeometry(false);
}

function normalizedGeometry(commit = false) {
  const options = state.generationOptions;
  const width = Math.round(clamp($('#width').value, 256, 2048, 960) / 32) * 32;
  const height = Math.round(clamp($('#height').value, 256, 2048, 544) / 32) * 32;
  const minFrames = Number(options.min_frames || 124); const maxFrames = Number(options.max_frames || 345);
  const alignment = Number(options.frame_alignment || 17); const remainder = Number(options.frame_remainder || 5);
  const rawFrames = clamp($('#frames').value, minFrames, maxFrames, minFrames);
  const frames = Math.min(maxFrames, Math.max(minFrames, Math.ceil((rawFrames - remainder) / alignment) * alignment + remainder));
  if (commit) { $('#width').value = width; $('#height').value = height; $('#frames').value = frames; }
  const fps = Number(options.native_fps || 24);
  $('#resolved').textContent = `${width} × ${height} · ${frames}f · ${(frames / fps).toFixed(2)}秒 · ${fps} FPS（推奨）`;
  return { width, height, frames };
}

function clearUpload(kind) {
  const isFirst = kind === 'first_frame'; const slot = $(isFirst ? '#firstSlot' : '#lastSlot');
  const operation = state.uploadOps[kind];
  operation.version += 1; operation.controller?.abort(); operation.controller = null; operation.pending = false;
  if (state.previewUrls[kind]) URL.revokeObjectURL(state.previewUrls[kind]);
  state.uploads[kind] = null; state.previewUrls[kind] = null;
  slot.classList.remove('ready', 'uploading', 'upload-error');
  $('.frame-preview', slot).replaceChildren(el('span', '', '＋'));
  $('.frame-copy small', slot).textContent = isFirst ? 'PNG・JPEG・WebP / 32MBまで' : '単独指定にも対応';
  $('.upload-state', slot).textContent = '';
  $('input[type="file"]', slot).value = '';
  updateGenerateAvailability();
}
function restoreUpload(kind, token) {
  clearUpload(kind);
  if (!token) return;
  const slot = $(kind === 'first_frame' ? '#firstSlot' : '#lastSlot');
  state.uploads[kind] = token;
  slot.classList.add('ready');
  $('.frame-copy small', slot).textContent = '履歴に保存された入力画像';
  $('.upload-state', slot).textContent = '再利用します';
}
async function uploadFrame(kind, file) {
  const slot = $(kind === 'first_frame' ? '#firstSlot' : '#lastSlot');
  const allowed = new Set(['image/png', 'image/jpeg', 'image/webp']);
  if (!allowed.has(file.type)) { $('.upload-state', slot).textContent = 'PNG・JPEG・WebPを選択してください'; slot.classList.add('upload-error'); return; }
  if (file.size > 32 * 1024 * 1024) { $('.upload-state', slot).textContent = '32MB以下の画像を選択してください'; slot.classList.add('upload-error'); return; }
  clearUpload(kind);
  const operation = state.uploadOps[kind]; const version = operation.version;
  operation.controller = new AbortController(); operation.pending = true;
  slot.classList.add('uploading'); $('.upload-state', slot).textContent = 'アップロード中…'; updateGenerateAvailability();
  const previewUrl = URL.createObjectURL(file); state.previewUrls[kind] = previewUrl;
  const image = new Image(); image.src = previewUrl; image.alt = '';
  $('.frame-preview', slot).replaceChildren(image);
  try {
    const form = new FormData(); form.append('file', file);
    const result = await api('/api/uploads', { method: 'POST', body: form, signal: operation.controller.signal });
    if (operation.version !== version || !operation.pending) return;
    state.uploads[kind] = result.id; slot.classList.remove('uploading'); slot.classList.add('ready');
    $('.frame-copy small', slot).textContent = `${file.name} · ${formatBytes(file.size)}`;
    $('.upload-state', slot).textContent = 'アップロード済み';
  } catch (error) {
    if (operation.version !== version || error.name === 'AbortError') return;
    state.uploads[kind] = null; slot.classList.remove('uploading'); slot.classList.add('upload-error');
    $('.upload-state', slot).textContent = error.message;
  } finally {
    if (operation.version === version) {
      operation.pending = false; operation.controller = null; updateGenerateAvailability();
    }
  }
}

function sourceMetaFromJob(job) {
  const resolved = job?.resolved || {}; const request = job?.request || {};
  return {
    name: String(job?.output_path || `${modeLabel(job?.kind)} ${String(job?.id || '').slice(0, 8)}`).split('/').pop(),
    width: resolved.output_width ?? resolved.width ?? request.width ?? null,
    height: resolved.output_height ?? resolved.height ?? request.height ?? null,
    fps: resolved.output_fps ?? resolved.fps ?? request.fps ?? null,
    duration_seconds: resolved.duration_seconds ?? resolved.output_duration_seconds ?? null,
    has_audio: resolved.has_audio,
  };
}

function formatDuration(seconds) {
  const value = Number(seconds);
  if (!Number.isFinite(value)) return '—';
  if (value < 60) return `${value.toFixed(2)}秒`;
  return `${Math.floor(value / 60)}:${String(Math.floor(value % 60)).padStart(2, '0')}`;
}

function clearPostSource(mode) {
  const operation = state.videoUploadOps[mode];
  operation.version += 1; operation.controller?.abort(); operation.controller = null; operation.pending = false;
  if (operation.previewUrl) URL.revokeObjectURL(operation.previewUrl);
  operation.previewUrl = null; state.postSources[mode] = null;
  const block = $(`[data-source-block="${mode}"]`); block.classList.remove('ready', 'uploading');
  const video = $(`#${mode}Preview`); video.pause(); video.removeAttribute('src'); video.load(); video.hidden = true;
  $(`#${mode}Source`).value = ''; $(`#${mode}File`).value = '';
  $(`[data-clear-source="${mode}"]`).hidden = true;
  renderSourceSpec(mode); updatePostAvailability(mode);
}

function setPostSource(mode, source, previewUrl) {
  const operation = state.videoUploadOps[mode];
  if (operation.previewUrl && operation.previewUrl !== previewUrl) URL.revokeObjectURL(operation.previewUrl);
  operation.previewUrl = source.type === 'upload' ? previewUrl : null;
  state.postSources[mode] = source;
  const block = $(`[data-source-block="${mode}"]`); block.classList.remove('uploading'); block.classList.add('ready');
  const video = $(`#${mode}Preview`); video.src = previewUrl; video.hidden = false; video.load();
  $(`[data-clear-source="${mode}"]`).hidden = false;
  renderSourceSpec(mode); updatePostAvailability(mode);
}

function setPostSourceFromJob(mode, job, switchMode = true) {
  if (!job?.id || !job.output_path || job.status !== 'completed') { toast('完了した映像だけを入力にできます', 'error'); return; }
  clearPostSource(mode);
  $(`#${mode}Source`).value = job.id;
  setPostSource(mode, { type: 'job', id: job.id, meta: sourceMetaFromJob(job) }, outputUrl(job.output_path));
  if (switchMode) { activateMode(mode, true); toast(`${modeLabel(job.kind)}映像を${modeLabel(mode)}の入力に設定しました`); }
}

function renderSourceSpec(mode) {
  const source = state.postSources[mode]; const meta = source?.meta || {};
  const list = $(`#${mode}SourceSpec`); const values = [
    ['入力', meta.width && meta.height ? `${meta.width} × ${meta.height}` : '—'],
    ['FPS', Number.isFinite(Number(meta.fps)) ? Number(meta.fps).toFixed(3).replace(/\.0+$/, '') : '—'],
    ['長さ', formatDuration(meta.duration_seconds)],
  ];
  list.replaceChildren(...values.map(([term, value]) => {
    const row = el('div'); row.append(el('dt', '', term), el('dd', '', value)); return row;
  }));
  updateResultSpec(mode);
}

function updateResultSpec(mode) {
  const meta = state.postSources[mode]?.meta || {}; const target = $(`#${mode}ResultSpec`);
  if (!state.postSources[mode]) { target.textContent = '入力映像を選択してください'; return; }
  if (mode === 'upscale') {
    const scale = Number($('#upscaleScale').value || 2);
    if (scale === 1) {
      target.textContent = meta.width && meta.height
        ? `${meta.width} × ${meta.height} · 元の解像度とFPSを維持（Neural Rendering）`
        : '1× · 元の解像度とFPSを維持';
      return;
    }
    target.textContent = meta.width && meta.height
      ? `${meta.width} × ${meta.height} → ${meta.width * scale} × ${meta.height * scale} · FPS維持`
      : `${scale}× · アスペクト比とFPSを維持`;
  } else {
    const factor = Number($('#interpolateFactor').value || 2);
    target.textContent = Number.isFinite(Number(meta.fps))
      ? `${Number(meta.fps).toFixed(3).replace(/\.0+$/, '')} → ${(Number(meta.fps) * factor).toFixed(3).replace(/\.0+$/, '')} FPS · 解像度維持`
      : `${factor}× FPS · 解像度と長さを維持`;
  }
}

function availablePostMethod(mode) {
  const methods = state.postprocessCapabilities?.[mode]?.methods || [];
  return methods.find((method) => method.id === $(`#${mode}Method`).value && method.available);
}

function fillPostMethods(mode) {
  const capability = state.postprocessCapabilities?.[mode] || {}; const methods = capability.methods || [];
  const select = $(`#${mode}Method`); const previous = select.value; select.replaceChildren();
  const available = methods.filter((method) => method.available);
  if (!available.length) select.add(new Option('利用できる方式がありません', ''));
  else available.forEach((method) => select.add(new Option(method.label, method.id)));
  const preferred = [previous, capability.default_method].find((id) => available.some((method) => method.id === id));
  select.value = preferred || available[0]?.id || ''; select.disabled = available.length === 0;
  const selected = availablePostMethod(mode); const factorSelect = $(mode === 'upscale' ? '#upscaleScale' : '#interpolateFactor');
  const previousFactor = factorSelect.value; factorSelect.replaceChildren();
  const acceptedFactors = mode === 'upscale' ? [1, 2, 3, 4] : [2, 3, 4];
  const factors = (selected?.factors || [2]).filter((factor) => acceptedFactors.includes(Number(factor)));
  (factors.length ? factors : [2]).forEach((factor) => {
    const label = Number(factor) === 1 ? '1×（元の解像度）' : `${factor}×`;
    factorSelect.add(new Option(label, String(factor)));
  });
  factorSelect.value = factors.map(String).includes(previousFactor) ? previousFactor : factors.map(String).includes('2') ? '2' : String(factors[0] || 2);
  factorSelect.disabled = !selected;
  const unavailable = methods.filter((method) => !method.available);
  const note = $(`#${mode}Capability`); note.classList.toggle('unavailable', !selected);
  if (selected) {
    const description = selected.description || (selected.kind === 'ai' ? 'AIモデル' : '従来方式');
    note.textContent = `${selected.label} · ${description} 元映像は変更せず、新しいファイルとして保存します。`;
    if (unavailable.length) note.textContent += ` 利用不可: ${unavailable.map((method) => `${method.label}（${method.reason || '未検証'}）`).join('、')}`;
  } else {
    note.textContent = unavailable.map((method) => `${method.label}: ${method.reason || '実行環境を確認できません'}`).join(' / ') || '対応エンジンが見つかりません。';
  }
  updateResultSpec(mode); updatePostAvailability(mode);
}

async function refreshPostprocessCapabilities() {
  try {
    state.postprocessCapabilities = await api('/api/postprocess/capabilities');
  } catch (error) {
    state.postprocessCapabilities = { upscale: { methods: [] }, interpolate: { methods: [] } };
    $('#upscaleCapability').textContent = `アップスケール機能を確認できません: ${error.message}`;
    $('#interpolateCapability').textContent = `フレーム補間機能を確認できません: ${error.message}`;
  }
  fillPostMethods('upscale'); fillPostMethods('interpolate'); updateHeaderStatus();
}

async function uploadVideoSource(mode, file) {
  const allowedTypes = new Set(['video/mp4', 'application/mp4', 'application/octet-stream', '']);
  const allowedExtension = /\.mp4$/i.test(file.name);
  if ((!allowedTypes.has(file.type) && !allowedExtension) || file.size > 16 * 1024 ** 3) {
    $(`#${mode}Error`).textContent = file.size > 16 * 1024 ** 3 ? '動画は16GB以下にしてください。' : 'MP4動画を選択してください。'; return;
  }
  clearPostSource(mode);
  const operation = state.videoUploadOps[mode]; const version = operation.version;
  operation.controller = new AbortController(); operation.pending = true;
  const block = $(`[data-source-block="${mode}"]`); block.classList.add('uploading');
  $(`#${mode}Error`).textContent = '動画を読み込んでいます…'; updatePostAvailability(mode);
  const previewUrl = URL.createObjectURL(file); operation.previewUrl = previewUrl;
  const video = $(`#${mode}Preview`); video.src = previewUrl; video.hidden = false; block.classList.add('ready'); video.load();
  try {
    const form = new FormData(); form.append('file', file);
    const uploaded = await api('/api/video-uploads', { method: 'POST', body: form, signal: operation.controller.signal });
    if (operation.version !== version || !operation.pending) return;
    setPostSource(mode, { type: 'upload', id: uploaded.id, meta: uploaded }, previewUrl);
    $(`#${mode}Source`).value = ''; $(`#${mode}Error`).textContent = '';
    toast(`${file.name}を入力映像に設定しました`);
  } catch (error) {
    if (operation.version !== version || error.name === 'AbortError') return;
    $(`#${mode}Error`).textContent = error.message; block.classList.remove('ready', 'uploading');
    video.hidden = true; state.postSources[mode] = null;
  } finally {
    if (operation.version === version) { operation.pending = false; operation.controller = null; updatePostAvailability(mode); }
  }
}

function postRequest(mode) {
  const source = state.postSources[mode];
  return {
    source: source?.type === 'job' ? { job_id: source.id } : { upload_id: source?.id },
    method: $(`#${mode}Method`).value,
    [mode === 'upscale' ? 'scale' : 'factor']: Number($(mode === 'upscale' ? '#upscaleScale' : '#interpolateFactor').value),
  };
}

function postProblem(mode) {
  if (state.videoUploadOps[mode].pending) return '動画のアップロード完了を待ってください。';
  if (!state.postSources[mode]) return '入力映像を選択してください。';
  if (!availablePostMethod(mode)) return '利用できる処理方式がありません。';
  return '';
}

function updatePostAvailability(mode) {
  const target = $(mode === 'upscale' ? '#enqueueUpscale' : '#enqueueInterpolate');
  target.disabled = Boolean(postProblem(mode)) || target.dataset.busy === 'true';
}

async function enqueuePostprocess(mode) {
  const errorNode = $(`#${mode}Error`); const problem = postProblem(mode); errorNode.textContent = '';
  if (problem) { errorNode.textContent = problem; return; }
  const target = $(mode === 'upscale' ? '#enqueueUpscale' : '#enqueueInterpolate');
  target.dataset.busy = 'true'; setButtonBusy(target, true, 'キューに追加中…');
  try {
    await api(`/api/jobs/${mode}`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(postRequest(mode)) });
    toast(`${modeLabel(mode)}をキューに追加しました`); await loadHistory(true);
  } catch (error) { errorNode.textContent = error.message; }
  finally { target.dataset.busy = 'false'; setButtonBusy(target, false); updatePostAvailability(mode); }
}

function requestBody() {
  const geometry = normalizedGeometry(true);
  const seedText = $('#seed').value.trim();
  const seed = seedText === '' ? null : clamp(seedText, 0, Number.MAX_SAFE_INTEGER, 1);
  return {
    prompt: $('#prompt').value.trim(),
    model: $('#model').value,
    first_frame: state.uploads.first_frame,
    last_frame: state.uploads.last_frame,
    ...geometry,
    seed,
    steps: clamp($('#steps').value, 1, 80, PRESETS[state.preset].steps),
    preset: state.preset,
    attention: $('#attention').value,
    precision: 'int8',
    memory_profile: $('#memoryProfile').value,
    loras: selectedLoras(),
    latent_refine: {
      enabled: $('#latentRefineEnabled').checked,
      model: $('#latentRefineEnabled').checked ? ($('#latentRefineModel').value || null) : null,
      scale: 2,
      strength: clamp($('#latentRefineStrength').value, 0.05, 0.6, 0.18),
      steps: Math.round(clamp($('#latentRefineSteps').value, 1, 20, 4)),
    },
  };
}
function formProblem() {
  if (!state.connected) return 'ローカルエンジンに接続できません。';
  if (!state.engineReady) return '生成エンジンのセットアップを完了してください。';
  if (!modelIsReady(selectedModel())) return '必要なファイルが揃ったモデルを選択してください。';
  if (!loadedModelMatchesSelection()) return '選択したモデルと実行設定を読み込んでください。';
  if (['queued', 'running'].includes(engineControl().state)) return 'モデル設定の適用完了を待ってください。';
  if (engineControl().state === 'error') return engineControl().error || 'モデル設定の適用に失敗しました。';
  if (!loadedLorasMatchSelection()) return 'LoRA設定の適用完了を待ってください。';
  if (!$('#prompt').value.trim()) return 'プロンプトを入力してください。';
  if (Object.values(state.uploadOps).some((operation) => operation.pending)) return '参照フレームのアップロード完了を待ってください。';
  const missingLora = selectedLoras().find((item) => item.enabled && !loraAvailable(item.path));
  if (missingLora) return `LoRAが見つかりません: ${missingLora.path}`;
  const acceleration = selectedLoras().filter((item) => item.enabled && TURBO_PATTERN.test(item.path));
  if (acceleration.length > 1) return '高速化LoRAは1件だけ有効にしてください。';
  if (acceleration.length === 1 && !Object.values(TURBO_LORAS).some((name) => acceleration[0].path.toLowerCase().endsWith(name))) return 'この高速化LoRAは独立エンジンで未検証です。';
  if (PRESETS[state.preset].turbo && !hasTurboLora(state.preset)) return 'このTurboプリセットに対応する公式LoRAを有効にしてください。';
  if (!PRESETS[state.preset].turbo && acceleration.length) return '高速化LoRAに対応するTurboプリセットを選択してください。';
  if ($('#latentRefineEnabled').checked && !$('#latentRefineModel').value) return 'Latent 2-pass用モデルを選択してください。';
  return '';
}
function updateGenerateAvailability() {
  const problem = formProblem();
  $('#generate').disabled = state.submitted || Boolean(problem && problem !== 'プロンプトを入力してください。');
}
async function createJob() {
  if (state.submitted) return;
  const problem = formProblem(); $('#formError').textContent = '';
  if (problem) { $('#formError').textContent = problem; if (!$('#prompt').value.trim()) $('#prompt').focus(); return; }
  state.submitted = true; const generate = $('#generate'); generate.disabled = true;
  try {
    await api('/api/jobs', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(requestBody()) });
    toast('生成キューに追加しました'); await loadHistory(true);
  } catch (error) { $('#formError').textContent = error.message; }
  finally { state.submitted = false; updateGenerateAvailability(); }
}

function jobSignature(job) {
  return JSON.stringify([job.kind, job.status, job.progress, job.phase, job.output_path, job.metadata_path, job.thumbnail_url, job.error, job.updated_at, job.request, job.resolved, job.id === state.activePreviewJobId]);
}
function mediaPlaceholder(job) {
  const wrap = el('div', 'media-placeholder');
  if (ACTIVE_STATES.has(job.status)) wrap.append(el('i', 'spinner'));
  else wrap.append(el('span', 'empty-symbol', job.status === 'failed' ? '!' : '◇'));
  wrap.append(el('span', '', job.status === 'failed' ? '出力は作成されていません' : STATUS_LABELS[job.status] || '出力なし'));
  return wrap;
}
function restoreSettings(request = {}, resolved = {}) {
  state.engineControl.suspended = true;
  $('#prompt').value = request.prompt || '';
  $('#promptCount').textContent = `${$('#prompt').value.length.toLocaleString('ja-JP')} / 20,000`;
  if (request.model) {
    const modelSelect = $('#model');
    const existing = [...modelSelect.options].find((option) => option.value === request.model);
    if (!existing) {
      const missing = new Option(`${request.model} — 見つかりません`, request.model);
      missing.disabled = true; modelSelect.add(missing); modelSelect.dataset.missingModel = request.model;
    } else {
      delete modelSelect.dataset.missingModel;
    }
    modelSelect.value = request.model;
  }
  $('#width').value = request.width ?? 960; $('#height').value = request.height ?? 544; $('#frames').value = request.frames ?? 124;
  $('#seed').value = request.seed ?? resolved.seed ?? '';
  restoreUpload('first_frame', request.first_frame);
  restoreUpload('last_frame', request.last_frame);
  $('#memoryProfile').value = request.memory_profile || 'auto';
  if ([...$('#attention').options].some((option) => option.value === request.attention && !option.disabled)) $('#attention').value = request.attention;
  $('#loraList').replaceChildren(); (request.loras || []).forEach((item) => addLora(item.path, item.weight, item.enabled));
  setPreset(request.preset || 'balanced'); $('#steps').value = request.steps ?? PRESETS[state.preset].steps;
  const latent = request.latent_refine || {};
  $('#latentRefineEnabled').checked = Boolean(latent.enabled) && !$('#latentRefineEnabled').disabled;
  if (latent.model) $('#latentRefineModel').value = latent.model;
  $('#latentRefineStrength').value = latent.strength ?? 0.18;
  $('#latentRefineSteps').value = latent.steps ?? 4;
  $('#latentRefineSettings').hidden = !$('#latentRefineEnabled').checked;
  state.engineControl.suspended = false;
  scheduleLoraApply();
  normalizedGeometry(true); updateLoraEmpty(); updatePresetCompatibility(); updateModelHealth(); updateGenerateAvailability();
  activateMode('generate', true);
  toast('生成設定をエディターに戻しました');
}
async function jobAction(action, job, trigger) {
  const busyLabel = action === 'cancel' ? '中止中…' : action === 'retry-finalize' ? '再試行中…' : '追加中…';
  setButtonBusy(trigger, true, busyLabel);
  try {
    await api(`/api/jobs/${encodeURIComponent(job.id)}/${action}`, { method: 'POST' });
    const messages = { cancel: '中止を要求しました', replay: '同じ設定でキューに追加しました', 'retry-finalize': '後処理を再試行しました' };
    toast(messages[action] || '処理を開始しました'); await loadHistory(true);
  } catch (error) { toast(error.message, 'error'); setButtonBusy(trigger, false); }
}
function buildJobCard(job) {
  const isActivePreview = job.id === state.activePreviewJobId;
  const card = el('article', `job-card ${job.status || ''}${isActivePreview ? ' active-preview' : ''}`); card.dataset.jobId = job.id; card.dataset.signature = jobSignature(job);
  const media = el('div', 'job-media');
  const badge = el('span', `job-state-badge ${job.status || ''}`, STATUS_LABELS[job.status] || job.status || '不明'); media.append(badge);
  if (job.output_path && (job.status === 'completed' || job.status === 'failed') && isActivePreview) {
    const video = document.createElement('video'); video.controls = true; video.preload = 'metadata'; video.src = outputUrl(job.output_path);
    video.setAttribute('playsinline', ''); bindExclusivePlayback(video); media.append(video);
  } else if (job.thumbnail_url) {
    const thumbnail = document.createElement('img'); thumbnail.className = 'job-thumbnail'; thumbnail.loading = 'lazy'; thumbnail.alt = '';
    thumbnail.src = job.thumbnail_url;
    thumbnail.addEventListener('error', () => { if (thumbnail.isConnected) thumbnail.replaceWith(mediaPlaceholder(job)); }, { once: true });
    media.append(thumbnail);
  } else media.append(mediaPlaceholder(job));

  const content = el('div', 'job-content'); const kind = job.kind || 'generate';
  content.append(el('span', `job-kind ${kind}`, modeLabel(kind)));
  const operationTitle = kind === 'upscale'
    ? `${postMethodLabel(kind, job.request?.method)} · ${job.request?.scale || 2}×`
    : kind === 'interpolate' ? `${postMethodLabel(kind, job.request?.method)} · ${job.request?.factor || 2}× FPS` : job.request?.prompt || 'プロンプトなし';
  content.append(el('p', 'job-prompt', operationTitle));
  const resolved = job.resolved || job.request || {};
  const details = el('div', 'job-details');
  if (kind === 'generate') {
    const model = el('span', '', String(job.request?.model || '').split('/').pop() || 'model'); model.title = job.request?.model || '';
    details.append(model, el('span', '', `${resolved.width || '—'} × ${resolved.height || '—'} · ${resolved.frames || '—'}f`));
  } else {
    const sourceJobId = job.request?.source?.job_id; const sourceJob = state.jobs.find((candidate) => candidate.id === sourceJobId);
    const provenance = sourceJob ? `${modeLabel(sourceJob.kind || 'generate')} #${sourceJob.id.slice(0, 6)}` : job.request?.source?.upload_id ? 'アップロード映像' : '入力映像';
    const width = resolved.output_width ?? resolved.width; const height = resolved.output_height ?? resolved.height; const fps = resolved.output_fps ?? resolved.fps;
    const outputSpec = width && height ? `${width} × ${height}${fps ? ` · ${Number(fps).toFixed(3).replace(/\.0+$/, '')} FPS` : ''}` : modeLabel(kind);
    details.append(el('span', '', `入力: ${provenance}`), el('span', '', outputSpec));
  }
  content.append(details);
  if (ACTIVE_STATES.has(job.status)) {
    const track = el('div', 'progress-track'); const bar = el('span'); bar.style.width = `${Math.max(0, Math.min(100, Number(job.progress || 0) * 100))}%`; track.append(bar); content.append(track);
    content.append(el('p', 'phase-line', job.phase || STATUS_LABELS[job.status]));
  }
  if (job.error) content.append(el('pre', 'job-error', job.error));
  const actions = el('div', 'job-actions');
  if (ACTIVE_STATES.has(job.status)) actions.append(button('中止', 'danger-action', (event) => jobAction('cancel', job, event.currentTarget)));
  if (job.output_path && (job.status === 'completed' || job.status === 'failed') && !isActivePreview) actions.append(button('プレビュー', 'preview-action', () => selectHistoryPreview(job)));
  if (job.output_path) {
    const save = el('a', '', 'MP4'); save.href = outputUrl(job.output_path); save.download = ''; actions.append(save);
  }
  if (job.metadata_path) {
    const metadata = el('a', '', '設定 JSON'); metadata.href = outputUrl(job.metadata_path); metadata.download = ''; actions.append(metadata);
  }
  if (job.status === 'completed' && job.output_path) {
    actions.append(button('拡大する', '', () => setPostSourceFromJob('upscale', job)));
    actions.append(button('補間する', '', () => setPostSourceFromJob('interpolate', job)));
  }
  if (job.status === 'failed' && job.output_path && job.metadata_path) actions.append(button('後処理を再試行', '', (event) => jobAction('retry-finalize', job, event.currentTarget)));
  if (!ACTIVE_STATES.has(job.status)) actions.append(button(kind === 'generate' ? '再生成' : '再実行', '', (event) => jobAction('replay', job, event.currentTarget)));
  if (kind === 'generate') actions.append(button('設定を戻す', '', () => restoreSettings(job.request || {}, job.resolved || {})));
  content.append(actions); card.append(media, content); return card;
}
function refreshSourceJobOptions() {
  const completed = state.jobs.filter((job) => job.status === 'completed' && job.output_path);
  const signature = JSON.stringify(completed.map((job) => [job.id, job.kind, job.output_path, job.finished_at]));
  if (signature === state.sourceJobSignature) return;
  state.sourceJobSignature = signature;
  ['upscale', 'interpolate'].forEach((mode) => {
    const select = $(`#${mode}Source`); const current = state.postSources[mode];
    select.replaceChildren(new Option('完了した映像を選択…', ''));
    completed.forEach((job) => {
      const meta = sourceMetaFromJob(job); const date = job.finished_at || job.created_at;
      const when = date ? new Intl.DateTimeFormat('ja-JP', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' }).format(new Date(date)) : '';
      select.add(new Option(`${modeLabel(job.kind || 'generate')} · ${meta.name}${when ? ` · ${when}` : ''}`, job.id));
    });
    if (current?.type === 'job' && completed.some((job) => job.id === current.id)) select.value = current.id;
  });
}
function renderHistory(jobs) {
  state.jobs = jobs;
  refreshSourceJobOptions();
  const gallery = $('#history'); gallery.setAttribute('aria-busy', 'false');
  if (!jobs.length) {
    state.activePreviewJobId = null;
    $$('.job-card', gallery).forEach(releaseCardMedia);
    gallery.replaceChildren();
    const empty = el('div', 'empty-state'); empty.append(el('span', 'empty-symbol', '◇'), el('b', '', 'まだ処理履歴はありません'), el('span', '', '生成、拡大、補間の結果がここに表示されます')); gallery.append(empty);
    $('#queueSummary').textContent = 'ジョブなし'; return;
  }
  const playable = jobs.filter((job) => job.output_path && (job.status === 'completed' || job.status === 'failed'));
  if (!playable.some((job) => job.id === state.activePreviewJobId)) state.activePreviewJobId = playable[0]?.id || null;
  const displayJobs = state.activePreviewJobId
    ? [jobs.find((job) => job.id === state.activePreviewJobId), ...jobs.filter((job) => job.id !== state.activePreviewJobId)].filter(Boolean)
    : jobs;
  const existing = new Map($$('.job-card', gallery).map((card) => [card.dataset.jobId, card]));
  const keep = new Set(displayJobs.map((job) => job.id));
  existing.forEach((card, id) => { if (!keep.has(id)) { releaseCardMedia(card); card.remove(); } });
  $$('.empty-state', gallery).forEach((empty) => empty.remove());
  displayJobs.forEach((job, index) => {
    let card = existing.get(job.id);
    if (!card || card.dataset.signature !== jobSignature(job)) {
      const next = buildJobCard(job);
      if (card) { releaseCardMedia(card); card.replaceWith(next); }
      card = next;
    }
    const currentAtIndex = gallery.children[index];
    if (currentAtIndex !== card) gallery.insertBefore(card, currentAtIndex || null);
  });
  const active = jobs.filter((job) => ACTIVE_STATES.has(job.status));
  $('#queueSummary').textContent = active.length ? `${active.length}件を処理中 · 全${jobs.length}件` : `全${jobs.length}件`;
}
async function loadHistory(force = false) {
  if (state.historyLoading && !force) return;
  state.historyLoading = true;
  try { renderHistory(await api('/api/jobs?limit=100')); }
  catch (error) {
    const gallery = $('#history'); gallery.setAttribute('aria-busy', 'false');
    if (!$$('.job-card', gallery).length) {
      const empty = el('div', 'empty-state'); empty.append(el('span', 'empty-symbol', '!'), el('b', '', '処理履歴を取得できません'), el('span', '', error.message)); gallery.replaceChildren(empty);
    }
    $('#queueSummary').textContent = '接続エラー';
  } finally { state.historyLoading = false; }
}

async function openOutputs() {
  const trigger = $('#openOutputs'); setButtonBusy(trigger, true, '開いています…');
  try { await api('/api/outputs/open', { method: 'POST' }); toast('outputsフォルダを開きました'); }
  catch (error) { toast(error.message, 'error'); }
  finally { setButtonBusy(trigger, false); }
}

function bindEvents() {
  bindExclusivePlayback($('#upscalePreview'));
  bindExclusivePlayback($('#interpolatePreview'));
  $$('.mode-tab').forEach((node) => node.addEventListener('click', () => activateMode(node.dataset.mode, true)));
  $$('#presets button').forEach((node) => node.addEventListener('click', () => setPreset(node.dataset.preset)));
  ['width', 'height', 'frames'].forEach((id) => {
    $(`#${id}`).addEventListener('input', () => normalizedGeometry(false));
    $(`#${id}`).addEventListener('change', () => { normalizedGeometry(true); renderGenerationOptions(); });
  });
  $('#prompt').addEventListener('input', () => {
    $('#promptCount').textContent = `${$('#prompt').value.length.toLocaleString('ja-JP')} / 20,000`;
    $('#formError').textContent = ''; updateGenerateAvailability();
  });
  $('#model').addEventListener('change', updateModelHealth);
  $('#loadModel').addEventListener('click', loadSelectedModel);
  $('#attention').addEventListener('change', renderEngineControl);
  $('#memoryProfile').addEventListener('change', renderEngineControl);
  $('#latentRefineEnabled').addEventListener('change', () => {
    $('#latentRefineSettings').hidden = !$('#latentRefineEnabled').checked;
    updateGenerateAvailability();
  });
  $('#randomSeed').addEventListener('click', () => { $('#seed').value = Math.floor(Math.random() * Number.MAX_SAFE_INTEGER); });
  $('#refresh').addEventListener('click', () => refreshInventory(true));
  $('#addLora').addEventListener('click', () => addLora());
  $('#generate').addEventListener('click', createJob);
  $('#openOutputs').addEventListener('click', openOutputs);
  $('#firstFile').addEventListener('change', (event) => { if (event.target.files[0]) uploadFrame('first_frame', event.target.files[0]); });
  $('#lastFile').addEventListener('change', (event) => { if (event.target.files[0]) uploadFrame('last_frame', event.target.files[0]); });
  $('#firstSlot .frame-remove').addEventListener('click', () => clearUpload('first_frame'));
  $('#lastSlot .frame-remove').addEventListener('click', () => clearUpload('last_frame'));
  ['upscale', 'interpolate'].forEach((mode) => {
    $(`#${mode}Source`).addEventListener('change', (event) => {
      const job = state.jobs.find((candidate) => candidate.id === event.target.value);
      if (job) setPostSourceFromJob(mode, job, false); else clearPostSource(mode);
    });
    $(`#${mode}File`).addEventListener('change', (event) => { if (event.target.files[0]) uploadVideoSource(mode, event.target.files[0]); });
    $(`[data-clear-source="${mode}"]`).addEventListener('click', () => clearPostSource(mode));
    $(`#${mode}Method`).addEventListener('change', () => fillPostMethods(mode));
  });
  $('#upscaleScale').addEventListener('change', () => updateResultSpec('upscale'));
  $('#interpolateFactor').addEventListener('change', () => updateResultSpec('interpolate'));
  $('#enqueueUpscale').addEventListener('click', () => enqueuePostprocess('upscale'));
  $('#enqueueInterpolate').addEventListener('click', () => enqueuePostprocess('interpolate'));
  document.addEventListener('keydown', (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
      event.preventDefault();
      if (state.currentMode === 'generate') createJob(); else enqueuePostprocess(state.currentMode);
    }
  });
}

async function boot() {
  state.engineControl.suspended = true;
  bindEvents(); activateMode('generate'); normalizedGeometry(); updateLoraEmpty(); updatePresetCompatibility();
  state.engineControl.suspended = false;
  await Promise.allSettled([refreshStatus(), refreshInventory(), refreshGenerationOptions(), refreshPostprocessCapabilities(), loadHistory()]);
  updateGenerateAvailability(); updatePostAvailability('upscale'); updatePostAvailability('interpolate');
  window.setInterval(loadHistory, 1800);
  window.setInterval(refreshStatus, 2500);
  window.setInterval(() => { if (!(state.inventory.models || []).some(modelIsReady)) refreshInventory(false, true); }, 10000);
}

boot();
