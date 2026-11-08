// Venno MOSS Studio: Text to Image with LoRAs and a comparison grid for testing MOSS-Image LoRAs, Image Info,
// Settings. Built on Venno Studio's styles and shared modules (job client, LoRA chips, helpers).

import { $, $$, api, downloadDataURL, escapeHtml, fillSelect, formatBytes, loadStore, saveStore, showLightbox, toast } from "/studio/assets/js/common.js";
import { cancelJob, finishedImages, jobProgress, runJob, skipJob } from "/studio/assets/js/jobs.js";
import { LoraChips } from "/studio/assets/js/loras.js";

const V = "/moss/v1";
const SIZES = [[1024, 1024], [832, 1216], [1216, 832], [896, 1152], [1152, 896], [768, 1344], [1344, 768], [512, 512]];
const DEFAULT = {
  prompt: "", negative: "", loras: [], steps: 50, cfg: 3, shift: 5, count: 1, width: 1024, height: 1024, seed: -1,
  skip22: true, save: true, model: "", newest: false,
  grid: { enabled: false, cells: false, axes: [{ type: "lora", values: "", target: "" }, { type: "none", values: "", target: "" }, { type: "none", values: "", target: "" }] },
};
const AXIS_HINTS = {
  lora: "LoRA files, e.g. a training run's epochs; \"none\" = without it. Use the list on the right to add them",
  lora_weight: "0, 0.5, 1, 1.5  or  0-1.5 [4]",
  prompt_sr: "the first value is searched in the prompt and replaced by each value: red hair, blue hair",
  seed: "1, 2, 3  or  100-104",
  steps: "20, 30, 50",
  cfg: "2, 3, 4  or  2-5 (+1)",
  shift: "3, 5, 7",
  model: "MOSS-Image-Base, MOSS-Image-SFT",
  size: "832x1216, 1024x1024",
  skip_layers: "22, 24, 26, none  (OpenMOSS: 22, 24 and 26 all help; none looks more realistic but can lose detail)",
};

const s = loadStore("venno-moss", DEFAULT);
s.grid = { ...DEFAULT.grid, ...s.grid };
const persist = () => saveStore("venno-moss", s);
let info = null;
let busy = false;
let jobId = null;
let results = []; // [{src, seed, info}]
let current = -1;
let picked = false;

// ---------------- tabs ----------------
function showTab(name) {
  for (const p of ["t2i", "info", "settings"]) $(`#page-${p}`).hidden = p !== name;
  $$("#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === name));
  $("#btn-settings").classList.toggle("on", name === "settings");
  if (name === "settings") loadCatalog();
}
$$("#tabs button").forEach((b) => (b.onclick = () => showTab(b.dataset.tab)));
$("#btn-settings").onclick = () => showTab($("#page-settings").hidden ? "settings" : "t2i");

// ---------------- form ----------------
const num = (id, key, after = () => {}) => {
  const el = $(`#${id}`);
  el.value = s[key];
  el.onchange = () => { const v = Number(el.value); if (Number.isFinite(v)) s[key] = v; persist(); after(); };
};
for (const [id, key] of [["steps", "steps"], ["cfg", "cfg"], ["shift", "shift"], ["count", "count"], ["width", "width"], ["height", "height"], ["seed", "seed"]]) num(id, key);
for (const [id, key] of [["prompt", "prompt"], ["negative", "negative"]]) {
  const el = $(`#${id}`);
  el.value = s[key];
  el.oninput = () => { s[key] = el.value; persist(); if (id === "prompt") countTokens(); };
}
for (const [id, key] of [["skip22", "skip22"], ["save", "save"], ["lora-newest", "newest"]]) {
  const el = $(`#${id}`);
  el.checked = s[key];
  el.onchange = () => { s[key] = el.checked; persist(); if (key === "newest") fillLoras(); };
}
$("#sizes").innerHTML = SIZES.map(([w, h]) => `<button data-w="${w}" data-h="${h}">${w}×${h}</button>`).join("") + `<button id="swap" title="Swap width/height">⇅</button>`;
$("#sizes").onclick = (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  if (b.id === "swap") [s.width, s.height] = [s.height, s.width];
  else [s.width, s.height] = [Number(b.dataset.w), Number(b.dataset.h)];
  $("#width").value = s.width;
  $("#height").value = s.height;
  persist();
};
$("#seed-random").onclick = () => { s.seed = -1; $("#seed").value = -1; persist(); };
$("#seed-reuse").onclick = () => { const r = results[current]; if (r) { s.seed = r.seed; $("#seed").value = r.seed; persist(); } };
$("#btn-defaults").onclick = () => {
  Object.assign(s, { steps: 50, cfg: 3, shift: 5, skip22: true });
  for (const k of ["steps", "cfg", "shift"]) $(`#${k}`).value = s[k];
  $("#skip22").checked = true;
  persist();
};
$("#sel-model").onchange = (e) => { s.model = e.target.value; persist(); };
addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && !$("#page-t2i").hidden) generate(); });

let tokenTimer = 0;
function countTokens() {
  clearTimeout(tokenTimer);
  tokenTimer = setTimeout(async () => {
    if (!s.prompt.trim()) return ($("#tokens").textContent = "");
    try { const r = await api(`${V}/tokens`, { text: s.prompt }); $("#tokens").textContent = r.count ? `${r.count} tokens` : ""; } catch {}
  }, 500);
}

// ---------------- LoRAs ----------------
const chips = new LoraChips($("#lora-chips"), { get: () => s.loras, set: (list) => { s.loras = list; persist(); renderAxes(); loraInfo(); } });
chips.render();
/** One line per LoRA in the list: the model it was trained on (when its file records it), rank and alpha. */
function loraInfo() {
  const known = new Map((info?.loras || []).map((l) => [l.name, l]));
  $("#lora-info").innerHTML = s.loras.map((x) => {
    const l = known.get(x.name);
    if (!l) return "";
    const bits = [`trained on ${l.trained_on ? `<b>${escapeHtml(l.trained_on)}</b>` : "(not recorded)"}`];
    if (l.rank) bits.push(`rank ${l.rank}`);
    if (l.alpha != null) bits.push(`alpha ${l.alpha}`);
    return `<div>${escapeHtml(x.name.replace(/\.safetensors$/i, ""))}: ${bits.join(" · ")}</div>`;
  }).join("");
}
function fillLoras() {
  const list = [...(info?.loras || [])];
  if (s.newest) list.sort((a, b) => b.mtime - a.mtime);
  const sel = $("#sel-lora");
  sel.innerHTML = `<option value="">+ LoRA… (${list.length})</option>` + list.map((l) => `<option value="${escapeHtml(l.name)}">${escapeHtml(l.name)}</option>`).join("");
  chips.setAvailable(list.map((l) => l.name));
  loraInfo();
}
$("#sel-lora").onchange = (e) => { if (e.target.value) chips.add(e.target.value, 1); e.target.value = ""; };

/** LoRA files grouped into training runs: "run-000004", "run-000008" → "run" (sorted by their number). */
function loraRuns() {
  const runs = new Map();
  for (const l of info?.loras || []) {
    const stem = l.name.replace(/\.safetensors$/i, "");
    const m = stem.match(/^(.*?)[-_. ]?(?:e|ep|epoch|step|steps|s)?[-_]?(\d+)$/i);
    if (!m || !m[1]) continue;
    const key = m[1];
    if (!runs.has(key)) runs.set(key, []);
    runs.get(key).push({ name: l.name, n: Number(m[2]) });
  }
  return [...runs.entries()].filter(([, v]) => v.length > 1).map(([k, v]) => [k, v.sort((a, b) => a.n - b.n).map((x) => x.name)]);
}

// ---------------- comparison grid ----------------
const quote = (v) => (/[,"]/.test(v) ? `"${v.replace(/"/g, '\\"')}"` : v);
function renderAxes() {
  const axes = info?.axes || [];
  const box = $("#axes");
  box.innerHTML = s.grid.axes.map((a, i) => `
    <div class="axis" data-i="${i}">
      <div class="row">
        <b>${"XYZ"[i]}</b>
        <select class="a-type"><option value="none">—</option>${axes.map((x) => `<option value="${x.key}"${x.key === a.type ? " selected" : ""}>${x.label}</option>`).join("")}</select>
        ${a.type === "lora" || a.type === "lora_weight" ? `<select class="a-target" title="${a.type === "lora" ? "The LoRA in the list whose file changes per cell (none in the list: added at weight 1)" : "The LoRA whose weight changes"}"><option value="">${a.type === "lora" ? "first LoRA in the list" : "first LoRA"}</option>${s.loras.map((l) => `<option${l.name === a.target ? " selected" : ""}>${escapeHtml(l.name)}</option>`).join("")}</select>` : ""}
        ${a.type === "lora" ? `<select class="a-add"><option value="">+ add…</option><option value="none">none (without it)</option>${loraRuns().map(([k, v]) => `<option value="run:${escapeHtml(k)}">all ${v.length} of ${escapeHtml(k)}</option>`).join("")}${(info?.loras || []).map((l) => `<option value="${escapeHtml(l.name)}">${escapeHtml(l.name)}</option>`).join("")}</select>` : ""}
      </div>
      ${a.type !== "none" ? `<input class="a-values" value="${escapeHtml(a.values)}" placeholder="${escapeHtml(AXIS_HINTS[a.type] || "")}">` : ""}
    </div>`).join("");
  $$(".axis", box).forEach((row) => {
    const a = s.grid.axes[Number(row.dataset.i)];
    $(".a-type", row).onchange = (e) => { a.type = e.target.value; a.values = ""; a.target = ""; persist(); renderAxes(); };
    const t = $(".a-target", row);
    if (t) t.onchange = () => { a.target = t.value; persist(); checkGrid(); };
    const v = $(".a-values", row);
    if (v) v.oninput = () => { a.values = v.value; persist(); checkGrid(); };
    const add = $(".a-add", row);
    if (add) add.onchange = () => {
      const val = add.value;
      if (!val) return;
      const items = val.startsWith("run:") ? loraRuns().find(([k]) => k === val.slice(4))[1] : [val];
      a.values = [a.values.trim(), ...items.map(quote)].filter(Boolean).join(", ");
      persist();
      renderAxes();
    };
  });
  checkGrid();
}
$("#grid-on").checked = s.grid.enabled;
$("#grid-on").onchange = (e) => { s.grid.enabled = e.target.checked; persist(); checkGrid(); };
$("#grid-on").onclick = (e) => e.stopPropagation(); // don't toggle the card
$("#grid-cells").checked = s.grid.cells;
$("#grid-cells").onchange = (e) => { s.grid.cells = e.target.checked; persist(); };

const gridJob = () => {
  if (!s.grid.enabled) return null;
  const axis = (a) => (a.type === "none" ? null : { type: a.type, values: a.values, target: a.target || "" });
  return { x: axis(s.grid.axes[0]), y: axis(s.grid.axes[1]), z: axis(s.grid.axes[2]), include_cells: s.grid.cells };
};
let gridTimer = 0;
function checkGrid() {
  clearTimeout(gridTimer);
  const g = gridJob();
  if (!g) return ($("#grid-sum").textContent = "");
  gridTimer = setTimeout(async () => {
    try {
      const r = await api(`${V}/grid/check`, g);
      $("#grid-sum").textContent = r.ok ? `${r.counts.join(" × ")} = ${r.total} picture${r.total > 1 ? "s" : ""}` : r.error;
    } catch {}
  }, 300);
}

// ---------------- generating ----------------
function buildJob() {
  if (!s.model) throw new Error("Pick a MOSS-Image model in the top bar (or download one in ⚙ Settings → Models)");
  return {
    prompt: s.prompt, negative: s.negative, model: s.model, loras: s.loras.filter((l) => l.weight !== 0).map(({ name, weight }) => ({ name, weight })),
    width: s.width, height: s.height, steps: s.steps, cfg: s.cfg, shift: s.shift, skip_layers: s.skip22 ? [22] : [],
    seed: s.seed, count: s.count, grid: gridJob(), save: s.save,
  };
}

async function generate() {
  if (busy) return;
  let job;
  try { job = buildJob(); } catch (e) { return toast(e.message); }
  busy = true;
  jobId = null;
  picked = false;
  $("#btn-generate").disabled = true;
  $("#btn-cancel").hidden = false;
  const bar = $("#progress");
  bar.hidden = false;
  $(".bar", bar).style.width = "0%";
  $("span", bar).textContent = "Starting… (the first picture loads the model)";
  const older = results;
  let shown = 0;
  const entries = (j) => j.images.map((src, i) => ({ src, seed: j.seeds[i], info: j.infotexts[i] || "" }));
  try {
    const done = await runJob(job, {
      onSubmit: (j) => (jobId = j.id),
      onUpdate: (j) => {
        const { pct, text } = jobProgress(j);
        $(".bar", bar).style.width = `${pct}%`;
        $("span", bar).textContent = j.status === "running" && !j.steps ? "Loading the model and encoding the prompt…" : text;
        $("#btn-skip").hidden = !(j.status === "running" && j.job_count > 1);
        if (!job.grid && j.images.length > shown) {
          if (current >= shown) current += j.images.length - shown;
          shown = j.images.length;
          results = [...entries(j), ...older].slice(0, 80);
          renderThumbs();
        }
      },
      onPreview: (src) => { if (!picked) showImage(src); },
    });
    const fresh = entries({ ...done, images: finishedImages(done) });
    for (const w of done.warnings || []) toast(w, "info");
    results = [...fresh, ...older].slice(0, 80);
    select(picked && current < fresh.length ? current : 0);
    if (job.grid) showLightbox(fresh[0].src, `moss-grid-${fresh[0].seed}.png`);
  } catch (e) {
    if (e.message !== "Cancelled") toast(e.message);
    if (current >= 0) select(current);
  } finally {
    busy = false;
    jobId = null;
    $("#btn-generate").disabled = false;
    $("#btn-cancel").hidden = true;
    $("#btn-skip").hidden = true;
    bar.hidden = true;
  }
}
$("#btn-generate").onclick = generate;
$("#btn-cancel").onclick = () => jobId && cancelJob(jobId);
$("#btn-skip").onclick = () => jobId && skipJob(jobId);

// ---------------- results ----------------
function showImage(src) {
  $("#r-img").src = src;
  $("#r-empty").hidden = true;
}
function renderThumbs() {
  $("#thumbs").innerHTML = "";
  results.forEach((r, k) => {
    const img = document.createElement("img");
    img.src = r.src;
    img.title = `Seed ${r.seed}`;
    img.classList.toggle("on", k === current);
    img.onclick = () => { if (busy) picked = true; select(k); };
    img.ondblclick = () => showLightbox(r.src, `moss-${r.seed}.png`);
    $("#thumbs").appendChild(img);
  });
}
function select(i) {
  current = i;
  const r = results[i];
  if (!r) return;
  showImage(r.src);
  $("#r-info").textContent = r.info;
  for (const id of ["r-settings", "r-view", "r-download"]) $(`#${id}`).disabled = false;
  renderThumbs();
}
$("#r-view").onclick = () => { const r = results[current]; if (r) showLightbox(r.src, `moss-${r.seed}.png`); };
$("#r-download").onclick = () => { const r = results[current]; if (r) downloadDataURL(r.src, `moss-${r.seed}.png`); };
$("#r-settings").onclick = async () => { const r = results[current]; if (r) applySettings((await readImage(await toDataURL(r.src))).settings); };

// ---------------- image info / settings round trip ----------------
const toDataURL = async (src) => {
  const blob = await (await fetch(src)).blob();
  return new Promise((res) => { const fr = new FileReader(); fr.onload = () => res(fr.result); fr.readAsDataURL(blob); });
};
const readImage = (dataURL) => api(`${V}/image/read`, { image: dataURL });

function applySettings(x) {
  if (!x) return toast("This picture has no Venno MOSS settings");
  for (const k of ["prompt", "negative", "steps", "cfg", "shift", "width", "height", "seed"]) if (x[k] != null) s[k] = x[k];
  s.skip22 = (x.skip_layers || []).includes(22);
  if (x.model && [...$("#sel-model").options].some((o) => o.value === x.model)) s.model = x.model;
  else if (x.model) toast(`Model "${x.model}" isn't installed; keeping ${s.model || "the current one"}`, "info");
  s.loras = (x.loras || []).map((l) => ({ name: l.name, weight: l.weight }));
  persist();
  for (const k of ["prompt", "negative", "steps", "cfg", "shift", "width", "height", "seed"]) $(`#${k}`).value = s[k];
  $("#skip22").checked = s.skip22;
  $("#sel-model").value = s.model;
  chips.render();
  renderAxes();
  showTab("t2i");
  toast("Settings loaded", "info");
}

let infoSettings = null;
async function openInfo(file) {
  const dataURL = await new Promise((res) => { const fr = new FileReader(); fr.onload = () => res(fr.result); fr.readAsDataURL(file); });
  try {
    const r = await readImage(dataURL);
    $(".info-out").hidden = false;
    $("#info-img").src = dataURL;
    $("#info-text").textContent = r.parameters || "(no settings text in this picture)";
    infoSettings = r.settings;
    $("#info-use").disabled = !r.settings;
  } catch (e) { toast(e.message); }
}
$("#info-open").onclick = () => $("#info-file").click();
$("#info-file").onchange = (e) => e.target.files[0] && openInfo(e.target.files[0]);
$("#info-use").onclick = () => applySettings(infoSettings);
for (const ev of ["dragover", "drop"]) {
  $("#page-info").addEventListener(ev, (e) => {
    e.preventDefault();
    if (ev === "drop" && e.dataTransfer.files[0]) openInfo(e.dataTransfer.files[0]);
  });
}

// ---------------- settings ----------------
$$(".settings-nav button").forEach((b) => (b.onclick = () => {
  $$(".settings-nav button").forEach((x) => x.classList.toggle("on", x === b));
  $$(".settings-body > section").forEach((sec) => (sec.hidden = sec.dataset.section !== b.dataset.section));
}));

let dlTimer = 0;
async function loadCatalog() {
  try {
    const c = await api(`${V}/models/catalog`);
    $("#catalog").innerHTML = c.catalog.map((m) => `
      <li><div style="flex:1"><b>${escapeHtml(m.name)}</b> <span class="hint">${m.installed ? "· installed" : `· ${m.size_gb} GB`}</span><br><span class="hint">${escapeHtml(m.description)}</span></div>
      <button class="${m.installed ? "" : "primary"}" data-variant="${m.variant}"${m.installed ? " disabled" : ""}>${m.installed ? "Installed" : "Download"}</button></li>`).join("");
    $$("#catalog button[data-variant]").forEach((b) => (b.onclick = async () => {
      try { await api(`${V}/models/download`, { variant: b.dataset.variant }); pollDownloads(); } catch (e) { toast(e.message); }
    }));
    renderDownloads(c.jobs);
    if (c.jobs.some((j) => j.status === "downloading" || j.status === "queued")) pollDownloads();
  } catch (e) { toast(e.message); }
}
function renderDownloads(list) {
  $("#dl-jobs").innerHTML = list.map((j) => `<li><div style="flex:1">${escapeHtml(j.name)}${j.shared ? " + text encoder and VAE" : ""} <span class="hint">· ${j.status === "error" ? `failed: ${escapeHtml(j.error)}` : j.status === "done" ? "done" : j.total ? `${formatBytes(j.done) || "0 B"} of ${formatBytes(j.total)}` : `${escapeHtml(j.phase || "starting…")} ${j.seconds}s`}</span>
    <div class="progress-inline"><div class="bar" style="width:${j.total ? Math.round((100 * j.done) / j.total) : j.status === "done" ? 100 : 0}%"></div></div></div></li>`).join("");
}
function pollDownloads() {
  clearTimeout(dlTimer);
  dlTimer = setTimeout(async () => {
    try {
      const list = await api(`${V}/models/jobs`);
      renderDownloads(list);
      if (list.some((j) => j.status === "downloading" || j.status === "queued")) return pollDownloads();
      if (list.some((j) => j.status === "done")) { await loadInfo(true); loadCatalog(); toast("Download finished", "info"); }
    } catch { pollDownloads(); }
  }, 2000);
}

async function saveSettings(body) {
  try {
    const r = await api(`${V}/settings`, body);
    applyInfo(r.info);
    renderSettings(r.settings);
    toast("Saved", "info");
  } catch (e) { toast(e.message); }
}
const lines = (id) => $(`#${id}`).value.split("\n").map((x) => x.trim()).filter(Boolean);
$("#save-model-dirs").onclick = () => saveSettings({ model_dirs: lines("model-dirs") });
$("#save-lora-dirs").onclick = () => saveSettings({ lora_dirs: lines("lora-dirs") });
$$("input[name=memory]").forEach((r) => (r.onchange = () => saveSettings({ memory: r.value })));

function renderSettings(saved) {
  $("#model-dirs").value = (saved.model_dirs || []).join("\n");
  $("#lora-dirs").value = (saved.lora_dirs || []).join("\n");
  const mode = saved.memory || "auto";
  $$("input[name=memory]").forEach((r) => (r.checked = r.value === mode));
}

// ---------------- info ----------------
function applyInfo(i) {
  info = i;
  const models = i.models.map((m) => ({ value: m.name, label: m.name }));
  if (!models.some((m) => m.value === s.model)) s.model = models[0]?.value || "";
  fillSelect($("#sel-model"), models, s.model, { empty: models.length ? null : "— no model yet —" });
  fillLoras();
  renderAxes();
  const e = i.engine;
  const mem = e.memory === "auto" ? `auto → ${e.memory_resolved}` : e.memory_resolved;
  $("#status").textContent = i.complete ? `${i.models.length} model(s) · ${i.loras.length} LoRA(s) · memory ${mem}` : "No MOSS-Image model yet: ⚙ Settings → Models";
  $("#lora-default").textContent = i.folders.loras[0] ? i.folders.loras.at(-1) : "";
  $("#memory-now").textContent = e.device === "cuda" ? `Your graphics card: ${e.gpu}, ${e.vram_gb} GB. Now using: ${e.memory_resolved}.` : "No graphics card in use: running on the CPU (very slow).";
  const row = (k, v) => `<tr><th>${k}</th><td><code>${escapeHtml(String(v))}</code></td></tr>`;
  $("#system").innerHTML = row("Version", i.version) + row("Device", e.device === "cuda" ? `${e.gpu} (${e.vram_gb} GB)` : "CPU") + row("Memory", mem)
    + row("Model folders", i.folders.models.join("\n")) + row("LoRA folders", i.folders.loras.join("\n")) + row("Outputs", i.folders.outputs)
    + row("Loaded", e.loaded || "nothing yet") + row("Prompts cached", e.prompt_cache);
  persist();
}

async function loadInfo(refresh = false) {
  try {
    applyInfo(await api(refresh ? `${V}/refresh` : `${V}/info`, refresh ? {} : undefined));
    renderSettings((await api(`${V}/settings`)).settings);
    if (!info.complete && !refresh) {
      showTab("settings");
      toast("Download MOSS-Image SFT or Base to start (or add the folder you already have it in).", "info");
    }
  } catch (e) {
    $("#status").textContent = "Server unreachable";
    toast(`Could not reach Venno MOSS: ${e.message}`);
  }
}
$("#btn-refresh").onclick = () => loadInfo(true);

loadInfo();
countTokens();
