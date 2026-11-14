// Venno MOSS Studio: Text to Image, Image to Image and Inpaint tabs (form.js) for testing MOSS-Image LoRAs, Image
// Info and Settings. Built on Venno Studio's styles and shared modules (job client, LoRA chips, cards, zoom).

import { $, $$, api, escapeHtml, fillSelect, formatBytes, loadStore, saveStore, toast } from "/studio/assets/js/common.js";
import { MossForm, readImage } from "/moss/assets/form.js";

const V = "/moss/v1";
const FORMS = ["txt2img", "img2img", "inpaint"];
const shared = loadStore("venno-moss-shared", { model: loadStore("venno-moss", { model: "" }).model || "" }); // the model is one choice for all tabs
let info = null;

// ---------------- tabs ----------------
function showTab(name) {
  for (const p of [...FORMS, "info", "settings"]) $(`#page-${p}`).hidden = p !== name;
  $$("#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === name));
  $("#btn-settings").classList.toggle("on", name === "settings");
  if (name === "settings") loadCatalog();
  if (forms[name]?.image) forms[name].fitStage();
}
const visibleForm = () => forms[FORMS.find((m) => !$(`#page-${m}`).hidden)];

/** The model of every tab; with a name: pick it (from a picture's settings) when it is installed. */
function model(name) {
  if (name === undefined || name === shared.model) return shared.model;
  if ([...$("#sel-model").options].some((o) => o.value === name)) {
    shared.model = name;
    $("#sel-model").value = name;
    saveStore("venno-moss-shared", shared);
  } else toast(`Model "${name}" isn't installed; keeping ${shared.model || "the current one"}`, "info");
  return shared.model;
}

const ctx = {
  info: () => info,
  model,
  show: showTab,
  /** A picture (and optionally the settings read from it) to another tab. */
  send: async (target, src, settings) => {
    showTab(target);
    if (target !== "txt2img" && src) await forms[target].setInputImage(src);
    if (settings) forms[target].applySettings(settings);
  },
};
const forms = Object.fromEntries(FORMS.map((m) => [m, new MossForm($(`#page-${m}`), m, ctx)]));

$$("#tabs button").forEach((b) => (b.onclick = () => showTab(b.dataset.tab)));
$("#btn-settings").onclick = () => showTab($("#page-settings").hidden ? "settings" : "txt2img");
$("#sel-model").onchange = (e) => { shared.model = e.target.value; saveStore("venno-moss-shared", shared); };
addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) visibleForm()?.generate(); });

// ---------------- image info ----------------
let infoSettings = null, infoSrc = null;
async function openInfo(file) {
  const dataURL = await new Promise((res) => { const fr = new FileReader(); fr.onload = () => res(fr.result); fr.readAsDataURL(file); });
  infoSrc = dataURL;
  $("#info-img").src = dataURL;
  $("#info-img").hidden = false;
  $("#info-empty").hidden = true;
  $("#info-send").hidden = false;
  let r = { parameters: "", fields: [], settings: null };
  try { r = await readImage(dataURL); } catch (e) { toast(e.message); }
  infoSettings = r.settings;
  $("#info-text").textContent = r.parameters || "No generation parameters in this image.";
  $('#info-send button[data-to="txt2img"]').disabled = !r.settings; // the image tabs take the picture either way
  $("#info-note").textContent = r.settings?.partial ? "from another UI: takes the prompt, negative prompt, seed and size" : "";
  $("#info-table").innerHTML = (r.fields || []).map(([k, v]) => `<tr><th>${escapeHtml(k)}</th><td>${escapeHtml(String(v))}</td></tr>`).join("");
}
{
  const drop = $("#info-drop");
  const image = (list) => [...list].find((f) => f.type.startsWith("image/"));
  $("#info-open").onclick = () => $("#info-file").click();
  drop.onclick = (e) => { if (e.target === drop || e.target.id === "info-img") $("#info-file").click(); };
  $("#info-file").onchange = (e) => { if (e.target.files[0]) openInfo(e.target.files[0]); e.target.value = ""; };
  $$("#info-send button").forEach((b) => (b.onclick = () => ctx.send(b.dataset.to, infoSrc, infoSettings)));
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("dragging"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("dragging"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("dragging");
    const f = image(e.dataTransfer.files);
    if (f) openInfo(f);
  });
  // paste: into Image Info, or as the input image of the Image to Image / Inpaint tab that is open
  document.addEventListener("paste", (e) => {
    const f = image(e.clipboardData?.files || []);
    if (!f) return;
    const form = visibleForm();
    if (!$("#page-info").hidden) openInfo(f);
    else if (form?.image) form.setInputImage(URL.createObjectURL(f));
    else return;
    e.preventDefault();
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
    // one row per model, a button per size: "Q4 · 9.2 GB" with the cards it is for underneath
    const button = (x, first) => `<button class="size${x.installed ? "" : first ? " primary" : ""}" data-variant="${x.variant}"${x.installed ? " disabled" : ""} title="${x.installed ? "Already here" : `Download ${x.size_gb} GB`}">
        <b>${escapeHtml(x.label)}${x.installed ? " ✓" : ` · ${x.size_gb} GB`}</b>${x.hint ? `<small>${escapeHtml(x.hint)}</small>` : ""}</button>`;
    $("#catalog").innerHTML = c.catalog.map((m) => `
      <li><div style="flex:1"><b>${escapeHtml(m.name)}</b>${m.kind === "lora" ? ` <span class="hint">· LoRA</span>` : ""}<br><span class="hint">${escapeHtml(m.description)}${m.small_repo ? ` Q8 / Q4: ${escapeHtml(m.small_repo)}.` : ""}</span></div>
      <div class="sizes">${m.sizes.map((x) => button(x, m.sizes.length === 1 || !m.sizes.some((y) => y.installed))).join("")}</div></li>`).join("");
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
$("#sel-te").onchange = (e) => saveSettings({ text_encoder: e.target.value });

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
  if (!models.some((m) => m.value === shared.model)) shared.model = models[0]?.value || "";
  fillSelect($("#sel-model"), models, shared.model, { empty: models.length ? null : "— no model yet —" });
  saveStore("venno-moss-shared", shared);
  for (const f of Object.values(forms)) f.setInfo(i);
  const e = i.engine;
  const mem = e.memory === "auto" ? `auto → ${e.memory_resolved}` : e.memory_resolved;
  $("#status").textContent = i.complete ? `${i.models.length} model(s) · ${i.loras.length} LoRA(s) · memory ${mem}` : "No MOSS-Image model yet: ⚙ Settings → Models";
  $("#lora-default").textContent = i.folders.loras[0] ? i.folders.loras.at(-1) : "";
  $("#te-card").hidden = i.text_encoders.length < 2;
  fillSelect($("#sel-te"), i.text_encoders.map((t) => ({ value: t.name, label: `${t.name} · ${formatBytes(t.size)}` })), i.text_encoder);
  $("#memory-now").textContent = e.device === "cuda" ? `Your graphics card: ${e.gpu}, ${e.vram_gb} GB. Now using: ${e.memory_resolved}.` : "No graphics card in use: running on the CPU (very slow).";
  const row = (k, v) => `<tr><th>${k}</th><td><code>${escapeHtml(String(v))}</code></td></tr>`;
  $("#system").innerHTML = row("Version", i.version) + row("Device", e.device === "cuda" ? `${e.gpu} (${e.vram_gb} GB)` : "CPU") + row("Memory", mem)
    + row("Model folders", i.folders.models.join("\n")) + row("LoRA folders", i.folders.loras.join("\n")) + row("Outputs", i.folders.outputs)
    + row("Text encoder", i.text_encoder || "none yet") + row("Loaded", e.loaded || "nothing yet") + row("Prompts cached", e.prompt_cache);
}

async function loadInfo(refresh = false) {
  try {
    applyInfo(await api(refresh ? `${V}/refresh` : `${V}/info`, refresh ? {} : undefined));
    renderSettings((await api(`${V}/settings`)).settings);
    if (!info.complete && !refresh) {
      showTab("settings");
      toast("Download MOSS-Image Base or SFT to start (or add the folder you already have it in).", "info");
    }
  } catch (e) {
    $("#status").textContent = "Server unreachable";
    toast(`Could not reach Venno MOSS: ${e.message}`);
  }
}
$("#btn-refresh").onclick = () => loadInfo(true);

loadInfo();
