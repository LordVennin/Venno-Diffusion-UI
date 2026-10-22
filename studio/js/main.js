import { Doc, LAYER_CONTROL, LAYER_PAINT, LAYER_REGION, ctx2d, hasContentIn, loadImage, makeCanvas, strokeRegionMask } from "./doc.js";
import { Generator } from "./generate.js";
import { jobProgress } from "./jobs.js";
import { TOOLS, View } from "./view.js";
import { GenForm } from "./forms.js";
import { ADETAILER_DEFAULT, api, presetForCheckpoint, showLightbox, withQualityPrefix } from "./common.js";
import { DetailerCard, HIRES_DEFAULT, HiresCard } from "./cards.js";
import { GRID_DEFAULT, GridCard } from "./gridui.js";
import { ModelsPage } from "./models.js";
import { SettingsPage } from "./settings.js";
import { InfoPage } from "./info.js";
import { LoraChips } from "./loras.js";
import { describeSettings, hasQualityTags, offerSettings } from "./params.js";
import { enableAutocomplete, setWildcards } from "./autocomplete.js";
import { attachTokenCounter, refreshTokenCounters } from "./tokens.js";
import { openLoraBrowser } from "./lorabrowser.js";

enableAutocomplete();
import { runSteps, SchedulePreview } from "./schedule.js";

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

// ---------------- state ----------------
const STORE_KEY = "venno-studio-v1";
const saved = (() => { try { return JSON.parse(localStorage.getItem(STORE_KEY)) || {}; } catch { return {}; } })();

const state = {
  info: null,
  model: { checkpoint: "", textEncoder: "", vae: "", ...(saved.model || {}) },
  gen: {
    prompt: "", negative: "", loras: [], switchPrompt: "", switchAt: 0.4, strength: 1, batch: 2, steps: 10, cfg: 1, shift: 3, sampler: "Euler", scheduler: "Simple",
    seed: 0, seedLocked: false, resolution: 1024, context: 0.5, feather: 12, qualityPrefix: true, saveImages: true, softInpaint: true,
    regionBase: 0.25, regionEnd: 1, regionAppendRoot: true, regionOverlap: "top", regionSeparation: 0.7, regionRefine: false, regionRefineDenoise: 0.4, regionsOnly: false, regionEdges: "soft", preset: "", ...HIRES_DEFAULT,
    ...(saved.gen || {}),
  },
  showProgressPreview: saved.showProgressPreview ?? true,
  liveOnCanvas: saved.liveOnCanvas ?? true,
};
const brush = { size: 12, opacity: 1, hardness: 0.8, color: "#000000", color2: "#ffffff", pressureSize: true, pressureOpacity: false, selectMode: "replace", fillTolerance: 24, ...(saved.brush || {}) };

let saveTimer = 0;
function save() {
  clearTimeout(saveTimer);
  saveTimer = 0;
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify({ model: state.model, gen: state.gen, brush, showProgressPreview: state.showProgressPreview, liveOnCanvas: state.liveOnCanvas }));
  } catch {}
}
function persist() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(save, 300);
}
addEventListener("pagehide", () => saveTimer && save()); // a change made just before closing or reloading

// ---------------- core objects ----------------
const doc = new Doc(1024, 1024);
const view = new View(doc, $("#viewport"), brush);
const gen = new Generator(doc, view, state);

// ---------------- tabs ----------------
let activeTab = "canvas";
function showTab(name, section = undefined) {
  activeTab = name;
  $$("#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === name));
  $("#btn-settings").classList.toggle("on", name === "settings");
  $$(".page").forEach((p) => (p.hidden = p.id !== `page-${name}`));
  $$(".canvas-only").forEach((el) => (el.style.visibility = name === "canvas" ? "" : "hidden"));
  if (name === "canvas") view.resize();
  if (name === "settings") settingsPage.show(section);
}
$$("#tabs button").forEach((b) => (b.onclick = () => showTab(b.dataset.tab)));
$("#btn-settings").onclick = () => showTab("settings");

const formCtx = {
  state,
  onSendToCanvas: async (src) => { showTab("canvas"); await addImageAsLayer(src, "From gallery"); },
  // a result (and the sending tab's settings) to another tab
  onSend: (target, src, formState) => { showTab(target); forms[target].receive(src, formState); },
  selectCheckpoint: (name) => selectCheckpoint(name),
};
const txt2imgForm = new GenForm($("#page-txt2img"), "txt2img", formCtx);
const img2imgForm = new GenForm($("#page-img2img"), "img2img", formCtx);
const inpaintForm = new GenForm($("#page-inpaint"), "inpaint", formCtx);
const forms = { txt2img: txt2imgForm, img2img: img2imgForm, inpaint: inpaintForm };
const infoPage = new InfoPage($("#page-info"), {
  // Image Info → a tab: the image (where it has one) and the settings read from it
  send: async (target, src, settings) => {
    showTab(target);
    if (target === "canvas") {
      if (src) await addImageAsLayer(src, "From Image Info");
      if (settings) applyCanvasSettings(settings);
      return;
    }
    if (target !== "txt2img" && src) await forms[target].setInputImage(src);
    if (settings) forms[target].applySettings(settings);
  },
});
const settingsPage = new SettingsPage($("#page-settings"), { modelsPage: null, onRescan: () => loadInfo(true) });
const modelsPage = new ModelsPage(settingsPage.modelsRoot, { onInstalled: () => loadInfo(true) });
settingsPage.o.modelsPage = modelsPage;

// ---------------- toast / status ----------------
let toastTimer = 0;
function toast(msg, kind = "error") {
  const t = $("#toast");
  t.textContent = msg;
  t.className = `toast on ${kind === "info" ? "info" : ""}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.className = "toast"), kind === "info" ? 2500 : 6000);
}
view.onToast = (m) => toast(m, "info");
const setStatus = (s) => ($("#status").textContent = s);

/** Venno's files were updated after the server started: new options in the Studio would be ignored until a restart. */
function showRestartBanner(on) {
  let el = $("#restart-note");
  if (!on) return el?.remove();
  if (el) return;
  el = document.createElement("div");
  el.id = "restart-note";
  el.innerHTML = `Venno was updated after it started. <b>Restart Venno</b> (close it and run venno.bat / venno.sh again): until then the server runs the old version and ignores new options. <button title="Hide">×</button>`;
  el.querySelector("button").onclick = () => el.remove();
  document.body.prepend(el);
}

// ---------------- toolbar ----------------
const ICONS = {
  brush: '<path d="M4 20c2 0 4-1 4-3a2 2 0 0 0-2-2c-2 0-2 3-2 5z"/><path d="M8 15 19 4l1 1L9 16"/>',
  eraser: '<path d="m7 21-4-4 10-10 6 6-8 8z"/><path d="M7 21h13"/><path d="m9 11 6 6"/>',
  fill: '<path d="m5 11 7-7 7 7-7 7z"/><path d="M5 11h14"/><path d="M20 16c0 1.5-1 2.5-1 3.5a1 1 0 0 0 2 0c0-1-1-2-1-3.5z"/>',
  picker: '<path d="m14 4 6 6"/><path d="m16 6-9 9-3 5 5-3 9-9"/>',
  "select-rect": '<rect x="4" y="5" width="16" height="14" stroke-dasharray="3 2"/>',
  "select-lasso": '<path d="M12 5c5 0 8 2 8 5s-3 5-8 5c-2 0-3 0-4-.5" stroke-dasharray="3 2"/><path d="M8 14.5C5 14 4 12 4 10c0-3 3-5 8-5"/><path d="M8 15c-1 1-1 3 1 4"/>',
  "select-brush": '<circle cx="12" cy="12" r="7" stroke-dasharray="3 2"/><circle cx="12" cy="12" r="2"/>',
  move: '<path d="M12 3v18M3 12h18"/><path d="m9 6 3-3 3 3M9 18l3 3 3-3M6 9l-3 3 3 3M18 9l3 3-3 3"/>',
  "shape-rect": '<rect x="4" y="6" width="16" height="12" rx="1" fill="currentColor" fill-opacity=".25" stroke="currentColor" stroke-width="2"/>',
  pan: '<path d="M8 13V6a1.5 1.5 0 0 1 3 0v6V4.5a1.5 1.5 0 0 1 3 0V12V6a1.5 1.5 0 0 1 3 0v8c0 4-2.5 7-6 7s-5-2-7-5l-1.5-2.5a1.5 1.5 0 0 1 2.5-1.5z"/>',
};
const toolbar = $("#toolbar");
Object.entries(TOOLS).forEach(([id, t], i) => {
  if (id === "select-rect" || id === "move") toolbar.appendChild(document.createElement("hr"));
  const b = document.createElement("button");
  b.dataset.tool = id;
  b.title = `${t.label} (${t.key.toUpperCase()})`;
  b.innerHTML = `<svg viewBox="0 0 24 24">${ICONS[id]}</svg>`;
  b.onclick = () => setTool(id);
  toolbar.appendChild(b);
});

function setTool(id) {
  view.tool = id;
  $$("#toolbar button").forEach((b) => b.classList.toggle("on", b.dataset.tool === id));
  const paints = ["brush", "eraser", "select-brush"].includes(id);
  $('.group[data-for="paint"]').style.display = paints ? "" : "none";
  $('.group[data-for="fill"]').style.display = id === "fill" ? "" : "none";
  updateSelectionOpts();
  view.canvas.style.cursor = id === "pan" ? "grab" : id === "move" ? "move" : paints ? "none" : "crosshair";
  view.render();
}

function updateSelectionOpts() {
  $('.group[data-for="select"]').style.display = view.tool.startsWith("select") || doc.hasSelection ? "" : "none";
}

// ---------------- tool options ----------------
function bindRange(id, get, set, fmt = (v) => v) {
  const input = $(`#${id}`);
  const out = $(`#out-${id.replace(/^opt-/, "")}`);
  input.value = get();
  if (out) out.textContent = fmt(input.value);
  input.addEventListener("input", () => {
    set(Number(input.value));
    if (out) out.textContent = fmt(input.value);
    persist();
  });
  return input;
}
bindRange("opt-size", () => brush.size, (v) => { brush.size = v; view.render(); });
bindRange("opt-opacity", () => Math.round(brush.opacity * 100), (v) => (brush.opacity = v / 100));
bindRange("opt-hardness", () => Math.round(brush.hardness * 100), (v) => (brush.hardness = v / 100));
bindRange("opt-tolerance", () => brush.fillTolerance, (v) => (brush.fillTolerance = v));
$("#opt-psize").checked = brush.pressureSize;
$("#opt-psize").onchange = (e) => { brush.pressureSize = e.target.checked; persist(); };

function setColor(c) {
  brush.color = c;
  $("#opt-color").value = c;
  persist();
}
$("#opt-color").value = brush.color;
$("#opt-color2").value = brush.color2;
$("#opt-color").oninput = (e) => setColor(e.target.value);
$("#opt-color2").oninput = (e) => { brush.color2 = e.target.value; persist(); };
function swapColors() {
  const a = brush.color;
  setColor(brush.color2);
  brush.color2 = a;
  $("#opt-color2").value = a;
}
$("#btn-swap").onclick = swapColors;
view.onColorPicked = setColor;
["#000000", "#ffffff", "#7f7f7f", "#e53935", "#fb8c00", "#fdd835", "#43a047", "#1e88e5", "#8e24aa", "#f8bbd0", "#8d6e63"].forEach((c) => {
  const b = document.createElement("button");
  b.style.background = c;
  b.title = c;
  b.onclick = () => setColor(c);
  $("#swatches").appendChild(b);
});

$$("#opt-selmode button").forEach((b) => (b.onclick = () => {
  brush.selectMode = b.dataset.mode;
  $$("#opt-selmode button").forEach((x) => x.classList.toggle("on", x === b));
  persist();
}));
$$("#opt-selmode button").forEach((x) => x.classList.toggle("on", x.dataset.mode === brush.selectMode));

$("#opt-showregions").onchange = (e) => { view.showRegions = e.target.checked; view.render(); };
view.onZoom = (z) => ($("#zoom-label").textContent = `${Math.round(z * 100)}%`);

// ---------------- actions ----------------
const actions = {
  new: () => openNewDialog(),
  open: () => $("#file-open").click(),
  export: () => exportPNG(),
  undo: () => doc.history.undo(),
  redo: () => doc.history.redo(),
  fit: () => view.fit(),
  "select-all": () => doc.selectAll(),
  deselect: () => doc.deselect(),
  "invert-selection": () => doc.invertSelection(),
  "fill-selection": () => fillSelection(),
  "clear-selection": () => clearSelection(),
  "add-paint": () => doc.addLayer(LAYER_PAINT),
  "add-region": () => addRegion(),
  "add-control": () => { const l = doc.addLayer(LAYER_CONTROL); const cns = state.info?.controlnets || [];
    l.controlModel = cns.find((m) => /any-?test|lineart|sketch/i.test(m)) || cns[0] || ""; setTool("brush"); setColor("#000000"); renderLayers(); },
  "layer-up": () => doc.moveLayer(doc.active, 1),
  "layer-down": () => doc.moveLayer(doc.active, -1),
  "layer-dup": () => doc.duplicateLayer(doc.active),
  "layer-merge": () => doc.mergeDown(doc.active),
  "layer-clear": () => doc.editPixels(doc.active.canvas, (c) => c.clearRect(0, 0, doc.width, doc.height)),
  "layer-delete": () => doc.removeLayer(doc.active),
};
document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-action]");
  if (b && actions[b.dataset.action]) actions[b.dataset.action]();
});

function syncBrushUI() {
  $("#opt-size").value = brush.size;
  $("#out-size").textContent = brush.size;
}

function fillSelection() {
  const l = doc.active;
  if (!doc.hasSelection) return toast("Nothing selected", "info");
  doc.editPixels(l.canvas, (c) => {
    const tmp = makeCanvas(doc.width, doc.height);
    const t = ctx2d(tmp);
    t.drawImage(doc.selection, 0, 0);
    t.globalCompositeOperation = "source-in";
    t.fillStyle = l.type === LAYER_REGION ? l.color : brush.color;
    t.fillRect(0, 0, doc.width, doc.height);
    c.drawImage(tmp, 0, 0);
  });
}

function clearSelection() {
  if (!doc.hasSelection) return;
  doc.editPixels(doc.active.canvas, (c) => {
    c.globalCompositeOperation = "destination-out";
    c.drawImage(doc.selection, 0, 0);
    c.globalCompositeOperation = "source-over";
  });
}

// ---------------- layers panel ----------------
const KIND = { [LAYER_PAINT]: "paint", [LAYER_REGION]: "region", [LAYER_CONTROL]: "guide" };
function renderLayers() {
  const list = $("#layer-list");
  list.innerHTML = "";
  for (const l of doc.layers) {
    const li = document.createElement("li");
    li.classList.toggle("on", l.id === doc.activeId);
    const thumb = makeCanvas(36, 36);
    const tc = ctx2d(thumb);
    const s = Math.min(36 / doc.width, 36 / doc.height);
    tc.drawImage(l.canvas, (36 - doc.width * s) / 2, (36 - doc.height * s) / 2, doc.width * s, doc.height * s);
    li.appendChild(thumb);
    if (l.type === LAYER_REGION) {
      const dot = document.createElement("span");
      dot.className = "dot";
      dot.style.background = l.color;
      li.appendChild(dot);
    }
    const name = document.createElement("div");
    name.className = "name";
    name.innerHTML = `<div></div><div class="kind"></div>`;
    name.firstChild.textContent = l.prompt ? l.prompt : l.name;
    name.lastChild.textContent = KIND[l.type] + (l.type !== LAYER_REGION && l.prompt.trim() ? " · region" : "");
    li.appendChild(name);
    const eye = document.createElement("button");
    eye.className = `eye ${l.visible ? "" : "off"}`;
    eye.textContent = "👁";
    eye.title = "Toggle visibility";
    eye.onclick = (e) => { e.stopPropagation(); l.visible = !l.visible; doc.emit("layer-props"); };
    li.appendChild(eye);
    li.onclick = () => { doc.activeId = l.id; doc.emit("active"); };
    li.ondblclick = () => {
      const n = prompt("Layer name", l.name);
      if (n) doc.setLayerProp(l, "name", n);
    };
    list.appendChild(li);
  }
  renderLayerProps();
}

function renderLayerProps() {
  const l = doc.active;
  const box = $("#layer-props");
  box.innerHTML = "";
  const opacity = rangeField("Opacity", Math.round(l.opacity * 100), 0, 100, (v) => { l.opacity = v / 100; view.render(); }, "%");
  if (l.type === LAYER_PAINT) {
    box.appendChild(opacity);
    layerPromptBlock(box, l);
  } else if (l.type === LAYER_REGION) {
    const lab = document.createElement("label");
    lab.textContent = "Region prompt";
    const ta = document.createElement("textarea");
    ta.rows = 2;
    ta.value = l.prompt;
    ta.placeholder = "e.g. 1girl, silver hair, blue kimono";
    ta.oninput = () => { l.prompt = ta.value; doc.emit("region-prompt"); updateLayerThumbLabel(); };
    lab.appendChild(ta);
    box.appendChild(lab);
    new SchedulePreview(ta, { getSteps: () => runSteps(g.steps, g.strength), allowAnd: false });
    box.appendChild(regionNegativeField(l));
    const color = document.createElement("label");
    color.textContent = "Overlay color";
    const ci = document.createElement("input");
    ci.type = "color";
    ci.value = l.color;
    ci.onchange = () => {
      // recolor existing paint so the overlay stays consistent
      l.color = ci.value;
      doc.editPixels(l.canvas, (c) => { c.globalCompositeOperation = "source-in"; c.fillStyle = l.color; c.fillRect(0, 0, doc.width, doc.height); c.globalCompositeOperation = "source-over"; });
    };
    color.appendChild(ci);
    box.appendChild(color);
    box.appendChild(rangeField("Region strength", Math.round(l.regionStrength * 100), 10, 200, (v) => (l.regionStrength = v / 100), "%"));
    hint(box, "Paint with Brush/Fill to mark where this prompt applies. Areas outside every region use the main prompt.");
  } else if (l.type === LAYER_CONTROL) {
    const lab = document.createElement("label");
    lab.textContent = "Guide model (ControlNet / ControlLLLite)";
    const sel = document.createElement("select");
    const models = state.info?.controlnets || [];
    sel.innerHTML = `<option value="">— none (layer ignored) —</option>` + models.map((m) => `<option>${escapeHtml(m)}</option>`).join("");
    sel.value = l.controlModel;
    sel.onchange = () => { l.controlModel = sel.value; doc.emit("layer-props"); };
    lab.appendChild(sel);
    box.appendChild(lab);
    const inv = document.createElement("label");
    inv.textContent = "Line polarity sent to the model";
    const isel = document.createElement("select");
    isel.innerHTML = '<option value="auto">Auto (invert for scribble models)</option><option value="no">Dark lines on white</option><option value="yes">Inverted: white lines on black</option>';
    isel.value = l.controlInvert;
    isel.onchange = () => { l.controlInvert = isel.value; gen.scheduleLive(); };
    inv.appendChild(isel);
    box.appendChild(inv);
    box.appendChild(rangeField("Guide strength", Math.round(l.controlWeight * 100), 0, 200, (v) => (l.controlWeight = v / 100), "%"));
    box.appendChild(rangeField("Guide from", Math.round(l.controlStart * 100), 0, 100, (v) => (l.controlStart = v / 100), "%"));
    box.appendChild(rangeField("Guide until", Math.round(l.controlEnd * 100), 0, 100, (v) => (l.controlEnd = v / 100), "%"));
    box.appendChild(opacity);
    layerPromptBlock(box, l);
    hint(box, models.length
      ? "Draw line art (dark lines on transparent/white). It steers composition without being painted into the image."
      : "No ControlNet / ControlLLLite models found. Put Anima ControlLLLite (lineart / scribble) models in models/ControlNet and press ⟳.");
  }
}

// Paint / guide layers can carry their own prompt: their strokes (grown, closed shapes filled) become a region.
let regionPreviewLayer = null;
function layerPromptBlock(box, l) {
  const lab = document.createElement("label");
  lab.textContent = "Layer prompt (turns this sketch into a region)";
  const ta = document.createElement("textarea");
  ta.rows = 2;
  ta.value = l.prompt;
  ta.placeholder = "e.g. 1girl, red dress — applies where this layer is drawn";
  lab.appendChild(ta);
  box.appendChild(lab);
  new SchedulePreview(ta, { getSteps: () => runSteps(g.steps, g.strength), allowAnd: false });
  const extra = document.createElement("div");
  extra.className = "layer-props";
  box.appendChild(extra);
  const renderExtra = () => {
    extra.innerHTML = "";
    if (!l.prompt.trim()) { if (regionPreviewLayer === l) setRegionPreview(null); return; }
    extra.appendChild(regionNegativeField(l));
    extra.appendChild(rangeField("Region spread", l.regionSpread, 8, 256, (v) => { l.regionSpread = v; refreshRegionPreview(); }, "px"));
    const fill = document.createElement("label");
    fill.className = "check";
    fill.innerHTML = `<input type="checkbox" ${l.regionFill ? "checked" : ""}> Fill enclosed shapes`;
    fill.firstChild.onchange = (e) => { l.regionFill = e.target.checked; refreshRegionPreview(); gen.scheduleLive(); };
    extra.appendChild(fill);
    const show = document.createElement("label");
    show.className = "check";
    show.innerHTML = `<input type="checkbox" ${regionPreviewLayer === l ? "checked" : ""}> Show region`;
    show.firstChild.onchange = (e) => setRegionPreview(e.target.checked ? l : null);
    extra.appendChild(show);
  };
  ta.oninput = () => {
    const had = !!l.prompt.trim();
    l.prompt = ta.value;
    doc.emit("region-prompt");
    updateLayerThumbLabel();
    if (had !== !!l.prompt.trim()) renderExtra();
  };
  renderExtra();
}

function setRegionPreview(layer) {
  regionPreviewLayer = layer;
  refreshRegionPreview();
}

function refreshRegionPreview() {
  const l = regionPreviewLayer;
  if (!l || !doc.layers.includes(l) || !l.prompt.trim()) { view.setRegionPreview(null); return; }
  const m = strokeRegionMask(l.canvas, l.regionSpread, l.regionFill);
  const c = ctx2d(m);
  c.globalCompositeOperation = "source-in";
  c.fillStyle = "#ff8a3d";
  c.fillRect(0, 0, m.width, m.height);
  view.setRegionPreview(m);
}

function updateLayerThumbLabel() {
  const li = $("#layer-list li.on .name div");
  if (li) li.textContent = doc.active.prompt || doc.active.name;
}

function rangeField(label, value, min, max, onInput, unit = "") {
  const lab = document.createElement("label");
  lab.innerHTML = `<span>${label} <output>${value}${unit}</output></span>`;
  const r = document.createElement("input");
  r.type = "range";
  r.min = min;
  r.max = max;
  r.value = value;
  r.oninput = () => { onInput(Number(r.value)); lab.querySelector("output").textContent = `${r.value}${unit}`; };
  r.onchange = () => gen.scheduleLive();
  lab.appendChild(r);
  return lab;
}

function hint(box, text) {
  const d = document.createElement("div");
  d.className = "hint";
  d.textContent = text;
  box.appendChild(d);
}

function escapeHtml(s) {
  return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

// ---------------- regions panel: every region with its prompt, without hunting through layers ----------------
function addRegion() {
  const l = doc.addLayer(LAYER_REGION);
  setTool("shape-rect");
  $("#regions-box").open = true;
  toast(`Drag rectangles where "${l.name}" applies (U), or paint it with the brush (B)`, "info");
  renderRegions();
  setTimeout(() => $(`#region-list li[data-id="${l.id}"] textarea`)?.focus(), 50);
}
$("#btn-add-region").onclick = (e) => { e.preventDefault(); e.stopPropagation(); addRegion(); };

let activeRegions = 0;
function updateRegionCount() {
  const items = doc.layers.filter((l) => l.type === LAYER_REGION || l.prompt?.trim());
  activeRegions = items.filter((l) => l.visible && l.prompt.trim() && hasContentIn(l.type === LAYER_REGION ? l.canvas : strokeRegionMask(l.canvas, l.regionSpread, l.regionFill), { x: 0, y: 0, w: doc.width, h: doc.height })).length;
  $("#regions-count").textContent = items.length ? `(${activeRegions} active)` : "";
  updateModeLabel();
}

/** "Region negative" textarea for the layer panel (the Regions list has its own). */
function regionNegativeField(l) {
  const lab = document.createElement("label");
  lab.textContent = "Region negative";
  lab.title = "Negative prompt for this region only (empty = the main negative). Needs CFG above 1.";
  const ta = document.createElement("textarea");
  ta.rows = 1;
  ta.className = "neg";
  ta.value = l.negative;
  ta.placeholder = "optional, e.g. hat, black hair";
  ta.oninput = () => { l.negative = ta.value; doc.emit("region-prompt"); };
  lab.appendChild(ta);
  return lab;
}

function renderRegions() {
  const list = $("#region-list");
  const items = doc.layers.filter((l) => l.type === LAYER_REGION || (l.prompt && l.prompt.trim())).reverse();
  list.innerHTML = "";
  updateRegionCount();
  if (!items.length) { list.innerHTML = `<li class="empty">No regions yet: add one to prompt part of the image separately (e.g. one character each).</li>`; return; }
  for (const l of items) {
    const li = document.createElement("li");
    li.dataset.id = l.id;
    li.classList.toggle("on", l.id === doc.activeId);
    li.classList.toggle("off", !l.visible);
    const color = l.type === LAYER_REGION ? l.color : "#9aa4b0";
    const painted = hasContentIn(l.canvas, { x: 0, y: 0, w: doc.width, h: doc.height });
    li.innerHTML = `
      <span class="dot" style="background:${color}" title="${l.type === LAYER_REGION ? "Region layer" : "Sketch layer with a prompt"}"></span>
      <div class="r-prompts">
        <textarea rows="2" placeholder="Prompt for this region, e.g. 1girl, silver hair, blue kimono"></textarea>
        <details class="r-neg"><summary>Negative</summary><textarea class="neg" rows="1" placeholder="Only for this region, e.g. hat, black hair (the main negative is added)"></textarea></details>
      </div>
      <div class="r-side">
        <button class="r-edit" title="Edit its area (selects the layer and the rectangle tool)">${painted ? "Area" : "Draw area"}</button>
        <label title="Region strength">×<input type="number" class="r-str" min="0.1" max="2" step="0.1"></label>
        <button class="r-eye" title="Use / ignore this region">${l.visible ? "👁" : "—"}</button>
        ${l.type === LAYER_REGION ? `<button class="r-del" title="Delete region">✕</button>` : ""}
      </div>`;
    const ta = li.querySelector("textarea");
    const neg = li.querySelector(".r-neg");
    const nta = neg.querySelector("textarea");
    nta.value = l.negative;
    neg.open = !!l.negative.trim();
    neg.classList.toggle("set", !!l.negative.trim());
    nta.oninput = () => { l.negative = nta.value; neg.classList.toggle("set", !!nta.value.trim()); doc.emit("region-prompt"); };
    nta.onfocus = () => { if (doc.activeId !== l.id) { doc.activeId = l.id; doc.emit("active"); } };
    ta.value = l.prompt;
    ta.oninput = () => { l.prompt = ta.value; doc.emit("region-prompt"); updateLayerThumbLabel(); };
    ta.onfocus = () => { if (doc.activeId !== l.id) { doc.activeId = l.id; doc.emit("active"); } };
    attachTokenCounter(ta);
    attachTokenCounter(nta);
    li.querySelector(".r-str").value = l.regionStrength;
    li.querySelector(".r-str").onchange = (e) => { l.regionStrength = Math.max(0.1, Number(e.target.value) || 1); gen.scheduleLive(); };
    li.querySelector(".r-edit").onclick = () => { doc.activeId = l.id; doc.emit("active"); setTool(l.type === LAYER_REGION ? "shape-rect" : "brush"); };
    li.querySelector(".r-eye").onclick = () => { l.visible = !l.visible; doc.emit("layer-props"); };
    li.querySelector(".r-del")?.addEventListener("click", () => { doc.removeLayer(l); });
    if (!painted) li.classList.add("unpainted");
    list.appendChild(li);
  }
}

let layersTimer = 0;
doc.on((what) => {
  if (["layers", "reset", "layer-props"].includes(what)) renderRegions();
  if (what === "region-prompt") $$("#region-list li").forEach((li) => {  // edited in the layer panel: mirror here
    const l = doc.layers.find((x) => String(x.id) === li.dataset.id), ta = li.querySelector("textarea"), nta = li.querySelector(".r-neg textarea");
    if (l && ta && document.activeElement !== ta) ta.value = l.prompt;
    if (l && nta && document.activeElement !== nta) { nta.value = l.negative; li.querySelector(".r-neg").classList.toggle("set", !!l.negative.trim()); }
  });
  if (what === "region-prompt") {  // a sketch layer gained or lost its prompt: it joins / leaves the list
    const listed = new Set($$("#region-list li[data-id]").map((li) => li.dataset.id));
    const wanted = doc.layers.filter((l) => l.type === LAYER_REGION || l.prompt?.trim()).map((l) => String(l.id));
    if (wanted.length !== listed.size || wanted.some((id) => !listed.has(id))) renderRegions();
  }
  if (what === "active") $$("#region-list li").forEach((li) => li.classList.toggle("on", li.dataset.id === String(doc.activeId)));
  if (what === "pixels") setTimeout(() => $$("#region-list li").forEach((li) => {
    const l = doc.layers.find((x) => String(x.id) === li.dataset.id);
    if (l) { const p = hasContentIn(l.canvas, { x: 0, y: 0, w: doc.width, h: doc.height }); li.classList.toggle("unpainted", !p); li.querySelector(".r-edit").textContent = p ? "Area" : "Draw area"; }
  }) || updateRegionCount(), 160);
  if (what === "region-prompt") updateRegionCount();
  if (["layers", "reset", "active", "layer-props"].includes(what)) renderLayers();
  else if (what === "pixels") { clearTimeout(layersTimer); layersTimer = setTimeout(() => { renderLayers(); if (regionPreviewLayer) refreshRegionPreview(); }, 150); }
  if (what === "reset" && regionPreviewLayer) setRegionPreview(null);
  if (["selection", "reset"].includes(what)) { updateModeLabel(); updateSelectionOpts(); }
  if (["pixels", "selection", "layers", "layer-props", "region-prompt"].includes(what)) gen.scheduleLive(what === "region-prompt" ? 1200 : 700);
  updateUndoButtons();
});
doc.history.onChange = updateUndoButtons;
function updateUndoButtons() {
  $('[data-action="undo"]').disabled = !doc.history.undoStack.length;
  $('[data-action="redo"]').disabled = !doc.history.redoStack.length;
}

// ---------------- generate panel ----------------
const g = state.gen;
// ADetailer / grid settings like the other tabs (older saves had a single "ADetailer (faces)" switch)
g.detailer = { ...structuredClone(ADETAILER_DEFAULT), ...(g.detailer || {}), ...(g.detailer ? {} : { enabled: g.adetailer !== false }) };
delete g.adetailer;
g.detailer.units = ADETAILER_DEFAULT.units.map((d, i) => ({ ...d, ...(g.detailer.units?.[i] || {}) })); // always both detector slots
g.grid = { ...structuredClone(GRID_DEFAULT), ...(g.grid || {}) };
const prompt_ = $("#prompt");
const negative = $("#negative");
prompt_.value = g.prompt;
negative.value = g.negative;
prompt_.oninput = () => { g.prompt = prompt_.value; persist(); gen.scheduleLive(1200); };
attachTokenCounter(prompt_, { text: () => withQualityPrefix(prompt_.value, g.qualityPrefix, state.info) }); // what the job sends
attachTokenCounter(negative);
negative.oninput = () => { g.negative = negative.value; persist(); gen.scheduleLive(1200); };

bindRange("strength", () => Math.round(g.strength * 100), (v) => { g.strength = v / 100; updateModeLabel(); gen.scheduleLive(); }, (v) => `${v}%`);
$("#batch").value = g.batch;
$("#batch").onchange = (e) => { g.batch = Math.max(1, Math.min(8, Number(e.target.value) || 1)); persist(); };

function updateModeLabel() {
  const sel = doc.hasSelection;
  const s = Math.round(g.strength * 100);
  $("#regions-only-box").hidden = !activeRegions;
  let label;
  if (g.regionsOnly && activeRegions) {
    const n = `${activeRegions} region${activeRegions > 1 ? "s" : ""}`;
    label = `${s >= 100 ? "Regenerate" : `Refine ${s}%:`} ${n}${sel ? " inside the selection" : ""}, one at a time (the rest stays)`;
  } else if (s >= 100) label = sel ? "Generate inside selection (inpaint)" : "Generate new image (replaces canvas view)";
  else label = sel ? `Refine selection ${s}% (inpaint)` : `Refine canvas ${s}% (image-to-image / sketch)`;
  $("#mode-label").textContent = label;
  $("#btn-generate").textContent = s >= 100 ? "Generate" : "Refine";
}
$("#regions-only").checked = !!g.regionsOnly;
$("#regions-only").onchange = (e) => { g.regionsOnly = e.target.checked; persist(); updateModeLabel(); gen.scheduleLive(); };

$("#btn-generate").onclick = () => gen.run();
$("#btn-cancel").onclick = () => gen.interrupt();
$("#btn-apply").onclick = () => gen.apply();
$("#btn-discard").onclick = () => gen.discardPreview();

$("#live").onchange = (e) => {
  gen.setLive(e.target.checked);
  $("#live-panel").hidden = !e.target.checked;
};
$("#live-oncanvas").checked = state.liveOnCanvas;
$("#live-oncanvas").onchange = (e) => {
  state.liveOnCanvas = e.target.checked;
  persist();
  if (!e.target.checked && gen.selected?.result === gen.live.result) gen.discardPreview();
  else if (e.target.checked && gen.live.result) gen.preview(gen.live.result, 0);
};
$("#btn-live-apply").onclick = () => gen.live.result && gen.apply(gen.live.result, 0);
$("#btn-live-seed").onclick = () => gen.newLiveSeed();

// settings
function bindValue(id, key, { num = true, onChange = null } = {}) {
  const el = $(`#${id}`);
  el.value = g[key];
  el.onchange = () => { g[key] = num ? Number(el.value) : el.value; persist(); onChange && onChange(); };
  return el;
}
bindValue("steps", "steps");
bindValue("cfg", "cfg");
bindValue("shift", "shift");
bindValue("sampler", "sampler", { num: false });
bindValue("scheduler", "scheduler", { num: false });
bindValue("seed", "seed");
bindValue("resolution", "resolution");
$("#seed-lock").checked = g.seedLocked;
$("#seed-lock").onchange = (e) => { g.seedLocked = e.target.checked; persist(); };
bindRange("context", () => Math.round(g.context * 100), (v) => (g.context = v / 100), (v) => `${v}%`);
bindRange("feather", () => g.feather, (v) => (g.feather = v));
bindRange("region-base", () => Math.round(g.regionBase * 100), (v) => (g.regionBase = v / 100), (v) => `${v}%`);
bindRange("region-end", () => Math.round(g.regionEnd * 100), (v) => (g.regionEnd = v / 100), (v) => `${v}%`);
bindRange("region-sep", () => Math.round(g.regionSeparation * 100), (v) => (g.regionSeparation = v / 100), (v) => `${v}%`);
bindRange("region-refine-denoise", () => Math.round(g.regionRefineDenoise * 100), (v) => (g.regionRefineDenoise = v / 100), (v) => `${v}%`);
const hrCard = new HiresCard($("#canvas-hr-card"), {
  state: () => g, persist, size: () => { const p = gen.plan(); return [p.gw, p.gh]; },
  note: "Canvas: used when generating a new picture (no selection, Strength 100% or an empty canvas); the result is scaled to the canvas. Not in live mode.",
});
const adCard = new DetailerCard($("#canvas-det-card"), { state: () => g.detailer, persist, note: "Not used in live mode." });
const gridCard = new GridCard($("#canvas-grid-card"), { state: () => g.grid, persist, info: () => state.info, loras: () => g.loras });
$("#resolution").addEventListener("change", () => hrCard.updateInfo());
doc.on((what) => { if (what === "reset") hrCard.updateInfo(); });
$("#region-overlap").value = g.regionOverlap;
$("#region-edges").value = g.regionEdges || "soft";
$("#region-edges").onchange = (e) => { g.regionEdges = e.target.value; persist(); gen.scheduleLive(); };
view.regionOverlap = g.regionOverlap;
$("#region-overlap").onchange = (e) => { g.regionOverlap = view.regionOverlap = e.target.value; persist(); view.render(); gen.scheduleLive(); };
for (const [id, key] of [["quality-prefix", "qualityPrefix"], ["save-images", "saveImages"], ["soft-inpaint", "softInpaint"], ["region-append", "regionAppendRoot"], ["region-refine", "regionRefine"]]) {
  $(`#${id}`).checked = g[key];
  $(`#${id}`).onchange = (e) => { g[key] = e.target.checked; persist(); if (key === "qualityPrefix") refreshTokenCounters(); };
}
$("#progress-preview").checked = state.showProgressPreview;
$("#progress-preview").onchange = (e) => { state.showProgressPreview = e.target.checked; persist(); };

$("#preset").onchange = (e) => applyPreset(e.target.value);
function applyPreset(name) {
  const p = state.info?.presets?.[name];
  if (!p) return;
  Object.assign(g, { preset: name, sampler: p.sampler, scheduler: p.scheduler, steps: p.steps, cfg: p.cfg, shift: p.shift });
  for (const k of ["sampler", "scheduler", "steps", "cfg", "shift"]) $(`#${k}`).value = g[k];
  $("#preset").value = name;
  persist();
}

const loraChips = new LoraChips($("#lora-chips"), { get: () => g.loras, set: (list) => { g.loras = list; persist(); gen.scheduleLive(700); } });
loraChips.watch(prompt_, (text) => { g.prompt = text; persist(); });
loraChips.watch(negative, (text) => { g.negative = text; persist(); });
// "then switch to" prompt + what-happens-when preview
const switchTa = $("#switch-prompt"), switchAt = $("#switch-at");
const showSwitch = () => {
  $("#out-switch-at").textContent = `${Math.round(g.switchAt * 100)}%`;
  $("#switch-state").textContent = g.switchPrompt.trim() ? `(on, after ${Math.round(g.switchAt * 100)}%)` : "";
};
switchTa.value = g.switchPrompt;
attachTokenCounter(switchTa, { text: () => withQualityPrefix(switchTa.value, g.qualityPrefix, state.info) });
switchAt.value = Math.round(g.switchAt * 100);
$("#switch-box").open = !!g.switchPrompt.trim();
const promptSchedule = new SchedulePreview(prompt_, {
  el: $("#prompt-schedule"),
  getSteps: () => runSteps(g.steps, g.strength),
  getSwitch: () => (g.switchPrompt.trim() ? { prompt: g.switchPrompt, at: g.switchAt } : null),
});
switchTa.oninput = () => { g.switchPrompt = switchTa.value; persist(); showSwitch(); promptSchedule.refresh(); gen.scheduleLive(1200); };
switchAt.oninput = () => { g.switchAt = Number(switchAt.value) / 100; persist(); showSwitch(); promptSchedule.refresh(); gen.scheduleLive(700); };
loraChips.watch(switchTa, (text) => { g.switchPrompt = text; persist(); });
showSwitch();
for (const id of ["steps", "strength"]) $(`#${id}`).addEventListener("input", () => promptSchedule.refresh());
for (const id of ["steps", "strength"]) $(`#${id}`).addEventListener("change", () => promptSchedule.refresh());

$("#btn-lora-browser").onclick = () => openLoraBrowser({
  chips: loraChips, prompt: prompt_,
  current: () => (gen.selected ? gen.selected.result.images[gen.selected.index]?.src : null),
});
$("#sel-lora").onchange = (e) => {
  if (e.target.value) loraChips.add(e.target.value);
  e.target.value = "";
};

// ---------------- generator events ----------------
gen.on((what, data) => {
  if (what === "busy") {
    $("#btn-generate").disabled = true;
    $("#btn-cancel").hidden = false;
    $("#progress").classList.add("on");
    setProgress(0, data.live ? "Live…" : "Starting…");
  } else if (what === "progress") {
    const { pct, text } = jobProgress(data);
    setProgress(pct, text);
  } else if (what === "idle") {
    $("#btn-generate").disabled = false;
    $("#btn-cancel").hidden = true;
    $("#progress").classList.remove("on");
  } else if (what === "results") {
    renderThumbs();
    $("#seed").value = g.seed;
  } else if (what === "selected") {
    $("#btn-apply").disabled = !gen.selected;
    $("#btn-discard").disabled = !gen.selected;
    $$("#thumbs img").forEach((img) => img.classList.toggle("on", gen.selected && img._r === gen.selected.result && img._i === gen.selected.index));
  } else if (what === "live-result") {
    $("#live-img").src = data.images[0].src;
  } else if (what === "grid") {
    showLightbox(data);
  } else if (what === "error") {
    toast(data);
  } else if (what === "detected") {  // ADetailer's report: what it found (or that it found nothing)
    toast(`ADetailer: ${data}${/\d+ found/.test(data) ? "" : " — nothing was refined"}. Right-click the result to see the detections.`, "info");
  }
});

function setProgress(pct, text) {
  $("#progress .bar").style.width = `${pct}%`;
  $("#progress span").textContent = text;
}

function renderThumbs() {
  const box = $("#thumbs");
  box.innerHTML = "";
  if (!gen.results.length) { box.innerHTML = '<div class="empty">Results appear here</div>'; return; }
  for (const r of gen.results) {
    const grp = document.createElement("div");
    grp.className = "group";
    r.images.forEach((im, i) => {
      const img = document.createElement("img");
      img.src = im.src;
      img._r = r;
      img._i = i;
      if (r.grid?.[i]) {
        img.title = "Comparison grid — click to view";
        img.classList.add("grid");
        img.onclick = () => showLightbox(im.src);
      } else {
        img.title = `Seed ${r.seeds?.[i] ?? "?"} — click to preview, double-click to apply`;
        img.onclick = () => gen.preview(r, i);
        img.ondblclick = () => gen.apply(r, i);
        if (r.detections?.[i]) {  // like Forge's ADetailer preview
          img.title += `\nADetailer: ${r.detectNotes[i]} — right-click to see the detections`;
          img.classList.add("detected");
          img.oncontextmenu = (e) => { e.preventDefault(); showLightbox(r.detections[i], `venno-detections-${r.seeds?.[i] ?? 0}.png`); };
        }
      }
      grp.appendChild(img);
    });
    box.appendChild(grp);
  }
  gen.emit("selected");
}

// ---------------- models ----------------
function fillSelect(sel, items, value, { empty = null } = {}) {
  sel.innerHTML = (empty !== null ? `<option value="">${empty}</option>` : "") + items.map((i) => `<option value="${escapeHtml(i.value)}">${escapeHtml(i.label)}</option>`).join("");
  if (value !== undefined && items.some((i) => i.value === value)) sel.value = value;
}

function guessModule(mods, words) {
  return mods.find((m) => words.some((w) => m.name.toLowerCase().includes(w)))?.name || "";
}

async function loadInfo(refresh = false) {
  try {
    setStatus(refresh ? "Rescanning models…" : "Connecting…");
    const info = refresh ? await api("/venno/v1/refresh", {}) : await api("/venno/v1/info");
    state.info = info;
    showRestartBanner(info.restart_needed);
    refreshTokenCounters(); // the quality prefix comes with the info
    const ckpts = info.checkpoints.map((c) => ({ value: c.title, label: c.name }));
    const mods = info.modules.filter((m) => !/qwen3[._-]?5[._-]?4b|qwen35_4b/i.test(m.name)); // Anima 3.8B's second encoder is picked automatically
    if (!ckpts.some((c) => c.value === state.model.checkpoint)) {
      const cur = ckpts.find((c) => c.value === info.current_checkpoint);
      state.model.checkpoint = cur ? cur.value : ckpts.find((c) => /anima/i.test(c.label))?.value || ckpts[0]?.value || "";
    }
    if (!mods.some((m) => m.name === state.model.textEncoder)) state.model.textEncoder = guessModule(mods, ["qwen_3", "qwen3", "06b", "0.6b"]);
    if (!mods.some((m) => m.name === state.model.vae)) state.model.vae = guessModule(mods, ["qwen_image_vae", "vae"]);

    fillSelect($("#sel-checkpoint"), ckpts, state.model.checkpoint, { empty: ckpts.length ? null : "— no models found —" });
    const modItems = mods.map((m) => ({ value: m.name, label: m.name }));
    fillSelect($("#sel-te"), modItems, state.model.textEncoder, { empty: "— none —" });
    fillSelect($("#sel-vae"), modItems, state.model.vae, { empty: "— none —" });
    fillSelect($("#sampler"), info.samplers.map((s) => ({ value: s, label: s })), g.sampler);
    fillSelect($("#scheduler"), info.schedulers.map((s) => ({ value: s, label: s })), g.scheduler);
    fillSelect($("#preset"), Object.entries(info.presets).map(([k, p]) => ({ value: k, label: p.label })), g.preset, { empty: "Custom" });
    const loraSel = $("#sel-lora");
    loraSel.innerHTML = `<option value="">+ LoRA… (${info.loras.length})</option>` + info.loras.map((l) => `<option value="${escapeHtml(l.name)}">${escapeHtml(l.name)}</option>`).join("");
    loraChips.setAvailable(info.loras.map((l) => l.name));
    setWildcards(info.wildcards);
    if (!g.negative && info.prompt_defaults?.negative) { g.negative = info.prompt_defaults.negative; negative.value = g.negative; }
    if (!g.preset) applyPreset(presetForCheckpoint(state.model.checkpoint));
    persist();
    renderLayerProps();
    for (const f of Object.values(forms)) f.setInfo(info);
    hrCard.setInfo(info);
    adCard.setInfo(info);
    gridCard.render();
    if (!ckpts.length && !refresh) showTab("settings", "models");
    setStatus(ckpts.length ? `${ckpts.length} model(s) · ${info.loras.length} LoRA(s) · ${info.controlnets.length} guide model(s)` : "No Anima checkpoint yet — download one in ⚙ Settings → Models");
  } catch (e) {
    setStatus("Backend unreachable");
    toast(`Could not reach the backend: ${e.message}`);
  }
}

$("#sel-checkpoint").onchange = (e) => {
  state.model.checkpoint = e.target.value;
  const preset = presetForCheckpoint(e.target.value);
  if (g.preset && g.preset !== preset) { applyPreset(preset); toast(`Switched to ${state.info.presets[preset].label} settings`, "info"); }
  persist();
};
$("#sel-te").onchange = (e) => { state.model.textEncoder = e.target.value; persist(); };

/** Select a checkpoint by name (from loaded settings) without switching presets. */
function selectCheckpoint(name) {
  const stem = (v) => v.replace(/\.[^.]+$/, "").replace(/\s*\[[0-9a-f]+\]$/i, "").toLowerCase();
  const opt = [...$("#sel-checkpoint").options].find((o) => stem(o.value) === stem(name) || stem(o.text) === stem(name));
  if (!opt) { toast(`Model "${name}" is not installed; keeping ${$("#sel-checkpoint").value || "the current model"}`, "info"); return false; }
  $("#sel-checkpoint").value = opt.value;
  state.model.checkpoint = opt.value;
  persist();
  return true;
}

/** Canvas: apply settings read from an image. */
function applyCanvasSettings(x) {
  if (x.prompt != null) g.prompt = prompt_.value = x.prompt;
  if (x.negative != null) g.negative = negative.value = x.negative;
  g.loras = x.loras || [];
  loraChips.render();
  if (hasQualityTags(g.prompt)) { g.qualityPrefix = false; $("#quality-prefix").checked = false; }
  for (const k of ["steps", "cfg", "shift", "seed"]) if (x[k] != null) { g[k] = x[k]; $(`#${k}`).value = x[k]; }
  if (x.seed != null) { g.seedLocked = true; $("#seed-lock").checked = true; }
  if (x.sampler && state.info?.samplers?.includes(x.sampler)) { g.sampler = x.sampler; $("#sampler").value = x.sampler; }
  if (x.scheduler && state.info?.schedulers?.includes(x.scheduler)) { g.scheduler = x.scheduler; $("#scheduler").value = x.scheduler; }
  g.preset = ""; $("#preset").value = "";
  g.switchPrompt = switchTa.value = x.switch_prompt || "";
  if (x.switch_at != null) { g.switchAt = x.switch_at; switchAt.value = Math.round(x.switch_at * 100); }
  $("#switch-box").open = !!g.switchPrompt;
  showSwitch();
  refreshTokenCounters();
  if (x.detailer) {
    g.detailer.enabled = x.detailer.length > 0;
    x.detailer.slice(0, g.detailer.units.length).forEach((d, i) => Object.assign(g.detailer.units[i], { model: d.model, confidence: d.confidence, denoise: d.denoise, prompt: d.prompt || "" }));
    adCard.render();
  }
  g.hr = !!x.hires;
  if (x.hires) {
    const h = x.hires;
    if (state.info?.upscalers?.includes(h.upscaler)) g.hrUpscaler = h.upscaler;
    Object.assign(g, { hrScale: h.scale || g.hrScale, hrResizeX: h.resize_x || 0, hrResizeY: h.resize_y || 0, hrSteps: h.steps || 0, hrDenoise: h.denoise ?? g.hrDenoise, hrCfg: h.cfg || 0, hrShift: h.shift || 0 });
  }
  hrCard.render();
  if (x.checkpoint) selectCheckpoint(x.checkpoint);
  persist();
  promptSchedule.refresh();
  toast(`Loaded: ${describeSettings(x)}${x.regions?.length ? " (region prompts need their region layers, which images don't store)" : ""}`, "info");
}
$("#sel-vae").onchange = (e) => { state.model.vae = e.target.value; persist(); };
$("#btn-refresh").onclick = () => loadInfo(true);

// ---------------- files ----------------
async function addImageAsLayer(src, name = "Image") {
  const img = await loadImage(src);
  if (isBlankDoc()) {
    // adopt the image size (snapped to multiples of 8) for a fresh canvas
    const w = Math.round(img.width / 8) * 8, h = Math.round(img.height / 8) * 8;
    if (w !== doc.width || h !== doc.height) doc.reset(w, h);
  }
  const c = makeCanvas(doc.width, doc.height);
  const s = Math.min(doc.width / img.width, doc.height / img.height, 1);
  const w = img.width * s, h = img.height * s;
  ctx2d(c).drawImage(img, (doc.width - w) / 2, (doc.height - h) / 2, w, h);
  const bg = doc.layers.find((l) => l.type === LAYER_PAINT);
  doc.addLayer(LAYER_PAINT, { name, canvas: c, above: bg });
}

function isBlankDoc() {
  return doc.history.undoStack.length === 0 && doc.layers.length === 2;
}

$("#file-open").onchange = async (e) => {
  const f = e.target.files[0];
  e.target.value = "";
  if (f) { addImageAsLayer(URL.createObjectURL(f), f.name.replace(/\.[^.]+$/, "")); offerSettings(f, applyCanvasSettings); }
};

const vp = $("#viewport");
vp.addEventListener("dragover", (e) => { e.preventDefault(); vp.classList.add("dragging"); });
vp.addEventListener("dragleave", () => vp.classList.remove("dragging"));
vp.addEventListener("drop", (e) => {
  e.preventDefault();
  vp.classList.remove("dragging");
  const f = [...e.dataTransfer.files].find((x) => x.type.startsWith("image/"));
  if (f) { addImageAsLayer(URL.createObjectURL(f), f.name.replace(/\.[^.]+$/, "")); offerSettings(f, applyCanvasSettings); }
});
document.addEventListener("paste", (e) => {
  if (e.target.closest("textarea, input")) return;
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
  if (!item) return;
  e.preventDefault();
  const file = item.getAsFile();
  const src = URL.createObjectURL(file);
  if (activeTab === "img2img" || activeTab === "inpaint") forms[activeTab].setInputImage(src, { offer: true });
  else if (activeTab === "info") infoPage.load(file);
  else if (activeTab === "canvas") { addImageAsLayer(src, "Pasted"); offerSettings(file, applyCanvasSettings); }
});

function exportPNG() {
  const c = doc.composite(null, null, null, { background: null });
  c.toBlob((blob) => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `venno-${Date.now()}.png`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }, "image/png");
}

// new canvas dialog
const SIZES = [["Square", 1024, 1024], ["Portrait", 832, 1216], ["Landscape", 1216, 832], ["Tall", 768, 1344], ["Wide", 1344, 768], ["Small", 768, 768]];
SIZES.forEach(([n, w, h]) => {
  const b = document.createElement("button");
  b.type = "button";
  b.innerHTML = `${n}<br><small>${w}×${h}</small>`;
  b.onclick = () => { $("#new-w").value = w; $("#new-h").value = h; };
  $("#size-presets").appendChild(b);
});
function openNewDialog() {
  $("#new-w").value = doc.width;
  $("#new-h").value = doc.height;
  $("#dlg-new").showModal();
}
$("#dlg-new").addEventListener("close", () => {
  if ($("#dlg-new").returnValue !== "ok") return;
  const w = Math.max(64, Math.min(4096, Number($("#new-w").value) || 1024));
  const h = Math.max(64, Math.min(4096, Number($("#new-h").value) || 1024));
  gen.discardPreview();
  if ($("#new-resize").checked) doc.resizeCanvas(w, h);
  else doc.reset(w, h);
  view.fit();
});

// ---------------- keyboard ----------------
document.addEventListener("keydown", (e) => {
  if (activeTab !== "canvas") return;
  const typing = e.target.closest("textarea, input:not([type=range]):not([type=checkbox]), select");
  const mod = e.ctrlKey || e.metaKey;
  if (mod && e.key === "Enter") { e.preventDefault(); gen.run(); return; }
  if (typing) return;
  const k = e.key.toLowerCase();
  if (mod && k === "z") { e.preventDefault(); e.shiftKey ? doc.history.redo() : doc.history.undo(); return; }
  if (mod && k === "y") { e.preventDefault(); doc.history.redo(); return; }
  if (mod && k === "a") { e.preventDefault(); doc.selectAll(); return; }
  if (mod && k === "d") { e.preventDefault(); doc.deselect(); return; }
  if (mod && e.shiftKey && k === "i") { e.preventDefault(); doc.invertSelection(); return; }
  if (mod && k === "s") { e.preventDefault(); exportPNG(); return; }
  if (mod) return;
  if (e.key === " ") { e.preventDefault(); view.spaceDown = true; view.canvas.style.cursor = "grab"; return; }
  if (e.key === "[") { brush.size = Math.max(1, Math.round(brush.size / 1.2)); syncBrushUI(); view.render(); return; }
  if (e.key === "]") { brush.size = Math.min(400, Math.round(brush.size * 1.2) + 1); syncBrushUI(); view.render(); return; }
  if (e.key === "Delete" || e.key === "Backspace") { e.preventDefault(); e.altKey ? fillSelection() : clearSelection(); return; }
  if (e.key === "Enter" && gen.selected) { gen.apply(); return; }
  if (e.key === "Escape") { if (gen.selected) gen.discardPreview(); else doc.deselect(); return; }
  if (k === "x") { swapColors(); return; }
  if (k === "0") { view.fit(); return; }
  if (k === "1") { view.setZoom(1); return; }
  const tool = Object.entries(TOOLS).find(([, t]) => t.key === k);
  if (tool) setTool(tool[0]);
});
document.addEventListener("keyup", (e) => {
  if (e.key === " ") { view.spaceDown = false; setTool(view.tool); }
});
window.addEventListener("beforeunload", (e) => {
  if (doc.history.undoStack.length) { e.preventDefault(); e.returnValue = ""; }
});

// ---------------- boot ----------------
setTool("brush");
renderLayers();
renderRegions();
updateModeLabel();
updateUndoButtons();
loadInfo();
window.venno = { doc, view, gen, state, forms }; // handy for debugging from the console
