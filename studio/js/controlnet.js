// ControlNet card for the form tabs: up to three guide units, each a picture (or the tab's input image) turned
// into a map by a preprocessor (venno/engine/preprocess.py) and applied by an Anima ControlLLLite model.

import { $, $$, api, escapeHtml, fillSelect, toast } from "./common.js";

const UNIT_DEFAULT = { enabled: false, useInput: false, preprocessor: "lineart", model: "", weight: 0.8, start: 0, end: 0.8, mode: "balanced", resize: 1, cannyLow: 100, cannyHigh: 200 };
export const CONTROLNET_DEFAULT = { enabled: false, unit: 0, units: [0, 1, 2].map(() => ({ ...UNIT_DEFAULT })) };

const MODES = [
  ["balanced", "Balanced", "The guide and the prompt weigh the same."],
  ["prompt", "Prompt", "My prompt is more important: the guide sets the layout and fades in the later blocks, so the prompt decides the details."],
  ["guide", "ControlNet", "ControlNet is more important: the guide acts on the prompt side only, so CFG amplifies it. Needs CFG above 1."],
];
const RESIZE = [[1, "Crop and resize"], [0, "Just resize"], [2, "Resize and fill"]];
const NEEDS = { canny: "LLLite lineart or any-test", lineart: "LLLite lineart or any-test", softedge: "LLLite scribble or any-test", depth: "LLLite depth" };

const readDataURL = (blob) => new Promise((ok, fail) => {
  const r = new FileReader();
  r.onload = () => ok(r.result);
  r.onerror = fail;
  r.readAsDataURL(blob);
});

export class ControlNetCard {
  /** opts: { state: () => CONTROLNET_DEFAULT-shaped, persist, size: () => [w, h] job frame, input?: () => data URL or null } */
  constructor(el, opts) {
    this.el = el;
    this.o = opts;
    this.pictures = [null, null, null]; // data URLs, in memory only (too big for the browser's settings storage)
    this.maps = [null, null, null]; // preview of each unit's map
    this.models = [];
    this.preprocessors = [];
    el.classList.add("cn-card");
    el.innerHTML = `
      <summary><input type="checkbox" class="cn-on" title="Use ControlNet"> <span>ControlNet</span> <span class="cn-sum"></span></summary>
      <div class="hint">Guide the picture with another one: its line art, edges or depth are extracted (<b>preprocessor</b>) and an Anima ControlLLLite model makes the result follow them. Get guide models in ⚙ Settings → Models.</div>
      <div class="seg cn-units">${[0, 1, 2].map((k) => `<button data-k="${k}">Unit ${k + 1}<i></i></button>`).join("")}</div>
      <div class="cn-unit">
        <label class="check"><input type="checkbox" class="cn-enabled"> Enable this unit</label>
        <div class="cn-pics">
          <div class="cn-drop" title="Click, paste or drop a picture"><img class="cn-pic" alt="" draggable="false"><span>Picture<br><small>click or drop</small></span><button class="cn-clear" title="Remove">×</button></div>
          <div class="cn-mapbox"><img class="cn-map" alt="" draggable="false"><span>Map preview</span></div>
        </div>
        ${opts.input ? `<label class="check"><input type="checkbox" class="cn-input"> Use the input image</label>` : ""}
        <div class="grid2">
          <label>Preprocessor <select class="cn-pre"></select></label>
          <label>Model <select class="cn-model"></select></label>
        </div>
        <div class="hint warn cn-need" hidden></div>
        <div class="grid2 cn-canny">
          <label>Canny low <input type="number" class="cn-low" min="0" max="255" step="5"></label>
          <label>Canny high <input type="number" class="cn-high" min="0" max="255" step="5"></label>
        </div>
        <div class="grid2">
          <label>Strength <input type="number" class="cn-weight" min="0" max="2" step="0.05"></label>
          <label>Resize <select class="cn-resize">${RESIZE.map(([v, l]) => `<option value="${v}">${l}</option>`).join("")}</select></label>
          <label>Guide from <input type="number" class="cn-start" min="0" max="1" step="0.05"></label>
          <label>Guide until <input type="number" class="cn-end" min="0" max="1" step="0.05"></label>
        </div>
        <label class="slider">Control mode <span class="seg cn-mode">${MODES.map(([k, l, t]) => `<button data-mode="${k}" title="${escapeHtml(t)}">${l}</button>`).join("")}</span></label>
        <div class="hint cn-mode-hint"></div>
      </div>`;

    const s = () => this.o.state();
    const u = () => s().units[s().unit];
    const on = $(".cn-on", el);
    on.onclick = (e) => e.stopPropagation();
    on.onchange = () => {
      s().enabled = on.checked;
      if (on.checked) { el.open = true; if (!s().units.some((x) => x.enabled)) u().enabled = true; }
      this.changed();
    };
    $$(".cn-units button", el).forEach((b) => (b.onclick = () => { s().unit = Number(b.dataset.k); this.changed(); }));
    $(".cn-enabled", el).onchange = (e) => { u().enabled = e.target.checked; if (e.target.checked) s().enabled = true; this.changed(); };
    const useInput = $(".cn-input", el);
    if (useInput) useInput.onchange = () => { u().useInput = useInput.checked; if (useInput.checked) u().enabled = s().enabled = true; this.changed(true); };
    $(".cn-pre", el).onchange = (e) => {
      u().preprocessor = e.target.value;
      const suggested = this.suggested(u().preprocessor);
      if (suggested) u().model = suggested; // the model trained on this kind of map
      this.changed(true);
    };
    $(".cn-model", el).onchange = (e) => { u().model = e.target.value; this.changed(true); }; // the model decides the line polarity
    for (const [cls, key, refresh] of [[".cn-weight", "weight"], [".cn-start", "start"], [".cn-end", "end"], [".cn-low", "cannyLow", true], [".cn-high", "cannyHigh", true]]) {
      $(cls, el).onchange = (e) => { u()[key] = Number(e.target.value); this.changed(refresh); };
    }
    $(".cn-resize", el).onchange = (e) => { u().resize = Number(e.target.value); this.changed(true); };
    $$(".cn-mode button", el).forEach((b) => (b.onclick = (e) => { e.preventDefault(); u().mode = b.dataset.mode; this.changed(); }));

    // picture: click to pick, paste or drop
    const picker = document.createElement("input");
    picker.type = "file";
    picker.accept = "image/png,image/jpeg,image/webp";
    picker.onchange = async () => { if (picker.files[0]) await this.setPicture(await readDataURL(picker.files[0])); picker.value = ""; };
    const drop = $(".cn-drop", el);
    drop.onclick = (e) => { if (!e.target.closest(".cn-clear")) picker.click(); };
    $(".cn-clear", el).onclick = (e) => { e.stopPropagation(); this.pictures[s().unit] = this.maps[s().unit] = null; this.render(); };
    el.addEventListener("dragover", (e) => e.preventDefault());
    el.addEventListener("drop", async (e) => {
      const f = [...(e.dataTransfer?.files || [])].find((x) => x.type.startsWith("image/"));
      if (!f) return;
      e.preventDefault();
      e.stopPropagation(); // not "load settings from this image"
      await this.setPicture(await readDataURL(f));
    });
    this.render();
  }

  suggested(pre) {
    const keys = this.preprocessors.find((p) => p.key === pre)?.suits || [];
    for (const key of keys) {
      const hit = this.models.find((m) => m.toLowerCase().includes(key));
      if (hit) return hit;
    }
    return "";
  }

  async setPicture(src) {
    const s = this.o.state(), k = s.unit;
    this.pictures[k] = src;
    s.units[k].useInput = false;
    s.units[k].enabled = s.enabled = true;
    this.el.open = true;
    this.changed(true);
  }

  /** The picture a unit uses: its own, or the tab's input image. */
  source(k) {
    const unit = this.o.state().units[k];
    return unit.useInput && this.o.input ? this.o.input() : this.pictures[k];
  }

  changed(refreshMap = false) {
    this.o.persist();
    this.render();
    if (refreshMap) this.refreshMap(this.o.state().unit);
  }

  async refreshMap(k) {
    const unit = this.o.state().units[k], src = this.source(k);
    this.maps[k] = null;
    if (!src) return this.render();
    const [width, height] = this.o.size();
    const seq = (this.seq = (this.seq || 0) + 1);
    $(".cn-mapbox", this.el).classList.add("busy");
    try {
      const res = await api("/venno/v1/guides/preview", { image: src, preprocessor: unit.preprocessor, model: unit.model, width, height, resize_mode: unit.resize, canny_low: unit.cannyLow, canny_high: unit.cannyHigh });
      if (seq === this.seq) this.maps[k] = res.image;
    } catch (e) {
      toast(`ControlNet preview: ${e.message}`);
    } finally {
      if (seq === this.seq) $(".cn-mapbox", this.el).classList.remove("busy");
      this.render();
    }
  }

  render() {
    const s = this.o.state(), el = this.el, k = s.unit, unit = s.units[k];
    $(".cn-on", el).checked = s.enabled;
    if (s.enabled) el.open = true;
    $$(".cn-units button", el).forEach((b, i) => {
      b.classList.toggle("on", i === k);
      b.classList.toggle("active", s.units[i].enabled && !!this.source(i));
    });
    $(".cn-enabled", el).checked = unit.enabled;
    const useInput = $(".cn-input", el);
    if (useInput) useInput.checked = unit.useInput;
    const src = this.source(k);
    $(".cn-pic", el).src = src || "";
    $(".cn-drop", el).classList.toggle("has", !!src);
    $(".cn-clear", el).hidden = !this.pictures[k] || unit.useInput;
    $(".cn-map", el).src = this.maps[k] || "";
    $(".cn-mapbox", el).classList.toggle("has", !!this.maps[k]);
    fillSelect($(".cn-pre", el), this.preprocessors.map((p) => ({ value: p.key, label: p.label })), unit.preprocessor);
    fillSelect($(".cn-model", el), this.models.map((m) => ({ value: m, label: m })), unit.model, { empty: this.models.length ? "— pick a guide model —" : "— none: download one in ⚙ Settings → Models —" });
    const need = $(".cn-need", el);
    need.hidden = !(unit.preprocessor in NEEDS) || !!this.suggested(unit.preprocessor);
    need.textContent = `${this.preprocessors.find((p) => p.key === unit.preprocessor)?.label || ""} maps work with ${NEEDS[unit.preprocessor] || ""}: download it in ⚙ Settings → Models.`;
    $(".cn-canny", el).hidden = unit.preprocessor !== "canny";
    $(".cn-low", el).value = unit.cannyLow;
    $(".cn-high", el).value = unit.cannyHigh;
    $(".cn-weight", el).value = unit.weight;
    $(".cn-start", el).value = unit.start;
    $(".cn-end", el).value = unit.end;
    $(".cn-resize", el).value = unit.resize;
    $$(".cn-mode button", el).forEach((b) => b.classList.toggle("on", b.dataset.mode === unit.mode));
    $(".cn-mode-hint", el).textContent = MODES.find(([m]) => m === unit.mode)?.[2] || "";
    const n = this.activeUnits().length;
    $(".cn-sum", el).textContent = s.enabled ? (n ? `(${n} unit${n > 1 ? "s" : ""})` : "(add a picture)") : "";
    $(".cn-sum", el).classList.toggle("warn", s.enabled && !n);
  }

  activeUnits() {
    const s = this.o.state();
    return s.units.map((unit, k) => ({ unit, k, src: this.source(k) })).filter((x) => x.unit.enabled && x.src);
  }

  setInfo(info) {
    this.models = info.controlnets || [];
    this.preprocessors = info.preprocessors || [{ key: "none", label: "None", suits: [] }];
    for (const unit of this.o.state().units) {
      if (!unit.model || !this.models.includes(unit.model)) unit.model = this.suggested(unit.preprocessor) || this.models[0] || "";
    }
    this.render();
  }

  /** The job's guides for the enabled units (throws when one can't run). */
  job() {
    const s = this.o.state();
    if (!s.enabled) return [];
    const active = this.activeUnits();
    if (!active.length) throw new Error("ControlNet: add a picture to a unit (or turn ControlNet off)");
    return active.map(({ unit, k, src }) => {
      if (!unit.model) throw new Error(`ControlNet unit ${k + 1}: pick a guide model (⚙ Settings → Models has them)`);
      return {
        model: unit.model, image: src, weight: unit.weight, start: unit.start, end: unit.end,
        preprocessor: unit.preprocessor, mode: unit.mode, resize_mode: unit.resize, canny_low: unit.cannyLow, canny_high: unit.cannyHigh,
      };
    });
  }
}
