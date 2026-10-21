// Option cards shared by the Canvas and the form tabs: Hires. fix and ADetailer.
// Each card renders into a <details class="card"> and reads / writes a plain state object.

import { $, $$, fillSelect } from "./common.js";

export const HIRES_DEFAULT = { hr: false, hrUpscaler: "Latent", hrScale: 1.5, hrResizeX: 0, hrResizeY: 0, hrSteps: 0, hrDenoise: 0.5, hrCfg: 0, hrShift: 0 };

/** The job's `hires` field for a state holding the HIRES_DEFAULT keys, or null when off. */
export function hiresJob(s) {
  if (!s.hr) return null;
  return {
    upscaler: s.hrUpscaler,
    scale: s.hrScale,
    resize_x: s.hrResizeX,
    resize_y: s.hrResizeY,
    steps: s.hrSteps,
    denoise: s.hrDenoise,
    cfg: s.hrCfg || null,
    shift: s.hrShift || null,
  };
}

/** The job's `detailer` list for an ADetailer state ({enabled, units}). */
export function detailerJob(ad) {
  if (!ad?.enabled) return [];
  return ad.units
    .filter((u) => u.model && u.model !== "None")
    .map((u) => ({ model: u.model, prompt: u.prompt || "", negative: u.negative || "", confidence: u.confidence, denoise: u.denoise, mask_blur: u.maskBlur, dilate: u.dilate, padding: u.padding }));
}

export class HiresCard {
  /** opts: { state: () => object with HIRES_DEFAULT keys, persist, size: () => [w, h] first-pass size, note?: html } */
  constructor(el, opts) {
    this.el = el;
    this.o = opts;
    el.classList.add("hr-card");
    el.innerHTML = `
      <summary><input type="checkbox" class="hr-on" title="Use hires fix"> <span>Hires. fix</span> <span class="hr-sum"></span></summary>
      ${opts.note ? `<div class="hint">${opts.note}</div>` : ""}
      <div class="grid2">
        <label>Upscaler <select class="hr-upscaler"></select></label>
        <label>Hires steps <input type="number" class="hr-steps" min="0" max="150" title="0 = same as first pass"></label>
        <label>Upscale by <input type="number" class="hr-scale" min="1" max="4" step="0.05"></label>
        <label>Denoising <input type="number" class="hr-denoise" min="0" max="1" step="0.05"></label>
        <label>Resize width to <input type="number" class="hr-x" min="0" max="4096" step="64" title="0 = use Upscale by"></label>
        <label>Resize height to <input type="number" class="hr-y" min="0" max="4096" step="64" title="0 = use Upscale by"></label>
        <label>Hires CFG <input type="number" class="hr-cfg" min="0" max="30" step="0.1" title="0 = same as first pass"></label>
        <label>Hires shift <input type="number" class="hr-shift" min="0" max="20" step="0.1" title="0 = same as first pass"></label>
      </div>
      <div class="hint hr-info"></div>`;
    const s = () => this.o.state();
    const on = $(".hr-on", el);
    on.onclick = (e) => e.stopPropagation(); // the box toggles the option, the title opens the settings
    on.onchange = () => { s().hr = on.checked; if (on.checked) el.open = true; this.changed(); };
    for (const [cls, key] of [["hr-steps", "hrSteps"], ["hr-scale", "hrScale"], ["hr-denoise", "hrDenoise"], ["hr-x", "hrResizeX"], ["hr-y", "hrResizeY"], ["hr-cfg", "hrCfg"], ["hr-shift", "hrShift"]]) {
      const input = $(`.${cls}`, el);
      input.onchange = () => { s()[key] = Number(input.value); this.changed(); };
    }
    $(".hr-upscaler", el).onchange = (e) => { s().hrUpscaler = e.target.value; this.changed(); };
    this.render();
  }

  changed() {
    this.o.persist();
    this.updateInfo();
  }

  render() {
    const s = this.o.state(), el = this.el;
    $(".hr-on", el).checked = s.hr;
    if (s.hr) el.open = true;
    for (const [cls, key] of [["hr-steps", "hrSteps"], ["hr-scale", "hrScale"], ["hr-denoise", "hrDenoise"], ["hr-x", "hrResizeX"], ["hr-y", "hrResizeY"], ["hr-cfg", "hrCfg"], ["hr-shift", "hrShift"]]) $(`.${cls}`, el).value = s[key];
    $(".hr-upscaler", el).value = s.hrUpscaler;
    this.updateInfo();
  }

  setInfo(info) {
    fillSelect($(".hr-upscaler", this.el), (info.upscalers || []).map((x) => ({ value: x, label: x })), this.o.state().hrUpscaler);
  }

  updateInfo() {
    const s = this.o.state();
    const [w0, h0] = this.o.size();
    let w = s.hrResizeX || Math.round((w0 * s.hrScale) / 64) * 64;
    let h = s.hrResizeY || Math.round((h0 * s.hrScale) / 64) * 64;
    if (s.hrResizeX && !s.hrResizeY) h = Math.round((h0 * s.hrResizeX) / w0 / 64) * 64;
    if (s.hrResizeY && !s.hrResizeX) w = Math.round((w0 * s.hrResizeY) / h0 / 64) * 64;
    $(".hr-sum", this.el).textContent = s.hr ? `(${w}×${h})` : "";
    $(".hr-info", this.el).textContent = s.hr ? `${w0}×${h0} → ${w}×${h}` : "Generate at low resolution, then upscale and refine (like Forge's Hires. fix).";
  }

  job() { return hiresJob(this.o.state()); }
}

export class DetailerCard {
  /** opts: { state: () => {enabled, units}, persist, note?: html } */
  constructor(el, opts) {
    this.el = el;
    this.o = opts;
    el.classList.add("det-card");
    el.innerHTML = `
      <summary><input type="checkbox" class="det-on" title="Use ADetailer"> <span>ADetailer</span></summary>
      ${opts.note ? `<div class="hint">${opts.note}</div>` : ""}
      ${[0, 1].map((n) => `
      <div class="det-unit" data-unit="${n}">
        <h4>${n === 0 ? "1st" : "2nd"} detector</h4>
        <div class="grid2">
          <label>Model <select class="det-model"></select></label>
          <label>Denoising <input type="number" class="det-denoise" min="0" max="1" step="0.05"></label>
          <label>Confidence <input type="number" class="det-confidence" min="0" max="1" step="0.05"></label>
          <label>Mask blur <input type="number" class="det-blur" min="0" max="64"></label>
          <label>Dilate / erode <input type="number" class="det-dilate" min="-64" max="64"></label>
          <label>Padding <input type="number" class="det-padding" min="0" max="256"></label>
        </div>
        <textarea class="det-prompt" rows="1" placeholder="Detail prompt (empty = main prompt)"></textarea>
      </div>`).join("")}`;
    const ad = () => this.o.state();
    const on = $(".det-on", el);
    on.onclick = (e) => e.stopPropagation();
    on.onchange = () => { ad().enabled = on.checked; if (on.checked) el.open = true; this.o.persist(); };
    $$(".det-unit", el).forEach((unit) => {
      const u = () => ad().units[Number(unit.dataset.unit)];
      for (const [cls, key, isNum] of this.fields()) {
        const f = $(cls, unit);
        f.onchange = () => { u()[key] = isNum ? Number(f.value) : f.value; this.o.persist(); };
      }
      $(".det-prompt", unit).oninput = (e) => { u().prompt = e.target.value; this.o.persist(); };
    });
    this.render();
  }

  fields() {
    return [[".det-model", "model", false], [".det-denoise", "denoise", true], [".det-confidence", "confidence", true], [".det-blur", "maskBlur", true], [".det-dilate", "dilate", true], [".det-padding", "padding", true]];
  }

  render() {
    const ad = this.o.state();
    $(".det-on", this.el).checked = ad.enabled;
    $$(".det-unit", this.el).forEach((unit) => {
      const u = ad.units[Number(unit.dataset.unit)];
      for (const [cls, key] of this.fields()) $(cls, unit).value = u[key];
      $(".det-prompt", unit).value = u.prompt || "";
    });
  }

  setInfo(info) {
    const models = [{ value: "None", label: "None" }, ...(info.adetailer_models || []).map((x) => ({ value: x, label: x }))];
    const ad = this.o.state();
    $$(".det-unit", this.el).forEach((unit) => fillSelect($(".det-model", unit), models, ad.units[Number(unit.dataset.unit)].model));
  }

  job() { return detailerJob(this.o.state()); }
}
