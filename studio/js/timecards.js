// Cards for Cosmos's time axis (venno/engine/reference.py): consistent sets (several frames generated together)
// and reference pictures. Same shape as cards.js: each renders into a <details class="card"> over a plain state object.

import { $, $$, escapeHtml, toast } from "./common.js";

export const SET_DEFAULT = { enabled: false, frames: 3, prompts: ["", "", "", ""], separation: 0.2 };
export const REFERENCE_DEFAULT = { enabled: false, useInput: false, follow: false, noise: 0.5, strength: 1 };
const MAX_REFERENCES = 3;

export class SetCard {
  /** opts: { state: () => SET_DEFAULT-shaped object, persist } */
  constructor(el, opts) {
    this.el = el;
    this.o = opts;
    el.classList.add("set-card");
    el.innerHTML = `
      <summary><input type="checkbox" class="set-on" title="Generate a consistent set"> <span>Consistent set</span> <span class="set-sum"></span></summary>
      <div class="hint">Generates several pictures <b>together</b>, as frames of one sequence: the same character, outfit and style, with small changes between them. Give a frame its own text to steer it (added in front of the prompt), e.g. an expression or a pose. Each image of the batch is its own set.</div>
      <label class="slider">Frames <span class="seg set-frames">${[2, 3, 4].map((n) => `<button data-n="${n}">${n}</button>`).join("")}</span></label>
      <div class="set-prompts"></div>
      <label class="slider set-sep-row" title="With frame texts: how far the frames are kept from each other. Higher follows each frame's text more; lower keeps the frames more alike.">Keep frames apart <input type="range" class="set-sep" min="0" max="100" step="5"><span class="set-sep-v"></span></label>
      <div class="hint">Sampling takes longer with every frame (about ×2 for 2 frames, ×5 for 4). Not used by the Canvas.</div>`;
    const s = () => this.o.state();
    const on = $(".set-on", el);
    on.onclick = (e) => e.stopPropagation();
    on.onchange = () => { s().enabled = on.checked; if (on.checked) el.open = true; this.changed(); };
    $$(".set-frames button", el).forEach((b) => (b.onclick = () => { s().frames = Number(b.dataset.n); this.changed(); this.render(); }));
    $(".set-sep", el).oninput = (e) => { s().separation = Number(e.target.value) / 100; this.changed(); };
    this.render();
  }

  changed() {
    this.o.persist();
    this.updateSummary();
  }

  render() {
    const s = this.o.state(), el = this.el;
    $(".set-on", el).checked = s.enabled;
    if (s.enabled) el.open = true;
    $$(".set-frames button", el).forEach((b) => b.classList.toggle("on", Number(b.dataset.n) === s.frames));
    const box = $(".set-prompts", el);
    box.innerHTML = Array.from({ length: s.frames }, (_, k) => `<label>Frame ${k + 1} <input type="text" data-k="${k}" placeholder="e.g. ${["smile", "angry", "surprised", "sleeping"][k]} (optional)" value="${escapeHtml(s.prompts[k] || "")}"></label>`).join("");
    $$("input", box).forEach((input) => (input.oninput = () => { s.prompts[Number(input.dataset.k)] = input.value; this.changed(); }));
    this.updateSummary();
  }

  updateSummary() {
    const s = this.o.state();
    $(".set-sum", this.el).textContent = s.enabled ? `(${s.frames} frames)` : "";
    $(".set-sep", this.el).value = Math.round(s.separation * 100);
    $(".set-sep-v", this.el).textContent = `${Math.round(s.separation * 100)}%`;
    $(".set-sep-row", this.el).classList.toggle("muted", !s.prompts.slice(0, s.frames).some((p) => p?.trim()));
  }

  /** Fields for the job (text to image), or {} when off. */
  job() {
    const s = this.o.state();
    if (!s.enabled) return {};
    return { frames: s.frames, frame_prompts: s.prompts.slice(0, s.frames).map((p) => p || ""), frame_separation: s.separation };
  }

  /** Settings read from an image (infotext.settings). */
  apply(x) {
    const s = this.o.state();
    s.enabled = (x.frames || 1) > 1;
    if (s.enabled) {
      s.frames = Math.min(4, Math.max(2, x.frames));
      s.prompts = Array.from({ length: 4 }, (_, k) => x.frame_prompts?.[k] || "");
      if (x.frame_separation !== undefined) s.separation = x.frame_separation;
    }
  }
}

export class ReferenceCard {
  /** opts: { state: () => REFERENCE_DEFAULT-shaped object, persist, input?: () => data URL of the tab's input image or null } */
  constructor(el, opts) {
    this.el = el;
    this.o = opts;
    this.pictures = []; // data URLs; kept in memory only (too big for the browser's settings storage)
    el.classList.add("ref-card");
    el.innerHTML = `
      <summary><input type="checkbox" class="ref-on" title="Use reference pictures"> <span>Reference</span> <span class="ref-sum"></span></summary>
      <div class="hint">The model sees these pictures next to the one it generates and takes over their character, outfit and style, without a LoRA. At full strength the result stays close to the reference (variations, Edit LoRAs); around 80–90% the prompt sets the scene and pose and the reference lends its look. Works best when the prompt describes the same character.</div>
      ${opts.input ? `<label class="check"><input type="checkbox" class="ref-input"> Use the input image as a reference</label>` : ""}
      <div class="ref-list"></div>
      <div class="row"><button class="ref-add">Add picture…</button><span class="muted">or drop one here · up to ${MAX_REFERENCES}</span></div>
      <label class="check" title="The references get the same noise level as the picture at every step. This is how the Anima Edit LoRAs were trained."><input type="checkbox" class="ref-follow"> Follow the steps (Anima Edit LoRAs)</label>
      <label class="slider ref-strength-row" title="How much the new picture looks at the references. Lower lets the prompt decide pose and scene, and keeps the character from the references.">Reference strength <input type="range" class="ref-strength" min="50" max="100" step="5"><span class="ref-strength-v"></span></label>
      <label class="slider ref-noise-row">Reference noise <input type="range" class="ref-noise" min="0" max="100" step="5"><span class="ref-noise-v"></span></label>
      <div class="hint ref-hint"></div>`;
    const s = () => this.o.state();
    const on = $(".ref-on", el);
    on.onclick = (e) => e.stopPropagation();
    on.onchange = () => { s().enabled = on.checked; if (on.checked) el.open = true; this.changed(); };
    $(".ref-follow", el).onchange = (e) => { s().follow = e.target.checked; this.changed(); };
    $(".ref-noise", el).oninput = (e) => { s().noise = Number(e.target.value) / 100; this.changed(); };
    $(".ref-strength", el).oninput = (e) => { s().strength = Number(e.target.value) / 100; this.changed(); };
    const useInput = $(".ref-input", el);
    if (useInput) useInput.onchange = () => { s().useInput = useInput.checked; if (useInput.checked) s().enabled = true; this.changed(); this.render(); };
    const picker = document.createElement("input");
    picker.type = "file";
    picker.accept = "image/png,image/jpeg,image/webp";
    picker.multiple = true;
    picker.onchange = () => { this.addFiles([...picker.files]); picker.value = ""; };
    $(".ref-add", el).onclick = () => picker.click();
    el.addEventListener("dragover", (e) => e.preventDefault());
    el.addEventListener("drop", (e) => {
      const files = [...(e.dataTransfer?.files || [])].filter((f) => f.type.startsWith("image/"));
      if (!files.length) return;
      e.preventDefault();
      e.stopPropagation(); // not "load settings from this image"
      this.addFiles(files);
    });
    this.render();
  }

  changed() {
    this.o.persist();
    this.updateInfo();
  }

  max() { return MAX_REFERENCES - (this.usesInput() ? 1 : 0); }

  usesInput() { return !!(this.o.input && this.o.state().useInput); }

  async addFiles(files) {
    for (const f of files) {
      if (this.pictures.length >= this.max()) { toast(`Up to ${MAX_REFERENCES} reference pictures`, "info"); break; }
      this.pictures.push(await new Promise((ok, fail) => {
        const r = new FileReader();
        r.onload = () => ok(r.result);
        r.onerror = fail;
        r.readAsDataURL(f);
      }));
    }
    this.o.state().enabled = true;
    this.changed();
    this.render();
  }

  /** Add a picture given as a data URL (e.g. a result: "Use as reference"). */
  add(src) {
    if (this.pictures.length >= this.max()) this.pictures.shift();
    this.pictures.push(src);
    this.o.state().enabled = true;
    this.el.open = true;
    this.changed();
    this.render();
  }

  render() {
    const s = this.o.state(), el = this.el;
    $(".ref-on", el).checked = s.enabled;
    if (s.enabled) el.open = true;
    const useInput = $(".ref-input", el);
    if (useInput) useInput.checked = s.useInput;
    $(".ref-follow", el).checked = s.follow;
    $(".ref-noise", el).value = Math.round(s.noise * 100);
    $(".ref-strength", el).value = Math.round(s.strength * 100);
    const list = $(".ref-list", el);
    list.innerHTML = this.pictures.map((src, k) => `<div class="ref-pic"><img src="${src}" alt="Reference ${k + 1}" draggable="false"><button data-k="${k}" title="Remove">×</button></div>`).join("");
    $$("button", list).forEach((b) => (b.onclick = () => { this.pictures.splice(Number(b.dataset.k), 1); this.changed(); this.render(); }));
    this.updateInfo();
  }

  updateInfo() {
    const s = this.o.state(), el = this.el;
    const n = this.pictures.length + (this.usesInput() ? 1 : 0);
    $(".ref-sum", el).textContent = s.enabled ? (n ? `(${n} picture${n > 1 ? "s" : ""}${s.follow ? ", follow" : ""})` : "(add a picture)") : "";
    $(".ref-sum", el).classList.toggle("warn", s.enabled && !n);
    $(".ref-noise", el).disabled = s.follow;
    $(".ref-noise-row", el).classList.toggle("muted", s.follow);
    $(".ref-noise-v", el).textContent = s.follow ? "follows the steps" : s.noise.toFixed(2);
    $(".ref-strength-v", el).textContent = `${Math.round(s.strength * 100)}%`;
    $(".ref-hint", el).innerHTML = s.follow
      ? "For an Edit LoRA: add it under LoRAs, use the input image as the reference and Denoising 1, and write the change you want in the prompt."
      : "How noisy the model is told the references are. Lower copies them more closely (below about 0.3 a dotted pattern can appear); higher follows the prompt more. 0.5 is a good start.";
  }

  /** Fields for the job, or {} when off / empty. */
  job() {
    const s = this.o.state();
    if (!s.enabled) return {};
    const refs = [...(this.usesInput() && this.o.input() ? [this.o.input()] : []), ...this.pictures];
    if (!refs.length) return {};
    return { references: refs, reference_noise: s.noise, reference_follow: s.follow, reference_strength: s.strength };
  }

  apply(x) {
    const s = this.o.state();
    if (x.reference_follow !== undefined) s.follow = x.reference_follow;
    if (x.reference_noise !== undefined) s.noise = x.reference_noise;
    if (x.reference_strength !== undefined) s.strength = x.reference_strength;
  }
}
