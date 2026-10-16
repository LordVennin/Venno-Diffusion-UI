// "Text to Image" and "Image to Image" tabs: classic form-style generation pages.

import {
  $, $$, ADETAILER_DEFAULT, downloadDataURL, escapeHtml, fillSelect, loadStore, presetForCheckpoint, saveStore, toast, withQualityPrefix,
} from "./common.js";
import { cancelJob, jobModel, jobProgress, runJob } from "./jobs.js";
import { jobLoras, LoraChips } from "./loras.js";
import { runSteps, SchedulePreview } from "./schedule.js";
import { describeSettings, hasQualityTags, offerSettings, readImageSettings } from "./params.js";
import { GRID_DEFAULT, GridCard } from "./gridui.js";

const SIZES = [[1024, 1024], [832, 1216], [1216, 832], [896, 1152], [1152, 896], [768, 1344], [1344, 768]];
const RESIZE_MODES = ["Just resize", "Crop and resize", "Resize and fill"];
const FILL_MODES = ["Fill", "Original", "Latent noise", "Latent nothing"];
const FILL_KEYS = ["fill", "original", "latent_noise", "latent_nothing"];
const MASK_COLOR = "#ff8a3d";

const DEFAULTS = {
  prompt: "", negative: "", loras: [], switchPrompt: "", switchAt: 0.4, qualityPrefix: true, width: 1024, height: 1024, batchSize: 1, batchCount: 1,
  steps: 10, cfg: 1, shift: 3, sampler: "Euler", scheduler: "Simple", seed: -1, preset: "",
  // hires fix (txt2img)
  hr: false, hrUpscaler: "Latent", hrScale: 1.5, hrResizeX: 0, hrResizeY: 0, hrSteps: 0, hrDenoise: 0.5, hrCfg: 0, hrShift: 0,
  // img2img
  denoise: 0.6, resizeMode: 0, resizeBy: "by", scaleBy: 1, maskBlur: 8, fill: 1, onlyMasked: false, onlyMaskedPadding: 64, softInpaint: true,
  saveImages: true,
  adetailer: ADETAILER_DEFAULT,
  grid: GRID_DEFAULT,
};

export class GenForm {
  constructor(root, mode, ctx) {
    this.root = root;
    this.mode = mode; // "txt2img" | "img2img" | "inpaint" (the last two send img2img jobs)
    this.ctx = ctx; // { state, onSendToCanvas, onSendToImg2Img }
    this.key = `venno-form-${mode}`;
    this.s = loadStore(this.key, structuredClone(DEFAULTS));
    this.s.adetailer = { ...structuredClone(ADETAILER_DEFAULT), ...(this.s.adetailer || {}) };
    this.s.grid = { ...structuredClone(GRID_DEFAULT), ...(this.s.grid || {}) };
    this.results = []; // [{src, seed, info}]
    this.current = -1;
    this.busy = false;
    this.inputImage = null; // img2img source (HTMLImageElement)
    this.mask = null; // img2img mask canvas (image resolution, white = inpaint)
    this.build();
  }

  persist() { saveStore(this.key, this.s); }

  // ---------------- DOM ----------------
  build() {
    const i2i = this.mode !== "txt2img";
    const inpaint = this.mode === "inpaint";
    this.root.innerHTML = `
      <div class="form-col">
        <section class="card">
          <textarea class="f-prompt" rows="4" placeholder="Prompt (Ctrl+Enter to generate)"></textarea>
          <textarea class="f-negative" rows="2" placeholder="Negative prompt"></textarea>
          <div class="f-lora-chips"></div>
          <details class="switch-box f-switch-box">
            <summary>Then switch to… <span class="f-switch-state"></span></summary>
            <textarea class="f-switch" rows="2" placeholder="Prompt for the later steps, e.g. the same scene with character or style details. Empty = off"></textarea>
            <label class="slider" title="Composition is mostly decided early; details late.">Switch after <input type="range" class="f-switch-at" min="5" max="95" step="5"><output></output></label>
          </details>
          <div class="f-schedule"></div>
          <div class="row">
            <select class="f-lora"></select>
            <label class="check"><input type="checkbox" class="f-quality"> Anima quality tags</label>
            <button class="f-load-settings" title="Load prompt and settings from an image made with Venno, Forge or A1111 (you can also drop the image onto this panel)">From image…</button>
          </div>
          <div class="row buttons">
            <button class="primary big f-generate">Generate</button>
            <button class="big f-cancel" hidden>Cancel</button>
          </div>
          <div class="f-progress progress-inline" hidden><div class="bar"></div><span></span></div>
        </section>

        <section class="card">
          <div class="grid2">
            <label>Preset <select class="f-preset"></select></label>
            <label>Sampler <select class="f-sampler"></select></label>
            <label>Scheduler <select class="f-scheduler"></select></label>
            <label>Steps <input type="number" class="f-steps" min="1" max="150"></label>
            <label>CFG <input type="number" class="f-cfg" min="1" max="30" step="0.1"></label>
            <label>Shift <input type="number" class="f-shift" min="0" max="20" step="0.1"></label>
          </div>
          ${i2i ? `
          <div class="seg resize-tabs"><button class="f-rt-by">Resize by</button><button class="f-rt-to">Resize to</button></div>
          <div class="f-resize-by">
            <label class="slider">Scale <input type="range" class="f-scale" min="25" max="400" step="5"><output></output></label>
            <div class="hint f-scale-info"></div>
          </div>
          <div class="f-resize-to">` : ""}
          <div class="grid2">
            <label>Width <input type="number" class="f-width" min="256" max="2048" step="64"></label>
            <label>Height <input type="number" class="f-height" min="256" max="2048" step="64"></label>
          </div>
          <div class="sizes">${SIZES.map(([w, h]) => `<button data-w="${w}" data-h="${h}">${w}×${h}</button>`).join("")}<button class="f-swap" title="Swap width/height">⇅</button></div>
          ${i2i ? `
          <div class="grid2">
            <label>Resize mode <select class="f-resize-mode">${RESIZE_MODES.map((m, i) => `<option value="${i}">${m}</option>`).join("")}</select></label>
            <label>Size from image <span class="row"><button class="f-size-from-image">Match input</button></span></label>
          </div>
          </div>
          <label class="slider">Denoising strength <input type="range" class="f-denoise" min="0" max="100"><output></output></label>` : ""}
          <div class="grid2">
            <label>Batch size <input type="number" class="f-batch-size" min="1" max="8"></label>
            <label>Batch count <input type="number" class="f-batch-count" min="1" max="100"></label>
          </div>
          <label>Seed <span class="row seed"><input type="number" class="f-seed"><button class="f-seed-random" title="Random (-1)">🎲</button><button class="f-seed-reuse" title="Reuse seed of the shown image">♻️</button></span></label>
        </section>

        ${inpaint ? `
        <details class="card" open>
          <summary>Inpainting</summary>
          <div class="grid2">
            <label title="What the masked area starts from. Original: the input (use any strength). Fill: colours from the surroundings (needs high strength). Latent noise: regenerate from scratch (strength is ignored). Latent nothing: blank.">Masked content <select class="f-fill">${FILL_MODES.map((m, i) => `<option value="${i}">${m}</option>`).join("")}</select></label>
            <label>Inpaint area <select class="f-only-masked"><option value="0">Whole picture</option><option value="1">Only masked</option></select></label>
            <label>Mask blur <input type="number" class="f-mask-blur" min="0" max="64"></label>
            <label>Only-masked padding <input type="number" class="f-padding" min="0" max="256"></label>
          </div>
          <label class="check"><input type="checkbox" class="f-soft"> Seamless (soft) inpainting</label>
        </details>` : i2i ? "" : `
        <details class="card f-hr-card">
          <summary><input type="checkbox" class="f-hr" title="Use hires fix"> <span>Hires. fix</span></summary>
          <div class="grid2">
            <label>Upscaler <select class="f-hr-upscaler"></select></label>
            <label>Hires steps <input type="number" class="f-hr-steps" min="0" max="150" title="0 = same as first pass"></label>
            <label>Upscale by <input type="number" class="f-hr-scale" min="1" max="4" step="0.05"></label>
            <label>Denoising <input type="number" class="f-hr-denoise" min="0" max="1" step="0.05"></label>
            <label>Resize width to <input type="number" class="f-hr-x" min="0" max="4096" step="64" title="0 = use Upscale by"></label>
            <label>Resize height to <input type="number" class="f-hr-y" min="0" max="4096" step="64" title="0 = use Upscale by"></label>
            <label>Hires CFG <input type="number" class="f-hr-cfg" min="0" max="30" step="0.1" title="0 = same as first pass"></label>
            <label>Hires shift <input type="number" class="f-hr-shift" min="0" max="20" step="0.1" title="0 = same as first pass"></label>
          </div>
          <div class="hint f-hr-info"></div>
        </details>`}

        <details class="card f-ad-card">
          <summary><input type="checkbox" class="f-ad" title="Use ADetailer"> <span>ADetailer</span></summary>
          ${[0, 1].map((n) => `
          <div class="ad-unit" data-unit="${n}">
            <h4>${n === 0 ? "1st" : "2nd"} detector</h4>
            <div class="grid2">
              <label>Model <select class="ad-model"></select></label>
              <label>Denoising <input type="number" class="ad-denoise" min="0" max="1" step="0.05"></label>
              <label>Confidence <input type="number" class="ad-confidence" min="0" max="1" step="0.05"></label>
              <label>Mask blur <input type="number" class="ad-blur" min="0" max="64"></label>
              <label>Dilate / erode <input type="number" class="ad-dilate" min="-64" max="64"></label>
              <label>Padding <input type="number" class="ad-padding" min="0" max="256"></label>
            </div>
            <textarea class="ad-prompt" rows="1" placeholder="Detail prompt (empty = main prompt)"></textarea>
          </div>`).join("")}
        </details>

        <details class="card f-grid-card grid-card"></details>

        <label class="check save"><input type="checkbox" class="f-save"> Save results to outputs folder</label>
      </div>

      <div class="result-col${i2i ? " work" : ""}">
        ${i2i ? `
        <div class="panes">
          <div class="pane in-pane">
            <div class="pane-head">
              <span class="pane-title">${inpaint ? "Image & mask" : "Input image"}</span>
              <div class="row mask-tools" hidden>
                ${inpaint ? `
                <div class="seg"><button class="f-mask-draw on">Paint</button><button class="f-mask-erase">Erase</button></div>
                <label>Brush <input type="range" class="f-mask-size" min="4" max="300" value="48"></label>
                <button class="f-mask-invert">Invert</button>
                <button class="f-mask-clear">Clear mask</button>` : ""}
                <button class="f-open-again" title="Open another image">Open…</button>
                <button class="f-image-clear" title="Remove image">✕</button>
              </div>
            </div>
            <div class="i2i-drop">
              <div class="i2i-empty">Drop, paste or <button class="f-open">open</button> an image<br><small>or use “Send to ${inpaint ? "Inpaint" : "Image to Image"}” from another tab</small></div>
              <div class="i2i-stage" hidden><img class="i2i-img" alt="">${inpaint ? `<canvas class="i2i-mask"></canvas>` : ""}</div>
            </div>
            ${inpaint ? `<div class="hint">Paint over the area to regenerate. No mask = the whole image.</div>` : ""}
          </div>
          <div class="pane out-pane">
            <div class="pane-head"><span class="pane-title">Result</span></div>
            <div class="result-main"><img class="r-img" alt=""><div class="r-empty">Results appear here</div></div>
          </div>
        </div>` : `
        <div class="result-main"><img class="r-img" alt=""><div class="r-empty">Results appear here</div></div>`}
        <div class="r-thumbs"></div>
        <div class="row r-actions">
          <button class="r-to-canvas" disabled>Open in Canvas</button>
          ${i2i ? `<button class="r-to-input" disabled>Use as input</button>` : ""}
          ${this.mode !== "img2img" ? `<button class="r-to-i2i" disabled>Send to Image to Image</button>` : ""}
          ${this.mode !== "inpaint" ? `<button class="r-to-inpaint" disabled>Send to Inpaint</button>` : ""}
          <button class="r-use-settings" disabled title="Load this image's prompt and settings into the form">Use settings</button>
          <button class="r-download" disabled>Download</button>
        </div>
        <pre class="r-info"></pre>
      </div>`;
    this.bind();
  }

  q(sel) { return $(sel, this.root); }

  bind() {
    const s = this.s;
    const i2i = this.mode !== "txt2img";
    const inpaint = this.mode === "inpaint";
    this.syncs = []; // re-apply state to the inputs (after loading settings from an image)
    const sync = (fn) => { fn(); this.syncs.push(fn); };
    const num = (sel, key, onChange) => {
      const el = this.q(sel);
      sync(() => (el.value = s[key]));
      el.onchange = () => { s[key] = Number(el.value); this.persist(); onChange && onChange(); };
    };
    const txt = (sel, key) => {
      const el = this.q(sel);
      sync(() => (el.value = s[key]));
      el.oninput = () => { s[key] = el.value; this.persist(); };
    };
    const chk = (sel, key, onChange) => {
      const el = this.q(sel);
      sync(() => (el.checked = s[key]));
      el.onchange = () => { s[key] = el.checked; this.persist(); onChange && onChange(); };
    };

    txt(".f-prompt", "prompt");
    txt(".f-negative", "negative");
    chk(".f-quality", "qualityPrefix");
    chk(".f-save", "saveImages");
    num(".f-steps", "steps");
    num(".f-cfg", "cfg");
    num(".f-shift", "shift");
    num(".f-width", "width", () => this.updateHrInfo());
    num(".f-height", "height", () => this.updateHrInfo());
    num(".f-batch-size", "batchSize");
    num(".f-batch-count", "batchCount");
    num(".f-seed", "seed");
    this.q(".f-sampler").onchange = (e) => { s.sampler = e.target.value; this.persist(); };
    this.q(".f-scheduler").onchange = (e) => { s.scheduler = e.target.value; this.persist(); };
    this.syncs.push(() => {
      this.q(".f-sampler").value = s.sampler;
      this.q(".f-scheduler").value = s.scheduler;
      this.q(".f-preset").value = s.preset || "";
    });
    this.q(".f-preset").onchange = (e) => this.applyPreset(e.target.value);
    this.q(".f-seed-random").onclick = () => { s.seed = -1; this.q(".f-seed").value = -1; this.persist(); };
    this.q(".f-seed-reuse").onclick = () => {
      const r = this.results[this.current];
      if (r) { s.seed = r.seed; this.q(".f-seed").value = r.seed; this.persist(); }
    };
    $$(".sizes button[data-w]", this.root).forEach((b) => (b.onclick = () => this.setSize(Number(b.dataset.w), Number(b.dataset.h))));
    this.q(".f-swap").onclick = () => this.setSize(s.height, s.width);
    this.q(".f-generate").onclick = () => this.generate();
    this.q(".f-cancel").onclick = () => this.jobId && cancelJob(this.jobId);
    this.loras = new LoraChips(this.q(".f-lora-chips"), { get: () => s.loras, set: (list) => { s.loras = list; this.persist(); this.gridCard?.render(); } });
    this.q(".f-lora").onchange = (e) => {
      if (e.target.value) this.loras.add(e.target.value);
      e.target.value = "";
    };
    this.loras.watch(this.q(".f-prompt"), (text) => { s.prompt = text; this.persist(); });
    this.loras.watch(this.q(".f-negative"), (text) => { s.negative = text; this.persist(); });
    this.bindSwitch();
    this.root.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); e.stopPropagation(); this.generate(); }
    });

    if (inpaint) {
      num(".f-mask-blur", "maskBlur");
      num(".f-padding", "onlyMaskedPadding");
      chk(".f-soft", "softInpaint");
      sync(() => {
        this.q(".f-fill").value = s.fill;
        this.q(".f-only-masked").value = s.onlyMasked ? "1" : "0";
      });
      this.q(".f-fill").onchange = (e) => { s.fill = Number(e.target.value); this.persist(); };
      this.q(".f-only-masked").onchange = (e) => { s.onlyMasked = e.target.value === "1"; this.persist(); };
    }
    if (i2i) {
      sync(() => {
        this.q(".f-resize-mode").value = s.resizeMode;
        this.q(".f-denoise").value = Math.round(s.denoise * 100);
        this.q(".f-denoise").nextElementSibling.textContent = s.denoise;
        this.q(".f-scale").value = Math.round(s.scaleBy * 100);
        this.showResize();
      });
      this.q(".f-rt-by").onclick = () => { s.resizeBy = "by"; this.persist(); this.showResize(); };
      this.q(".f-rt-to").onclick = () => { s.resizeBy = "to"; this.persist(); this.showResize(); };
      this.q(".f-scale").oninput = (e) => { s.scaleBy = Number(e.target.value) / 100; this.persist(); this.showResize(); };
      this.q(".f-resize-mode").onchange = (e) => { s.resizeMode = Number(e.target.value); this.persist(); };
      const den = this.q(".f-denoise");
      den.value = Math.round(s.denoise * 100);
      den.nextElementSibling.textContent = den.value / 100;
      den.oninput = () => { s.denoise = den.value / 100; den.nextElementSibling.textContent = s.denoise; this.persist(); };
      this.q(".f-size-from-image").onclick = () => this.sizeFromImage();
      this.bindImageInput();
    } else {
      chk(".f-hr", "hr", () => this.updateHrInfo());
      this.q(".f-hr").onclick = (e) => e.stopPropagation(); // don't toggle the <details>
      num(".f-hr-steps", "hrSteps");
      num(".f-hr-scale", "hrScale", () => this.updateHrInfo());
      num(".f-hr-denoise", "hrDenoise");
      num(".f-hr-x", "hrResizeX", () => this.updateHrInfo());
      num(".f-hr-y", "hrResizeY", () => this.updateHrInfo());
      num(".f-hr-cfg", "hrCfg");
      num(".f-hr-shift", "hrShift");
      this.q(".f-hr-upscaler").onchange = (e) => { s.hrUpscaler = e.target.value; this.persist(); };
      this.syncs.push(() => { this.q(".f-hr-upscaler").value = s.hrUpscaler; if (s.hr) this.q(".f-hr-card").open = true; });
      this.updateHrInfo();
    }

    // ADetailer

    const adChk = this.q(".f-ad");
    sync(() => (adChk.checked = s.adetailer.enabled));
    adChk.onclick = (e) => e.stopPropagation();
    adChk.onchange = () => { s.adetailer.enabled = adChk.checked; this.persist(); };
    $$(".ad-unit", this.root).forEach((el) => {
      const u = () => s.adetailer.units[Number(el.dataset.unit)];
      const bindU = (sel, key, isNum = true) => {
        const f = $(sel, el);
        sync(() => (f.value = u()[key]));
        f.onchange = () => { u()[key] = isNum ? Number(f.value) : f.value; this.persist(); };
      };
      bindU(".ad-model", "model", false);
      bindU(".ad-denoise", "denoise");
      bindU(".ad-confidence", "confidence");
      bindU(".ad-blur", "maskBlur");
      bindU(".ad-dilate", "dilate");
      bindU(".ad-padding", "padding");
      const p = $(".ad-prompt", el);
      sync(() => (p.value = u().prompt));
      p.oninput = () => { u().prompt = p.value; this.persist(); };
    });

    // comparison grid
    this.gridCard = new GridCard(this.q(".f-grid-card"), { state: () => s.grid, persist: () => this.persist(), info: () => this.ctx.state.info, loras: () => s.loras });
    this.syncs.push(() => this.gridCard.render());

    // results
    this.q(".r-to-canvas").onclick = () => this.withCurrent((r) => this.ctx.onSendToCanvas(r.src));
    this.q(".r-to-input")?.addEventListener("click", () => this.withCurrent((r) => this.setInputImage(r.src)));
    this.q(".r-to-i2i")?.addEventListener("click", () => this.withCurrent((r) => this.ctx.onSend("img2img", r.src, this.exportState(r))));
    this.q(".r-to-inpaint")?.addEventListener("click", () => this.withCurrent((r) => this.ctx.onSend("inpaint", r.src, this.exportState(r))));
    this.q(".r-download").onclick = () => this.withCurrent((r) => downloadDataURL(r.src, `venno-${r.seed}.png`));
    this.q(".r-use-settings").onclick = () => this.withCurrent((r) => this.loadSettingsFrom(r.src));

    // settings from an image: button, or drop an image anywhere on the form column (not the input-image box)
    const picker = document.createElement("input");
    picker.type = "file";
    picker.accept = "image/png,image/jpeg,image/webp";
    picker.onchange = () => { if (picker.files[0]) this.loadSettingsFrom(picker.files[0]); picker.value = ""; };
    this.q(".f-load-settings").onclick = () => picker.click();
    const col = this.q(".form-col");
    col.addEventListener("dragover", (e) => { if (!e.target.closest(".i2i-drop")) e.preventDefault(); });
    col.addEventListener("drop", (e) => {
      if (e.target.closest(".i2i-drop")) return;
      const f = [...(e.dataTransfer?.files || [])].find((x) => x.type.startsWith("image/"));
      if (!f) return;
      e.preventDefault();
      this.loadSettingsFrom(f);
    });
  }

  /** Everything another tab needs to continue from a result of this one. */
  exportState(result = null) {
    const s = this.s;
    return structuredClone({
      prompt: s.prompt, negative: s.negative, loras: s.loras, qualityPrefix: s.qualityPrefix,
      steps: s.steps, cfg: s.cfg, shift: s.shift, sampler: s.sampler, scheduler: s.scheduler, preset: s.preset,
      switchPrompt: s.switchPrompt, switchAt: s.switchAt, adetailer: s.adetailer, saveImages: s.saveImages,
      seed: result?.seed ?? s.seed,
    });
  }

  /** Take an image (and optionally the sending tab's settings) as this tab's input. */
  async receive(src, state = null) {
    if (state) {
      Object.assign(this.s, structuredClone(state));
      this.persist();
      this.syncs.forEach((fn) => fn());
      this.loras.render();
      this.schedule.refresh();
      this.gridCard.render();
    }
    if (src && this.mode !== "txt2img") await this.setInputImage(src);
  }

  async loadSettingsFrom(source) {
    const res = await readImageSettings(source);
    if (!res.found) return toast(res.note || "No generation settings found in this image.", "info");
    this.applySettings(res.settings);
  }

  /** Apply settings read from an image (see params.js). Missing fields keep their current values. */
  applySettings(x) {
    const s = this.s;
    const set = (key, v) => { if (v !== undefined && v !== null && v !== "") s[key] = v; };
    s.prompt = x.prompt ?? s.prompt;
    s.negative = x.negative ?? s.negative;
    s.loras = x.loras || [];
    if (hasQualityTags(s.prompt)) s.qualityPrefix = false; // the prompt brings its own
    set("steps", x.steps); set("cfg", x.cfg); set("shift", x.shift); set("seed", x.seed);
    const info = this.ctx.state.info || {};
    if (x.sampler && info.samplers?.includes(x.sampler)) s.sampler = x.sampler;
    if (x.scheduler && info.schedulers?.includes(x.scheduler)) s.scheduler = x.scheduler;
    s.preset = "";
    if (this.mode === "txt2img" || !this.inputImage) { set("width", x.width); set("height", x.height); }
    if (this.mode !== "txt2img") set("denoise", x.denoise);
    s.switchPrompt = x.switch_prompt || "";
    set("switchAt", x.switch_at);
    if (this.mode === "txt2img") {
      const h = x.hires;
      s.hr = !!h;
      if (h) {
        s.hrUpscaler = info.upscalers?.includes(h.upscaler) ? h.upscaler : s.hrUpscaler;
        Object.assign(s, { hrScale: h.scale || s.hrScale, hrResizeX: h.resize_x || 0, hrResizeY: h.resize_y || 0, hrSteps: h.steps || 0, hrDenoise: h.denoise, hrCfg: h.cfg || 0, hrShift: h.shift || 0 });
      }
    }
    if (x.detailer) {
      s.adetailer.enabled = x.detailer.length > 0;
      x.detailer.slice(0, s.adetailer.units.length).forEach((d, i) => Object.assign(s.adetailer.units[i], { model: d.model, confidence: d.confidence, denoise: d.denoise, prompt: d.prompt || "" }));
    }
    this.persist();
    this.syncs.forEach((fn) => fn());
    this.loras.render();
    this.schedule.refresh();
    this.updateHrInfo();
    if (x.checkpoint) this.ctx.selectCheckpoint?.(x.checkpoint);
    const notes = [];
    if (x.regions?.length) notes.push(`${x.regions.length} region prompt(s) are only usable on the Canvas`);
    if (x.template) notes.push("it used wildcards: its resolved prompt was loaded");
    toast(`Loaded: ${describeSettings(x)}${notes.length ? ` (${notes.join("; ")})` : ""}`, "info");
  }

  withCurrent(fn) {
    const r = this.results[this.current];
    if (r) fn(r);
  }

  // called whenever /venno/v1/info is (re)loaded
  setInfo(info) {
    const s = this.s;
    fillSelect(this.q(".f-sampler"), info.samplers.map((x) => ({ value: x, label: x })), s.sampler);
    fillSelect(this.q(".f-scheduler"), info.schedulers.map((x) => ({ value: x, label: x })), s.scheduler);
    fillSelect(this.q(".f-preset"), Object.entries(info.presets).map(([k, p]) => ({ value: k, label: p.label })), s.preset, { empty: "Custom" });
    this.q(".f-lora").innerHTML = `<option value="">+ LoRA… (${info.loras.length})</option>` + info.loras.map((l) => `<option value="${escapeHtml(l.name)}">${escapeHtml(l.name)}</option>`).join("");
    this.loras.setAvailable(info.loras.map((l) => l.name));
    if (this.q(".f-hr-upscaler")) fillSelect(this.q(".f-hr-upscaler"), info.upscalers.map((x) => ({ value: x, label: x })), s.hrUpscaler);
    const adModels = [{ value: "None", label: "None" }, ...info.adetailer_models.map((x) => ({ value: x, label: x }))];
    $$(".ad-unit", this.root).forEach((el) => fillSelect($(".ad-model", el), adModels, s.adetailer.units[Number(el.dataset.unit)].model));
    if (!s.negative && info.prompt_defaults?.negative) { s.negative = info.prompt_defaults.negative; this.q(".f-negative").value = s.negative; }
    if (!s.preset) this.applyPreset(presetForCheckpoint(this.ctx.state.model.checkpoint));
  }

  applyPreset(name) {
    const p = this.ctx.state.info?.presets?.[name];
    if (!p) return;
    Object.assign(this.s, { preset: name, sampler: p.sampler, scheduler: p.scheduler, steps: p.steps, cfg: p.cfg, shift: p.shift });
    this.q(".f-sampler").value = p.sampler;
    this.q(".f-scheduler").value = p.scheduler;
    this.q(".f-steps").value = p.steps;
    this.q(".f-cfg").value = p.cfg;
    this.q(".f-shift").value = p.shift;
    this.q(".f-preset").value = name;
    this.persist();
  }

  setSize(w, h) {
    this.s.width = w;
    this.s.height = h;
    this.q(".f-width").value = w;
    this.q(".f-height").value = h;
    this.persist();
    this.updateHrInfo();
  }

  updateHrInfo() {
    const el = this.q(".f-hr-info");
    if (!el) return;
    const s = this.s;
    let w = s.hrResizeX || Math.round((s.width * s.hrScale) / 64) * 64;
    let h = s.hrResizeY || Math.round((s.height * s.hrScale) / 64) * 64;
    if (s.hrResizeX && !s.hrResizeY) h = Math.round((s.height * s.hrResizeX) / s.width / 64) * 64;
    if (s.hrResizeY && !s.hrResizeX) w = Math.round((s.width * s.hrResizeY) / s.height / 64) * 64;
    el.textContent = s.hr ? `${s.width}×${s.height} → ${w}×${h}` : "Generate at low resolution, then upscale and refine (like Forge's Hires. fix).";
  }

  // ---------------- img2img input & mask ----------------
  /** Fit the input image (and its mask canvas) to the pane, scaling small images up too. */
  fitStage() {
    const img = this.inputImage, drop = this.q(".i2i-drop"), stage = this.q(".i2i-stage");
    if (!img || !drop || stage.hidden) return;
    const r = drop.getBoundingClientRect();
    const k = Math.min((r.width - 16) / img.naturalWidth, (r.height - 16) / img.naturalHeight);
    if (!(k > 0)) return;
    stage.style.width = `${Math.floor(img.naturalWidth * k)}px`;
    stage.style.height = `${Math.floor(img.naturalHeight * k)}px`;
  }

  bindImageInput() {
    const drop = this.q(".i2i-drop");
    new ResizeObserver(() => this.fitStage()).observe(drop);
    const file = document.createElement("input");
    file.type = "file";
    file.accept = "image/*";
    file.onchange = () => { if (file.files[0]) this.setInputImage(URL.createObjectURL(file.files[0]), { offer: true }); file.value = ""; };
    this.q(".f-open").onclick = () => file.click();
    drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("dragging"); });
    drop.addEventListener("dragleave", () => drop.classList.remove("dragging"));
    drop.addEventListener("drop", (e) => {
      e.preventDefault();
      drop.classList.remove("dragging");
      const f = [...e.dataTransfer.files].find((x) => x.type.startsWith("image/"));
      if (f) this.setInputImage(URL.createObjectURL(f), { offer: true });
    });

    this.q(".f-open-again").onclick = () => file.click();
    this.q(".f-image-clear").onclick = () => this.setInputImage(null);
    const mc = this.q(".i2i-mask");
    if (!mc) return; // Image to Image: no mask
    let erase = false, size = 48, drawing = false, last = null;
    const cursor = document.createElement("div");
    cursor.className = "mask-cursor";
    this.q(".i2i-stage").appendChild(cursor);
    this.q(".f-mask-draw").onclick = () => { erase = false; this.q(".f-mask-draw").classList.add("on"); this.q(".f-mask-erase").classList.remove("on"); };
    this.q(".f-mask-erase").onclick = () => { erase = true; this.q(".f-mask-erase").classList.add("on"); this.q(".f-mask-draw").classList.remove("on"); };
    this.q(".f-mask-size").oninput = (e) => (size = Number(e.target.value));
    this.q(".f-mask-clear").onclick = () => this.clearMask();
    this.q(".f-mask-invert").onclick = () => this.invertMask();
    const pos = (e) => {
      const r = mc.getBoundingClientRect();
      return { x: ((e.clientX - r.left) / r.width) * mc.width, y: ((e.clientY - r.top) / r.height) * mc.height, k: mc.width / r.width };
    };
    const paint = (p) => {
      const c = mc.getContext("2d");
      c.globalCompositeOperation = erase ? "destination-out" : "source-over";
      c.strokeStyle = c.fillStyle = MASK_COLOR;
      c.lineWidth = size * p.k;
      c.lineCap = c.lineJoin = "round";
      c.beginPath();
      c.moveTo((last || p).x, (last || p).y);
      c.lineTo(p.x, p.y);
      c.stroke();
      last = p;
    };
    mc.addEventListener("pointerdown", (e) => { mc.setPointerCapture(e.pointerId); drawing = true; last = null; paint(pos(e)); });
    mc.addEventListener("pointermove", (e) => {
      const r = mc.getBoundingClientRect();
      Object.assign(cursor.style, { display: "block", width: `${size}px`, height: `${size}px`, left: `${e.clientX - r.left}px`, top: `${e.clientY - r.top}px` });
      if (drawing) paint(pos(e));
    });
    mc.addEventListener("pointerleave", () => (cursor.style.display = "none"));
    mc.addEventListener("pointerup", () => { drawing = false; last = null; });
    mc.addEventListener("wheel", (e) => {  // brush size with the wheel over the image
      e.preventDefault();
      size = Math.max(4, Math.min(300, size * (e.deltaY < 0 ? 1.12 : 1 / 1.12)));
      this.q(".f-mask-size").value = size;
      cursor.style.width = cursor.style.height = `${size}px`;
    }, { passive: false });
  }

  async setInputImage(src, { offer = false } = {}) {
    if (!src) {
      this.inputImage = null;
      this.q(".i2i-stage").hidden = true;
      this.q(".i2i-empty").hidden = false;
      this.q(".mask-tools").hidden = true;
      return;
    }
    const img = new Image();
    img.src = src;
    await img.decode();
    this.inputImage = img;
    this.q(".i2i-img").src = src;
    if (offer) offerSettings(src, (st) => this.applySettings(st));
    const mc = this.q(".i2i-mask");
    if (mc) {
      mc.width = img.naturalWidth;
      mc.height = img.naturalHeight;
    }
    this.q(".i2i-stage").hidden = false;
    this.q(".i2i-empty").hidden = true;
    this.q(".mask-tools").hidden = false;
    this.fitStage();
    if (this.s.resizeBy !== "by") this.sizeFromImage();
    this.showResize();
  }

  /** Output size for image modes: input × scale ("Resize by"), or width / height ("Resize to"). */
  outputSize() {
    const s = this.s, img = this.inputImage;
    if (s.resizeBy === "by" && img) {
      const snap = (v) => Math.max(64, Math.round((v * s.scaleBy) / 8) * 8);
      return [snap(img.naturalWidth), snap(img.naturalHeight)];
    }
    return [s.width, s.height];
  }

  showResize() {
    if (this.mode === "txt2img") return;
    const by = this.s.resizeBy === "by";
    this.q(".f-rt-by").classList.toggle("on", by);
    this.q(".f-rt-to").classList.toggle("on", !by);
    this.q(".f-resize-by").hidden = !by;
    this.q(".f-resize-to").hidden = by;
    this.q(".f-scale").nextElementSibling.textContent = `${Math.round(this.s.scaleBy * 100)}%`;
    const img = this.inputImage;
    const [w, h] = this.outputSize();
    this.q(".f-scale-info").textContent = img ? `${img.naturalWidth}×${img.naturalHeight} → ${w}×${h}` : "Scales the input image (output = input size × scale).";
  }

  sizeFromImage() {
    if (!this.inputImage) return;
    const { naturalWidth: w, naturalHeight: h } = this.inputImage;
    // keep the aspect ratio, ~1 megapixel for Anima, multiples of 64
    const scale = Math.sqrt((1024 * 1024) / (w * h));
    const snap = (v) => Math.max(256, Math.round((v * scale) / 64) * 64);
    this.setSize(snap(w), snap(h));
  }

  maskHasContent() {
    const mc = this.q(".i2i-mask");
    if (!mc || !this.inputImage) return false;
    const d = mc.getContext("2d").getImageData(0, 0, mc.width, mc.height).data;
    for (let i = 3; i < d.length; i += 16) if (d[i] > 0) return true;
    return false;
  }

  clearMask() {
    const mc = this.q(".i2i-mask");
    mc.getContext("2d").clearRect(0, 0, mc.width, mc.height);
  }

  invertMask() {
    const mc = this.q(".i2i-mask");
    const c = mc.getContext("2d");
    const t = document.createElement("canvas");
    t.width = mc.width;
    t.height = mc.height;
    const tc = t.getContext("2d");
    tc.fillStyle = MASK_COLOR;
    tc.fillRect(0, 0, t.width, t.height);
    tc.globalCompositeOperation = "destination-out";
    tc.drawImage(mc, 0, 0);
    c.clearRect(0, 0, mc.width, mc.height);
    c.drawImage(t, 0, 0);
  }

  maskDataURL() {
    const mc = this.q(".i2i-mask");
    const out = document.createElement("canvas");
    out.width = mc.width;
    out.height = mc.height;
    const c = out.getContext("2d");
    // painted alpha -> white-on-black mask
    const white = document.createElement("canvas");
    white.width = mc.width;
    white.height = mc.height;
    const wc = white.getContext("2d");
    wc.drawImage(mc, 0, 0);
    wc.globalCompositeOperation = "source-in";
    wc.fillStyle = "#fff";
    wc.fillRect(0, 0, white.width, white.height);
    c.fillStyle = "#000";
    c.fillRect(0, 0, out.width, out.height);
    c.drawImage(white, 0, 0);
    return out.toDataURL("image/png");
  }

  inputDataURL() {
    const img = this.inputImage;
    const c = document.createElement("canvas");
    c.width = img.naturalWidth;
    c.height = img.naturalHeight;
    c.getContext("2d").drawImage(img, 0, 0);
    return c.toDataURL("image/png");
  }

  // ---------------- generation ----------------
  bindSwitch() {
    const s = this.s;
    const ta = this.q(".f-switch"), at = this.q(".f-switch-at"), out = at.nextElementSibling, state = this.q(".f-switch-state");
    const show = () => {
      out.textContent = `${Math.round(s.switchAt * 100)}%`;
      state.textContent = s.switchPrompt.trim() ? `(on, after ${Math.round(s.switchAt * 100)}% of the steps)` : "";
    };
    ta.value = s.switchPrompt;
    at.value = Math.round(s.switchAt * 100);
    this.q(".f-switch-box").open = !!s.switchPrompt.trim();
    ta.oninput = () => { s.switchPrompt = ta.value; this.persist(); show(); this.schedule.refresh(); };
    at.oninput = () => { s.switchAt = Number(at.value) / 100; this.persist(); show(); this.schedule.refresh(); };
    this.loras.watch(ta, (text) => { s.switchPrompt = text; this.persist(); });
    show();
    this.schedule = new SchedulePreview(this.q(".f-prompt"), {
      el: this.q(".f-schedule"),
      getSteps: () => (this.mode !== "txt2img" && this.inputImage ? runSteps(s.steps, s.denoise) : s.steps),
      getSwitch: () => (s.switchPrompt.trim() ? { prompt: s.switchPrompt, at: s.switchAt } : null),
    });
    this.root.addEventListener("change", (e) => { if (e.target.matches(".f-steps, .f-denoise")) this.schedule.refresh(); });
    this.root.addEventListener("input", (e) => { if (e.target.matches(".f-denoise")) this.schedule.refresh(); });
  }

  buildJob() {
    const s = this.s;
    const st = this.ctx.state;
    const [width, height] = this.mode === "txt2img" ? [s.width, s.height] : this.outputSize();
    const job = {
      mode: this.mode === "txt2img" ? "txt2img" : "img2img",
      model: jobModel(st.model),
      prompt: withQualityPrefix(s.prompt, s.qualityPrefix, st.info),
      negative: s.negative,
      loras: jobLoras(s.loras),
      switch_prompt: s.switchPrompt.trim() ? withQualityPrefix(s.switchPrompt, s.qualityPrefix, st.info) : "",
      switch_at: s.switchAt,
      width,
      height,
      batch: s.batchSize,
      count: s.batchCount,
      seed: s.seed,
      sampler: s.sampler,
      scheduler: s.scheduler,
      steps: s.steps,
      cfg: s.cfg,
      shift: s.shift,
      save: s.saveImages,
      tag: this.mode,
    };
    if (this.mode === "txt2img" && s.hr) {
      job.hires = {
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
    if (this.mode !== "txt2img") {
      if (!this.inputImage) throw new Error("Add an input image first");
      Object.assign(job, { init_image: this.inputDataURL(), strength: s.denoise, resize_mode: s.resizeMode });
      if (this.maskHasContent()) {
        Object.assign(job, {
          mask: this.maskDataURL(),
          mask_blur: s.maskBlur,
          fill: FILL_KEYS[s.fill],
          only_masked: s.onlyMasked,
          only_masked_padding: s.onlyMaskedPadding,
          soft_inpaint: s.softInpaint,
        });
      }
    }
    const grid = this.gridCard.job();
    if (grid) job.grid = grid;
    job.detailer = !s.adetailer.enabled ? [] : s.adetailer.units
      .filter((u) => u.model && u.model !== "None")
      .map((u) => ({ model: u.model, prompt: u.prompt || "", negative: u.negative || "", confidence: u.confidence, denoise: u.denoise, mask_blur: u.maskBlur, dilate: u.dilate, padding: u.padding }));
    return job;
  }

  async generate() {
    if (this.busy) return;
    if (!this.ctx.state.model.checkpoint) return toast("Select or download an Anima checkpoint first (Models tab).");
    let job;
    try { job = this.buildJob(); } catch (e) { return toast(e.message); }

    this.busy = true;
    this.jobId = null;
    this.q(".f-generate").disabled = true;
    this.q(".f-cancel").hidden = false;
    const bar = this.q(".f-progress");
    bar.hidden = false;
    $(".bar", bar).style.width = "0%";
    $("span", bar).textContent = "Starting…";
    try {
      const done = await runJob(job, {
        onSubmit: (j) => (this.jobId = j.id),
        onUpdate: (j) => {
          const { pct, text } = jobProgress(j);
          $(".bar", bar).style.width = `${pct}%`;
          $("span", bar).textContent = text;
        },
        onPreview: (src) => this.showPreview(src),
      });
      if (done.status === "cancelled") throw new Error("Cancelled"); // ignore half-denoised images from an interrupted run
      if (!done.images.length) throw new Error("No image returned");
      const fresh = done.images.map((src, i) => ({ src, seed: done.seeds[i] ?? job.seed, info: done.infotexts[i] || "" }));
      this.results = [...fresh, ...this.results].slice(0, 60);
      this.select(0);
    } catch (e) {
      if (e.message !== "Cancelled") toast(e.message);
      if (this.current >= 0) this.select(this.current);
    } finally {
      this.busy = false;
      this.jobId = null;
      this.q(".f-generate").disabled = false;
      this.q(".f-cancel").hidden = true;
      bar.hidden = true;
    }
  }

  showPreview(src) {
    this.q(".r-img").src = src;
    this.q(".r-empty").hidden = true;
  }

  select(i) {
    this.current = i;
    const r = this.results[i];
    if (!r) return;
    this.showPreview(r.src);
    this.q(".r-info").textContent = r.info;
    $$(".r-actions button", this.root).forEach((b) => (b.disabled = false));
    const thumbs = this.q(".r-thumbs");
    thumbs.innerHTML = "";
    this.results.forEach((x, k) => {
      const img = document.createElement("img");
      img.src = x.src;
      img.title = `Seed ${x.seed}`;
      img.classList.toggle("on", k === i);
      img.onclick = () => this.select(k);
      thumbs.appendChild(img);
    });
  }
}

