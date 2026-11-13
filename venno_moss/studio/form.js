// A Venno MOSS tab: Text to Image, Image to Image (with Outpaint) or Inpaint, laid out like Venno Studio's tabs
// (same markup classes and styles). Each tab keeps its own prompt, settings and results.

import { $, $$, api, downloadDataURL, escapeHtml, loadStore, saveStore, showLightbox, toast } from "/studio/assets/js/common.js";
import { cancelJob, finishedImages, jobProgress, runJob, sheetFirst, skipJob } from "/studio/assets/js/jobs.js";
import { LoraChips } from "/studio/assets/js/loras.js";
import { HIRES_DEFAULT, HiresCard, hiresJob } from "/studio/assets/js/cards.js";
import { ZoomPane } from "/studio/assets/js/zoompane.js";
import { openLoraBrowser } from "/studio/assets/js/lorabrowser.js";

const V = "/moss/v1";
const SIZES = [[1024, 1024], [832, 1216], [1216, 832], [896, 1152], [1152, 896], [768, 1344], [1344, 768], [512, 512]];
const RESIZE_MODES = ["Just resize", "Crop and resize", "Resize and fill"];
const FILL_MODES = ["Fill", "Original", "Latent noise", "Latent nothing"];
const FILL_KEYS = ["fill", "original", "latent_noise", "latent_nothing"];
const MASK_COLOR = "#ff8a3d";
const MAX_SIDE = 2048;
const TITLES = { txt2img: "Text to Image", img2img: "Image to Image", inpaint: "Inpaint" };
const SETTING_KEYS = ["prompt", "negative", "steps", "cfg", "shift", "width", "height", "seed"];

const DEFAULT = {
  prompt: "", negative: "", loras: [], steps: 50, cfg: 3, shift: 5, count: 1, width: 1024, height: 1024, seed: -1,
  skip22: true, save: true, sheet: true, newest: false, fast: false,
  ...HIRES_DEFAULT, hrUpscaler: "Lanczos", // hires fix (Text to Image)
  // image tabs
  denoise: 0.6, resizeMode: 0, resizeBy: "by", scaleBy: 1, maskBlur: 8, fill: 1, onlyMasked: false, onlyMaskedPadding: 64, softInpaint: true,
  outpaint: { enabled: false, left: 0, right: 0, top: 0, bottom: 0, overlap: 32, denoise: 1 },
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
  strength: "0.3, 0.5, 0.7",
  hires_denoise: "0.3, 0.45, 0.6",
  hires_scale: "1.25, 1.5, 2",
  hires_steps: "10, 20, 30",
  hires_upscaler: "Lanczos, Latent, or an upscaler model's name",
  skip_layers: "22, 24, 26, none  (OpenMOSS: 22, 24 and 26 all help; none looks more realistic but can lose detail)",
};
const quote = (v) => (/[,"]/.test(v) ? `"${v.replace(/"/g, '\\"')}"` : v);
const snap16 = (v) => Math.min(MAX_SIDE, Math.max(256, Math.round(v / 16) * 16));

export const toDataURL = async (src) => {
  const blob = await (await fetch(src)).blob();
  return new Promise((res) => { const fr = new FileReader(); fr.onload = () => res(fr.result); fr.readAsDataURL(blob); });
};
export const readImage = (dataURL) => api(`${V}/image/read`, { image: dataURL });

export class MossForm {
  /**
   * @param {HTMLElement} root
   * @param {"txt2img"|"img2img"|"inpaint"} mode
   * @param {{ info: () => object|null, model: () => string, send: (target: string, src: string, settings?: object) => void, show: (tab: string) => void }} ctx
   */
  constructor(root, mode, ctx) {
    this.root = root;
    this.mode = mode;
    this.ctx = ctx;
    this.key = `venno-moss-${mode}`;
    const old = mode === "txt2img" && !localStorage.getItem(this.key) ? loadStore("venno-moss", {}) : {}; // the single form of 0.2
    this.s = { ...structuredClone(DEFAULT), ...old, ...loadStore(this.key, {}) };
    this.s.grid = { ...structuredClone(DEFAULT.grid), ...this.s.grid };
    this.s.outpaint = { ...structuredClone(DEFAULT.outpaint), ...this.s.outpaint };
    this.results = []; // [{src, seed, info}]
    this.current = -1;
    this.busy = false;
    this.jobId = null;
    this.picked = false;
    this.inputImage = null; // HTMLImageElement
    this.build();
    this.bind();
  }

  get image() { return this.mode !== "txt2img"; }
  q(sel) { return $(sel, this.root); }
  persist() { saveStore(this.key, this.s); }

  // ---------------- DOM ----------------
  build() {
    const i2i = this.image, inpaint = this.mode === "inpaint";
    const sizeInputs = `
          <div class="grid2">
            <label>Width <input type="number" class="f-width" min="256" max="${MAX_SIDE}" step="16"></label>
            <label>Height <input type="number" class="f-height" min="256" max="${MAX_SIDE}" step="16"></label>
          </div>
          <div class="sizes">${SIZES.map(([w, h]) => `<button data-w="${w}" data-h="${h}">${w}×${h}</button>`).join("")}<button class="f-swap" title="Swap width/height">⇅</button></div>`;
    this.root.innerHTML = `
      <div class="form-col">
        <section class="card">
          <textarea class="f-prompt" rows="5" placeholder="Prompt: tags, natural language or both (Ctrl+Enter to generate)"></textarea>
          <textarea class="f-negative" rows="2" placeholder="Negative prompt (MOSS-Image works well without one)"></textarea>
          <div class="hint f-tokens"></div>
          <div class="row">
            <select class="f-lora"></select>
            <button class="f-lora-browse" title="Browse LoRAs: folders, previews, tags and search">🗂 Browse</button>
            <label class="check" title="Show the newest LoRAs first (handy while a training run saves epochs)"><input type="checkbox" class="f-newest"> newest first</label>
          </div>
          <div class="f-lora-chips"></div>
          <div class="hint f-lora-info"></div>
          <div class="row buttons">
            <button class="primary big f-generate">Generate</button>
            <button class="big f-skip" hidden title="Drop the image being made and go on with the next one">Skip</button>
            <button class="big f-cancel" hidden title="Stop; the images already finished are kept">Cancel</button>
          </div>
          <label class="check" title="8 steps at CFG 1 with lodestones' turbo LoRA (added by itself): about 12× faster, for trying prompts and comparing epochs. It changes the look a little, so judge a LoRA without it."><input type="checkbox" class="f-fast"> Fast preview <span class="hint f-fast-note"></span></label>
          <div class="f-progress progress-inline" hidden><div class="bar"></div><span></span></div>
        </section>

        <section class="card">
          <div class="grid2">
            <label>Steps <input type="number" class="f-steps" min="1" max="200"></label>
            <label>CFG <input type="number" class="f-cfg" min="1" max="20" step="0.1"></label>
            <label title="Higher spends more steps on composition (MOSS-Image default 5)">Shift <input type="number" class="f-shift" min="0.5" max="20" step="0.5"></label>
            <label title="Pictures made one after another, seed +1 each">Batch count <input type="number" class="f-count" min="1" max="100"></label>
          </div>
          ${i2i ? `
          <div class="seg resize-tabs"><button class="f-rt-by">Resize by</button><button class="f-rt-to">Resize to</button></div>
          <div class="f-resize-by">
            <label class="slider">Scale <input type="range" class="f-scale" min="25" max="400" step="5"><output></output></label>
            <div class="hint f-scale-info"></div>
          </div>
          <div class="f-resize-to">${sizeInputs}
          <div class="grid2">
            <label>Resize mode <select class="f-resize-mode">${RESIZE_MODES.map((m, i) => `<option value="${i}">${m}</option>`).join("")}</select></label>
            <label>Size from image <span class="row"><button class="f-size-from-image">Match input</button></span></label>
          </div>
          </div>
          <label class="slider" title="How much of the picture is redone: the job runs the last part of its steps. MOSS-Image's shift of 5 makes a value change more than the same value in Venno.">Denoising strength <input type="range" class="f-denoise" min="0" max="100"><output></output></label>` : sizeInputs}
          <label>Seed <span class="row seed"><input type="number" class="f-seed"><button class="f-seed-random" title="Random (-1)">🎲</button><button class="f-seed-reuse" title="Reuse seed of the shown image">♻️</button></span></label>
          <label class="check" title="MOSS-Image's guidance: the negative branch skips transformer block 22 (the model card's default). Off: plain CFG."><input type="checkbox" class="f-skip22"> Skip block 22 on the negative branch (recommended)</label>
          <div class="row"><button class="f-defaults" title="Steps 50, CFG 3, shift 5, block 22 skipped: the model card's settings">Model defaults</button>${i2i ? "" : `<span class="hint">Same settings and seed as a venno-trainer sample give the same picture.</span>`}</div>
        </section>

        ${inpaint ? `
        <details class="card" open>
          <summary>Inpainting</summary>
          <div class="grid2">
            <label title="What the masked area starts from. Original: the input (use any strength). Fill: colours from the surroundings (needs high strength). Latent noise: regenerate from scratch (strength is ignored). Latent nothing: blank.">Masked content <select class="f-fill">${FILL_MODES.map((m, i) => `<option value="${i}">${m}</option>`).join("")}</select></label>
            <label title="Only masked: the area around the mask is redrawn at the size above and pasted back, so small things get the model's full resolution">Inpaint area <select class="f-only-masked"><option value="0">Whole picture</option><option value="1">Only masked</option></select></label>
            <label>Mask blur <input type="number" class="f-mask-blur" min="0" max="64"></label>
            <label>Only-masked padding <input type="number" class="f-padding" min="0" max="256"></label>
          </div>
          <label class="check" title="The feathered mask edge joins the generation gradually, so the new part blends into the picture (MOSS-Image's latents are coarse, 16 px a cell: keep this on with a mask blur of 8 or more)"><input type="checkbox" class="f-soft"> Seamless (soft) inpainting</label>
        </details>` : i2i ? `
        <details class="card f-op-card">
          <summary><input type="checkbox" class="f-op" title="Use outpainting"> <span>Outpaint</span> <span class="f-op-sum"></span></summary>
          <div class="hint">Extend the picture: the new border is generated to continue it, the original stays as it is.</div>
          <div class="grid2">
            <label>Left <input type="number" class="f-op-left" min="0" max="2048" step="16"></label>
            <label>Right <input type="number" class="f-op-right" min="0" max="2048" step="16"></label>
            <label>Top <input type="number" class="f-op-top" min="0" max="2048" step="16"></label>
            <label>Bottom <input type="number" class="f-op-bottom" min="0" max="2048" step="16"></label>
          </div>
          <div class="sizes f-op-quick"><button data-op="all">All sides +128</button><button data-op="wide">Wider +256</button><button data-op="tall">Taller +256</button><button data-op="reset">Reset</button></div>
          <div class="grid2">
            <label title="How far into the original picture the new content may blend">Overlap <input type="number" class="f-op-overlap" min="0" max="256" step="8"></label>
            <label title="For the new area (1 = generate it from scratch)">Denoising <input type="number" class="f-op-denoise" min="0.3" max="1" step="0.05"></label>
          </div>
          <div class="hint f-op-info"></div>
        </details>` : `
        <details class="card f-hr-card"></details>`}

        <details class="card f-grid-card">
          <summary><input type="checkbox" class="f-grid-on" title="Use the comparison grid"> <span>Comparison grid (X/Y/Z)</span> <span class="hint f-grid-sum"></span></summary>
          <div class="hint">One picture per combination, same seed. Put a training run's LoRA files on X to compare epochs, weights on Y.</div>
          <div class="f-axes"></div>
          <label class="check"><input type="checkbox" class="f-grid-cells"> Also keep each cell as its own picture</label>
        </details>

        <label class="check save"><input type="checkbox" class="f-save"> Save results to the outputs folder</label>
        <label class="check save" title="With two or more pictures in a job, they are also stitched into one picture for comparing them (like Forge's grid); it is shown first and saved with the grids"><input type="checkbox" class="f-sheet"> Batch sheet: stitch a batch into one picture</label>
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
                <label>Brush <input type="range" class="f-mask-size" min="1" max="300" value="48"></label>
                <button class="f-mask-invert">Invert</button>
                <button class="f-mask-clear" title="Clear the mask">Clear</button>` : ""}
                <button class="f-zoom" disabled title="Zoom: scroll over the image (it zooms where the pointer is). Click to fit the image again.">100%</button>
                <button class="f-open-again" title="Open another image">Open…</button>
                <button class="f-image-clear" title="Remove image">✕</button>
              </div>
            </div>
            <div class="i2i-drop">
              <div class="i2i-empty">Drop, paste or <button class="f-open">open</button> an image<br><small>or use “Send to ${TITLES[this.mode]}” from another tab</small></div>
              <div class="i2i-stage" hidden><img class="i2i-img" alt="" draggable="false">${inpaint ? `<canvas class="i2i-mask"></canvas>` : ""}</div>
            </div>
            <div class="hint f-pane-hint">${inpaint ? "Paint over the area to regenerate. No mask = the whole image.<br>" : ""}<b>Scroll</b> zooms in where the pointer is; drag with the <b>middle button</b> (or hold <b>Space</b> and drag) to move around${inpaint ? "; <b>Shift + scroll</b> changes the brush size" : ""}.</div>
          </div>
          <div class="pane out-pane">
            <div class="pane-head"><span class="pane-title">Result</span></div>
            <div class="result-main"><img class="r-img" alt=""><div class="r-empty">Results appear here</div></div>
          </div>
        </div>` : `
        <div class="result-main"><img class="r-img" alt=""><div class="r-empty">Results appear here</div></div>`}
        <div class="r-thumbs"></div>
        <div class="row r-actions">
          ${i2i ? `<button class="r-to-input" disabled title="Continue from this result">Use as input</button>` : ""}
          ${this.mode !== "img2img" ? `<button class="r-to-img2img" disabled>Send to Image to Image</button>` : ""}
          ${this.mode !== "inpaint" ? `<button class="r-to-inpaint" disabled>Send to Inpaint</button>` : ""}
          <button class="r-use-settings" disabled title="Load this picture's prompt and settings into the form">Use settings</button>
          <button class="r-view" disabled title="Full window">View</button>
          <button class="r-download" disabled>Download</button>
        </div>
        <pre class="r-info"></pre>
      </div>`;
  }

  bind() {
    const s = this.s;
    const num = (cls, key, after = () => {}) => {
      const el = this.q(cls);
      el.value = s[key];
      el.onchange = () => { const v = Number(el.value); if (Number.isFinite(v)) s[key] = v; this.persist(); after(); };
    };
    const sized = () => { this.hires?.updateInfo(); this.showResize(); };
    for (const [cls, key] of [[".f-steps", "steps"], [".f-cfg", "cfg"], [".f-shift", "shift"], [".f-count", "count"], [".f-seed", "seed"]]) num(cls, key);
    num(".f-width", "width", sized);
    num(".f-height", "height", sized);
    for (const [cls, key] of [[".f-prompt", "prompt"], [".f-negative", "negative"]]) {
      const el = this.q(cls);
      el.value = s[key];
      el.oninput = () => { s[key] = el.value; this.persist(); if (key === "prompt") this.countTokens(); };
    }
    for (const [cls, key] of [[".f-skip22", "skip22"], [".f-save", "save"], [".f-sheet", "sheet"], [".f-newest", "newest"]]) {
      const el = this.q(cls);
      el.checked = s[key];
      el.onchange = () => { s[key] = el.checked; this.persist(); if (key === "newest") this.fillLoras(); };
    }
    this.q(".sizes").onclick = (e) => {
      const b = e.target.closest("button");
      if (!b || b.closest(".f-op-quick")) return;
      if (b.classList.contains("f-swap")) this.setSize(s.height, s.width);
      else this.setSize(Number(b.dataset.w), Number(b.dataset.h));
    };
    this.q(".f-seed-random").onclick = () => { s.seed = -1; this.q(".f-seed").value = -1; this.persist(); };
    this.q(".f-seed-reuse").onclick = () => { const r = this.results[this.current]; if (r) { s.seed = r.seed; this.q(".f-seed").value = r.seed; this.persist(); } };
    this.q(".f-defaults").onclick = () => {
      Object.assign(s, { steps: 50, cfg: 3, shift: 5, skip22: true });
      for (const k of ["steps", "cfg", "shift"]) this.q(`.f-${k}`).value = s[k];
      this.q(".f-skip22").checked = true;
      this.persist();
    };

    // LoRAs
    this.chips = new LoraChips(this.q(".f-lora-chips"), { get: () => s.loras, set: (list) => { s.loras = list; this.persist(); this.renderAxes(); this.loraInfo(); } });
    this.q(".f-lora").onchange = (e) => { if (e.target.value) this.chips.add(e.target.value, 1); e.target.value = ""; };
    this.q(".f-lora-browse").onclick = () => openLoraBrowser({
      chips: this.chips, prompt: this.q(".f-prompt"), defaultWeight: 1,
      current: () => this.results[this.current]?.src || null,
      emptyHint: "No LoRAs found. Add your LoRA folders in ⚙ Settings → LoRAs (sub-folders become folders here) and press ⟳.",
    });

    // fast preview
    this.q(".f-fast").onchange = (e) => {
      if (e.target.checked && this.ctx.info() && !this.ctx.info().turbo) {
        toast("Fast preview needs the 8-step turbo LoRA (1.2 GB): download it here, then switch it on.", "info");
        this.ctx.show("settings");
        e.target.checked = false;
        return;
      }
      s.fast = e.target.checked;
      this.persist();
      this.renderFast();
    };

    // hires fix (Venno Studio's card)
    if (this.mode === "txt2img") {
      this.hires = new HiresCard(this.q(".f-hr-card"), {
        state: () => s, persist: () => this.persist(), size: () => [s.width, s.height], round: 16,
        note: "A second pass at a higher resolution. <b>Lanczos</b> or an upscaler model (the ones in Venno's upscaler folders) keep the picture: denoising 0.3–0.5. <b>Latent</b> needs 0.55 or more. LoRAs apply in both passes. MOSS-Image is made for up to about 4 megapixels (2048×2048).",
      });
    }

    // image tabs
    if (this.image) {
      const den = this.q(".f-denoise");
      den.value = Math.round(s.denoise * 100);
      den.nextElementSibling.textContent = s.denoise;
      den.oninput = () => { s.denoise = Number(den.value) / 100; den.nextElementSibling.textContent = s.denoise; this.persist(); };
      const scale = this.q(".f-scale");
      scale.value = Math.round(s.scaleBy * 100);
      scale.oninput = () => { s.scaleBy = Number(scale.value) / 100; this.persist(); this.showResize(); };
      this.q(".f-rt-by").onclick = () => { s.resizeBy = "by"; this.persist(); this.showResize(); };
      this.q(".f-rt-to").onclick = () => { s.resizeBy = "to"; this.persist(); this.showResize(); };
      const rm = this.q(".f-resize-mode");
      rm.value = s.resizeMode;
      rm.onchange = () => { s.resizeMode = Number(rm.value); this.persist(); };
      this.q(".f-size-from-image").onclick = () => this.sizeFromImage();
      this.bindImageInput();
      this.showResize();
    }
    if (this.mode === "inpaint") {
      num(".f-mask-blur", "maskBlur");
      num(".f-padding", "onlyMaskedPadding");
      const fill = this.q(".f-fill"), om = this.q(".f-only-masked"), soft = this.q(".f-soft");
      fill.value = s.fill;
      fill.onchange = () => { s.fill = Number(fill.value); this.persist(); };
      om.value = s.onlyMasked ? 1 : 0;
      om.onchange = () => { s.onlyMasked = om.value === "1"; this.persist(); };
      soft.checked = s.softInpaint;
      soft.onchange = () => { s.softInpaint = soft.checked; this.persist(); };
    }
    if (this.mode === "img2img") this.bindOutpaint();

    // comparison grid
    const gridOn = this.q(".f-grid-on"), cells = this.q(".f-grid-cells");
    gridOn.checked = s.grid.enabled;
    gridOn.onclick = (e) => e.stopPropagation(); // don't toggle the card
    gridOn.onchange = () => { s.grid.enabled = gridOn.checked; this.persist(); this.checkGrid(); };
    cells.checked = s.grid.cells;
    cells.onchange = () => { s.grid.cells = cells.checked; this.persist(); };

    // generating, results
    this.q(".f-generate").onclick = () => this.generate();
    this.q(".f-cancel").onclick = () => this.jobId && cancelJob(this.jobId);
    this.q(".f-skip").onclick = () => this.jobId && skipJob(this.jobId);
    const withCurrent = (fn) => () => { const r = this.results[this.current]; if (r) fn(r); };
    this.q(".r-view").onclick = withCurrent((r) => showLightbox(r.src, `moss-${r.seed}.png`));
    this.q(".r-download").onclick = withCurrent((r) => downloadDataURL(r.src, `moss-${r.seed}.png`));
    this.q(".r-use-settings").onclick = withCurrent(async (r) => this.applySettings((await readImage(await toDataURL(r.src))).settings));
    for (const target of ["img2img", "inpaint"]) {
      const b = this.q(`.r-to-${target}`);
      if (b) b.onclick = withCurrent(async (r) => this.ctx.send(target, r.src, await this.settingsOf(r)));
    }
    const again = this.q(".r-to-input");
    if (again) again.onclick = withCurrent((r) => this.setInputImage(r.src));

    this.chips.render();
    this.renderFast();
    this.countTokens();
  }

  /** What a result was made with, for the tab it is sent to: the settings saved in the picture (it may be an
   *  older result than the form shows), else this form's prompt and settings with the result's seed. */
  async settingsOf(r) {
    try {
      const read = (await readImage(await toDataURL(r.src))).settings;
      if (read && !read.partial) return read;
    } catch {}
    const s = this.s;
    return { prompt: s.prompt, negative: s.negative, loras: s.loras, steps: s.steps, cfg: s.cfg, shift: s.shift, skip_layers: s.skip22 ? [22] : [], seed: r.seed, fast: s.fast };
  }

  setSize(w, h) {
    this.s.width = w;
    this.s.height = h;
    this.q(".f-width").value = w;
    this.q(".f-height").value = h;
    this.persist();
    this.hires?.updateInfo();
    this.showResize();
  }

  countTokens() {
    clearTimeout(this.tokenTimer);
    this.tokenTimer = setTimeout(async () => {
      const el = this.q(".f-tokens");
      if (!this.s.prompt.trim()) return (el.textContent = "");
      try { const r = await api(`${V}/tokens`, { text: this.s.prompt }); el.textContent = r.count ? `${r.count} tokens` : ""; } catch {}
    }, 500);
  }

  // ---------------- info from the server ----------------
  setInfo(info) {
    this.fillLoras();
    this.renderAxes();
    this.hires?.setInfo(info);
    if (this.s.fast && !info.turbo) this.s.fast = false;
    this.renderFast();
  }

  renderFast() {
    const s = this.s, info = this.ctx.info();
    this.q(".f-fast").checked = s.fast;
    const f = info?.fast || { steps: 8, cfg: 1 };
    this.q(".f-fast-note").textContent = s.fast ? `· ${f.steps} steps, CFG ${f.cfg}, turbo LoRA: Steps and CFG below are not used` : info && !info.turbo ? "· needs the turbo LoRA (⚙ Settings → Models)" : `· ${f.steps} steps with the turbo LoRA`;
    for (const k of ["steps", "cfg"]) this.q(`.f-${k}`).classList.toggle("unused", s.fast);
  }

  // ---------------- LoRAs ----------------
  /** One line per LoRA in the list: the model it was trained on (when its file records it), rank and alpha. */
  loraInfo() {
    const known = new Map((this.ctx.info()?.loras || []).map((l) => [l.name, l]));
    this.q(".f-lora-info").innerHTML = this.s.loras.map((x) => {
      const l = known.get(x.name);
      if (!l) return "";
      const bits = [`trained on ${l.trained_on ? `<b>${escapeHtml(l.trained_on)}</b>` : "(not recorded)"}`];
      if (l.rank) bits.push(`rank ${l.rank}`);
      if (l.alpha != null) bits.push(`alpha ${l.alpha}`);
      return `<div>${escapeHtml(x.name.replace(/\.safetensors$/i, ""))}: ${bits.join(" · ")}</div>`;
    }).join("");
  }

  fillLoras() {
    const list = [...(this.ctx.info()?.loras || [])];
    if (this.s.newest) list.sort((a, b) => b.mtime - a.mtime);
    this.q(".f-lora").innerHTML = `<option value="">+ LoRA… (${list.length})</option>` + list.map((l) => `<option value="${escapeHtml(l.name)}">${escapeHtml(l.name)}</option>`).join("");
    this.chips.setAvailable(list.map((l) => l.name));
    this.loraInfo();
  }

  /** LoRA files grouped into training runs: "run-000004", "run-000008" → "run" (sorted by their number). */
  loraRuns() {
    const runs = new Map();
    for (const l of this.ctx.info()?.loras || []) {
      const m = l.name.replace(/\.safetensors$/i, "").match(/^(.*?)[-_. ]?(?:e|ep|epoch|step|steps|s)?[-_]?(\d+)$/i);
      if (!m || !m[1]) continue;
      if (!runs.has(m[1])) runs.set(m[1], []);
      runs.get(m[1]).push({ name: l.name, n: Number(m[2]) });
    }
    return [...runs.entries()].filter(([, v]) => v.length > 1).map(([k, v]) => [k, v.sort((a, b) => a.n - b.n).map((x) => x.name)]);
  }

  // ---------------- comparison grid ----------------
  renderAxes() {
    const s = this.s, info = this.ctx.info();
    // hires axes belong to Text to Image, the denoising axis to the image tabs
    const axes = (info?.axes || []).filter((a) => (this.image ? !a.key.startsWith("hires_") : a.key !== "strength"));
    const box = this.q(".f-axes");
    box.innerHTML = s.grid.axes.map((a, i) => `
      <div class="axis" data-i="${i}">
        <div class="row">
          <b>${"XYZ"[i]}</b>
          <select class="a-type"><option value="none">—</option>${axes.map((x) => `<option value="${x.key}"${x.key === a.type ? " selected" : ""}>${x.label}</option>`).join("")}</select>
          ${a.type === "lora" || a.type === "lora_weight" ? `<select class="a-target" title="${a.type === "lora" ? "The LoRA in the list whose file changes per cell (none in the list: added at weight 1)" : "The LoRA whose weight changes"}"><option value="">${a.type === "lora" ? "first LoRA in the list" : "first LoRA"}</option>${s.loras.map((l) => `<option${l.name === a.target ? " selected" : ""}>${escapeHtml(l.name)}</option>`).join("")}</select>` : ""}
          ${a.type === "lora" ? `<select class="a-add"><option value="">+ add…</option><option value="none">none (without it)</option>${this.loraRuns().map(([k, v]) => `<option value="run:${escapeHtml(k)}">all ${v.length} of ${escapeHtml(k)}</option>`).join("")}${(info?.loras || []).map((l) => `<option value="${escapeHtml(l.name)}">${escapeHtml(l.name)}</option>`).join("")}</select>` : ""}
        </div>
        ${a.type !== "none" ? `<input class="a-values" value="${escapeHtml(a.values)}" placeholder="${escapeHtml(AXIS_HINTS[a.type] || "")}">` : ""}
      </div>`).join("");
    $$(".axis", box).forEach((row) => {
      const a = s.grid.axes[Number(row.dataset.i)];
      $(".a-type", row).onchange = (e) => { a.type = e.target.value; a.values = ""; a.target = ""; this.persist(); this.renderAxes(); };
      const t = $(".a-target", row);
      if (t) t.onchange = () => { a.target = t.value; this.persist(); this.checkGrid(); };
      const v = $(".a-values", row);
      if (v) v.oninput = () => { a.values = v.value; this.persist(); this.checkGrid(); };
      const add = $(".a-add", row);
      if (add) add.onchange = () => {
        if (!add.value) return;
        const items = add.value.startsWith("run:") ? this.loraRuns().find(([k]) => k === add.value.slice(4))[1] : [add.value];
        a.values = [a.values.trim(), ...items.map(quote)].filter(Boolean).join(", ");
        this.persist();
        this.renderAxes();
      };
    });
    this.checkGrid();
  }

  gridJob() {
    const g = this.s.grid;
    if (!g.enabled) return null;
    const axis = (a) => (a.type === "none" ? null : { type: a.type, values: a.values, target: a.target || "" });
    return { x: axis(g.axes[0]), y: axis(g.axes[1]), z: axis(g.axes[2]), include_cells: g.cells };
  }

  checkGrid() {
    clearTimeout(this.gridTimer);
    const g = this.gridJob(), el = this.q(".f-grid-sum");
    if (!g) return (el.textContent = "");
    this.gridTimer = setTimeout(async () => {
      try {
        const r = await api(`${V}/grid/check`, g);
        el.textContent = r.ok ? `${r.counts.join(" × ")} = ${r.total} picture${r.total > 1 ? "s" : ""}` : r.error;
      } catch {}
    }, 300);
  }

  // ---------------- input image, mask, outpaint (as Venno's Image to Image / Inpaint tabs) ----------------
  /** Outpaint settings when outpainting is on and extends at least one side, else null. */
  outpaint() {
    const o = this.s.outpaint;
    return this.mode === "img2img" && o?.enabled && o.left + o.right + o.top + o.bottom > 0 ? o : null;
  }

  /** Size of the picture sent to the engine: the input, plus the outpaint border. */
  sourceSize() {
    const img = this.inputImage;
    if (!img) return [this.s.width, this.s.height];
    const o = this.outpaint();
    return o ? [img.naturalWidth + o.left + o.right, img.naturalHeight + o.top + o.bottom] : [img.naturalWidth, img.naturalHeight];
  }

  /** Output size: input × scale ("Resize by", kept within MOSS-Image's 2048 a side), or width / height ("Resize to"). */
  outputSize() {
    const s = this.s;
    if (s.resizeBy === "by" && this.inputImage) {
      const [w, h] = this.sourceSize();
      const k = Math.min(s.scaleBy, MAX_SIDE / Math.max(w, h));
      return [snap16(w * k), snap16(h * k)];
    }
    return [s.width, s.height];
  }

  showResize() {
    if (!this.image) return;
    const s = this.s, by = s.resizeBy === "by";
    this.q(".f-rt-by").classList.toggle("on", by);
    this.q(".f-rt-to").classList.toggle("on", !by);
    this.q(".f-resize-by").hidden = !by;
    this.q(".f-resize-to").hidden = by;
    this.q(".f-scale").nextElementSibling.textContent = `${Math.round(s.scaleBy * 100)}%`;
    const [w, h] = this.outputSize();
    const [sw, sh] = this.sourceSize();
    this.q(".f-scale-info").textContent = this.inputImage ? `${sw}×${sh}${this.outpaint() ? " (outpainted)" : ""} → ${w}×${h}` : "Scales the input image (output = input size × scale, at most 2048 a side).";
  }

  sizeFromImage() {
    if (!this.inputImage) return;
    const [w, h] = this.sourceSize();
    const scale = Math.sqrt((1024 * 1024) / (w * h)); // keep the aspect ratio at about 1 megapixel
    this.setSize(snap16(w * scale), snap16(h * scale));
  }

  /** Fit the input image (and its mask canvas) to the pane, scaling small images up too; × the zoom. */
  fitStage() {
    const img = this.inputImage, drop = this.q(".i2i-drop"), stage = this.q(".i2i-stage");
    if (!img || !drop || stage.hidden) return;
    const op = this.outpaint();
    const [W, H] = this.sourceSize();
    const r = drop.getBoundingClientRect();
    const k = Math.min((r.width - 16) / W, (r.height - 16) / H);
    if (!(k > 0)) return;
    this.zoomPane.apply(Math.floor(W * k), Math.floor(H * k));
    stage.classList.toggle("outpainting", !!op);
    Object.assign(this.q(".i2i-img").style, op
      ? { position: "absolute", left: `${(op.left / W) * 100}%`, top: `${(op.top / H) * 100}%`, width: `${(img.naturalWidth / W) * 100}%`, height: `${(img.naturalHeight / H) * 100}%` }
      : { position: "", left: "", top: "", width: "", height: "" });
  }

  bindImageInput() {
    const drop = this.q(".i2i-drop"), stage = this.q(".i2i-stage"), zoomBtn = this.q(".f-zoom");
    this.zoomPane = new ZoomPane(drop, stage, {
      onChange: (z) => { zoomBtn.textContent = `${Math.round(z * 100)}%`; zoomBtn.disabled = z === 1; },
      onBrush: (dir) => this.brushStep?.(dir),
    });
    zoomBtn.onclick = () => this.zoomPane.reset();
    new ResizeObserver(() => this.fitStage()).observe(drop);
    const file = document.createElement("input");
    file.type = "file";
    file.accept = "image/*";
    file.onchange = () => { if (file.files[0]) this.setInputImage(URL.createObjectURL(file.files[0])); file.value = ""; };
    this.q(".f-open").onclick = () => file.click();
    this.q(".f-open-again").onclick = () => file.click();
    this.q(".f-image-clear").onclick = () => this.setInputImage(null);
    drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("dragging"); });
    drop.addEventListener("dragleave", () => drop.classList.remove("dragging"));
    drop.addEventListener("drop", (e) => {
      e.preventDefault();
      drop.classList.remove("dragging");
      const f = [...e.dataTransfer.files].find((x) => x.type.startsWith("image/"));
      if (f) this.setInputImage(URL.createObjectURL(f));
    });

    const mc = this.q(".i2i-mask");
    if (!mc) return; // Image to Image: no mask
    stage.classList.add("paintable");
    const tool = { erase: false, size: 48 };
    let drawing = false, last = null;
    const cursor = document.createElement("div");
    cursor.className = "mask-cursor";
    stage.appendChild(cursor);
    const showTool = () => {
      this.q(".f-mask-draw").classList.toggle("on", !tool.erase);
      this.q(".f-mask-erase").classList.toggle("on", tool.erase);
      this.q(".f-mask-size").value = tool.size;
    };
    this.q(".f-mask-draw").onclick = () => { tool.erase = false; showTool(); };
    this.q(".f-mask-erase").onclick = () => { tool.erase = true; showTool(); };
    this.q(".f-mask-size").oninput = (e) => (tool.size = Number(e.target.value));
    this.q(".f-mask-clear").onclick = () => this.clearMask();
    this.q(".f-mask-invert").onclick = () => this.invertMask();
    const pos = (e) => {
      const r = mc.getBoundingClientRect();
      return { x: ((e.clientX - r.left) / r.width) * mc.width, y: ((e.clientY - r.top) / r.height) * mc.height, k: mc.width / r.width };
    };
    const paint = (p) => {  // the brush keeps its size on screen: zoomed in, it paints finer
      const c = mc.getContext("2d");
      c.globalCompositeOperation = tool.erase ? "destination-out" : "source-over";
      c.strokeStyle = c.fillStyle = MASK_COLOR;
      c.lineWidth = tool.size * p.k;
      c.lineCap = c.lineJoin = "round";
      c.beginPath();
      c.moveTo((last || p).x, (last || p).y);
      c.lineTo(p.x, p.y);
      c.stroke();
      last = p;
    };
    stage.addEventListener("pointerdown", (e) => { if (e.button !== 0) return; stage.setPointerCapture(e.pointerId); drawing = true; last = null; paint(pos(e)); });
    stage.addEventListener("pointermove", (e) => {
      const r = stage.getBoundingClientRect();
      Object.assign(cursor.style, { display: "block", width: `${tool.size}px`, height: `${tool.size}px`, left: `${e.clientX - r.left}px`, top: `${e.clientY - r.top}px` });
      if (drawing) paint(pos(e));
    });
    stage.addEventListener("pointerleave", () => (cursor.style.display = "none"));
    const end = () => { drawing = false; last = null; };
    stage.addEventListener("pointerup", end);
    stage.addEventListener("pointercancel", end);
    this.brushStep = (dir) => {  // Shift + wheel over the image (the plain wheel zooms)
      tool.size = Math.max(1, Math.min(300, tool.size * (dir > 0 ? 1.12 : 1 / 1.12)));
      this.q(".f-mask-size").value = tool.size;
      cursor.style.width = cursor.style.height = `${tool.size}px`;
    };
    showTool();
  }

  async setInputImage(src) {
    if (!src) {
      this.inputImage = null;
      this.zoomPane.reset();
      this.q(".i2i-stage").hidden = true;
      this.q(".i2i-empty").hidden = false;
      this.q(".mask-tools").hidden = true;
      this.showResize();
      this.updateOutpaintInfo();
      return;
    }
    const img = new Image();
    img.src = src;
    await img.decode();
    this.inputImage = img;
    this.zoomPane.reset();
    this.q(".i2i-img").src = src;
    const mc = this.q(".i2i-mask");
    if (mc) { mc.width = img.naturalWidth; mc.height = img.naturalHeight; } // also clears the mask
    this.q(".i2i-stage").hidden = false;
    this.q(".i2i-empty").hidden = true;
    this.q(".mask-tools").hidden = false;
    this.fitStage();
    if (this.s.resizeBy !== "by") this.sizeFromImage();
    this.showResize();
    this.updateOutpaintInfo();
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
    const mc = this.q(".i2i-mask"), c = mc.getContext("2d");
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

  /** The painted mask as white on black. */
  maskDataURL() {
    const mc = this.q(".i2i-mask");
    const white = document.createElement("canvas");
    white.width = mc.width;
    white.height = mc.height;
    const wc = white.getContext("2d");
    wc.drawImage(mc, 0, 0);
    wc.globalCompositeOperation = "source-in";
    wc.fillStyle = "#fff";
    wc.fillRect(0, 0, white.width, white.height);
    const out = document.createElement("canvas");
    out.width = mc.width;
    out.height = mc.height;
    const c = out.getContext("2d");
    c.fillStyle = "#000";
    c.fillRect(0, 0, out.width, out.height);
    c.drawImage(white, 0, 0);
    return out.toDataURL("image/png");
  }

  /** The input image as sent: with the outpaint border around it when outpainting. */
  inputDataURL() {
    const img = this.inputImage, o = this.outpaint();
    const [W, H] = this.sourceSize();
    const c = document.createElement("canvas");
    c.width = W;
    c.height = H;
    c.getContext("2d").drawImage(img, o ? o.left : 0, o ? o.top : 0);
    return c.toDataURL("image/png");
  }

  /** Outpaint mask: the new border, reaching `overlap` px into the original on the extended sides. */
  outpaintMaskDataURL() {
    const img = this.inputImage, o = this.outpaint();
    const [W, H] = this.sourceSize();
    const c = document.createElement("canvas");
    c.width = W;
    c.height = H;
    const x = c.getContext("2d");
    x.fillStyle = "#fff";
    x.fillRect(0, 0, W, H);
    const ov = (side) => (o[side] > 0 ? o.overlap : 0);
    x.fillStyle = "#000";
    x.fillRect(o.left + ov("left"), o.top + ov("top"), Math.max(0, img.naturalWidth - ov("left") - ov("right")), Math.max(0, img.naturalHeight - ov("top") - ov("bottom")));
    return c.toDataURL("image/png");
  }

  bindOutpaint() {
    const op = () => this.s.outpaint;
    const changed = () => { this.persist(); this.fitStage(); this.showResize(); this.updateOutpaintInfo(); };
    const render = () => {
      this.q(".f-op").checked = op().enabled;
      if (op().enabled) this.q(".f-op-card").open = true;
      for (const key of ["left", "right", "top", "bottom", "overlap", "denoise"]) this.q(`.f-op-${key}`).value = op()[key];
    };
    this.renderOutpaint = render;
    const chk = this.q(".f-op");
    chk.onclick = (e) => e.stopPropagation();
    chk.onchange = () => { op().enabled = chk.checked; render(); changed(); };
    for (const key of ["left", "right", "top", "bottom", "overlap", "denoise"]) {
      const el = this.q(`.f-op-${key}`);
      el.onchange = () => {
        const v = Math.max(0, Number(el.value) || 0);
        op()[key] = key === "denoise" ? Math.min(1, Math.max(0.3, v)) : v;
        el.value = op()[key];
        changed();
      };
    }
    $$(".f-op-quick button", this.root).forEach((b) => (b.onclick = () => {
      const o = op(), k = b.dataset.op;
      if (k === "reset") Object.assign(o, { left: 0, right: 0, top: 0, bottom: 0 });
      if (k === "all") for (const side of ["left", "right", "top", "bottom"]) o[side] += 128;
      if (k === "wide") { o.left += 128; o.right += 128; }
      if (k === "tall") { o.top += 128; o.bottom += 128; }
      o.enabled = k !== "reset" || o.enabled;
      render();
      changed();
    }));
    render();
    this.updateOutpaintInfo();
  }

  updateOutpaintInfo() {
    const el = this.q(".f-op-info");
    if (!el) return;
    const o = this.outpaint(), img = this.inputImage;
    const [W, H] = this.sourceSize();
    this.q(".f-op-sum").textContent = o ? `(+${o.left + o.right}×${o.top + o.bottom})` : "";
    el.textContent = !img ? "Add an input image first." : o ? `${img.naturalWidth}×${img.naturalHeight} → ${W}×${H} before resizing. The Denoising strength above is not used.` : "Set how many pixels to add on each side.";
  }

  // ---------------- generating ----------------
  buildJob() {
    const s = this.s, model = this.ctx.model();
    if (!model) throw new Error("Pick a MOSS-Image model in the top bar (or download one in ⚙ Settings → Models)");
    const [width, height] = this.image ? this.outputSize() : [s.width, s.height];
    const job = {
      mode: this.image ? "img2img" : "txt2img",
      prompt: s.prompt, negative: s.negative, model, loras: s.loras.filter((l) => l.weight !== 0).map(({ name, weight }) => ({ name, weight })),
      width, height, steps: s.steps, cfg: s.cfg, shift: s.shift, skip_layers: s.skip22 ? [22] : [],
      seed: s.seed, count: s.count, grid: this.gridJob(), save: s.save, sheet: s.sheet, fast: s.fast, tag: this.mode,
    };
    if (!this.image) {
      job.hires = hiresJob(s);
      return job;
    }
    if (!this.inputImage) throw new Error("Add an input image first");
    const op = this.outpaint();
    if (op) {
      return Object.assign(job, {
        init_image: this.inputDataURL(), strength: op.denoise, resize_mode: 0,
        mask: this.outpaintMaskDataURL(), mask_blur: Math.max(4, Math.round(op.overlap / 2)), fill: "fill", only_masked: false, soft_inpaint: true,
      });
    }
    Object.assign(job, { init_image: this.inputDataURL(), strength: s.denoise, resize_mode: s.resizeMode });
    if (this.maskHasContent()) {
      Object.assign(job, { mask: this.maskDataURL(), mask_blur: s.maskBlur, fill: FILL_KEYS[s.fill], only_masked: s.onlyMasked, only_masked_padding: s.onlyMaskedPadding, soft_inpaint: s.softInpaint });
    }
    return job;
  }

  async generate() {
    if (this.busy) return;
    let job;
    try { job = this.buildJob(); } catch (e) { return toast(e.message); }
    this.busy = true;
    this.jobId = null;
    this.picked = false;
    this.q(".f-generate").disabled = true;
    this.q(".f-cancel").hidden = false;
    const bar = this.q(".f-progress");
    bar.hidden = false;
    $(".bar", bar).style.width = "0%";
    $("span", bar).textContent = "Starting… (the first picture loads the model)";
    const older = this.results;
    let shown = 0;
    const entries = (j) => j.images.map((src, i) => ({ src, seed: j.seeds[i], info: j.infotexts[i] || "" }));
    try {
      const done = await runJob(job, {
        onSubmit: (j) => (this.jobId = j.id),
        onUpdate: (j) => {
          const { pct, text } = jobProgress(j);
          $(".bar", bar).style.width = `${pct}%`;
          $("span", bar).textContent = j.status === "running" && !j.steps ? "Loading the model and encoding the prompt…" : text;
          this.q(".f-skip").hidden = !(j.status === "running" && j.job_count > 1);
          if (!job.grid && j.images.length > shown) {
            if (this.current >= shown) this.current += j.images.length - shown;
            shown = j.images.length;
            this.results = [...entries(j), ...older].slice(0, 80);
            this.renderThumbs();
          }
        },
        onPreview: (src) => { if (!this.picked) this.showImage(src); },
      });
      const made = entries({ ...done, images: finishedImages(done) });
      const fresh = sheetFirst(made); // the batch sheet leads, like Forge's grid
      for (const w of done.warnings || []) toast(w, "info");
      this.results = [...fresh, ...older].slice(0, 80);
      this.select(this.picked && this.current < made.length ? Math.min(fresh.length - 1, this.current + (fresh === made ? 0 : 1)) : 0);
      if (job.grid) showLightbox(fresh[0].src, `moss-grid-${fresh[0].seed}.png`);
    } catch (e) {
      if (e.message !== "Cancelled") toast(e.message);
      if (this.current >= 0) this.select(this.current);
    } finally {
      this.busy = false;
      this.jobId = null;
      this.q(".f-generate").disabled = false;
      this.q(".f-cancel").hidden = true;
      this.q(".f-skip").hidden = true;
      bar.hidden = true;
    }
  }

  // ---------------- results ----------------
  showImage(src) {
    this.q(".r-img").src = src;
    this.q(".r-empty").hidden = true;
  }

  renderThumbs() {
    const box = this.q(".r-thumbs");
    box.innerHTML = "";
    this.results.forEach((r, k) => {
      const img = document.createElement("img");
      img.src = r.src;
      img.title = `Seed ${r.seed}`;
      img.classList.toggle("on", k === this.current);
      img.onclick = () => { if (this.busy) this.picked = true; this.select(k); };
      img.ondblclick = () => showLightbox(r.src, `moss-${r.seed}.png`);
      box.appendChild(img);
    });
  }

  select(i) {
    this.current = i;
    const r = this.results[i];
    if (!r) return;
    this.showImage(r.src);
    this.q(".r-info").textContent = r.info;
    $$(".r-actions button", this.root).forEach((b) => (b.disabled = false));
    this.renderThumbs();
  }

  // ---------------- settings read from a picture ----------------
  applySettings(x) {
    if (!x) return toast("This picture has no settings to load");
    const s = this.s;
    const put = (keys) => { for (const k of keys) if (x[k] != null) s[k] = x[k]; };
    if (x.partial) { // a picture from another UI: its prompt, seed and size; steps, CFG and LoRAs belong to its model
      put(["prompt", "negative", "width", "height", "seed"]);
    } else {
      s.fast = !!x.fast && !!this.ctx.info()?.turbo;
      // a Fast preview picture recorded the 8 steps / CFG 1 it ran with: the form keeps its own
      put(SETTING_KEYS.filter((k) => !(x.fast && (k === "steps" || k === "cfg"))));
      s.skip22 = (x.skip_layers || []).includes(22);
      s.loras = (x.loras || []).map((l) => ({ name: l.name, weight: l.weight }));
      if (x.model) this.ctx.model(x.model);
      const hr = x.hires;
      if (this.mode === "txt2img") {
        s.hr = !!hr;
        if (hr) Object.assign(s, { hrUpscaler: hr.upscaler, hrScale: hr.scale, hrResizeX: hr.resize_x, hrResizeY: hr.resize_y, hrSteps: hr.steps, hrDenoise: hr.denoise, hrCfg: hr.cfg || 0, hrShift: hr.shift || 0 });
      }
      if (this.image && x.mode === "img2img") {
        if (x.strength != null) s.denoise = x.strength;
        if (x.resize_mode != null) s.resizeMode = x.resize_mode;
        if (this.mode === "inpaint") {
          if (x.mask_blur != null) s.maskBlur = x.mask_blur;
          if (FILL_KEYS.includes(x.fill)) s.fill = FILL_KEYS.indexOf(x.fill);
          if (x.only_masked != null) s.onlyMasked = x.only_masked;
          if (x.only_masked_padding != null) s.onlyMaskedPadding = x.only_masked_padding;
          if (x.soft_inpaint != null) s.softInpaint = x.soft_inpaint;
        }
      }
    }
    if (this.image) s.resizeBy = "by"; // the size of a picture's settings is its own; the input image decides here
    this.persist();
    this.render();
    toast(x.partial ? "Prompt, seed and size loaded (the other settings belong to another model)" : "Settings loaded", "info");
  }

  /** The form's inputs from its state (after settings were loaded). */
  render() {
    const s = this.s;
    for (const k of SETTING_KEYS) this.q(`.f-${k}`).value = s[k];
    this.q(".f-count").value = s.count;
    this.q(".f-skip22").checked = s.skip22;
    this.hires?.render();
    if (this.image) {
      const den = this.q(".f-denoise");
      den.value = Math.round(s.denoise * 100);
      den.nextElementSibling.textContent = s.denoise;
      this.q(".f-resize-mode").value = s.resizeMode;
      this.showResize();
    }
    if (this.mode === "inpaint") {
      this.q(".f-mask-blur").value = s.maskBlur;
      this.q(".f-padding").value = s.onlyMaskedPadding;
      this.q(".f-fill").value = s.fill;
      this.q(".f-only-masked").value = s.onlyMasked ? 1 : 0;
      this.q(".f-soft").checked = s.softInpaint;
    }
    this.renderFast();
    this.chips.render();
    this.loraInfo();
    this.renderAxes();
    this.countTokens();
  }
}
