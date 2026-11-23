// The Upscale tab, shared by Venno Studio and Venno MOSS: a picture through an upscaler model from the upscaler
// folders (models/ESRGAN, …), optionally on the processor only (CPU mode). The server side is
// venno/pipeline/upscaler.py; both servers serve it under their own API path.

import { $, api, downloadDataURL, escapeHtml, fillSelect, loadStore, saveStore, showLightbox, toast } from "./common.js";

const V = globalThis.VENNO_API_BASE || "/venno/v1";
const DEFAULTS = { upscaler: "", scale: 2, tile: 512, cpu: false, save: true };

const blobToDataURL = (blob) => new Promise((resolve, reject) => {
  const reader = new FileReader();
  reader.onload = () => resolve(reader.result);
  reader.onerror = () => reject(reader.error);
  reader.readAsDataURL(blob);
});

export class UpscaleTab {
  /**
   * @param {HTMLElement} root
   * @param {{ storeKey: string, modelsHint: string, send?: { label: string, title?: string, run: (src: string) => void }[] }} opts
   *   send: buttons under the result that pass it on to another tab
   */
  constructor(root, opts) {
    this.root = root;
    this.opts = opts;
    this.s = { ...DEFAULTS, ...loadStore(opts.storeKey, DEFAULTS) };
    this.file = null; // the picture as the file it came in (its bytes go to the server untouched, so its settings text survives)
    this.size = null; // [width, height] of the picture
    this.taskId = null;
    this.result = null; // { src, name }
    this.gpu = "";
    this.build();
  }

  q(sel) { return $(sel, this.root); }
  persist() { saveStore(this.opts.storeKey, this.s); }

  build() {
    this.root.classList.add("form-page", "upscale-page");
    this.root.innerHTML = `
      <div class="form-col">
        <section class="card">
          <label>Upscaler model <span class="row"><select class="u-model" style="flex:1"></select><button class="u-refresh" title="Look for new models in the upscaler folders">⟳</button></span></label>
          <div class="hint u-models-hint" hidden></div>
          <div class="grid2">
            <label title="The final size over the picture's size. The model runs at its own factor (2× or 4×, twice if needed) and the result is resized to this.">Scale by <input type="number" class="u-scale" min="1" max="8" step="0.25"></label>
            <label title="The picture is upscaled in overlapping pieces of this size, blended at their borders. Smaller pieces need less memory.">Tile size <select class="u-tile"></select></label>
          </div>
          <div class="hint u-size"></div>
          <label class="check" title="The whole upscale runs on the processor: the model never goes to the graphics card."><input type="checkbox" class="u-cpu"> CPU mode: use only the processor</label>
          <div class="hint u-cpu-hint"></div>
          <label class="check"><input type="checkbox" class="u-save"> Save the result to the outputs folder</label>
          <div class="row buttons">
            <button class="primary big u-go">Upscale</button>
            <button class="big u-cancel" hidden>Cancel</button>
          </div>
          <div class="u-progress progress-inline" hidden><div class="bar"></div><span></span></div>
          <div class="hint u-status"></div>
        </section>
      </div>
      <div class="result-col work">
        <div class="panes">
          <div class="pane in-pane">
            <div class="pane-head"><span class="pane-title">Picture</span>
              <span class="row"><button class="u-open-again" hidden title="Open another picture">Open…</button><button class="u-clear" hidden title="Remove the picture">✕</button></span>
            </div>
            <div class="i2i-drop u-drop">
              <div class="i2i-empty">Drop, paste or <button class="u-open">open</button> a picture<br><small>or use “Send to Upscale” from another tab</small></div>
              <img class="u-img" alt="" hidden>
              <input type="file" class="u-file" accept="image/*" hidden>
            </div>
          </div>
          <div class="pane out-pane">
            <div class="pane-head"><span class="pane-title">Result</span></div>
            <div class="result-main"><img class="u-result" alt=""><div class="r-empty">The upscaled picture appears here</div></div>
          </div>
        </div>
        <div class="row r-actions">
          <button class="u-view" disabled title="Full window">View</button>
          <button class="u-download" disabled>Download</button>
          <button class="u-again" disabled title="Upscale the result once more">Use as picture</button>
          ${(this.opts.send || []).map((b, i) => `<button class="u-send" data-i="${i}" disabled${b.title ? ` title="${escapeHtml(b.title)}"` : ""}>${escapeHtml(b.label)}</button>`).join("")}
        </div>
        <pre class="r-info u-info"></pre>
      </div>`;

    const s = this.s;
    this.q(".u-scale").value = s.scale;
    this.q(".u-cpu").checked = s.cpu;
    this.q(".u-save").checked = s.save;
    this.q(".u-model").onchange = (e) => { s.upscaler = e.target.value; this.persist(); };
    this.q(".u-scale").onchange = (e) => { s.scale = Math.min(8, Math.max(1, Number(e.target.value) || 2)); e.target.value = s.scale; this.persist(); this.showSize(); };
    this.q(".u-tile").onchange = (e) => { s.tile = Number(e.target.value); this.persist(); };
    this.q(".u-cpu").onchange = (e) => { s.cpu = e.target.checked; this.persist(); this.showDevice(); };
    this.q(".u-save").onchange = (e) => { s.save = e.target.checked; this.persist(); };
    this.q(".u-refresh").onclick = () => this.refresh(true);
    this.q(".u-go").onclick = () => this.run();
    this.q(".u-cancel").onclick = () => this.taskId && api(`${V}/upscale/${this.taskId}/cancel`, {}).catch(() => {});

    const file = this.q(".u-file"), drop = this.q(".u-drop");
    const open = () => file.click();
    this.q(".u-open").onclick = open;
    this.q(".u-open-again").onclick = open;
    this.q(".u-clear").onclick = () => this.setImage(null);
    file.onchange = () => { if (file.files[0]) this.setImage(file.files[0]); file.value = ""; };
    drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("dragging"); };
    drop.ondragleave = () => drop.classList.remove("dragging");
    drop.ondrop = (e) => {
      e.preventDefault();
      drop.classList.remove("dragging");
      const f = [...e.dataTransfer.files].find((x) => x.type.startsWith("image/"));
      if (f) this.setImage(f);
    };

    this.q(".u-view").onclick = () => this.result && showLightbox(this.result.src, this.result.name);
    this.q(".u-download").onclick = () => this.result && downloadDataURL(this.result.src, this.result.name);
    this.q(".u-again").onclick = () => this.result && this.setImage(this.result.src);
    for (const b of this.root.querySelectorAll(".u-send")) b.onclick = () => this.result && this.opts.send[Number(b.dataset.i)].run(this.result.src);
    this.showDevice();
  }

  /** Called when the tab is shown: the model list may have changed (a download, a new file). */
  show() { return this.refresh(); }

  async refresh(notify = false) {
    try {
      const r = await api(`${V}/upscale/models`);
      this.gpu = r.gpu || "";
      const s = this.s;
      if (!r.models.includes(s.upscaler)) s.upscaler = r.models[0] || "";
      fillSelect(this.q(".u-model"), r.models.map((m) => ({ value: m, label: m })), s.upscaler, { empty: r.models.length ? null : "— no upscaler model yet —" });
      const hint = this.q(".u-models-hint");
      hint.hidden = r.models.length > 0;
      hint.innerHTML = `${this.opts.modelsHint} Folder: <code>${escapeHtml(r.folders[0] || "")}</code>`;
      if (!r.tiles.includes(s.tile)) s.tile = 512;
      fillSelect(this.q(".u-tile"), r.tiles.map((t) => ({ value: String(t), label: `${t} px` })), String(s.tile));
      this.showDevice();
      if (notify) toast(`${r.models.length} upscaler model(s)`, "info");
    } catch (e) { toast(e.message); }
  }

  showDevice() {
    const box = this.q(".u-cpu"), hint = this.q(".u-cpu-hint");
    box.disabled = !this.gpu;
    if (!this.gpu) {
      box.checked = true;
      hint.textContent = "No graphics card in use here: upscaling runs on the processor.";
    } else {
      box.checked = this.s.cpu;
      hint.textContent = this.s.cpu
        ? "Slower, but the graphics card stays free: no VRAM is used, and it runs while pictures are being generated."
        : `Runs on the graphics card (${this.gpu}); it waits while a picture is being generated.`;
    }
  }

  showSize() {
    const el = this.q(".u-size");
    if (!this.size) return (el.textContent = "");
    const [w, h] = this.size;
    el.textContent = `${w}×${h} → ${Math.round(w * this.s.scale)}×${Math.round(h * this.s.scale)}`;
  }

  /** A picture to upscale: a File / Blob, or the address of one (a result of another tab). */
  async setImage(source) {
    const img = this.q(".u-img");
    if (!source) {
      this.file = this.size = null;
      img.hidden = true;
      img.removeAttribute("src");
    } else {
      try {
        this.file = typeof source === "string" ? await (await fetch(source)).blob() : source;
      } catch (e) { return toast(`Could not read the picture: ${e.message}`); }
      img.onload = () => { this.size = [img.naturalWidth, img.naturalHeight]; this.showSize(); };
      img.src = URL.createObjectURL(this.file);
      img.hidden = false;
    }
    this.q(".u-drop .i2i-empty").hidden = !!source;
    for (const cls of [".u-open-again", ".u-clear"]) this.q(cls).hidden = !source;
    this.showSize();
  }

  setBusy(busy) {
    this.q(".u-go").disabled = busy;
    this.q(".u-cancel").hidden = !busy;
    this.q(".u-progress").hidden = !busy;
  }

  async run() {
    const s = this.s;
    if (!this.file) return toast("Add a picture first");
    if (!s.upscaler) return toast("No upscaler model yet");
    this.setBusy(true);
    this.q(".u-status").textContent = "Sending the picture…";
    this.q(".u-progress .bar").style.width = "0%";
    this.q(".u-progress span").textContent = "";
    try {
      const image = await blobToDataURL(this.file);
      let t = await api(`${V}/upscale`, { image, upscaler: s.upscaler, scale: s.scale, tile: s.tile, cpu: s.cpu || !this.gpu, save: s.save });
      this.taskId = t.id;
      while (t.status === "queued" || t.status === "running") {
        this.progress(t);
        await new Promise((r) => setTimeout(r, 400));
        t = await api(`${V}/upscale/${t.id}`);
      }
      this.progress(t);
      if (t.status === "done") this.showResult(t);
      else if (t.status === "error") { toast(t.error); this.q(".u-status").textContent = t.error; }
      else this.q(".u-status").textContent = "Cancelled";
    } catch (e) {
      toast(e.message);
      this.q(".u-status").textContent = e.message;
    } finally {
      this.taskId = null;
      this.setBusy(false);
    }
  }

  progress(t) {
    const share = t.total ? t.done / t.total : 0;
    this.q(".u-progress .bar").style.width = `${Math.round(share * 100)}%`;
    this.q(".u-progress span").textContent = t.total ? `tile ${Math.min(t.done + 1, t.total)} of ${t.total}` : "";
    this.q(".u-status").textContent = `${t.phase} · ${Math.round(t.seconds)} s`;
  }

  showResult(t) {
    const src = `${V}/upscale/${t.id}/image`;
    this.result = { src, name: `upscaled-${t.upscaler}-${t.width}x${t.height}.png` };
    this.q(".u-result").src = src;
    this.q(".out-pane .r-empty").hidden = true;
    for (const b of this.root.querySelectorAll(".u-view, .u-download, .u-again, .u-send")) b.disabled = false;
    this.q(".u-status").textContent = `Done in ${Math.round(t.seconds)} s`;
    this.q(".u-info").textContent = `${t.source_width}×${t.source_height} → ${t.width}×${t.height} · ${t.upscaler} · on the ${t.device === "cpu" ? "processor" : "graphics card"} · ${Math.round(t.seconds)} s`
      + (t.saved ? `\nSaved: ${t.saved}` : "");
  }
}
