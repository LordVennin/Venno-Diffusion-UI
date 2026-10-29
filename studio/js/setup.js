// First-run setup: what a new install needs for a first picture (an Anima checkpoint, the Qwen3 text encoder and
// the Qwen-Image VAE), downloaded in one go. Opens by itself while any of them is missing, and from Settings → Models.

import { $, $$, api, escapeHtml, formatBytes, toast } from "./common.js";

const BLURBS = {
  "anima-turbo": "Recommended to start: fast (about 10 steps at CFG 1), made for live mode and smaller graphics cards.",
  "anima-aesthetic": "A polished, consistent default style. Slower: 30–50 steps at CFG 4–5.",
  "anima-base": "The most flexible: best with style LoRAs and the reference adapter. Slower: 30–50 steps at CFG 4–5.",
};
const LATER = "venno-setup-later"; // "Not now" for this browser session

export class SetupDialog {
  /** onDone(checkpointFile | null): model lists need refreshing (and the new checkpoint selecting). */
  constructor({ onDone }) {
    this.onDone = onDone;
    this.el = document.createElement("dialog");
    this.el.className = "setup";
    document.body.appendChild(this.el);
    this.timer = 0;
  }

  /** Opens when something essential is missing (or always with force); false when there was nothing to do. */
  async open(force = false) {
    if (!force && sessionStorage.getItem(LATER)) return false;
    try {
      this.state = await api("/venno/v1/setup");
    } catch {
      return false;
    }
    if (this.state.complete && !force) return false;
    this.renderChoose();
    if (!this.el.open) this.el.showModal();
    return true;
  }

  close() {
    clearTimeout(this.timer);
    if (this.el.open) this.el.close();
  }

  renderChoose() {
    const s = this.state, have = s.have;
    const models = s.models.map((m, k) => {
      const checked = !have.checkpoint && k === 0;
      return `<label class="setup-choice${m.installed ? " installed" : ""}">
        <input type="radio" name="setup-model" value="${m.id}"${checked ? " checked" : ""}${m.installed ? " disabled" : ""}>
        <span><b>${escapeHtml(m.name)}</b> <span class="muted">${m.installed ? "· installed" : `· ${formatBytes(m.size)}`}</span><br>
        <span class="muted">${escapeHtml(BLURBS[m.id] || m.description)}</span></span></label>`;
    }).join("");
    const keep = have.checkpoint ? `<label class="setup-choice"><input type="radio" name="setup-model" value="" checked><span><b>Keep my checkpoint</b><br><span class="muted">Only add what's missing below.</span></span></label>` : "";
    const required = Object.entries(s.required).map(([kind, r]) =>
      `<li>${escapeHtml(r.name)} <span class="muted">${have[kind] ? "· installed" : `· ${formatBytes(r.size)}`}</span></li>`).join("");
    const m = s.memory, gb = (mb) => `${(mb / 1024).toFixed(1)} GB`;
    const device = m.total_vram_mb
      ? `Your graphics card has ${gb(m.total_vram_mb)} of VRAM.`
      : "No graphics card was found that PyTorch can use: Venno will run on the CPU, which takes minutes per picture.";
    const suggest = s.suggested_vram_mode === "lowvram" && m.saved === "auto" && !m.from_command_line;
    this.el.innerHTML = `
      <form method="dialog" class="setup-body">
        <h2>Welcome to Venno</h2>
        <p>To make pictures, Venno needs an <b>Anima checkpoint</b> plus the <b>text encoder</b> and <b>VAE</b> every Anima model uses. Pick a checkpoint and Venno downloads everything in one go.</p>
        <h4>Checkpoint</h4>
        <div class="setup-choices">${keep}${models}</div>
        <h4>Always needed</h4>
        <ul class="setup-required">${required}</ul>
        <h4>Memory</h4>
        <p class="muted">${device}</p>
        ${suggest ? `<label class="check"><input type="checkbox" class="setup-lowvram" checked> Use <b>Low VRAM</b> mode (recommended for your card; applies after restarting Venno)</label>` : ""}
        <p class="setup-total muted"></p>
        <div class="row setup-buttons">
          <button type="button" class="primary setup-go">Download and set up</button>
          <button type="button" class="setup-manual">I'll add the files myself</button>
          <button type="button" class="setup-later">Not now</button>
        </div>
      </form>`;
    const update = () => {
      const files = this.selection();
      const total = files.reduce((sum, f) => sum + (f.size || 0), 0);
      const free = s.free_bytes;
      $(".setup-total", this.el).innerHTML = files.length
        ? `Downloads ${formatBytes(total)}${free != null ? ` · ${formatBytes(free)} free in the models folder` : ""}${free != null && free < total * 1.05 ? ` <span class="warn">· not enough space</span>` : ""}`
        : "Nothing to download.";
      $(".setup-go", this.el).disabled = !files.length && !$(".setup-lowvram", this.el)?.checked;
    };
    $$("input[name=setup-model]", this.el).forEach((r) => (r.onchange = update));
    update();
    $(".setup-go", this.el).onclick = () => this.start();
    $(".setup-manual", this.el).onclick = () => this.renderManual();
    $(".setup-later", this.el).onclick = () => { sessionStorage.setItem(LATER, "1"); this.close(); };
  }

  /** Catalog entries to download: the chosen checkpoint and whatever required file is missing. */
  selection() {
    const s = this.state;
    const id = $("input[name=setup-model]:checked", this.el)?.value;
    const model = s.models.find((m) => m.id === id && !m.installed);
    return [...(model ? [model] : []), ...Object.entries(s.required).filter(([kind]) => !s.have[kind]).map(([, r]) => r)];
  }

  async start() {
    const files = this.selection();
    const checkpoint = files.find((f) => f.kind === "checkpoint")?.filename || null;
    this.lowvram = !!$(".setup-lowvram", this.el)?.checked;
    if (this.lowvram) {
      try { await api("/venno/v1/settings", { vram_mode: "lowvram" }); } catch (e) { toast(e.message); this.lowvram = false; }
    }
    if (!files.length) return this.finish(checkpoint);
    try {
      for (const f of files) await api("/venno/v1/models/download", { id: f.id });
    } catch (e) {
      toast(`Could not start the download: ${e.message}`);
      return;
    }
    this.files = files;
    this.checkpoint = checkpoint;
    this.el.innerHTML = `
      <div class="setup-body">
        <h2>Downloading</h2>
        <p class="muted">You can keep using Venno meanwhile; downloads continue in the background (Settings → Models shows them too).</p>
        <ul class="setup-progress">${files.map((f) => `<li data-file="${escapeHtml(f.filename)}"><span>${escapeHtml(f.name)}</span><progress max="1" value="0"></progress><span class="muted setup-pct"></span></li>`).join("")}</ul>
        <div class="row setup-buttons"><button type="button" class="setup-hide">Hide</button></div>
      </div>`;
    $(".setup-hide", this.el).onclick = () => this.el.close(); // polling continues: finish() still runs
    this.poll();
  }

  poll() {
    clearTimeout(this.timer);
    this.timer = setTimeout(async () => {
      let jobs;
      try { jobs = await api("/venno/v1/models/jobs"); } catch { return this.poll(); }
      const mine = this.files.map((f) => [...jobs].reverse().find((j) => j.filename === f.filename));
      mine.forEach((j, k) => {
        const li = $(`li[data-file="${CSS.escape(this.files[k].filename)}"]`, this.el);
        if (!li || !j) return;
        const frac = j.total ? j.done / j.total : 0;
        $("progress", li).value = j.status === "done" ? 1 : frac;
        $(".setup-pct", li).textContent = j.status === "done" ? "done" : j.status === "error" ? `failed: ${j.error}` : j.status === "cancelled" ? "cancelled" : `${Math.round(frac * 100)}%`;
      });
      const failed = mine.filter((j) => j && (j.status === "error" || j.status === "cancelled"));
      if (failed.length) {
        if (!this.el.open) this.el.showModal();
        $(".setup-buttons", this.el).innerHTML = `<button type="button" class="primary setup-retry">Try again</button><button type="button" class="setup-hide">Close</button>`;
        $(".setup-retry", this.el).onclick = () => this.open(true);
        $(".setup-hide", this.el).onclick = () => this.close();
        return;
      }
      if (mine.every((j) => j && j.status === "done")) return this.finish(this.checkpoint);
      this.poll();
    }, 1000);
  }

  async finish(checkpoint) {
    await this.onDone(checkpoint);
    this.el.innerHTML = `
      <div class="setup-body">
        <h2>All set</h2>
        <p>Write a prompt and press <b>Generate</b>. ${checkpoint ? "The new checkpoint is selected, with its recommended settings." : ""}</p>
        ${this.lowvram ? `<p class="warn">Restart Venno once to switch to Low VRAM mode.</p>` : ""}
        <p class="muted">Guides (sketch to picture), upscalers, the reference adapter and other models are in ⚙ Settings → Models.</p>
        <div class="row setup-buttons"><button type="button" class="primary setup-close">Start</button></div>
      </div>`;
    if (!this.el.open) this.el.showModal();
    $(".setup-close", this.el).onclick = () => this.close();
  }

  renderManual() {
    const f = this.state.folders;
    const row = (label, path) => `<tr><th>${label}</th><td><code>${escapeHtml(path)}</code></td></tr>`;
    this.el.innerHTML = `
      <div class="setup-body">
        <h2>Add the files yourself</h2>
        <p>Put the files in these folders (sub-folders are fine), then press <b>Check again</b>. Files from a Forge / A1111 / ComfyUI install work as they are.</p>
        <table class="kv">
          ${row("Checkpoint (e.g. anima-turbo-v1.1.safetensors)", f.checkpoint)}
          ${row("Text encoder (qwen_3_06b_base.safetensors)", f.text_encoder)}
          ${row("VAE (qwen_image_vae.safetensors)", f.vae)}
        </table>
        <p class="muted">Another models folder can be used with <code>--models-dir</code> when starting Venno.</p>
        <div class="row setup-buttons">
          <button type="button" class="primary setup-again">Check again</button>
          <button type="button" class="setup-back">Back</button>
        </div>
      </div>`;
    $(".setup-back", this.el).onclick = () => this.renderChoose();
    $(".setup-again", this.el).onclick = async () => {
      await this.onDone(null);
      this.state = await api("/venno/v1/setup");
      if (this.state.complete) this.finish(null);
      else toast("Still missing: " + Object.entries(this.state.have).filter(([, v]) => !v).map(([k]) => k.replace("_", " ")).join(", "));
    };
  }
}
