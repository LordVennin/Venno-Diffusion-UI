// "Models" tab: curated downloads (Anima, guides, upscalers, detectors) and download-from-URL.

import { $, api, escapeHtml, formatBytes, toast } from "./common.js";

export class ModelsPage {
  constructor(root, { onInstalled }) {
    this.root = root;
    this.onInstalled = onInstalled; // refresh model lists after a download finishes
    this.data = null;
    this.timer = 0;
    this.known = new Set(); // job ids already reported as done
    root.innerHTML = `
      <div class="models-page">
        <section class="card intro">
          <h3>Models</h3>
          <p>Anima needs three files: a <b>checkpoint</b>, the <b>Qwen3 text encoder</b> and the <b>Qwen-Image VAE</b>. Guides (ControlLLLite) make the Canvas sketch guides work; upscalers are used by hires fix. ADetailer's face / hand detectors download automatically on first start.</p>
        </section>
        <div class="catalog"></div>
        <section class="card custom">
          <h3>Download from URL</h3>
          <div class="row">
            <input type="url" class="u-url" placeholder="https://huggingface.co/…/resolve/main/model.safetensors">
            <select class="u-kind"></select>
            <input type="text" class="u-name" placeholder="file name (optional)">
            <button class="primary u-go">Download</button>
          </div>
          <div class="hint">Hugging Face, Civitai (append <code>?token=…</code> if required) or any direct https link. Gated Hugging Face repos use the <code>HF_TOKEN</code> environment variable.</div>
        </section>
        <section class="card jobs-card" hidden>
          <h3>Downloads</h3>
          <ul class="jobs"></ul>
        </section>
      </div>`;
    $(".u-go", root).onclick = () => this.downloadURL();
  }

  async load() {
    try {
      this.data = await api("/venno/v1/models/catalog");
    } catch (e) {
      toast(`Could not load the model catalog: ${e.message}`);
      return;
    }
    const kindSel = $(".u-kind", this.root);
    if (!kindSel.options.length) kindSel.innerHTML = Object.entries(this.data.kinds).map(([k, v]) => `<option value="${k}">${escapeHtml(v)}</option>`).join("");
    $(".custom", this.root).hidden = !this.data.custom_urls;
    for (const j of this.data.jobs) if (j.status === "done") this.known.add(j.id);
    this.renderCatalog();
    this.renderJobs(this.data.jobs);
    if (this.data.jobs.some((j) => j.status === "downloading" || j.status === "queued")) this.poll();
  }

  renderCatalog() {
    const groups = {};
    for (const c of this.data.catalog) (groups[c.kind_label] ||= []).push(c);
    const jobsByFile = Object.fromEntries((this.data.jobs || []).filter((j) => j.status === "downloading" || j.status === "queued").map((j) => [j.filename, j]));
    $(".catalog", this.root).innerHTML = Object.entries(groups).map(([label, items]) => `
      <section class="card">
        <h3>${escapeHtml(label)}</h3>
        <ul class="catalog-list">
          ${items.map((c) => `
            <li>
              <div class="c-text"><b>${escapeHtml(c.name)}</b> <span class="muted">${formatBytes(c.size)}</span><div class="muted">${escapeHtml(c.description)}</div></div>
              ${c.installed ? `<span class="badge">Installed</span>` : jobsByFile[c.filename] ? `<span class="badge busy">Downloading…</span>` : `<button class="c-get" data-id="${c.id}">Download</button>`}
            </li>`).join("")}
        </ul>
      </section>`).join("");
    this.root.querySelectorAll(".c-get").forEach((b) => (b.onclick = () => this.download({ id: b.dataset.id })));
  }

  renderJobs(jobs) {
    const card = $(".jobs-card", this.root);
    card.hidden = !jobs.length;
    $(".jobs", this.root).innerHTML = jobs.map((j) => {
      const pct = j.total ? Math.round((j.done / j.total) * 100) : 0;
      const status = j.status === "downloading" ? `${pct}% · ${formatBytes(j.done)} of ${formatBytes(j.total)}` : j.status === "error" ? `failed: ${escapeHtml(j.error)}` : j.status;
      return `<li class="job ${j.status}">
        <div class="j-text"><b>${escapeHtml(j.name)}</b> <span class="muted">${escapeHtml(j.filename)}</span><div class="muted">${status}</div>
        ${j.status === "downloading" ? `<div class="progress-inline"><div class="bar" style="width:${pct}%"></div></div>` : ""}</div>
        ${j.status === "downloading" || j.status === "queued" ? `<button class="j-cancel" data-id="${j.id}">Cancel</button>` : ""}
      </li>`;
    }).join("");
    this.root.querySelectorAll(".j-cancel").forEach((b) => (b.onclick = () => api("/venno/v1/models/cancel", { id: b.dataset.id }).then((js) => this.renderJobs(js))));
  }

  async download(body) {
    try {
      await api("/venno/v1/models/download", body);
      toast("Download started", "info");
      await this.load();
      this.poll();
    } catch (e) {
      toast(e.message);
    }
  }

  downloadURL() {
    const url = $(".u-url", this.root).value.trim();
    if (!url) return toast("Paste a download link first");
    this.download({ url, kind: $(".u-kind", this.root).value, filename: $(".u-name", this.root).value.trim() || undefined });
  }

  poll() {
    clearTimeout(this.timer);
    this.timer = setTimeout(async () => {
      let jobs;
      try { jobs = await api("/venno/v1/models/jobs"); } catch { return this.poll(); }
      this.renderJobs(jobs);
      const finished = jobs.filter((j) => j.status === "done" && !this.known.has(j.id));
      if (finished.length) {
        finished.forEach((j) => { this.known.add(j.id); toast(`Installed ${j.filename}`, "info"); });
        await this.onInstalled();
        await this.load();
        return;
      }
      if (jobs.some((j) => j.status === "downloading" || j.status === "queued")) this.poll();
      else this.load();
    }, 1000);
  }
}
