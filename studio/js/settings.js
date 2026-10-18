// Settings (⚙ in the top bar): Models (downloads), Plugins (coming) and System (version, device, folders).
// New sections go in SECTIONS and get a <section data-section="…"> in the page.

import { $, $$, api, escapeHtml, toast } from "./common.js";

const SECTIONS = [
  { key: "models", label: "Models", hint: "Download checkpoints, guides, upscalers" },
  { key: "plugins", label: "Plugins", hint: "Coming soon" },
  { key: "system", label: "System", hint: "Version, device, folders" },
];

export class SettingsPage {
  /** opts: { modelsPage, onRescan } */
  constructor(root, opts) {
    this.root = root;
    this.o = opts;
    this.section = "models";
    root.innerHTML = `
      <nav class="settings-nav">
        <h2>Settings</h2>
        ${SECTIONS.map((s) => `<button data-section="${s.key}"><b>${s.label}</b><small>${s.hint}</small></button>`).join("")}
      </nav>
      <div class="settings-body">
        <section data-section="models" class="settings-models"></section>
        <section data-section="plugins" hidden>
          <div class="settings-inner">
            <section class="card">
              <h3>Plugins</h3>
              <p>Venno will get its own plugin system: add-ons dropped into a <code>plugins/</code> folder that can add generation options, panels and tools to the Studio. It is not available yet; this is where installed plugins will be listed and switched on or off.</p>
              <p>Extensions made for Forge or AUTOMATIC1111 are not supported.</p>
            </section>
          </div>
        </section>
        <section data-section="system" hidden><div class="settings-inner system-info"></div></section>
      </div>`;
    $$(".settings-nav button", root).forEach((b) => (b.onclick = () => this.show(b.dataset.section)));
  }

  /** The element the Models page renders into. */
  get modelsRoot() { return $(".settings-models", this.root); }

  show(section = this.section) {
    this.section = section;
    $$(".settings-nav button", this.root).forEach((b) => b.classList.toggle("on", b.dataset.section === section));
    $$(".settings-body > section", this.root).forEach((s) => (s.hidden = s.dataset.section !== section));
    if (section === "models") this.o.modelsPage.load();
    if (section === "system") this.loadSystem();
  }

  async loadSystem() {
    const box = $(".system-info", this.root);
    let s;
    try {
      s = await api("/venno/v1/system");
    } catch (e) {
      box.innerHTML = `<section class="card"><p>Could not load system information: ${escapeHtml(e.message)}</p></section>`;
      return;
    }
    const row = (k, v) => `<tr><th>${k}</th><td>${v}</td></tr>`;
    const path = (p) => `<code>${escapeHtml(p)}</code>`;
    box.innerHTML = `
      <section class="card">
        <h3>Venno</h3>
        <table class="kv">
          ${row("Version", escapeHtml(s.version))}
          ${row("Device", escapeHtml(s.device))}
          ${row("Python / PyTorch", `${escapeHtml(s.python)} · ${escapeHtml(s.torch)}`)}
          ${row("Loaded model", escapeHtml(s.loaded_model || "none yet (loads on the first generation)"))}
          ${row("Address", `${escapeHtml(s.listen)}:${s.port}`)}
          ${row("Started with", s.command_line ? path(s.command_line) : "<span class=\"muted\">no options</span>")}
        </table>
        <p class="muted">Options are set when starting Venno, e.g. <code>venno.bat --listen --lowvram</code>; <code>--help</code> lists them all.</p>
      </section>
      <section class="card">
        <h3>Folders</h3>
        <table class="kv">
          ${row("Models", path(s.models))}
          ${row("Outputs", path(s.outputs))}
          ${row("Data (wildcards, tags)", path(s.data))}
        </table>
        <table class="kv folders">
          <tr><th></th><th>New files go to</th><th>Also searched</th></tr>
          ${s.folders.map((f) => `<tr><th>${escapeHtml(f.kind)}</th><td>${path(f.new_files)}</td><td>${f.searched.filter((d) => d !== f.new_files).map(path).join("<br>") || "<span class=\"muted\">—</span>"}</td></tr>`).join("")}
        </table>
        <div class="row"><button class="s-rescan">Rescan model folders</button><span class="muted">after adding files by hand</span></div>
      </section>`;
    $(".s-rescan", box).onclick = async () => { await this.o.onRescan(); toast("Model folders rescanned", "info"); };
  }
}
