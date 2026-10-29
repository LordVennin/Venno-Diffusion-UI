// Settings (⚙ in the top bar): Models (downloads), Studio (preferences), Plugins (coming) and System (version, device, folders).
// New sections go in SECTIONS and get a <section data-section="…"> in the page.

import { warmTags } from "./autocomplete.js";
import { refreshTokenCounters } from "./tokens.js";
import { $, $$, api, escapeHtml, prefs, setPref, toast } from "./common.js";

const SECTIONS = [
  { key: "models", label: "Models", hint: "Download checkpoints, guides, upscalers" },
  { key: "studio", label: "Studio", hint: "Prompt boxes and other preferences" },
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
        <section data-section="studio" hidden>
          <div class="settings-inner">
            <section class="card">
              <h3>Prompt boxes</h3>
              <label class="check"><input type="checkbox" class="pref-autocomplete"> Tag autocomplete</label>
              <p>Suggests Danbooru tags (and <code>__wildcard__</code> files after <code>__</code>) while you type in any prompt box.</p>
              <label class="check"><input type="checkbox" class="pref-tokens"> Token counter</label>
              <p>Shows each prompt's length in tokens in the corner of its box (Anima is trained on prompts up to 512). Saved in this browser.</p>
            </section>
          </div>
        </section>
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
    const ac = $(".pref-autocomplete", root);
    ac.checked = prefs.autocomplete;
    ac.onchange = () => { setPref("autocomplete", ac.checked); warmTags(); };
    const tc = $(".pref-tokens", root);
    tc.checked = prefs.tokenCounter;
    tc.onchange = () => { setPref("tokenCounter", tc.checked); refreshTokenCounters(); };
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
      <section class="card s-updates">
        <h3>Updates</h3>
        <div class="row"><button class="s-check">Check for updates</button><span class="muted s-update-state">Venno ${escapeHtml(s.version)} · updates come from the <code>main</code> branch on GitHub</span></div>
        <div class="s-update-result"></div>
      </section>
      <section class="card s-memory">
        <h3>Memory</h3>
        ${memoryCard(s.memory)}
      </section>
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
    $(".s-check", box).onclick = () => this.checkUpdates();
    $$(".s-memory input[name=vram]", box).forEach((input) => (input.onchange = async () => {
      try {
        const r = await api("/venno/v1/settings", { vram_mode: input.value });
        $(".s-memory-state", box).innerHTML = memoryState(r.memory);
        await this.o.onRescan(); // refreshes the restart banner
      } catch (e) {
        toast(e.message);
      }
    }));
  }

  async checkUpdates() {
    const box = $(".s-update-result", this.root);
    const button = $(".s-check", this.root);
    button.disabled = true;
    box.innerHTML = `<p class="muted">Checking GitHub…</p>`;
    let u;
    try {
      u = await api("/venno/v1/update");
    } catch (e) {
      box.innerHTML = `<p class="warn">${escapeHtml(e.message)}</p>`;
      return;
    } finally {
      button.disabled = false;
    }
    const notes = u.changes.map((c) => `<h4>${escapeHtml(c.version)}</h4><pre class="s-notes">${escapeHtml(c.notes)}</pre>`).join("");
    const commits = u.commits?.length
      ? `<details><summary>${u.commits.length}${u.commits.length >= 50 ? "+" : ""} change${u.commits.length > 1 ? "s" : ""}</summary><ul class="s-commits">${u.commits.map((c) => `<li><code>${escapeHtml(c.id)}</code> ${escapeHtml(c.title)}</li>`).join("")}</ul></details>`
      : "";
    if (!u.update_available) {
      box.innerHTML = `<p>Venno ${escapeHtml(u.version)} is up to date.</p>${u.reason && u.reason !== "Up to date." ? `<p class="muted">${escapeHtml(u.reason)}</p>` : ""}`;
      return;
    }
    const title = u.latest && u.latest !== u.version ? `Venno ${escapeHtml(u.latest)} is available (you have ${escapeHtml(u.version)}).` : "Updates are available.";
    const action = u.can_update
      ? `<div class="row"><button class="primary s-update">Update</button><span class="muted">Downloads the new version with git; restart Venno afterwards to use it.</span></div>`
      : u.download ? `<p><a href="${escapeHtml(u.download)}" target="_blank" rel="noopener">Download the new version</a></p>` : "";
    box.innerHTML = `<p><b>${title}</b></p>${notes}${commits}${u.reason ? `<p class="muted">${escapeHtml(u.reason)}</p>` : ""}${action}`;
    const update = $(".s-update", box);
    if (update) update.onclick = async () => {
      update.disabled = true;
      update.textContent = "Updating…";
      try {
        const r = await api("/venno/v1/update", {});
        box.innerHTML = `<p><b>Updated to ${escapeHtml(r.to || r.commit)}.</b> Restart Venno (close it and start it again) to use the new version; changed requirements are installed on that start.</p>`;
        toast("Updated: restart Venno to use the new version", "info");
      } catch (e) {
        update.disabled = false;
        update.textContent = "Update";
        toast(e.message);
      }
    };
  }
}

const VRAM_CHOICES = [
  ["auto", "Automatic", "Loads as much of a model as fits in VRAM and streams the rest when it runs short. Right for most cards."],
  ["lowvram", "Low VRAM", "Always splits the diffusion model and keeps only the part being used on the GPU. For about 6–8 GB, or larger models (Anima 3.8B) on mid-size cards. Slower."],
  ["novram", "No VRAM", "Keeps almost nothing on the GPU between uses. For 4 GB or less, or when other programs need the VRAM. Slowest."],
];

/** Settings → System → Memory: the memory mode for the next start. */
function memoryCard(m) {
  const gb = (mb) => `${(mb / 1024).toFixed(1)} GB`;
  const choices = VRAM_CHOICES.map(([value, label, hint]) => `
    <label class="s-choice"><input type="radio" name="vram" value="${value}"${m.saved === value ? " checked" : ""}${m.from_command_line ? " disabled" : ""}>
      <span><b>${label}</b><br><span class="muted">${hint}</span></span></label>`).join("");
  return `<p class="muted">How models use the graphics card's memory (${m.total_vram_mb ? `${gb(m.total_vram_mb)} VRAM` : "no graphics card memory found"}, ${gb(m.total_ram_mb)} RAM). Applies from the next start.</p>
    <div class="s-choices">${choices}</div>
    <p class="s-memory-state">${memoryState(m)}</p>`;
}

function memoryState(m) {
  const name = (v) => (VRAM_CHOICES.find(([value]) => value === v) || [v, v])[1];
  if (m.from_command_line) return `<span class="muted">Set on the command line (<code>--${escapeHtml(m.running)}</code>), which overrides this setting: remove it from the command line (or VENNO_ARGS) to choose here.</span>`;
  if (m.saved !== m.running) return `<span class="warn">Saved: restart Venno to switch from ${escapeHtml(name(m.running))} to ${escapeHtml(name(m.saved))}.</span>`;
  return `<span class="muted">In use now: ${escapeHtml(name(m.running))}.</span>`;
}
