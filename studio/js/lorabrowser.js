// LoRA browser: a pop-up with the LoRA folders as tiles you open, preview cards, search, tags, and an editor for
// each LoRA's preview image, tags, trigger words, preferred weight and notes (saved next to the file, as in Forge:
// venno/engine/lora_library.py). Clicking a card adds the LoRA to the prompt it was opened from, or removes it.

import { $, $$, api, escapeHtml, loadStore, saveStore, toast } from "./common.js";
import { DEFAULT_WEIGHT } from "./loras.js";

const VIEW_KEY = "venno-lora-browser";
const SORTS = { name: "Name", newest: "Newest", folder: "Folder" };

const readDataURL = (blob) => new Promise((ok, fail) => {
  const r = new FileReader();
  r.onload = () => ok(r.result);
  r.onerror = fail;
  r.readAsDataURL(blob);
});

const initials = (name) => name.replace(/^(style|artist|concept|outfit|oc|ba|genshin)_/i, "").split(/[_\-\s]+/).filter(Boolean).slice(0, 2).map((w) => w[0].toUpperCase()).join("") || "?";
const sizeText = (n) => (n >= 1e9 ? `${(n / 1e9).toFixed(1)} GB` : `${Math.max(1, Math.round(n / 1e6))} MB`);

class LoraBrowser {
  constructor() {
    this.items = [];
    this.view = loadStore(VIEW_KEY, { folder: "", sort: "name", size: 160 });
    this.query = "";
    this.tags = new Set();
    this.editing = null; // name of the LoRA in the editor
    this.opts = null; // { chips, prompt?, current? } of the tab that opened the browser
    const el = (this.el = document.createElement("div"));
    el.id = "lora-browser";
    el.hidden = true;
    el.innerHTML = `
      <div class="lb-win" role="dialog" aria-label="LoRA browser">
        <header>
          <h2>LoRAs</h2>
          <input type="search" class="lb-search" placeholder="Search names, tags, trigger words, notes…" data-noac>
          <select class="lb-sort">${Object.entries(SORTS).map(([k, v]) => `<option value="${k}">${v}</option>`).join("")}</select>
          <input type="range" class="lb-size" min="110" max="260" step="10" title="Card size">
          <span class="lb-count"></span>
          <button class="lb-refresh" title="Rescan the LoRA folders">⟳</button>
          <button class="lb-close" title="Close (Esc)">×</button>
        </header>
        <div class="lb-body">
          <aside class="lb-side">
            <h4>Folders</h4><div class="lb-tree"></div>
            <h4>Tags <button class="lb-clear-tags" hidden>clear</button></h4><div class="lb-tags"></div>
          </aside>
          <main class="lb-main">
            <div class="lb-crumbs"></div>
            <div class="lb-grid"></div>
          </main>
          <section class="lb-edit" hidden></section>
        </div>
      </div>`;
    document.body.appendChild(el);
    el.addEventListener("mousedown", (e) => { if (e.target === el) this.close(); });
    $(".lb-close", el).onclick = () => this.close();
    $(".lb-refresh", el).onclick = () => this.load();
    const search = $(".lb-search", el);
    search.oninput = () => { this.query = search.value.trim().toLowerCase(); this.render(); };
    $(".lb-sort", el).onchange = (e) => { this.view.sort = e.target.value; this.saveView(); this.render(); };
    $(".lb-size", el).oninput = (e) => { this.view.size = Number(e.target.value); this.saveView(); this.applySize(); };
    $(".lb-clear-tags", el).onclick = () => { this.tags.clear(); this.render(); };
    document.addEventListener("keydown", (e) => {
      if (this.el.hidden || e.key !== "Escape") return;
      e.stopPropagation();
      if (this.editing) this.closeEditor();
      else this.close();
    }, true);
    // the grid: open folders, toggle LoRAs, edit
    $(".lb-grid", el).addEventListener("click", (e) => {
      const folder = e.target.closest(".lb-folder");
      if (folder) return this.openFolder(folder.dataset.folder);
      const card = e.target.closest(".lb-card");
      if (!card) return;
      if (e.target.closest(".lb-edit-btn")) return this.openEditor(card.dataset.name);
      this.toggle(card.dataset.name);
    });
    $(".lb-grid", el).addEventListener("contextmenu", (e) => {
      const card = e.target.closest(".lb-card");
      if (!card) return;
      e.preventDefault();
      this.openEditor(card.dataset.name);
    });
    $(".lb-tree", el).addEventListener("click", (e) => { const b = e.target.closest("[data-folder]"); if (b) this.openFolder(b.dataset.folder); });
    $(".lb-crumbs", el).addEventListener("click", (e) => { const b = e.target.closest("[data-folder]"); if (b) this.openFolder(b.dataset.folder); });
    $(".lb-tags", el).addEventListener("click", (e) => {
      const b = e.target.closest("[data-tag]");
      if (!b) return;
      this.tags.has(b.dataset.tag) ? this.tags.delete(b.dataset.tag) : this.tags.add(b.dataset.tag);
      this.render();
    });
  }

  saveView() { saveStore(VIEW_KEY, this.view); }

  /** opts: { chips: LoraChips, prompt?: textarea for trigger words, current?: () => image src of the selected result } */
  async open(opts) {
    this.opts = opts;
    this.el.hidden = false;
    $(".lb-sort", this.el).value = this.view.sort;
    $(".lb-size", this.el).value = this.view.size;
    this.applySize();
    $(".lb-search", this.el).focus();
    if (!this.items.length) await this.load();
    else this.render();
  }

  close() {
    this.closeEditor();
    this.el.hidden = true;
  }

  applySize() { this.el.style.setProperty("--lb-card", `${this.view.size}px`); }

  async load() {
    try {
      this.items = await api("/venno/v1/loras");
    } catch (e) {
      toast(`Could not list LoRAs: ${e.message}`);
      this.items = [];
    }
    if (this.view.folder && !this.items.some((i) => i.folder === this.view.folder || i.folder.startsWith(`${this.view.folder}/`))) this.view.folder = "";
    this.render();
  }

  openFolder(folder) {
    this.view.folder = folder;
    this.saveView();
    this.render();
    $(".lb-main", this.el).scrollTop = 0;
  }

  inUse(name) { return this.opts?.chips.list.find((l) => l.name === name); }

  toggle(name) {
    const chips = this.opts?.chips;
    if (!chips) return;
    const item = this.items.find((i) => i.name === name);
    if (this.inUse(name)) {
      chips.remove(name);
    } else {
      chips.add(name, item?.weight || DEFAULT_WEIGHT);
      this.addTriggerWords(item);
    }
    this.renderGrid();
  }

  /** Trigger words go into the prompt the browser was opened from, once (as Forge does with "activation text"). */
  addTriggerWords(item) {
    const ta = this.opts?.prompt, words = item?.activation?.trim();
    if (!ta || !words) return;
    if (ta.value.toLowerCase().includes(words.toLowerCase())) return;
    ta.value = ta.value.trim() ? `${ta.value.replace(/[\s,]+$/, "")}, ${words}` : words;
    ta.dispatchEvent(new Event("input", { bubbles: true }));
  }

  // ---------------- filtering ----------------

  inFolder(item, folder = this.view.folder) {
    return !folder || item.folder === folder || item.folder.startsWith(`${folder}/`);
  }

  matches(item) {
    if (this.tags.size && ![...this.tags].every((t) => item.tags.includes(t))) return false;
    if (!this.query) return true;
    const hay = `${item.name} ${item.folder} ${item.tags.join(" ")} ${item.activation} ${item.notes} ${item.description} ${item.base}`.toLowerCase();
    return this.query.split(/\s+/).every((w) => hay.includes(w.replace(/_/g, " ")) || hay.includes(w));
  }

  sorted(list) {
    const by = this.view.sort;
    return [...list].sort((a, b) => (by === "newest" ? b.modified - a.modified : by === "folder" ? a.folder.localeCompare(b.folder) || a.name.localeCompare(b.name) : a.name.localeCompare(b.name, undefined, { sensitivity: "base" })));
  }

  /** Sub-folders directly inside `folder`: [{ path, name, count, previews }] */
  subfolders(folder) {
    const out = new Map();
    for (const item of this.items) {
      if (!this.inFolder(item, folder) || item.folder === folder) continue;
      const rest = folder ? item.folder.slice(folder.length + 1) : item.folder;
      const name = rest.split("/")[0];
      const path = folder ? `${folder}/${name}` : name;
      const f = out.get(path) || { path, name, count: 0, previews: [] };
      f.count++;
      if (item.preview && f.previews.length < 4) f.previews.push(item.preview);
      out.set(path, f);
    }
    return [...out.values()].sort((a, b) => a.name.localeCompare(b.name));
  }

  // ---------------- rendering ----------------

  render() {
    this.renderTree();
    this.renderTags();
    this.renderGrid();
  }

  renderTree() {
    const counts = new Map();
    for (const item of this.items) {
      const parts = item.folder ? item.folder.split("/") : [];
      for (let i = 1; i <= parts.length; i++) {
        const p = parts.slice(0, i).join("/");
        counts.set(p, (counts.get(p) || 0) + 1);
      }
    }
    const rows = [`<button data-folder="" class="${this.view.folder === "" ? "on" : ""}">All LoRAs <small>${this.items.length}</small></button>`];
    const cur = this.view.folder;
    for (const p of [...counts.keys()].sort()) {
      const depth = p.split("/").length;
      const parent = p.split("/").slice(0, -1).join("/");
      if (parent && cur !== parent && !cur.startsWith(`${parent}/`)) continue; // inside a folder that isn't open
      rows.push(`<button data-folder="${escapeHtml(p)}" class="${this.view.folder === p ? "on" : ""}" style="padding-left:${8 + depth * 12}px">📁 ${escapeHtml(p.split("/").pop())} <small>${counts.get(p)}</small></button>`);
    }
    $(".lb-tree", this.el).innerHTML = rows.join("");
  }

  renderTags() {
    const counts = new Map();
    for (const item of this.items) if (this.inFolder(item)) for (const t of item.tags) counts.set(t, (counts.get(t) || 0) + 1);
    for (const t of this.tags) if (!counts.has(t)) counts.set(t, 0);
    const tags = [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
    $(".lb-tags", this.el).innerHTML = tags.length
      ? tags.map(([t, n]) => `<button data-tag="${escapeHtml(t)}" class="${this.tags.has(t) ? "on" : ""}">${escapeHtml(t)} <small>${n}</small></button>`).join("")
      : `<p class="muted">No tags yet. Right-click a LoRA (or its ✎) to add some.</p>`;
    $(".lb-clear-tags", this.el).hidden = !this.tags.size;
  }

  renderGrid() {
    const folder = this.view.folder;
    const filtering = !!(this.query || this.tags.size);
    const crumbs = [`<button data-folder="">All LoRAs</button>`];
    if (folder) folder.split("/").forEach((part, i, all) => crumbs.push(`<span>›</span><button data-folder="${escapeHtml(all.slice(0, i + 1).join("/"))}">${escapeHtml(part)}</button>`));
    if (filtering) crumbs.push(`<span class="muted">· ${folder ? "results in this folder and below" : "results in all folders"}</span>`);
    $(".lb-crumbs", this.el).innerHTML = crumbs.join("");

    let tiles = "";
    let list;
    if (filtering) {
      list = this.sorted(this.items.filter((i) => this.inFolder(i) && this.matches(i))); // flat: every match below this folder
    } else {
      tiles = this.subfolders(folder).map((f) => `
        <div class="lb-folder" data-folder="${escapeHtml(f.path)}" title="Open ${escapeHtml(f.path)}">
          <div class="lb-mosaic n${f.previews.length}">${f.previews.map((p) => `<img src="${p}" loading="lazy" alt="">`).join("") || "<span>📁</span>"}</div>
          <div class="lb-name">📁 ${escapeHtml(f.name)}</div><div class="lb-meta">${f.count} LoRA${f.count === 1 ? "" : "s"}</div>
        </div>`).join("");
      list = this.sorted(this.items.filter((i) => i.folder === folder));
    }
    const cards = list.map((i) => this.card(i, filtering)).join("");
    const grid = $(".lb-grid", this.el);
    grid.innerHTML = tiles + cards || `<p class="lb-empty">${this.items.length ? "No LoRA matches." : "No LoRAs found. Put .safetensors files in models/Lora (sub-folders become folders here) and press ⟳."}</p>`;
    $(".lb-count", this.el).textContent = filtering ? `${list.length} of ${this.items.length}` : `${this.items.length} LoRAs`;
  }

  card(i, showFolder) {
    const used = this.inUse(i.name);
    const img = i.preview ? `<img src="${i.preview}" loading="lazy" alt="">` : `<span class="lb-initials">${escapeHtml(initials(i.name))}</span>`;
    const meta = [showFolder && i.folder ? `📁 ${escapeHtml(i.folder)}` : "", i.base ? escapeHtml(i.base.replace(/^.*\//, "")) : ""].filter(Boolean).join(" · ");
    return `
      <div class="lb-card${used ? " used" : ""}" data-name="${escapeHtml(i.name)}" title="${escapeHtml(i.name)}${i.activation ? `\nTrigger words: ${escapeHtml(i.activation)}` : ""}\nClick to ${used ? "remove" : "add"} · right-click to edit">
        <div class="lb-img">${img}${used ? `<span class="lb-badge">✓ ${used.weight}</span>` : ""}</div>
        <button class="lb-edit-btn" title="Edit preview, tags and notes">✎</button>
        <div class="lb-name">${escapeHtml(i.name)}</div>
        ${meta ? `<div class="lb-meta">${meta}</div>` : ""}
        ${i.tags.length ? `<div class="lb-ctags">${i.tags.slice(0, 4).map((t) => `<span>${escapeHtml(t)}</span>`).join("")}</div>` : ""}
      </div>`;
  }

  // ---------------- editor ----------------

  openEditor(name) {
    const i = this.items.find((x) => x.name === name);
    if (!i) return;
    this.editing = name;
    const ed = $(".lb-edit", this.el);
    ed.hidden = false;
    const current = this.opts?.current?.();
    ed.innerHTML = `
      <div class="lb-edit-head"><h3 title="${escapeHtml(i.file)}">${escapeHtml(i.name)}</h3><button class="lb-edit-close" title="Close">×</button></div>
      <div class="lb-meta">${i.folder ? `📁 ${escapeHtml(i.folder)} · ` : ""}${escapeHtml(i.file)} · ${sizeText(i.size)}${i.base ? ` · ${escapeHtml(i.base)}` : ""}</div>
      <div class="lb-edit-img" title="Drop or paste a picture">${i.preview ? `<img src="${i.preview}" alt="">` : `<span>No preview<br><small>drop or paste a picture here</small></span>`}</div>
      <div class="row">
        <button class="lb-up">Upload…</button>
        ${current ? `<button class="lb-cur" title="The result selected in the tab you came from">Use current result</button>` : ""}
        ${i.preview ? `<button class="lb-rm">Remove</button>` : ""}
      </div>
      <label>Tags <span class="muted">(comma or Enter)</span>
        <div class="lb-tagedit">${i.tags.map((t) => `<span data-t="${escapeHtml(t)}">${escapeHtml(t)} <b>×</b></span>`).join("")}<input class="lb-taginput" list="lb-taglist" placeholder="add a tag" data-noac></div>
      </label>
      <datalist id="lb-taglist">${[...new Set(this.items.flatMap((x) => x.tags))].sort().map((t) => `<option value="${escapeHtml(t)}">`).join("")}</datalist>
      <label>Trigger words <span class="muted">(added to the prompt with the LoRA)</span><textarea class="lb-act" rows="2" data-noac>${escapeHtml(i.activation || "")}</textarea></label>
      <label>Preferred weight <span class="muted">(0 = default ${DEFAULT_WEIGHT})</span><input type="number" class="lb-w" min="-2" max="3" step="0.05" value="${i.weight ?? 0}"></label>
      <label>Notes<textarea class="lb-notes" rows="4">${escapeHtml(i.notes || "")}</textarea></label>
      <div class="row"><button class="primary lb-save">Save</button><span class="muted lb-saved"></span></div>`;
    const tags = new Set(i.tags);
    const tagBox = $(".lb-tagedit", ed), tagInput = $(".lb-taginput", ed);
    const renderTags = () => {
      $$("span", tagBox).forEach((s) => s.remove());
      tagInput.insertAdjacentHTML("beforebegin", [...tags].map((t) => `<span data-t="${escapeHtml(t)}">${escapeHtml(t)} <b>×</b></span>`).join(""));
    };
    const addTag = () => {
      for (const t of tagInput.value.split(",").map((x) => x.trim()).filter(Boolean)) tags.add(t);
      tagInput.value = "";
      renderTags();
    };
    tagInput.onkeydown = (e) => {
      if (e.key === "Enter" || e.key === ",") { e.preventDefault(); addTag(); }
      if (e.key === "Backspace" && !tagInput.value && tags.size) { tags.delete([...tags].pop()); renderTags(); }
    };
    tagInput.onchange = addTag;
    tagBox.onclick = (e) => { const s = e.target.closest("span[data-t]"); if (s) { tags.delete(s.dataset.t); renderTags(); } else tagInput.focus(); };
    $(".lb-edit-close", ed).onclick = () => this.closeEditor();
    $(".lb-save", ed).onclick = async () => {
      addTag();
      try {
        const updated = await api("/venno/v1/loras/meta", {
          name, tags: [...tags], activation: $(".lb-act", ed).value, weight: Number($(".lb-w", ed).value) || 0, notes: $(".lb-notes", ed).value,
        });
        this.replace(updated);
        $(".lb-saved", ed).textContent = "Saved";
      } catch (e) { toast(e.message); }
    };
    // preview: upload, paste, drop, current result, remove
    const setPreview = async (image) => {
      try {
        this.replace(await api("/venno/v1/loras/preview", { name, image }));
        this.openEditor(name);
      } catch (e) { toast(e.message); }
    };
    const picker = document.createElement("input");
    picker.type = "file";
    picker.accept = "image/png,image/jpeg,image/webp";
    picker.onchange = async () => { if (picker.files[0]) await setPreview(await readDataURL(picker.files[0])); };
    $(".lb-up", ed).onclick = () => picker.click();
    $(".lb-rm", ed)?.addEventListener("click", () => setPreview(null));
    $(".lb-cur", ed)?.addEventListener("click", async () => {
      const src = this.opts.current();
      setPreview(src.startsWith("data:") ? src : await readDataURL(await (await fetch(src)).blob()));
    });
    const box = $(".lb-edit-img", ed);
    box.ondragover = (e) => e.preventDefault();
    box.ondrop = async (e) => {
      e.preventDefault();
      const f = [...(e.dataTransfer?.files || [])].find((x) => x.type.startsWith("image/"));
      if (f) await setPreview(await readDataURL(f));
    };
    ed.onpaste = async (e) => {
      const f = [...(e.clipboardData?.files || [])].find((x) => x.type.startsWith("image/"));
      if (!f) return;
      e.preventDefault();
      await setPreview(await readDataURL(f));
    };
    this.renderGrid();
    $$(".lb-card", this.el).forEach((c) => c.classList.toggle("editing", c.dataset.name === name));
  }

  closeEditor() {
    this.editing = null;
    $(".lb-edit", this.el).hidden = true;
    $$(".lb-card.editing", this.el).forEach((c) => c.classList.remove("editing"));
  }

  replace(updated) {
    if (!updated?.name) return;
    this.items = this.items.map((x) => (x.name === updated.name ? updated : x));
    this.renderTree();
    this.renderTags();
    this.renderGrid();
    $$(".lb-card", this.el).forEach((c) => c.classList.toggle("editing", c.dataset.name === this.editing));
  }
}

let browser = null;

/** Open the LoRA browser for a prompt's LoRA chips. */
export function openLoraBrowser(opts) {
  browser = browser || new LoraBrowser();
  return browser.open(opts);
}
