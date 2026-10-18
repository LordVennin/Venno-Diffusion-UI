// "Comparison grid (X/Y/Z)" card for the Text to Image / Image to Image tabs.
// The server parses the values (same parser as generation) to show "3 × 4 = 12 images" while typing.

import { api, escapeHtml } from "./common.js";

export const AXES = [
  { key: "none", label: "—" },
  { key: "artist", label: "Artist", hint: "wlop, ebifurya, \"someone (else)\"  — added as @artist, or replaces @ARTIST in the prompt" },
  { key: "prompt_sr", label: "Prompt S/R", hint: "first value is searched in the prompt and replaced by each value: red hair, blue hair, green hair" },
  { key: "cfg", label: "CFG", hint: "3, 4, 5  or  2-6 (+1)  or  2-6 [5]" },
  { key: "steps", label: "Steps", hint: "10, 20, 30  or  10-30 (+10)" },
  { key: "shift", label: "Shift", hint: "2, 3, 4  or  1-5 [5]" },
  { key: "seed", label: "Seed", hint: "1, 2, 3  or  100-104" },
  { key: "sampler", label: "Sampler", list: "samplers" },
  { key: "scheduler", label: "Scheduler", list: "schedulers" },
  { key: "checkpoint", label: "Checkpoint", list: "checkpoints" },
  { key: "lora_weight", label: "LoRA weight", hint: "0, 0.4, 0.8, 1.2  or  0-1.2 [4]" },
  { key: "denoise", label: "Denoising", hint: "0.3, 0.45, 0.6 (hires denoise when hires fix is on)" },
  { key: "hires_upscaler", label: "Hires upscaler", list: "upscalers" },
  { key: "size", label: "Size", hint: "832x1216, 1024x1024, 1216x832" },
  { key: "switch_at", label: "Switch at", hint: "0.2, 0.4, 0.6 (needs a 'then switch to' prompt)" },
  { key: "detailer_denoise", label: "Detailer denoise", hint: "0.3, 0.4, 0.5" },
  { key: "region_base", label: "Main prompt in regions", hint: "0, 0.25, 0.5 (Canvas regions)" },
  { key: "region_separation", label: "Region separation", hint: "0, 0.5, 1 (Canvas regions)" },
  { key: "region_refine", label: "Region pass", hint: "0, 0.35, 0.5 (denoising, 0 = off; Canvas regions)" },
  { key: "frame_separation", label: "Frame separation", hint: "0, 0.3, 0.5 (consistent sets with frame texts)" },
];

export const GRID_DEFAULT = {
  enabled: false,
  cells: false,
  axes: [{ type: "artist", values: "", target: "" }, { type: "none", values: "", target: "" }, { type: "none", values: "", target: "" }],
};

const listValues = (info, list) => {
  if (list === "checkpoints") return (info.checkpoints || []).map((c) => c.name);
  if (list === "samplers" && info.engine_samplers?.length) return info.engine_samplers;
  if (list === "schedulers" && info.engine_schedulers?.length) return info.engine_schedulers;
  return info[list] || [];
};
const quote = (v) => (/[,"]/.test(v) ? `"${v.replace(/"/g, '\\"')}"` : v);

/** The job's `grid` field for a grid state, or null when off. */
export function gridJob(g) {
  if (!g?.enabled) return null;
  const axis = (a) => (a.type === "none" ? null : { type: a.type, values: a.values, target: a.target || "" });
  return { x: axis(g.axes[0]), y: axis(g.axes[1]), z: axis(g.axes[2]), include_cells: g.cells };
}

export class GridCard {
  /** opts: { state: () => gridState, persist, info: () => /info, loras: () => [{name}] } */
  constructor(el, opts) {
    this.el = el;
    this.o = opts;
    this.timer = 0;
    this.el.innerHTML = `
      <summary><input type="checkbox" class="g-on" title="Use the comparison grid"> <span>Comparison grid (X/Y/Z)</span> <span class="g-sum"></span></summary>
      ${["X", "Y", "Z"].map((n, i) => `
        <div class="g-axis" data-i="${i}">
          <span class="g-name">${n}</span>
          <select class="g-type">${AXES.map((a) => `<option value="${a.key}">${a.label}</option>`).join("")}</select>
          <select class="g-target" hidden></select>
          <input class="g-values" type="text" spellcheck="false">
          <button class="g-fill" title="Fill with every available option" hidden>All</button>
        </div>`).join("")}
      <div class="row g-foot"><label class="check"><input type="checkbox" class="g-cells"> Also keep each image</label><span class="g-count"></span></div>`;
    const g = () => this.o.state();
    const on = this.el.querySelector(".g-on");
    on.onclick = (e) => e.stopPropagation();
    on.onchange = () => { g().enabled = on.checked; if (on.checked) this.el.open = true; this.save(); };
    this.el.querySelector(".g-cells").onchange = (e) => { g().cells = e.target.checked; this.save(); };
    this.el.querySelectorAll(".g-axis").forEach((row) => {
      const ax = () => g().axes[Number(row.dataset.i)];
      row.querySelector(".g-type").onchange = (e) => { ax().type = e.target.value; ax().values = ""; this.render(); this.save(); };
      row.querySelector(".g-values").oninput = (e) => { ax().values = e.target.value; this.save(); };
      row.querySelector(".g-target").onchange = (e) => { ax().target = e.target.value; this.save(); };
      row.querySelector(".g-fill").onclick = () => {
        const def = AXES.find((a) => a.key === ax().type);
        ax().values = listValues(this.o.info() || {}, def.list).map(quote).join(", ");
        this.render();
        this.save();
      };
    });
    this.render();
  }

  save() {
    this.o.persist();
    this.renderSummary();
    clearTimeout(this.timer);
    this.timer = setTimeout(() => this.check(), 250);
  }

  render() {
    const g = this.o.state();
    this.el.querySelector(".g-on").checked = g.enabled;
    this.el.querySelector(".g-cells").checked = g.cells;
    if (g.enabled) this.el.open = true;
    this.el.querySelectorAll(".g-axis").forEach((row) => {
      const ax = g.axes[Number(row.dataset.i)];
      const def = AXES.find((a) => a.key === ax.type) || AXES[0];
      row.querySelector(".g-type").value = def.key;
      const vals = row.querySelector(".g-values");
      vals.value = ax.values;
      vals.disabled = def.key === "none";
      vals.placeholder = def.key === "none" ? "" : def.hint || "comma-separated, or All";
      row.querySelector(".g-fill").hidden = !def.list;
      const tgt = row.querySelector(".g-target");
      tgt.hidden = def.key !== "lora_weight";
      if (!tgt.hidden) {
        const names = (this.o.loras() || []).map((l) => l.name);
        if (!ax.target && names.length) ax.target = names[0];
        tgt.innerHTML = (names.length ? names : [ax.target || "(add a LoRA first)"]).map((n) => `<option>${escapeHtml(n)}</option>`).join("");
        tgt.value = ax.target;
      }
    });
    this.renderSummary();
    this.check();
  }

  renderSummary(text = null) {
    const g = this.o.state();
    this.el.querySelector(".g-sum").textContent = g.enabled ? text ?? "" : "";
  }

  /** The job's `grid` field, or null when off. */
  job() { return gridJob(this.o.state()); }

  async check() {
    const g = this.o.state();
    const count = this.el.querySelector(".g-count");
    if (!g.enabled) { count.textContent = ""; return; }
    const j = this.job();
    if (!j.x && !j.y && !j.z) { count.textContent = "Pick at least one axis"; count.className = "g-count err"; return; }
    try {
      const res = await api("/venno/v1/grid/check", j);
      if (!res.ok) { count.textContent = res.error; count.className = "g-count err"; this.renderSummary("(check the values)"); return; }
      const dims = res.axes.map((a) => a.values.length).join(" × ");
      count.textContent = `${dims} = ${res.cells} image${res.cells === 1 ? "" : "s"}`;
      count.className = "g-count";
      this.renderSummary(`(${dims})`);
    } catch (e) {
      count.textContent = e.message;
      count.className = "g-count err";
    }
  }
}
