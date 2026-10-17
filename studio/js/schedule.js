// Prompt preview: when a prompt uses scheduling ([a:b:0.5], [a|b]), AND, or the "then switch to" box,
// a small line under it says what the model gets at each step. Uses the server's parser (same as generation).

import { api, escapeHtml } from "./common.js";

const MAX_SEGMENTS = 6;

/** Steps actually sampled at a given strength (same rule as the engine). */
export const runSteps = (steps, strength = 1) => (strength >= 1 ? steps : Math.max(1, Math.min(steps, Math.floor(steps * strength + 0.5))));

function commonPrefix(texts) {
  let p = texts[0] || "";
  for (const t of texts) while (!t.startsWith(p)) p = p.slice(0, -1);
  return p;
}

/** Only the part of each text that changes between segments, trimmed to whole words. */
function changingParts(texts) {
  if (texts.length < 2) return texts;
  let pre = commonPrefix(texts);
  pre = pre.slice(0, Math.max(0, pre.search(/[\s,][^\s,]*$/) + 1)); // back to a word boundary
  const rev = texts.map((t) => [...t.slice(pre.length)].reverse().join(""));
  let suf = [...commonPrefix(rev)].reverse().join("");
  const cut = suf.search(/[\s,]/);
  suf = cut < 0 ? "" : suf.slice(cut);
  return texts.map((t) => {
    const mid = t.slice(pre.length, t.length - suf.length).trim().replace(/^,\s*|,\s*$/g, "");
    return (pre.trim() ? "…" : "") + (mid || "∅") + (suf.trim() ? "…" : "");
  });
}

const short = (t, n = 70) => (t.length > n ? t.slice(0, n - 1) + "…" : t);

export class SchedulePreview {
  /**
   * @param {HTMLTextAreaElement} textarea  the prompt box (preview goes right after it unless `el` is given)
   * @param {{ getSteps: () => number, getSwitch?: () => ({ prompt: string, at: number } | null), allowAnd?: boolean, el?: HTMLElement }} opts
   */
  constructor(textarea, { getSteps, getSwitch = () => null, allowAnd = true, el = null }) {
    this.ta = textarea;
    this.getSteps = getSteps;
    this.getSwitch = getSwitch;
    this.allowAnd = allowAnd;
    this.el = el || document.createElement("div");
    this.el.classList.add("prompt-preview");
    this.el.hidden = true;
    if (!el) textarea.insertAdjacentElement("afterend", this.el);
    this.timer = 0;
    this.seq = 0;
    textarea.addEventListener("input", () => this.refresh());
    this.refresh(0);
  }

  refresh(delay = 250) {
    clearTimeout(this.timer);
    this.timer = setTimeout(() => this.update(), delay);
  }

  async update() {
    const prompt = this.ta.value;
    const sw = this.getSwitch();
    const quick = /[[\]|]|\bAND\b/.test(prompt) || (sw && sw.prompt.trim());
    if (!quick) { this.el.hidden = true; return; }
    const seq = ++this.seq;
    let res;
    try {
      res = await api("/venno/v1/prompt/schedule", {
        prompt, switch_prompt: sw ? sw.prompt : "", switch_at: sw ? sw.at : 0.4, steps: Math.max(1, this.getSteps() | 0), allow_and: this.allowAnd,
      });
    } catch { this.el.hidden = true; return; }
    if (seq !== this.seq) return; // a newer edit is on its way
    this.render(res);
  }

  render({ segments, changes, constructs = [] }) {
    if (!changes) { this.el.hidden = true; return; }
    this.el.title = segments.map((s) => `Steps ${s.from}–${s.to}: ${s.parts.map((p) => p.text.trim() + (p.weight !== 1 ? ` (×${p.weight})` : "")).join("  AND  ")}`).join("\n");
    if (constructs.some((c) => c.kind === "alternate")) {
      // alternation makes one segment per step: describe the constructs instead
      const q = (t) => escapeHtml(short(t || "∅", 30));
      const items = constructs.map((c) => {
        if (c.kind === "alternate") return `<span class="seg"><b>each step</b> ${c.options.map(q).join(" ⇄ ")}</span>`;
        if (!c.from) return `<span class="seg"><b>from ${c.after + 1}</b> + ${q(c.to)}</span>`;
        if (!c.to) return `<span class="seg"><b>until ${c.after}</b> ${q(c.from)}</span>`;
        return `<span class="seg"><b>after ${c.after}</b> ${q(c.from)} → ${q(c.to)}</span>`;
      });
      const sw = this.getSwitch();
      if (sw) items.push(`<span class="seg"><b>after ${Math.floor(sw.at * Math.max(1, this.getSteps() | 0))}</b> switch prompt</span>`);
      this.el.innerHTML = `<span class="lbl">Steps</span>${items.join("")}`;
      this.el.hidden = false;
      return;
    }
    const composed = segments.some((s) => s.parts.length > 1 || s.parts[0].weight !== 1);
    const labels = composed
      ? segments.map((s) => s.parts.map((p) => `${short(p.text.trim(), 40)}${p.weight !== 1 ? ` ×${p.weight}` : ""}`).join(" + "))
      : changingParts(segments.map((s) => s.parts[0].text)).map((t) => short(t));
    const shown = segments.slice(0, MAX_SEGMENTS).map((s, i) => {
      const range = s.from === s.to ? `${s.from}` : `${s.from}–${s.to}`;
      return `<span class="seg"><b>${range}</b> ${escapeHtml(labels[i])}</span>`;
    });
    if (segments.length > MAX_SEGMENTS) shown.push(`<span class="more">+${segments.length - MAX_SEGMENTS} more changes</span>`);
    this.el.innerHTML = `<span class="lbl">Steps</span>${shown.join("")}`;
    this.el.hidden = false;
  }
}
