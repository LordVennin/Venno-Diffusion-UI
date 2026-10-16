// LoRA chips: the job's LoRAs live in a list shown under the prompts instead of <lora:…> text.
// Click a chip's name to remove it; the number is its weight (−/+ step 0.05, type, or scroll over it).

import { escapeHtml } from "./common.js";

const LORA_TAG = /<lora:([^>:]+)(?::([^>:]*))?(?::[^>]*)?>/gi;
export const DEFAULT_WEIGHT = 0.8;

/** "a cat, <lora:x:0.6>, smile" → { text: "a cat, smile", loras: [{ name: "x", weight: 0.6 }] } */
export function extractLoraTags(text) {
  const loras = [];
  const out = text.replace(LORA_TAG, (_, name, w) => {
    const weight = w === undefined || w === "" || Number.isNaN(Number(w)) ? 1 : Number(w);
    loras.push({ name: name.trim(), weight });
    return "";
  });
  if (!loras.length) return { text, loras };
  return { text: out.replace(/\s*,(\s*,)+/g, ",").replace(/[ \t]{2,}/g, " ").replace(/^[\s,]+|[\s,]+$/g, ""), loras };
}

/** LoRAs a job should carry (weight 0 = off). */
export const jobLoras = (list) => list.filter((l) => l.weight !== 0).map(({ name, weight }) => ({ name, weight }));

export class LoraChips {
  /**
   * @param {HTMLElement} el container
   * @param {{ get: () => {name: string, weight: number}[], set: (list) => void }} store
   */
  constructor(el, store) {
    this.el = el;
    this.store = store;
    this.available = null; // Set of names, null = not known yet
    this.el.classList.add("lora-chips");
    this.el.addEventListener("click", (e) => {
      const chip = e.target.closest(".lora-chip");
      if (!chip || e.target.closest("input")) return;
      const step = e.target.closest("button.step");
      if (step) {  // − / + : 0.05 per click
        const input = chip.querySelector("input.w");
        const v = Math.round((Number(input.value) + Number(step.dataset.d)) * 100) / 100;
        input.value = v;
        this.setWeight(chip.dataset.name, v);
        return;
      }
      this.remove(chip.dataset.name);
    });
    this.el.addEventListener("change", (e) => {
      const input = e.target.closest("input.w");
      if (!input) return;
      const v = Number(input.value);
      this.setWeight(input.closest(".lora-chip").dataset.name, Number.isFinite(v) ? v : DEFAULT_WEIGHT);
    });
    this.el.addEventListener("wheel", (e) => {
      const input = e.target.closest("input.w");
      if (!input) return;
      e.preventDefault();
      const v = Math.round((Number(input.value) + (e.deltaY < 0 ? 0.05 : -0.05)) * 100) / 100;
      input.value = v;
      this.setWeight(input.closest(".lora-chip").dataset.name, v);
    }, { passive: false });
    this.render();
  }

  get list() { return this.store.get(); }

  setAvailable(names) {
    this.available = new Set(names);
    this.render();
  }

  add(name, weight = DEFAULT_WEIGHT) {
    const list = this.list.filter((l) => l.name !== name);
    list.push({ name, weight });
    this.store.set(list);
    this.render();
  }

  addMany(loras) {
    for (const l of loras) this.add(l.name, l.weight);
  }

  remove(name) {
    this.store.set(this.list.filter((l) => l.name !== name));
    this.render();
  }

  setWeight(name, weight) {
    this.store.set(this.list.map((l) => (l.name === name ? { ...l, weight } : l)));
    this.el.querySelector(`.lora-chip[data-name="${CSS.escape(name)}"]`)?.classList.toggle("off", weight === 0);
  }

  /** Moves <lora:…> tags typed or pasted into a textarea into chips. */
  watch(textarea, onChange = () => {}) {
    const pull = () => {
      const { text, loras } = extractLoraTags(textarea.value);
      if (!loras.length) return;
      textarea.value = text;
      this.addMany(loras);
      onChange(text);
    };
    textarea.addEventListener("input", pull);
    textarea.addEventListener("change", pull);
    pull();
    return pull;
  }

  render() {
    const list = this.list;
    this.el.hidden = !list.length;
    this.el.innerHTML = list.map((l) => {
      const missing = this.available && !this.available.has(l.name);
      const title = missing ? `${l.name} is not in your LoRA folder — click to remove` : `${l.name} — click to remove`;
      return `<span class="lora-chip${missing ? " missing" : ""}${l.weight === 0 ? " off" : ""}" data-name="${escapeHtml(l.name)}" title="${escapeHtml(title)}">`
        + `<span class="n">${escapeHtml(l.name)}</span>`
        + `<button class="step" data-d="-0.05" title="−0.05">−</button>`
        + `<input class="w" type="number" step="0.05" value="${l.weight}" title="Weight (scroll to adjust, 0 = off)">`
        + `<button class="step" data-d="0.05" title="+0.05">+</button>`
        + `<span class="x" aria-hidden="true">×</span></span>`;
    }).join("");
  }
}
