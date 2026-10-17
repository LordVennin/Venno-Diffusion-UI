// "Use settings from image": read the generation settings stored in a PNG (A1111 / Forge / Venno) or
// A1111-style JPEG / WebP, and offer to apply them.

import { api, escapeHtml } from "./common.js";

async function toDataURL(source) {
  if (typeof source === "string" && source.startsWith("data:")) return source;
  const blob = source instanceof Blob ? source : await (await fetch(source)).blob();
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = () => reject(r.error);
    r.readAsDataURL(blob);
  });
}

/** {found, note, text, settings} for a File / Blob / URL / data URL. */
export async function readImageSettings(source) {
  try {
    return await api("/venno/v1/image/params", { image: await toDataURL(source) });
  } catch (e) {
    return { found: false, note: e.message };
  }
}

/** One-line description of loaded settings. */
export function describeSettings(st) {
  const bits = [];
  if (st.checkpoint) bits.push(st.checkpoint);
  if (st.sampler) bits.push(`${st.sampler}${st.scheduler ? ` / ${st.scheduler}` : ""}`);
  if (st.steps) bits.push(`${st.steps} steps`);
  if (st.cfg != null) bits.push(`CFG ${st.cfg}`);
  if (st.seed != null) bits.push(`seed ${st.seed}`);
  if (st.loras?.length) bits.push(`${st.loras.length} LoRA${st.loras.length > 1 ? "s" : ""}`);
  if (st.hires) bits.push("hires");
  return bits.join(" · ");
}

/** Prompts that carry their own quality / score tags shouldn't get Studio's quality prefix on top. */
export const hasQualityTags = (prompt = "") => /\b(masterpiece|best quality|good quality|score_\d)\b/i.test(prompt);

let actionTimer = 0;
/** A toast with a button (e.g. "This image has generation settings — Use them"). */
export function toastAction(message, label, onClick, { timeout = 12000 } = {}) {
  let el = document.getElementById("toast-action");
  if (!el) {
    el = document.createElement("div");
    el.id = "toast-action";
    el.className = "toast-action";
    document.body.appendChild(el);
  }
  el.innerHTML = `<span>${escapeHtml(message)}</span><button class="primary">${escapeHtml(label)}</button><button class="x" title="Dismiss">×</button>`;
  el.classList.add("on");
  const close = () => el.classList.remove("on");
  el.querySelector(".primary").onclick = () => { close(); onClick(); };
  el.querySelector(".x").onclick = close;
  clearTimeout(actionTimer);
  actionTimer = setTimeout(close, timeout);
}

/** Read settings from an image and, if any, offer to apply them via `apply(settings)`. */
export async function offerSettings(source, apply, { where = "" } = {}) {
  const res = await readImageSettings(source);
  if (!res.found) return false;
  toastAction(`This image has generation settings (${describeSettings(res.settings)})`, `Use them${where ? ` in ${where}` : ""}`, () => apply(res.settings));
  return true;
}
